"""
Self-play training pipeline orchestrator.

Drives the AlphaZero-style loop: simulate → train → benchmark → repeat.
Each iteration is called a "generation". Generation 0 uses local greedy PUCT priors
(no NN prior); subsequent generations use the previous generation's model
as the MCTS prior evaluator.

The pipeline supports **resume-on-interrupt**: if the process is stopped
mid-run, restarting it will automatically detect which generation and step
to resume from by scanning artifact directories.

Usage:
    python -m gimbur_nn.pipeline --config pipeline.json
    python -m gimbur_nn.pipeline --config pipeline.json --start-gen 3
    python -m gimbur_nn.pipeline --config pipeline.json --chart-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulateConfig:
    """Parameters for the ``gimbur simulate`` step."""

    games: int = 1000
    players: int = 2
    search_time_ms: int = 500
    max_simulations: int = 200
    max_rollout_depth: int = 500
    action_rollout_limit: int | None = None
    symmetries: bool = True
    verbosity: str = "quiet"
    oversample: float = 1.0
    parallelism: int | None = None
    max_pending_evaluations: int = 32
    leaf_evaluation_timeout_ms: int = 500
    drain_timeout_ms: int = 1000
    max_errors_per_game: int = 5
    max_error_rate_per_game: float = 0.02
    minimum_requests_for_rate: int = 50
    discard_games_with_fallbacks: bool = False
    max_discarded_games: int = 20
    max_discard_rate: float = 0.05
    minimum_attempts_for_discard_rate: int = 50
    max_consecutive_discards: int = 5
    greedy_prior: bool = True
    greedy_prior_uniform_mix: float = 0.25


@dataclass
class TrainConfig:
    """Parameters for ``python -m gimbur_nn.train``."""

    epochs: int = 0
    patience: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    val_split: float = 0.1
    test_split: float = 0.0
    log_interval: int = 50
    checkpoint_dir: bool = True
    checkpoint_retention: int = 2
    resume_from_previous: bool = True
    replay_generations: int = 3
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    mcts_value_weight_start: float = 0.9
    mcts_value_weight_end: float = 0.1
    victory_point_sampling_statistic: str = "median"
    victory_point_sampling_upper_percentage: float = 0.10


@dataclass
class ServeConfig:
    """Parameters for ``python -m gimbur_nn.serve``."""

    port: int = 8000
    host: str = "127.0.0.1"
    log_level: str = "warning"
    batch_window_ms: float = 0.0
    compile_model: bool = False


@dataclass
class InferenceConfig:
    export_fp16: bool = False
    simulation_precision: str = "fp32"
    benchmark_precisions: list[str] = field(default_factory=lambda: ["fp32"])
    promotion_precision: str = "fp32"


@dataclass
class MonitoringConfig:
    enabled: bool = False
    interval_seconds: float = 120.0
    output_dir: str | None = None


@dataclass
class GimburServerConfig:
    """Parameters for the C# Gimbur.Server (MCTS game server)."""

    port: int = 5123
    host: str = "127.0.0.1"
    dotnet_project: str = "src/Gimbur.Server"


@dataclass
class BenchmarkConfig:
    """A single benchmark run.

    Benchmarks run after each complete-model training generation.
    """

    name: str = "nn-vs-greedy"
    games: int = 10000
    ai: list[str] = field(default_factory=lambda: ["nn", "greedy"])
    search_time_ms: int | None = None
    server_prior_mode: str | None = None
    parallelism: int | None = None
    progress_interval: int = 10


@dataclass
class PromotionGateConfig:
    """One direct or MCTS promotion gate."""

    enabled: bool = True
    compare_with_greedy: bool = True
    games: int = 10_000
    ai: str = "nn-placement-state"
    minimum_improvement_vs_greedy: float = 0.0
    minimum_improvement_vs_champion: float = 0.0
    minimum_score_vs_greedy: float | None = None
    minimum_score_vs_champion: float | None = None
    search_time_ms: int | None = None
    parallelism: int | None = None
    progress_interval: int = 10


@dataclass
class PromotionConfig:
    """Champion/challenger promotion and failed-gate retry policy."""

    enabled: bool = False
    additional_training_games: int = 500
    max_retries: int = 2
    direct: PromotionGateConfig = field(default_factory=PromotionGateConfig)
    hybrid: PromotionGateConfig = field(
        default_factory=lambda: PromotionGateConfig(
            games=1_000,
            ai="nn-mcts-placement-state",
            search_time_ms=1_000,
        )
    )


@dataclass
class BaselineBenchmarkConfig:
    """A reference benchmark used as a horizontal line on the progress chart.

    Unlike :class:`BenchmarkConfig`, baselines are run once (not per
    generation) and cached under ``{results_dir}/baselines/{name}.json``.
    They typically pit a non-NN strategy (e.g. ``mcts``, ``server-mcts``)
    against the same opponent as one of the per-generation benchmarks, so
    the chart shows how the trained NN compares to a no-prior baseline.
    """

    name: str = "mcts-vs-greedy"
    games: int = 200
    ai: list[str] = field(default_factory=lambda: ["mcts", "greedy"])
    search_time_ms: int | None = None
    progress_interval: int = 10
    server_prior_mode: str | None = None


@dataclass
class PipelineConfig:
    """Top-level orchestrator configuration."""

    # Shared identifiers.
    map_config: str = "mini"
    game_config: str = "mini_2p"
    model_config: str = "small"

    # Reproducibility.
    seed: int | None = None

    # Directories (relative to project root).
    data_dir: str = "pipeline/data"
    model_dir: str = "pipeline/models"
    results_dir: str = "pipeline/results"

    # How many generations to run.
    generations: int = 10

    # Optional larger bootstrap dataset for generation 0.
    gen0_games: int | None = None
    gen0_milestones: list[int] = field(default_factory=list)

    # Skip generations before this threshold (treat them as complete).
    skip_until_gen: int | None = None

    # Paths to the CLI tools (relative to project root).
    dotnet_project: str = "src/Gimbur.Cli"
    python_module: str = "gimbur_nn"

    # Section configs.
    simulate: SimulateConfig = field(default_factory=SimulateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    gimbur_server: GimburServerConfig = field(default_factory=GimburServerConfig)
    benchmarks: list[BenchmarkConfig] = field(
        default_factory=lambda: [
            BenchmarkConfig(name="nn-vs-greedy", games=10000, ai=["nn", "greedy"]),
            BenchmarkConfig(name="nn-vs-random", games=10000, ai=["nn", "random"]),
        ]
    )
    baselines: list[BaselineBenchmarkConfig] = field(default_factory=list)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    config_path: Path | None = field(default=None, repr=False, compare=False)


def _strip_json_comments(text: str) -> str:
    """Remove single-line // comments from JSON text (outside strings)."""

    # Remove // comments that are not inside strings.
    # This is a simple heuristic: split by lines and strip trailing comments.
    lines = []
    for line in text.splitlines():
        # Find // outside of quoted strings by tracking quote state.
        in_string = False
        escape = False
        for i, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
            elif ch == "/" and not in_string and i + 1 < len(line) and line[i + 1] == "/":
                line = line[:i].rstrip()
                break
        lines.append(line)
    return "\n".join(lines)


def _load_config(path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a JSON file. Supports // comments."""
    text = _strip_json_comments(path.read_text())
    raw = json.loads(text)
    cfg = PipelineConfig(config_path=path.resolve())

    # Top-level scalars.
    for attr in (
        "map_config",
        "game_config",
        "model_config",
        "seed",
        "data_dir",
        "model_dir",
        "results_dir",
        "generations",
        "gen0_games",
        "gen0_milestones",
        "dotnet_project",
        "python_module",
        "skip_until_gen",
    ):
        json_key = _to_camel(attr)
        if json_key in raw:
            setattr(cfg, attr, raw[json_key])

    # Section configs.
    if "simulate" in raw:
        cfg.simulate = _load_section(SimulateConfig, raw["simulate"])
    if "train" in raw:
        cfg.train = _load_section(TrainConfig, raw["train"])
    if "serve" in raw:
        cfg.serve = _load_section(ServeConfig, raw["serve"])
    if "inference" in raw:
        cfg.inference = _load_section(InferenceConfig, raw["inference"])
    if "monitoring" in raw:
        cfg.monitoring = _load_section(MonitoringConfig, raw["monitoring"])
    if "gimburServer" in raw:
        cfg.gimbur_server = _load_section(GimburServerConfig, raw["gimburServer"])
    if "benchmarks" in raw:
        cfg.benchmarks = [_load_section(BenchmarkConfig, b) for b in raw["benchmarks"]]
    if "baselines" in raw:
        cfg.baselines = [_load_section(BaselineBenchmarkConfig, b) for b in raw["baselines"]]
    if "promotion" in raw:
        promotion_raw = raw["promotion"]
        scalar_raw = {
            key: value for key, value in promotion_raw.items() if key not in ("direct", "hybrid")
        }
        cfg.promotion = _load_section(PromotionConfig, scalar_raw)
        if "direct" in promotion_raw:
            cfg.promotion.direct = _load_section(PromotionGateConfig, promotion_raw["direct"])
        if "hybrid" in promotion_raw:
            cfg.promotion.hybrid = _load_section(PromotionGateConfig, promotion_raw["hybrid"])

    _validate_promotion_config(cfg)
    precisions = [cfg.inference.simulation_precision, cfg.inference.promotion_precision]
    precisions.extend(cfg.inference.benchmark_precisions)
    if not cfg.inference.benchmark_precisions or any(p not in ("fp32", "fp16") for p in precisions):
        raise ValueError("Inference precisions must be fp32 or fp16, with benchmarks non-empty.")
    if len(set(cfg.inference.benchmark_precisions)) != len(cfg.inference.benchmark_precisions):
        raise ValueError("benchmarkPrecisions must be unique.")
    if "fp16" in precisions and not cfg.inference.export_fp16:
        raise ValueError("FP16 inference requires inference.exportFp16=true.")
    if cfg.monitoring.interval_seconds <= 0:
        raise ValueError("monitoring.intervalSeconds must be positive.")
    if cfg.gen0_milestones:
        if cfg.gen0_milestones != sorted(set(cfg.gen0_milestones)):
            raise ValueError("gen0Milestones must be strictly increasing and unique.")
        if any(games <= 0 for games in cfg.gen0_milestones):
            raise ValueError("gen0Milestones values must be positive.")
        cfg.gen0_games = cfg.gen0_milestones[-1]
    return cfg


def _reload_config(cfg: PipelineConfig) -> PipelineConfig:
    """Refresh *cfg* in place from its source file before a pipeline step."""
    if cfg.config_path is None:
        return cfg
    refreshed = _load_config(cfg.config_path)
    for config_field in fields(PipelineConfig):
        setattr(cfg, config_field.name, getattr(refreshed, config_field.name))
    return cfg


def _validate_promotion_config(cfg: PipelineConfig) -> None:
    promotion = cfg.promotion
    if not promotion.enabled:
        return
    if promotion.additional_training_games < 0 or promotion.max_retries < 0:
        raise ValueError("Promotion additional games and retry count must be non-negative.")
    if cfg.serve.port >= 65535:
        raise ValueError("Promotion requires serve.port + 1 for the champion server.")
    for name, gate in (("direct", promotion.direct), ("hybrid", promotion.hybrid)):
        if gate.enabled and gate.games <= 0:
            raise ValueError(f"Promotion {name} gate games must be positive.")
        if gate.minimum_improvement_vs_greedy < -0.5:
            raise ValueError(f"Promotion {name} greedy improvement is below -0.5.")
        if gate.minimum_improvement_vs_champion < -0.5:
            raise ValueError(f"Promotion {name} champion improvement is below -0.5.")
        for opponent, score in (
            ("greedy", gate.minimum_score_vs_greedy),
            ("champion", gate.minimum_score_vs_champion),
        ):
            if score is not None and not 0 <= score <= 1:
                raise ValueError(
                    f"Promotion {name} minimum score versus {opponent} must be in [0, 1]."
                )


def _to_camel(snake: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _load_section(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a camelCase JSON dict.

    Warns on unrecognised keys so config typos don't silently use defaults.
    """
    known_keys = {_to_camel(f) for f in cls.__dataclass_fields__}
    unknown = set(data.keys()) - known_keys
    if unknown:
        import warnings

        warnings.warn(
            f"Unknown key(s) in {cls.__name__} config: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known_keys))}",
            stacklevel=2,
        )

    kwargs: dict[str, Any] = {}
    for f_name in cls.__dataclass_fields__:
        json_key = _to_camel(f_name)
        if json_key in data:
            kwargs[f_name] = data[json_key]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _data_path(cfg: PipelineConfig, gen: int) -> Path:
    return Path(cfg.data_dir) / f"gen{gen}"


def _model_path(cfg: PipelineConfig, gen: int) -> Path:
    return Path(cfg.model_dir) / f"gen{gen}.pt"


def _precision_model_path(model: Path, precision: str) -> Path:
    return model if precision == "fp32" else model.with_name(f"{model.stem}.fp16{model.suffix}")


def _ensure_inference_artifacts(cfg: PipelineConfig, model: Path, project_root: Path) -> None:
    if not cfg.inference.export_fp16 or not model.is_file():
        return
    destination = _precision_model_path(model, "fp16")
    if destination.is_file() and destination.stat().st_mtime_ns >= model.stat().st_mtime_ns:
        return
    _run(
        [
            sys.executable,
            "-m",
            f"{cfg.python_module}.quantize",
            "--source",
            str(model),
            "--out",
            str(destination),
        ],
        label="Export FP16 inference model",
        cwd=project_root,
    )


def _results_path(cfg: PipelineConfig, gen: int, name: str) -> Path:
    return Path(cfg.results_dir) / f"gen{gen}" / f"{name}.json"


def _bootstrap_model_path(cfg: PipelineConfig, games: int) -> Path:
    return Path(cfg.model_dir) / "bootstrap" / str(games) / "model.pt"


def _bootstrap_checkpoint_path(cfg: PipelineConfig, games: int) -> Path:
    return Path(cfg.model_dir) / "bootstrap" / str(games) / "checkpoints"


def _bootstrap_result_path(cfg: PipelineConfig, games: int, name: str) -> Path:
    return Path(cfg.results_dir) / "bootstrap" / str(games) / f"{name}.json"


def _candidate_model_path(cfg: PipelineConfig, gen: int, attempt: int) -> Path:
    return Path(cfg.model_dir) / "candidates" / f"gen{gen}" / f"attempt{attempt}" / "model.pt"


def _champion_model_path(cfg: PipelineConfig, gen: int) -> Path:
    return Path(cfg.model_dir) / "champions" / f"gen{gen}" / "model.pt"


def _champion_manifest_path(cfg: PipelineConfig) -> Path:
    return Path(cfg.model_dir) / "champion.json"


def _promotion_attempt_path(cfg: PipelineConfig, gen: int, attempt: int) -> Path:
    return Path(cfg.results_dir) / "promotion" / f"gen{gen}" / f"attempt{attempt}"


def _promotion_generation_path(cfg: PipelineConfig, gen: int) -> Path:
    return Path(cfg.results_dir) / "promotion" / f"gen{gen}" / "generation.json"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_champion(cfg: PipelineConfig) -> dict[str, Any] | None:
    path = _champion_manifest_path(cfg)
    return json.loads(path.read_text()) if path.is_file() else None


def _baseline_path(cfg: PipelineConfig, name: str) -> Path:
    """Path for a cached baseline benchmark result (run once, not per-gen)."""
    return Path(cfg.results_dir) / "baselines" / f"{name}.json"


def _checkpoint_path(cfg: PipelineConfig, gen: int) -> Path:
    return Path(cfg.model_dir) / f"gen{gen}_checkpoints"


def _config_dir(cfg: PipelineConfig) -> Path:
    """Directory for generated subprocess config files."""
    return Path(cfg.model_dir) / ".configs"


def _write_config(cfg: PipelineConfig, name: str, data: dict[str, Any]) -> Path:
    """Write a JSON config file and return its path."""
    config_dir = _config_dir(cfg)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------


def _simulation_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if the generation's data directory has enough game files.

    When ``cfg.skip_until_gen`` is set and *gen* is below that
    threshold, the generation is considered complete (i.e. skipped).
    """
    if cfg.skip_until_gen is not None and gen < cfg.skip_until_gen:
        return True
    sim = cfg.simulate
    data_dir = _data_path(cfg, gen)
    if not data_dir.is_dir():
        return False
    return _count_json_files(data_dir) >= sim.games


def _training_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if the generation's model checkpoint exists."""
    return _model_path(cfg, gen).is_file()


def _benchmark_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if all applicable benchmark result files exist.

    When *phase* is given, only benchmarks matching that phase are
    checked.  Otherwise all benchmarks are checked.
    """
    return all(
        _results_path(cfg, gen, _precision_benchmark_name(cfg, bench.name, precision)).is_file()
        for bench in cfg.benchmarks
        for precision in cfg.inference.benchmark_precisions
    )


def _precision_benchmark_name(cfg: PipelineConfig, name: str, precision: str) -> str:
    return name if cfg.inference.benchmark_precisions == ["fp32"] else f"{name}-{precision}"


def _generation_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if simulate, train, and benchmark are all done for a generation."""
    if cfg.promotion.enabled:
        decision_path = _promotion_generation_path(cfg, gen)
        if not decision_path.is_file():
            return False
        decision = json.loads(decision_path.read_text())
        return decision.get("status") == "rejected" or _benchmark_complete(cfg, gen)
    return (
        _simulation_complete(cfg, gen)
        and _training_complete(cfg, gen)
        and _benchmark_complete(cfg, gen)
    )


def _detect_resume_gen(cfg: PipelineConfig) -> int:
    """Scan artifacts to find the first incomplete generation.

    Returns the generation number to resume from.  If all configured
    generations are complete, returns ``cfg.generations`` (i.e. nothing
    left to do).
    """
    for gen in range(cfg.generations):
        if not _generation_complete(cfg, gen):
            return gen
    return cfg.generations


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _run(
    args: list[str],
    *,
    label: str,
    cwd: Path | None = None,
    monitor_cfg: PipelineConfig | None = None,
    inference_url: str | None = None,
) -> None:
    """Run a command, streaming stdout and capturing stderr. Raises on non-zero exit."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(args)}")
    print(f"{'=' * 60}\n", flush=True)

    monitor = _MonitorProcess(monitor_cfg, label, inference_url) if monitor_cfg else None
    try:
        result = subprocess.run(args, cwd=cwd, stderr=subprocess.PIPE, text=True)
    finally:
        if monitor:
            monitor.stop()
    if result.returncode != 0:
        stderr_msg = result.stderr.strip() if result.stderr else ""
        detail = f"{label} failed with exit code {result.returncode}"
        if stderr_msg:
            detail += f"\nstderr:\n{stderr_msg}"
        raise RuntimeError(detail)


def _build_serve_config(
    *,
    serve_cfg: ServeConfig,
    game_config: str,
    model_path: Path,
    model_config: str,
) -> dict[str, Any]:
    return {
        "port": serve_cfg.port,
        "host": serve_cfg.host,
        "logLevel": serve_cfg.log_level,
        "gameConfig": game_config,
        "model": str(model_path),
        "modelConfig": model_config,
        "batchWindowMs": serve_cfg.batch_window_ms,
        "compileModel": serve_cfg.compile_model,
    }


class _ServerProcess:
    """Manages the lifecycle of the inference server subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def start(
        self,
        *,
        serve_cfg: ServeConfig,
        game_config: str,
        python_module: str,
        model_path: Path,
        model_config: str,
        pipeline_cfg: PipelineConfig | None = None,
        cwd: Path | None = None,
    ) -> None:
        if self._proc is not None:
            self.stop()

        # Fail fast if the port is already occupied by another process.
        self._check_port_available(serve_cfg.host, serve_cfg.port)

        # Build serve config JSON.
        serve_config = _build_serve_config(
            serve_cfg=serve_cfg,
            game_config=game_config,
            model_path=model_path,
            model_config=model_config,
        )

        if pipeline_cfg is not None:
            config_path = _write_config(pipeline_cfg, "serve", serve_config)
            args = [
                sys.executable,
                "-m",
                f"{python_module}.serve",
                "--config",
                str(config_path),
            ]
        else:
            # Build CLI args directly.
            args = [
                sys.executable,
                "-m",
                f"{python_module}.serve",
                "--game-config",
                game_config,
                "--port",
                str(serve_cfg.port),
                "--host",
                serve_cfg.host,
                "--log-level",
                serve_cfg.log_level,
                "--batch-window-ms",
                str(serve_cfg.batch_window_ms),
            ]
            if serve_cfg.compile_model:
                args.append("--compile-model")
            args.extend(["--model", str(model_path), "--model-config", model_config])

        print(f"\n--- Starting inference server: {' '.join(args)}")
        self._proc = subprocess.Popen(args, cwd=cwd)

        # Wait for health check.
        url = f"http://{serve_cfg.host}:{serve_cfg.port}/health"
        try:
            self._wait_for_health(url, timeout=300 if serve_cfg.compile_model else 60)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._proc is None:
            return
        print("--- Stopping inference server...")
        self._proc.send_signal(signal.SIGINT)
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("--- Server did not exit, killing...")
            self._proc.kill()
            self._proc.wait()
        self._proc = None


    @staticmethod
    def _check_port_available(host: str, port: int) -> None:
        """Ensure no other process is listening on host:port.

        If the port is occupied (typically by a leftover inference server
        or Gimbur.Server from a previous pipeline run that was killed
        before its cleanup ran), attempt to identify and terminate the
        offending process automatically. Raises only if the port is still
        in use after the cleanup attempt.
        """
        import socket

        def _is_in_use() -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                return sock.connect_ex((host, port)) == 0

        if not _is_in_use():
            return

        # Try to find and kill the process holding the port.
        pids = _ServerProcess._pids_listening_on(port)
        if pids:
            print(
                f"--- Port {port} on {host} is in use by PID(s) {pids}; "
                f"sending SIGTERM (leftover from a previous run)."
            )
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    print(f"--- WARN: no permission to kill PID {pid}.")
            # Give them a chance to exit gracefully, then escalate.
            for _ in range(20):  # up to ~5s
                time.sleep(0.25)
                if not _is_in_use():
                    print(f"--- Port {port} freed.")
                    return
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for _ in range(8):  # up to ~2s
                time.sleep(0.25)
                if not _is_in_use():
                    print(f"--- Port {port} freed (SIGKILL).")
                    return

        raise RuntimeError(
            f"Port {port} on {host} is already in use and could not be "
            f"freed automatically. Kill the owning process and retry."
        )

    @staticmethod
    def _pids_listening_on(port: int) -> list[int]:
        """Return PIDs listening on TCP ``port`` using ss/lsof/fuser.

        Returns an empty list if no tool is available or no process is
        found.
        """
        # Try `ss` first (usually available on modern Linux distros).
        try:
            out = subprocess.run(
                ["ss", "-lptn", f"sport = :{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids: set[int] = set()
            for line in out.stdout.splitlines():
                # Lines look like: ... users:(("python",pid=12345,fd=7))
                for token in line.split("pid=")[1:]:
                    digits = ""
                    for ch in token:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        pids.add(int(digits))
            if pids:
                return sorted(pids)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fall back to `lsof`.
        try:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return sorted({int(s) for s in out.stdout.split() if s.isdigit()})
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fall back to `fuser`.
        try:
            out = subprocess.run(
                ["fuser", "-n", "tcp", str(port)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return sorted({int(s) for s in out.stdout.split() if s.isdigit()})
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return []

    def _wait_for_health(self, url: str, timeout: float) -> None:
        """Poll the /health endpoint until the server is ready."""
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # If the server process has exited, stop waiting immediately.
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"Inference server exited with code {self._proc.returncode} "
                    f"before becoming healthy."
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"--- Server healthy at {url}")
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Inference server did not become healthy within {timeout}s")


class _MonitorProcess:
    def __init__(self, cfg: PipelineConfig, step: str, inference_url: str | None = None) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        if not cfg.monitoring.enabled:
            return
        output_dir = Path(cfg.monitoring.output_dir or Path(cfg.results_dir) / "monitoring")
        safe_step = "".join(
            character if character.isalnum() else "-" for character in step
        ).strip("-")
        output = output_dir / f"{safe_step}.csv"
        args = [
            sys.executable,
            "-m",
            f"{cfg.python_module}.monitor",
            "--output",
            str(output),
            "--interval-seconds",
            str(cfg.monitoring.interval_seconds),
            "--step",
            step,
        ]
        if inference_url:
            args.extend(["--inference-url", inference_url])
        self._proc = subprocess.Popen(args)

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def __enter__(self) -> _MonitorProcess:
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


class _GimburServerProcess:
    """Manages the lifecycle of the C# Gimbur.Server subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def start(
        self,
        *,
        gimbur_server_cfg: GimburServerConfig,
        cwd: Path | None = None,
    ) -> None:
        if self._proc is not None:
            self.stop()

        host = gimbur_server_cfg.host
        port = gimbur_server_cfg.port
        _ServerProcess._check_port_available(host, port)

        args = [
            "dotnet",
            "run",
            "-c",
            "Release",
            "--project",
            gimbur_server_cfg.dotnet_project,
            "--",
            "--urls",
            f"http://{host}:{port}",
        ]
        print(f"\n--- Starting Gimbur.Server: {' '.join(args)}")
        self._proc = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for health check.
        url = f"http://{host}:{port}/health"
        self._wait_for_health(url, timeout=60)

    def stop(self) -> None:
        if self._proc is None:
            return
        print("--- Stopping Gimbur.Server...")
        # Send SIGINT to the entire process group so that both the
        # 'dotnet run' wrapper and the child application process exit.
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("--- Gimbur.Server did not exit, killing process group...")
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                self._proc.kill()
            self._proc.wait()
        self._proc = None

    def _wait_for_health(self, url: str, timeout: float) -> None:
        """Poll the /health endpoint until the server is ready."""
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"Gimbur.Server exited with code {self._proc.returncode} "
                    f"before becoming healthy."
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"--- Gimbur.Server healthy at {url}")
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Gimbur.Server did not become healthy within {timeout}s")


_SERVER_AI_KINDS = frozenset(
    {
        "server-mcts",
        "server-mcts-nn",
        "nn-mcts-placement",
        "nn-mcts-placement-random",
        "nn-mcts-placement-state",
        "nn-mcts-state",
        "mcts-placement",
        "mcts-placement-random",
    }
)


def _benchmarks_need_game_server(benchmarks: list[BenchmarkConfig]) -> bool:
    """Return True if any benchmark uses server-mcts or server-mcts-nn AI kinds."""
    return any(ai in _SERVER_AI_KINDS for bench in benchmarks for ai in bench.ai)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _step_simulate(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    nn_url: str | None,
    sim_override: SimulateConfig | None = None,
    target_games_override: int | None = None,
    seed_offset: int = 0,
    config_suffix: str = "",
) -> None:
    """Run self-play simulation for a generation.

    Uses ``--export-format json`` so each game is written to its own
    ``.json`` file in the generation data directory.  When
    ``oversample > 1.0``, requests more games than needed and monitors
    the output folder, terminating the CLI once the target game count
    is reached.  This avoids blocking on long-tail slow games.

    On resume, counts existing ``.json`` files and only requests the
    remaining games needed to reach the target.
    """
    _reload_config(cfg)
    out_dir = _data_path(cfg, gen)
    out_dir.mkdir(parents=True, exist_ok=True)
    simulation_precision = (
        cfg.inference.simulation_precision if nn_url is not None else "non-neural"
    )
    precision_marker = out_dir / ".inference-precision"
    existing = _count_json_files(out_dir)
    if precision_marker.is_file():
        recorded_precision = precision_marker.read_text().strip()
        if recorded_precision != simulation_precision:
            raise RuntimeError(
                f"Simulation data at {out_dir} uses {recorded_precision}, "
                f"but configuration requests {simulation_precision}."
            )
    elif existing > 0 and simulation_precision != "non-neural":
        raise RuntimeError(
            f"Simulation data at {out_dir} predates inference precision tracking; "
            "archive it or add a matching .inference-precision marker before resuming."
        )
    else:
        precision_marker.write_text(simulation_precision + "\n")

    # Skip if this generation is below the skip threshold.
    if cfg.skip_until_gen is not None and gen < cfg.skip_until_gen:
        print(f"  Simulate: gen {gen} < skipUntilGen {cfg.skip_until_gen}, skipping.")
        return

    sim = sim_override or cfg.simulate
    configured_games = cfg.gen0_games if gen == 0 and cfg.gen0_games is not None else sim.games
    target_games = target_games_override or configured_games
    if existing >= target_games:
        print(f"  Simulate: {existing}/{target_games} games already exist, skipping.")
        return

    remaining = target_games - existing
    requested_games = math.ceil(remaining * max(sim.oversample, 1.0))
    existing_discard_reason = _discard_stop_reason(out_dir, sim)
    if existing_discard_reason is not None:
        raise RuntimeError(f"Simulation discard policy already exceeded: {existing_discard_reason}")

    # Build config JSON for gimbur simulate.
    sim_config: dict[str, Any] = {
        "games": requested_games,
        "players": sim.players,
        "mapConfig": cfg.map_config,
        "export": str(out_dir),
        "exportFormat": "json",
        "searchTimeMs": sim.search_time_ms,
        "maxSimulations": sim.max_simulations,
        "maxRolloutDepth": sim.max_rollout_depth,
    }
    if sim.action_rollout_limit is not None:
        sim_config["actionRolloutLimit"] = sim.action_rollout_limit
    if sim.parallelism is not None:
        sim_config["parallelism"] = sim.parallelism
    sim_config["maxPendingEvaluations"] = sim.max_pending_evaluations
    sim_config["leafEvaluationTimeoutMs"] = sim.leaf_evaluation_timeout_ms
    sim_config["drainTimeoutMs"] = sim.drain_timeout_ms
    sim_config["maxErrorsPerGame"] = sim.max_errors_per_game
    sim_config["maxErrorRatePerGame"] = sim.max_error_rate_per_game
    sim_config["minimumRequestsForRate"] = sim.minimum_requests_for_rate
    sim_config["discardGamesWithFallbacks"] = sim.discard_games_with_fallbacks
    sim_config["maxDiscardedGames"] = sim.max_discarded_games
    sim_config["maxDiscardRate"] = sim.max_discard_rate
    sim_config["minimumAttemptsForDiscardRate"] = sim.minimum_attempts_for_discard_rate
    sim_config["maxConsecutiveDiscards"] = sim.max_consecutive_discards
    sim_config["greedyPrior"] = gen == 0 and sim.greedy_prior
    sim_config["greedyPriorUniformMix"] = sim.greedy_prior_uniform_mix
    if not sim.symmetries:
        sim_config["noSymmetries"] = True
    if sim.verbosity:
        sim_config["verbosity"] = sim.verbosity
    if cfg.seed is not None:
        sim_config["seed"] = cfg.seed + gen + seed_offset
    if nn_url is not None:
        sim_config["prior"] = True
        sim_config["nnUrl"] = nn_url
    sim_config["exportType"] = "GameState"

    config_path = _write_config(cfg, f"simulate_gen{gen}{config_suffix}", sim_config)

    args = [
        "dotnet",
        "run",
        "--project",
        cfg.dotnet_project,
        "--",
        "simulate",
        "--config",
        str(config_path),
    ]

    label = f"Gen {gen}: Simulate ({remaining} remaining of {target_games} games"
    if existing > 0:
        label += f", {existing} already done"
    if requested_games > remaining:
        label += f", requesting {requested_games} with oversample={sim.oversample}"
    label += ")"

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(args)}")
    print(f"  config: {config_path}")
    print(f"{'=' * 60}\n", flush=True)

    if requested_games <= remaining:
        # No oversampling — just run and wait.
        monitor = _MonitorProcess(cfg, f"gen{gen}-simulate", nn_url)
        try:
            result = subprocess.run(args, cwd=project_root, stderr=subprocess.PIPE, text=True)
        finally:
            monitor.stop()
        if result.returncode != 0:
            stderr_msg = result.stderr.strip() if result.stderr else ""
            detail = f"{label} failed with exit code {result.returncode}"
            if stderr_msg:
                detail += f"\nstderr:\n{stderr_msg}"
            raise RuntimeError(detail)
        return

    # Oversample mode: launch as a background process and monitor the folder.
    proc = subprocess.Popen(
        args,
        cwd=project_root,
        # Do not buffer stderr in a pipe: the simulation is monitored while it
        # runs, so an undrained pipe could fill and deadlock the child process.
        stderr=None,
        start_new_session=True,
    )
    monitor = _MonitorProcess(cfg, f"gen{gen}-simulate", nn_url)
    try:
        while proc.poll() is None:
            count = _count_json_files(out_dir)
            discard_reason = _discard_stop_reason(out_dir, sim)
            if discard_reason is not None:
                print(f"  Simulation discard policy exceeded: {discard_reason}")
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise RuntimeError(f"{label} stopped: {discard_reason}")
            if count >= target_games:
                print(
                    f"  Target reached: {count}/{target_games} games. Terminating simulation early."
                )
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                return
            time.sleep(1.0)

        # Process exited on its own — check exit code.
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

        # Verify we got enough games even though process exited normally.
        count = _count_json_files(out_dir)
        if count < target_games:
            print(f"  WARNING: Simulation finished but only produced {count}/{target_games} games.")
    except BaseException:
        # Ensure subprocess is cleaned up on any error (including KeyboardInterrupt).
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    finally:
        monitor.stop()


def _count_json_files(directory: Path) -> int:
    """Count ``.json`` files in *directory* (non-recursive)."""
    return sum(1 for _ in directory.glob("*.json"))


def _discard_stop_reason(directory: Path, sim: SimulateConfig) -> str | None:
    """Return a stop reason from accepted and discarded per-game files."""
    accepted = list(directory.glob("*.json"))
    discarded = list((directory / "discarded").glob("*.json"))
    attempts = len(accepted) + len(discarded)
    if len(discarded) > sim.max_discarded_games:
        return f"discarded games {len(discarded)} exceeded {sim.max_discarded_games}"
    if (
        attempts >= sim.minimum_attempts_for_discard_rate
        and len(discarded) / max(1, attempts) > sim.max_discard_rate
    ):
        return f"discard rate {len(discarded) / attempts:.2%} exceeded {sim.max_discard_rate:.2%}"

    events = sorted(
        [(path.stat().st_mtime_ns, False) for path in accepted]
        + [(path.stat().st_mtime_ns, True) for path in discarded]
    )
    consecutive = 0
    for _, is_discarded in events:
        consecutive = consecutive + 1 if is_discarded else 0
    if consecutive > sim.max_consecutive_discards:
        return f"consecutive discards {consecutive} exceeded {sim.max_consecutive_discards}"
    return None


def _step_train(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    sim_override: SimulateConfig | None = None,
    out_path_override: Path | None = None,
    resume_path_override: Path | None = None,
    checkpoint_path_override: Path | None = None,
    config_suffix: str = "",
) -> None:
    """Train the model for a generation. Skips if the checkpoint already exists."""
    _reload_config(cfg)
    out_path = out_path_override or _model_path(cfg, gen)
    tr = cfg.train
    checkpoint_dir = (
        checkpoint_path_override or _checkpoint_path(cfg, gen) if tr.checkpoint_dir else None
    )
    interrupted = (
        checkpoint_dir is not None
        and checkpoint_dir.is_dir()
        and any(checkpoint_dir.glob("epoch_*.pt"))
        and not (checkpoint_dir / "training_complete").is_file()
    )
    if out_path.is_file() and not interrupted:
        print(f"  Train: Model already exists at {out_path}, skipping.")
        _ensure_inference_artifacts(cfg, out_path, project_root)
        return

    replay_start = max(0, gen - max(1, tr.replay_generations) + 1)
    data_paths = [_data_path(cfg, replay_gen) for replay_gen in range(replay_start, gen + 1)]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build config JSON for training.
    train_config: dict[str, Any] = {
        "data": [str(path) for path in data_paths],
        "gameConfig": cfg.game_config,
        "modelConfig": cfg.model_config,
        "out": str(out_path),
        "epochs": tr.epochs,
        "patience": tr.patience,
        "batchSize": tr.batch_size,
        "lr": tr.lr,
        "valSplit": tr.val_split,
        "testSplit": tr.test_split,
        "logInterval": tr.log_interval,
        "valueLossWeight": tr.value_loss_weight,
        "policyLossWeight": tr.policy_loss_weight,
        "mctsValueWeightStart": tr.mcts_value_weight_start,
        "mctsValueWeightEnd": tr.mcts_value_weight_end,
        "victoryPointSamplingStatistic": tr.victory_point_sampling_statistic,
        "victoryPointSamplingUpperPercentage": tr.victory_point_sampling_upper_percentage,
        "checkpointRetention": tr.checkpoint_retention,
    }
    # Enable per-epoch checkpointing if configured.
    if checkpoint_dir is not None:
        train_config["checkpointDir"] = str(checkpoint_dir)

    # Resume from previous generation's model if available.
    if interrupted:
        train_config["resume"] = str(checkpoint_dir)
    elif resume_path_override is not None:
        train_config["resume"] = str(resume_path_override)
    elif tr.resume_from_previous and gen > 0:
        prev_model = _model_path(cfg, gen - 1)
        if prev_model.exists():
            train_config["resume"] = str(prev_model)

    config_path = _write_config(cfg, f"train_gen{gen}{config_suffix}", train_config)

    args = [
        sys.executable,
        "-m",
        f"{cfg.python_module}.train",
        "--config",
        str(config_path),
    ]

    _run(args, label=f"Gen {gen}: Train", cwd=project_root, monitor_cfg=cfg)
    _ensure_inference_artifacts(cfg, out_path, project_root)


def _step_benchmark(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    nn_url: str,
    gimbur_server: _GimburServerProcess | None = None,
    all_results: dict[int, dict[str, Any]] | None = None,
    output_path_for: Callable[[str], Path] | None = None,
    config_prefix: str | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Run benchmarks for a generation. Returns aggregated results.

    Skips individual benchmarks whose result files already exist (resume).
    """
    _reload_config(cfg)
    gen_results: dict[str, Any] = {}

    benchmarks = list(cfg.benchmarks)

    started_game_server = False
    try:
        for configured_bench in benchmarks:
            _reload_config(cfg)
            bench = next(
                (item for item in cfg.benchmarks if item.name == configured_bench.name), None
            )
            if bench is None:
                print(f"  Benchmark '{configured_bench.name}': removed from config, skipping.")
                continue
            if (
                gimbur_server is not None
                and not started_game_server
                and _benchmarks_need_game_server([bench])
            ):
                gimbur_server.start(
                    gimbur_server_cfg=cfg.gimbur_server,
                    cwd=project_root,
                )
                started_game_server = True
            result_name = _precision_benchmark_name(cfg, bench.name, precision)
            out_path = (
                output_path_for(result_name)
                if output_path_for is not None
                else _results_path(cfg, gen, result_name)
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # If results already exist, load and skip.
            if out_path.is_file():
                print(f"  Benchmark '{result_name}': results already exist, skipping.")
                gen_results[result_name] = json.loads(out_path.read_text())
                if all_results is not None:
                    all_results.setdefault(gen, {})[result_name] = gen_results[result_name]
                    _save_summary(cfg, all_results)
                    _save_progress_chart(cfg, all_results)
                continue

            # Build config JSON for benchmark.
            bench_config: dict[str, Any] = {
                "games": bench.games,
                "ai": bench.ai,
                "mapConfig": cfg.map_config,
                "output": str(out_path),
                "nnUrl": nn_url,
                "verbosity": "quiet",
            }
            if cfg.seed is not None:
                bench_config["seed"] = cfg.seed + gen * 1000
            if bench.search_time_ms is not None:
                bench_config["searchTimeMs"] = bench.search_time_ms
            if any(ai in _SERVER_AI_KINDS for ai in bench.ai):
                gs = cfg.gimbur_server
                bench_config["serverUrl"] = f"http://{gs.host}:{gs.port}"
            if bench.server_prior_mode is not None:
                bench_config["serverPriorMode"] = bench.server_prior_mode
            if bench.parallelism is not None:
                bench_config["parallelism"] = bench.parallelism
            bench_config["progressInterval"] = bench.progress_interval

            config_name = config_prefix or f"benchmark_gen{gen}"
            config_path = _write_config(cfg, f"{config_name}_{result_name}", bench_config)

            args = [
                "dotnet",
                "run",
                "--project",
                cfg.dotnet_project,
                "--",
                "benchmark",
                "--config",
                str(config_path),
            ]

            _run(
                args,
                label=f"Gen {gen}: Benchmark '{result_name}' ({bench.games} games)",
                cwd=project_root,
                monitor_cfg=cfg,
                inference_url=nn_url,
            )

            # Parse results.
            if not out_path.exists():
                raise RuntimeError(
                    f"Benchmark '{result_name}' completed without writing {out_path}."
                )
            results = json.loads(out_path.read_text())
            gen_results[result_name] = results
            if all_results is not None:
                all_results.setdefault(gen, {})[result_name] = results
                _save_summary(cfg, all_results)
                _save_progress_chart(cfg, all_results)

    finally:
        # Always stop the Gimbur.Server if we started it, even on error.
        if started_game_server and gimbur_server is not None:
            gimbur_server.stop()

    return gen_results


def _benchmark_score(result: dict[str, Any], label: str, expected_games: int) -> float:
    total_games = int(result.get("totalGames", 0))
    if total_games != expected_games:
        raise RuntimeError(
            f"Promotion benchmark completed {total_games}/{expected_games} required games."
        )
    entry = next((item for item in result.get("winRates", []) if item.get("label") == label), None)
    if entry is None:
        raise RuntimeError(f"Promotion benchmark has no win rate for label '{label}'.")
    draws = int(result.get("draws", 0))
    return (int(entry.get("wins", 0)) + 0.5 * draws) / total_games


def _run_promotion_match(
    cfg: PipelineConfig,
    gen: int,
    attempt: int,
    project_root: Path,
    *,
    name: str,
    gate: PromotionGateConfig,
    opponent_ai: str,
    challenger_url: str,
    opponent_url: str,
    gimbur_server: _GimburServerProcess,
) -> tuple[dict[str, Any], float]:
    _reload_config(cfg)
    gate = getattr(cfg.promotion, name.split("-", 1)[0])
    precision = cfg.inference.promotion_precision
    artifact_name = f"{name}-{precision}"
    out_path = _promotion_attempt_path(cfg, gen, attempt) / f"{artifact_name}.json"
    if out_path.is_file():
        try:
            result = json.loads(out_path.read_text())
            if int(result.get("totalGames", 0)) == gate.games:
                return result, _benchmark_score(result, "challenger", gate.games)
        except (json.JSONDecodeError, RuntimeError):
            pass
        out_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench_config: dict[str, Any] = {
        "games": gate.games,
        "ai": [gate.ai, opponent_ai],
        "playerLabels": ["challenger", "opponent"],
        "nnUrls": [challenger_url, opponent_url],
        "mapConfig": cfg.map_config,
        "output": str(out_path),
        "verbosity": "quiet",
    }
    if cfg.seed is not None:
        bench_config["seed"] = cfg.seed + gen * 100_000 + attempt * 10_000
    if gate.search_time_ms is not None:
        bench_config["searchTimeMs"] = gate.search_time_ms
    if gate.parallelism is not None:
        bench_config["parallelism"] = gate.parallelism
    bench_config["progressInterval"] = gate.progress_interval
    if gate.ai in _SERVER_AI_KINDS:
        gs = cfg.gimbur_server
        bench_config["serverUrl"] = f"http://{gs.host}:{gs.port}"
        gimbur_server.start(gimbur_server_cfg=cfg.gimbur_server, cwd=project_root)

    config_path = _write_config(
        cfg, f"promotion_gen{gen}_attempt{attempt}_{artifact_name}", bench_config
    )
    try:
        _run(
            [
                "dotnet",
                "run",
                "--project",
                cfg.dotnet_project,
                "--",
                "benchmark",
                "--config",
                str(config_path),
            ],
            label=f"Gen {gen} attempt {attempt}: promotion {name} ({gate.games} games)",
            cwd=project_root,
            monitor_cfg=cfg,
            inference_url=challenger_url,
        )
    finally:
        if gate.ai in _SERVER_AI_KINDS:
            gimbur_server.stop()

    result = json.loads(out_path.read_text())
    return result, _benchmark_score(result, "challenger", gate.games)


def _evaluate_promotion_gate(
    gate: PromotionGateConfig, greedy_score: float | None, champion_score: float
) -> dict[str, Any]:
    greedy_required = (
        gate.minimum_score_vs_greedy
        if gate.minimum_score_vs_greedy is not None
        else 0.5 + gate.minimum_improvement_vs_greedy
    )
    champion_required = (
        gate.minimum_score_vs_champion
        if gate.minimum_score_vs_champion is not None
        else 0.5 + gate.minimum_improvement_vs_champion
    )
    passed_greedy = not gate.compare_with_greedy or (
        greedy_score is not None and greedy_score >= greedy_required
    )
    passed_champion = champion_score >= champion_required
    return {
        "games": gate.games,
        "ai": gate.ai,
        "greedyScore": greedy_score,
        "greedyRequired": greedy_required,
        "championScore": champion_score,
        "championRequired": champion_required,
        "passedGreedy": passed_greedy,
        "passedChampion": passed_champion,
        "passed": passed_greedy and passed_champion,
    }


def _promote_candidate(cfg: PipelineConfig, gen: int, attempt: int) -> dict[str, Any]:
    source = _candidate_model_path(cfg, gen, attempt)
    destination = _champion_model_path(cfg, gen)
    if not source.is_file():
        raise FileNotFoundError(f"Candidate complete model not found at {source}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    legacy = _model_path(cfg, gen)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, legacy)
    if cfg.inference.export_fp16:
        source_fp16 = _precision_model_path(source, "fp16")
        shutil.copy2(source_fp16, _precision_model_path(destination, "fp16"))
        shutil.copy2(source_fp16, _precision_model_path(legacy, "fp16"))
    manifest: dict[str, Any] = {
        "generation": gen,
        "attempt": attempt,
        "model": str(destination),
    }
    _write_json_atomic(_champion_manifest_path(cfg), manifest)
    return manifest


def _train_promotion_candidate(
    cfg: PipelineConfig,
    gen: int,
    attempt: int,
    project_root: Path,
    champion: dict[str, Any] | None,
) -> None:
    destination = _candidate_model_path(cfg, gen, attempt)
    checkpoint_dir = destination.parent / "checkpoints"
    resume = (
        checkpoint_dir
        if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("epoch_*.pt"))
        else Path(champion["model"])
        if champion is not None
        else None
    )
    _step_train(
        cfg,
        gen,
        project_root,
        out_path_override=destination,
        resume_path_override=resume,
        checkpoint_path_override=checkpoint_dir,
        config_suffix=f"_attempt{attempt}",
    )


def _start_complete_model_server(
    server: _ServerProcess,
    cfg: PipelineConfig,
    serve_cfg: ServeConfig,
    model: Path,
    project_root: Path,
    precision: str = "fp32",
) -> str:
    original_port_offset = serve_cfg.port - cfg.serve.port
    _reload_config(cfg)
    _ensure_inference_artifacts(cfg, model, project_root)
    serve_cfg = replace(cfg.serve, port=cfg.serve.port + original_port_offset)
    server.start(
        serve_cfg=serve_cfg,
        game_config=cfg.game_config,
        python_module=cfg.python_module,
        model_path=_precision_model_path(model, precision),
        model_config=cfg.model_config,
        cwd=project_root,
    )
    return f"http://{serve_cfg.host}:{serve_cfg.port}"


def _run_promotion_gates(
    cfg: PipelineConfig,
    gen: int,
    attempt: int,
    project_root: Path,
    champion: dict[str, Any],
    gimbur_server: _GimburServerProcess,
) -> dict[str, Any]:
    challenger_server = _ServerProcess()
    champion_server = _ServerProcess()
    challenger_cfg = cfg.serve
    champion_cfg = replace(cfg.serve, port=cfg.serve.port + 1)
    gate_results: dict[str, Any] = {}
    try:
        challenger_url = _start_complete_model_server(
            challenger_server,
            cfg,
            challenger_cfg,
            _candidate_model_path(cfg, gen, attempt),
            project_root,
            cfg.inference.promotion_precision,
        )
        champion_url = _start_complete_model_server(
            champion_server,
            cfg,
            champion_cfg,
            Path(champion["model"]),
            project_root,
            cfg.inference.promotion_precision,
        )
        for name, gate in (("direct", cfg.promotion.direct), ("hybrid", cfg.promotion.hybrid)):
            if not gate.enabled:
                continue
            greedy_score = None
            if gate.compare_with_greedy:
                _, greedy_score = _run_promotion_match(
                    cfg,
                    gen,
                    attempt,
                    project_root,
                    name=f"{name}-challenger-vs-greedy",
                    gate=gate,
                    opponent_ai="greedy",
                    challenger_url=challenger_url,
                    opponent_url=champion_url,
                    gimbur_server=gimbur_server,
                )
            _, champion_score = _run_promotion_match(
                cfg,
                gen,
                attempt,
                project_root,
                name=f"{name}-challenger-vs-champion",
                gate=gate,
                opponent_ai=gate.ai,
                challenger_url=challenger_url,
                opponent_url=champion_url,
                gimbur_server=gimbur_server,
            )
            gate_results[name] = _evaluate_promotion_gate(gate, greedy_score, champion_score)
    finally:
        challenger_server.stop()
        champion_server.stop()
    return gate_results


def _run_promoted_generation(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    gimbur_server: _GimburServerProcess,
) -> None:
    generation_path = _promotion_generation_path(cfg, gen)
    if generation_path.is_file():
        return

    champion = _load_champion(cfg)
    base_games = cfg.gen0_games if gen == 0 and cfg.gen0_games is not None else cfg.simulate.games
    for attempt in range(cfg.promotion.max_retries + 1):
        attempt_path = _promotion_attempt_path(cfg, gen, attempt)
        decision_path = attempt_path / "decision.json"
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text())
            if decision.get("passed"):
                active = _load_champion(cfg)
                if active is None or active.get("generation") != gen:
                    active = _promote_candidate(cfg, gen, attempt)
                _write_json_atomic(
                    generation_path,
                    {
                        "generation": gen,
                        "status": "promoted",
                        "attempt": attempt,
                        "trainingGames": decision["trainingGames"],
                        "championGenerationAfter": active["generation"],
                    },
                )
                return
            continue

        target_games = base_games + attempt * cfg.promotion.additional_training_games
        simulation_server = _ServerProcess()
        simulation_url: str | None = None
        try:
            if champion is not None:
                simulation_url = _start_complete_model_server(
                    simulation_server,
                    cfg,
                    cfg.serve,
                    Path(champion["model"]),
                    project_root,
                    cfg.inference.simulation_precision,
                )
            _step_simulate(
                cfg,
                gen,
                project_root,
                simulation_url,
                target_games_override=target_games,
                seed_offset=attempt * 100_000,
                config_suffix=f"_attempt{attempt}",
            )
        finally:
            simulation_server.stop()

        _train_promotion_candidate(cfg, gen, attempt, project_root, champion)
        if champion is None:
            decision = {"passed": True, "bootstrap": True, "gates": {}}
        else:
            gates = _run_promotion_gates(cfg, gen, attempt, project_root, champion, gimbur_server)
            passed = all(gate["passed"] for gate in gates.values())
            decision = {"passed": passed, "bootstrap": False, "gates": gates}

        decision.update({"generation": gen, "attempt": attempt, "trainingGames": target_games})
        _write_json_atomic(decision_path, decision)
        if decision["passed"]:
            promoted = _promote_candidate(cfg, gen, attempt)
            _write_json_atomic(
                generation_path,
                {
                    "generation": gen,
                    "status": "promoted",
                    "attempt": attempt,
                    "trainingGames": target_games,
                    "championGenerationAfter": promoted["generation"],
                },
            )
            return

    _write_json_atomic(
        generation_path,
        {
            "generation": gen,
            "status": "rejected",
            "attempt": cfg.promotion.max_retries,
            "trainingGames": base_games
            + cfg.promotion.max_retries * cfg.promotion.additional_training_games,
            "championGenerationAfter": champion["generation"] if champion else None,
        },
    )


def _run_promoted_benchmarks(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    server: _ServerProcess,
    gimbur_server: _GimburServerProcess,
    all_results: dict[int, dict[str, Any]],
) -> None:
    decision_path = _promotion_generation_path(cfg, gen)
    if not decision_path.is_file():
        return
    decision = json.loads(decision_path.read_text())
    if decision.get("status") != "promoted" or _benchmark_complete(cfg, gen):
        return

    model = _champion_model_path(cfg, gen)
    if not model.is_file():
        raise FileNotFoundError(f"Promoted model file is missing: {model}")

    try:
        for precision in cfg.inference.benchmark_precisions:
            nn_url = _start_complete_model_server(
                server, cfg, cfg.serve, model, project_root, precision
            )
            results = _step_benchmark(
                cfg,
                gen,
                project_root,
                nn_url,
                gimbur_server=gimbur_server,
                all_results=all_results,
                precision=precision,
            )
            if results:
                all_results.setdefault(gen, {}).update(results)
                _save_summary(cfg, all_results)
                _save_progress_chart(cfg, all_results)
            server.stop()
    finally:
        server.stop()


def _run_gen0_milestones(
    cfg: PipelineConfig,
    project_root: Path,
    server: _ServerProcess,
    gimbur_server: _GimburServerProcess,
) -> None:
    """Train and benchmark cumulative generation-zero dataset milestones."""
    if not cfg.gen0_milestones:
        return

    bootstrap_summary_path = Path(cfg.results_dir) / "bootstrap-summary.json"
    summary_document = (
        json.loads(bootstrap_summary_path.read_text())
        if bootstrap_summary_path.is_file()
        else {"milestones": []}
    )
    summary: list[dict[str, Any]] = summary_document.get("milestones", [])
    summary_by_games = {int(entry["games"]): entry for entry in summary}

    # Reuse an already completed conventional 200-game Gen-0 model/result set.
    first = cfg.gen0_milestones[0]
    legacy_model = _champion_model_path(cfg, 0)
    if first == 200 and legacy_model.is_file() and not _bootstrap_model_path(cfg, first).is_file():
        destination = _bootstrap_model_path(cfg, first)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_model, destination)
        for bench in cfg.benchmarks:
            legacy_result = _results_path(cfg, 0, bench.name)
            if legacy_result.is_file():
                result = _bootstrap_result_path(cfg, first, bench.name)
                result.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_result, result)

    for games in cfg.gen0_milestones:
        print(f"\n{'#' * 60}\n  GEN-0 BOOTSTRAP: {games} GAMES\n{'#' * 60}\n")
        _step_simulate(
            cfg,
            0,
            project_root,
            nn_url=None,
            target_games_override=games,
            seed_offset=games,
            config_suffix=f"_bootstrap_{games}",
        )

        model_path = _bootstrap_model_path(cfg, games)
        _step_train(
            cfg,
            0,
            project_root,
            out_path_override=model_path,
            resume_path_override=None,
            checkpoint_path_override=_bootstrap_checkpoint_path(cfg, games),
            config_suffix=f"_bootstrap_{games}",
        )

        try:
            results = {}
            for precision in cfg.inference.benchmark_precisions:
                nn_url = _start_complete_model_server(
                    server, cfg, cfg.serve, model_path, project_root, precision
                )
                results.update(
                    _step_benchmark(
                        cfg,
                        0,
                        project_root,
                        nn_url,
                        gimbur_server=gimbur_server,
                        output_path_for=lambda name, games=games: _bootstrap_result_path(
                            cfg, games, name
                        ),
                        config_prefix=f"benchmark_bootstrap_{games}",
                        precision=precision,
                    )
                )
                server.stop()
        finally:
            server.stop()

        summary_by_games[games] = {
            "games": games,
            "model": str(model_path),
            "benchmarks": {
                name: {
                    "winRates": _normalize_win_rates(result),
                    "draws": result.get("draws", 0),
                    "totalGames": result.get("totalGames", 0),
                }
                for name, result in results.items()
            },
        }
        _write_json_atomic(
            bootstrap_summary_path,
            {"milestones": [summary_by_games[key] for key in sorted(summary_by_games)]},
        )
        _save_bootstrap_progress_chart(cfg, summary_by_games)

    final_games = cfg.gen0_milestones[-1]
    final_model = _bootstrap_model_path(cfg, final_games)
    champion = _champion_model_path(cfg, 0)
    champion.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_model, champion)
    shutil.copy2(final_model, _model_path(cfg, 0))
    if cfg.inference.export_fp16:
        shutil.copy2(
            _precision_model_path(final_model, "fp16"),
            _precision_model_path(champion, "fp16"),
        )
        shutil.copy2(
            _precision_model_path(final_model, "fp16"),
            _precision_model_path(_model_path(cfg, 0), "fp16"),
        )
    for bench in cfg.benchmarks:
        for precision in cfg.inference.benchmark_precisions:
            name = _precision_benchmark_name(cfg, bench.name, precision)
            source = _bootstrap_result_path(cfg, final_games, name)
            destination = _results_path(cfg, 0, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    _write_json_atomic(
        _champion_manifest_path(cfg),
        {"generation": 0, "attempt": 0, "games": final_games, "model": str(champion)},
    )
    _write_json_atomic(
        _promotion_generation_path(cfg, 0),
        {
            "generation": 0,
            "status": "promoted",
            "attempt": 0,
            "trainingGames": final_games,
            "championGenerationAfter": 0,
        },
    )


def _save_bootstrap_progress_chart(
    cfg: PipelineConfig,
    summary_by_games: dict[int, dict[str, Any]],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not summary_by_games:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    games = sorted(summary_by_games)
    for benchmark in cfg.benchmarks:
        for precision in cfg.inference.benchmark_precisions:
            name = _precision_benchmark_name(cfg, benchmark.name, precision)
            points: list[tuple[int, float]] = []
            for game_count in games:
                result = summary_by_games[game_count].get("benchmarks", {}).get(name)
                if not result:
                    continue
                rates = result.get("winRates", {})
                candidate = next(
                    (rate for label, rate in rates.items() if label == benchmark.ai[0]), None
                )
                if candidate is not None:
                    points.append((game_count, candidate))
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] * 100 for point in points],
                    marker="o",
                    label=name,
                )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Cumulative Gen-0 self-play games")
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Gen-0 Bootstrap Model Strength by Dataset Size")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(cfg.results_dir) / "bootstrap-progress.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    os.replace(temporary, output)
    return True


def _step_baselines(
    cfg: PipelineConfig,
    project_root: Path,
    gimbur_server: _GimburServerProcess | None = None,
) -> dict[str, dict[str, Any]]:
    """Run any configured baseline benchmarks once, caching results.

    Baselines are reference matchups (e.g. ``mcts vs greedy``) used as
    horizontal lines on the progress chart. Each baseline runs a single
    fixed number of games and the result is cached at
    ``{results_dir}/baselines/{name}.json``; on subsequent pipeline runs
    the cached file is loaded and the benchmark is skipped.

    Returns ``{baseline_name: result_dict}`` for every configured baseline,
    populated from cache where available.
    """
    _reload_config(cfg)
    results: dict[str, dict[str, Any]] = {}
    if not cfg.baselines:
        return results

    # Determine which baselines need to run vs which are cached.
    pending: list[BaselineBenchmarkConfig] = []
    for bench in cfg.baselines:
        out_path = _baseline_path(cfg, bench.name)
        if out_path.is_file():
            results[bench.name] = json.loads(out_path.read_text())
            print(f"  Baseline '{bench.name}': cached, skipping.")
        else:
            pending.append(bench)

    if not pending:
        return results

    # Start Gimbur.Server if any pending baseline needs it.
    started_game_server = False
    if gimbur_server is not None and any(
        ai in _SERVER_AI_KINDS for bench in pending for ai in bench.ai
    ):
        gimbur_server.start(
            gimbur_server_cfg=cfg.gimbur_server,
            cwd=project_root,
        )
        started_game_server = True

    try:
        for bench in pending:
            out_path = _baseline_path(cfg, bench.name)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            bench_config: dict[str, Any] = {
                "games": bench.games,
                "ai": bench.ai,
                "mapConfig": cfg.map_config,
                "output": str(out_path),
                "verbosity": "quiet",
            }
            if cfg.seed is not None:
                bench_config["seed"] = cfg.seed
            if bench.search_time_ms is not None:
                bench_config["searchTimeMs"] = bench.search_time_ms
            if any(ai in _SERVER_AI_KINDS for ai in bench.ai):
                gs = cfg.gimbur_server
                bench_config["serverUrl"] = f"http://{gs.host}:{gs.port}"
            bench_config["progressInterval"] = bench.progress_interval
            if bench.server_prior_mode is not None:
                bench_config["serverPriorMode"] = bench.server_prior_mode

            config_path = _write_config(cfg, f"baseline_{bench.name}", bench_config)

            args = [
                "dotnet",
                "run",
                "--project",
                cfg.dotnet_project,
                "--",
                "benchmark",
                "--config",
                str(config_path),
            ]

            _run(
                args,
                label=f"Baseline '{bench.name}' ({bench.games} games)",
                cwd=project_root,
            )

            if out_path.exists():
                results[bench.name] = json.loads(out_path.read_text())

    finally:
        if started_game_server and gimbur_server is not None:
            gimbur_server.stop()

    return results


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------


def _print_generation_summary(gen: int, results: dict[str, Any]) -> None:
    """Print a summary of benchmark results for a generation.

    Handles both the C# benchmark JSON format (``winRates`` as a list of
    ``{"ai": "nn", "rate": 0.33, "wins": 1}`` objects) and the
    normalised summary format (``winRates`` as ``{"nn": 0.33}``).
    """
    print(f"\n{'=' * 60}")
    print(f"  Generation {gen} — Results Summary")
    print(f"{'=' * 60}")

    for name, data in results.items():
        total = data.get("totalGames", 0)
        print(f"\n  {name}:")
        raw = data.get("winRates", {})
        if isinstance(raw, list):
            for wr in raw:
                pct = wr["rate"] * 100
                print(f"    {wr['ai']:>8s}: {pct:5.1f}% ({wr['wins']}/{total})")
        else:
            # Flat dict format from summary.json: {"nn": 0.33, ...}
            for ai, rate in raw.items():
                pct = rate * 100
                wins = round(rate * total) if total else 0
                print(f"    {ai:>8s}: {pct:5.1f}% ({wins}/{total})")
        draws = data.get("draws", 0)
        if draws > 0:
            print(f"    {'draws':>8s}: {draws}/{total}")

    print()


def _normalize_win_rates(data: dict[str, Any]) -> dict[str, float]:
    """Extract win rates as ``{ai: rate}`` from either format.

    The C# benchmark JSON emits ``winRates`` as a list of
    ``{"ai": "nn", "rate": 0.33, "wins": 1}`` objects.  The summary
    file stores them as a flat dict ``{"nn": 0.33}``.  This helper
    accepts both.
    """
    raw = data.get("winRates", {})
    if isinstance(raw, list):
        return {wr.get("label", wr["ai"]): wr["rate"] for wr in raw}
    # Already a dict (loaded from summary.json).
    return dict(raw)


def _normalize_confidence(data: dict[str, Any], field: str) -> dict[str, float]:
    """Extract per-AI confidence margins from raw or normalized results."""
    raw = data.get("winRates", {})
    if isinstance(raw, list):
        return {wr.get("label", wr["ai"]): wr[field] for wr in raw if field in wr}
    confidence = data.get(field, {})
    return dict(confidence) if isinstance(confidence, dict) else {}


def _save_summary(cfg: PipelineConfig, all_results: dict[int, dict[str, Any]]) -> None:
    """Save a summary JSON tracking results across all generations."""
    summary_path = Path(cfg.results_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for gen, results in sorted(all_results.items()):
        entry: dict[str, Any] = {"generation": gen, "benchmarks": {}}
        for name, data in results.items():
            entry["benchmarks"][name] = {
                "winRates": _normalize_win_rates(data),
                "confidence95Margin": _normalize_confidence(data, "confidence95Margin"),
                "worstCaseConfidence95Margin": _normalize_confidence(
                    data, "worstCaseConfidence95Margin"
                ),
                "draws": data.get("draws", 0),
                "totalGames": data.get("totalGames", 0),
            }
        summary.append(entry)

    with tempfile.NamedTemporaryFile("w", dir=summary_path.parent, delete=False) as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, summary_path)
    print(f"Summary saved to {summary_path}")


def _save_progress_chart(cfg: PipelineConfig, all_results: dict[int, dict[str, Any]]) -> bool:
    """Save a win-rate progress chart (PNG) covering all generations so far.

    Plots the subject (first AI in each benchmark config) win rate on the
    y-axis against generation on the x-axis.  All benchmark configs appear
    as separate series on the same chart.  The file is overwritten after
    every benchmark phase so it always reflects the latest state.

    Also overlays:

    * One dashed horizontal line per distinct player count, marking the
      equal-play baseline ``100 / player_count`` (e.g. 50% for 2 players).
    * One dashed horizontal line per configured baseline benchmark
      (loaded from ``{results_dir}/baselines/*.json``), marking how a
      reference no-prior strategy fares against the same opponent.

    Requires ``matplotlib`` (optional ``pipeline`` dependency).  If not
    installed, silently skips.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not all_results:
        return False

    # Build a lookup from benchmark name -> subject AI name.
    bench_subjects: dict[str, str] = {}
    for bench in cfg.benchmarks:
        subject = bench.ai[0].lower()
        for precision in cfg.inference.benchmark_precisions:
            bench_subjects[_precision_benchmark_name(cfg, bench.name, precision)] = subject

    # Collect series: bench_name -> [(gen, win_rate, confidence margin), ...]
    series: dict[str, list[tuple[int, float, float | None]]] = {}
    for gen in sorted(all_results):
        for bench_name, data in all_results[gen].items():
            rates = _normalize_win_rates(data)
            margins = _normalize_confidence(data, "confidence95Margin")
            subject = bench_subjects.get(bench_name)
            if subject is None:
                # Benchmark not in current config (leftover from earlier run);
                # fall back to first key in rates dict.
                subject = next(iter(rates), None)
            rate = rates.get(subject)
            if rate is None and subject is not None:
                # Read benchmark files produced before canonical hyphenated names.
                legacy_subject = subject.replace("-", "")
                rate = rates.get(legacy_subject)
                subject = legacy_subject
            if rate is not None:
                series.setdefault(bench_name, []).append((gen, rate, margins.get(subject)))

    if not series:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))

    for bench_name, points in sorted(series.items()):
        gens = [g for g, _, _ in points]
        rates = [r * 100 for _, r, _ in points]
        margins = [m * 100 if m is not None else 0 for _, _, m in points]
        ax.errorbar(
            gens,
            rates,
            yerr=margins,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            label=bench_name,
        )

    # Equal-play horizontal lines, one per distinct player count across
    # the configured benchmarks (typically just one).
    player_counts = {len(b.ai) for b in cfg.benchmarks if b.ai}
    for pc in sorted(player_counts):
        equal = 100.0 / pc
        ax.axhline(
            equal,
            linestyle=":",
            color="gray",
            linewidth=1.0,
            label=f"equal play ({pc}p, {equal:.1f}%)",
        )

    # Baseline horizontal lines (mcts vs greedy/random with same params).
    for bench in cfg.baselines:
        out_path = _baseline_path(cfg, bench.name)
        if not out_path.is_file():
            continue
        try:
            data = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rates = _normalize_win_rates(data)
        subject = bench.ai[0].lower() if bench.ai else None
        rate = rates.get(subject) if subject else None
        if rate is None and subject:
            rate = rates.get(subject.replace("-", ""))
        if rate is None:
            continue
        ax.axhline(
            rate * 100,
            linestyle="--",
            linewidth=1.0,
            label=f"{bench.name} ({rate * 100:.1f}%)",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Benchmark Win Rate by Generation")
    ax.set_ylim(-5, 105)

    # Integer ticks on x-axis.
    all_gens = sorted(all_results.keys())
    ax.set_xticks(all_gens)

    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)

    chart_path = Path(cfg.results_dir) / "progress.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=chart_path.parent,
            prefix=f".{chart_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        fig.savefig(temporary_path, format="png", dpi=120, bbox_inches="tight")
        os.replace(temporary_path, chart_path)
    finally:
        plt.close(fig)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"Progress chart saved to {chart_path}")
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _run_single_generation(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    server: _ServerProcess,
    all_results: dict[int, dict[str, Any]],
    gimbur_server: _GimburServerProcess | None = None,
) -> None:
    """Run one complete-model generation."""

    # --- Simulate ---
    # Gen 0: no NN model; local greedy one-hot priors steer placement and state PUCT.
    # Gen N>0: start server with gen N-1 model for prior evaluation.
    sim_already_done = _simulation_complete(cfg, gen)
    gen_nn_url: str | None = None
    if gen > 0 and not sim_already_done:
        _reload_config(cfg)
        prev_model = _model_path(cfg, gen - 1)
        if not prev_model.exists():
            raise FileNotFoundError(
                f"Model for gen {gen - 1} not found at {prev_model}. "
                f"Cannot run gen {gen} with NN prior."
            )
        _ensure_inference_artifacts(cfg, prev_model, project_root)
        server.start(
            serve_cfg=cfg.serve,
            game_config=cfg.game_config,
            python_module=cfg.python_module,
            model_path=_precision_model_path(prev_model, cfg.inference.simulation_precision),
            model_config=cfg.model_config,
            pipeline_cfg=cfg,
            cwd=project_root,
        )
        gen_nn_url = f"http://{cfg.serve.host}:{cfg.serve.port}"

    _step_simulate(cfg, gen, project_root, gen_nn_url)

    # Stop server after simulation (not needed for training).
    if not sim_already_done:
        server.stop()

    # --- Train ---
    _step_train(cfg, gen, project_root)

    # --- Benchmark ---
    # Start server with this generation's model (only if needed).
    bench_already_done = _benchmark_complete(cfg, gen)
    _reload_config(cfg)
    model_path = _model_path(cfg, gen)
    if not model_path.exists():
        print(f"WARNING: Model not found at {model_path}, skipping benchmarks.")
        return

    gen_results = {}
    if not bench_already_done:
        for precision in cfg.inference.benchmark_precisions:
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                model_path=_precision_model_path(model_path, precision),
                model_config=cfg.model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )
            gen_results.update(
                _step_benchmark(
                    cfg,
                    gen,
                    project_root,
                    f"http://{cfg.serve.host}:{cfg.serve.port}",
                    gimbur_server=gimbur_server,
                    all_results=all_results,
                    precision=precision,
                )
            )
            server.stop()

    # --- Report ---
    if gen_results:
        _print_generation_summary(gen, gen_results)
        all_results[gen] = gen_results
        _save_summary(cfg, all_results)
        _save_progress_chart(cfg, all_results)


def run_pipeline(cfg: PipelineConfig, start_gen: int, project_root: Path) -> None:
    """Execute the full self-play training pipeline.

    Supports step-level resume: each step (simulate, train, benchmark)
    checks whether its artifacts already exist and skips if so.

    Runs the current complete full-state model flow.
    """
    server = _ServerProcess()
    gimbur_server = _GimburServerProcess()
    all_results: dict[int, dict[str, Any]] = {}

    # Load any existing summary.
    summary_path = Path(cfg.results_dir) / "summary.json"
    if summary_path.exists():
        for entry in json.loads(summary_path.read_text()):
            all_results[entry["generation"]] = entry.get("benchmarks", {})

    try:
        # Run baseline benchmarks (cached; only first run executes them).
        # These are model-independent reference matchups for the chart.
        if cfg.baselines:
            print(f"\n{'#' * 60}")
            print("  BASELINES")
            print(f"{'#' * 60}\n")
            _step_baselines(cfg, project_root, gimbur_server=gimbur_server)

        if cfg.gen0_milestones and start_gen == 0:
            _run_gen0_milestones(cfg, project_root, server, gimbur_server)
            start_gen = 1

        gen = start_gen
        while True:
            _reload_config(cfg)
            if gen >= cfg.generations:
                break
            # Check if entire generation is already done.
            if _generation_complete(cfg, gen):
                print(f"\n--- Generation {gen} already complete, skipping.")
                # Ensure results are loaded for summary tracking.
                if gen not in all_results:
                    gen_results: dict[str, Any] = {}
                    for bench in cfg.benchmarks:
                        for precision in cfg.inference.benchmark_precisions:
                            name = _precision_benchmark_name(cfg, bench.name, precision)
                            rp = _results_path(cfg, gen, name)
                            if rp.is_file():
                                gen_results[name] = json.loads(rp.read_text())
                    if gen_results:
                        all_results[gen] = gen_results
                gen += 1
                continue

            print(f"\n{'#' * 60}")
            print(f"  GENERATION {gen}")
            print(f"{'#' * 60}\n")

            if cfg.promotion.enabled:
                _run_promoted_generation(
                    cfg,
                    gen,
                    project_root,
                    gimbur_server,
                )
                _run_promoted_benchmarks(
                    cfg,
                    gen,
                    project_root,
                    server,
                    gimbur_server,
                    all_results,
                )
            else:
                _run_single_generation(
                    cfg, gen, project_root, server, all_results, gimbur_server=gimbur_server
                )
            gen += 1

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    finally:
        server.stop()
        gimbur_server.stop()
        if all_results:
            _save_summary(cfg, all_results)
            _save_progress_chart(cfg, all_results)
            print(f"\nPipeline stopped after {len(all_results)} generation(s).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AlphaZero-style self-play training pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to pipeline configuration JSON file.",
    )
    parser.add_argument(
        "--start-gen",
        type=int,
        default=None,
        help=(
            "Generation to start from. Default: auto-detect from existing "
            "artifacts (resumes where the previous run left off)."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory. Default: auto-detect from config file location.",
    )
    parser.add_argument(
        "--chart-only",
        action="store_true",
        help="Regenerate progress.png from completed benchmark results, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config(args.config)

    if args.chart_only:
        summary_path = Path(cfg.results_dir) / "summary.json"
        if not summary_path.exists():
            print(f"Error: Summary file not found: {summary_path}", file=sys.stderr)
            sys.exit(1)
        all_results = {
            entry["generation"]: entry.get("benchmarks", {})
            for entry in json.loads(summary_path.read_text())
        }
        if not _save_progress_chart(cfg, all_results):
            print(
                "Error: Progress chart could not be generated. Install the pipeline "
                "dependencies and ensure the summary contains benchmark results.",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    # Resolve project root: explicit arg > two levels up from python/gimbur_nn/.
    if args.project_root is not None:
        project_root = args.project_root.resolve()
    else:
        # Assume config is somewhere accessible; use the repo root.
        project_root = Path(__file__).resolve().parent.parent.parent

    # Auto-detect resume generation if not explicitly provided.
    if args.start_gen is not None:
        start_gen = args.start_gen
    else:
        start_gen = _detect_resume_gen(cfg)
        if start_gen > 0:
            print(f"Auto-detected resume point: generation {start_gen}")
        if start_gen >= cfg.generations:
            print(f"All {cfg.generations} generations are already complete.")
            # Still ensure baselines are computed and chart reflects them.
            if cfg.baselines:
                gimbur_server = _GimburServerProcess()
                try:
                    _step_baselines(cfg, project_root, gimbur_server=gimbur_server)
                finally:
                    gimbur_server.stop()
            all_results: dict[int, dict[str, Any]] = {}
            summary_path = Path(cfg.results_dir) / "summary.json"
            if summary_path.exists():
                for entry in json.loads(summary_path.read_text()):
                    all_results[entry["generation"]] = entry.get("benchmarks", {})
            if all_results:
                _save_progress_chart(cfg, all_results)
            return

    print(f"Project root: {project_root}")
    print(f"Generations:  {start_gen} .. {cfg.generations - 1}")
    print(f"Data dir:     {cfg.data_dir}")
    print(f"Model dir:    {cfg.model_dir}")
    print(f"Results dir:  {cfg.results_dir}")

    run_pipeline(cfg, start_gen, project_root)


if __name__ == "__main__":
    main()
