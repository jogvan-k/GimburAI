"""
Self-play training pipeline orchestrator.

Drives the AlphaZero-style loop: simulate → train → benchmark → repeat.
Each iteration is called a "generation". Generation 0 uses greedy rollouts
(no NN prior); subsequent generations use the previous generation's model
as the MCTS prior evaluator.

The pipeline supports **resume-on-interrupt**: if the process is stopped
mid-run, restarting it will automatically detect which generation and step
to resume from by scanning artifact directories.

Usage:
    python -m gimbur_nn.pipeline --config pipeline.json
    python -m gimbur_nn.pipeline --config pipeline.json --start-gen 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    max_prior_depth: int | None = None
    simulations_per_action: int | None = None
    symmetries: bool = True
    verbosity: str = "quiet"
    oversample: float = 1.0


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
    loss: str = "hard"
    loss_sigma: float = 2.0
    resume_from_previous: bool = True
    target: str = "winrate"


@dataclass
class ServeConfig:
    """Parameters for ``python -m gimbur_nn.serve``."""

    port: int = 8000
    host: str = "127.0.0.1"
    log_level: str = "warning"


@dataclass
class GimburServerConfig:
    """Parameters for the C# Gimbur.Server (MCTS game server)."""

    port: int = 5123
    host: str = "127.0.0.1"
    dotnet_project: str = "src/Gimbur.Server"


@dataclass
class BenchmarkConfig:
    """A single benchmark run.

    The *phase* field controls when the benchmark runs in combined-mode
    pipelines:

    - ``"placement"`` — run after placement training (step 3).
    - ``"combined"`` — run after value training (step 6).
    - ``"both"`` — run after both steps.

    For single-model pipelines (``model_type`` = ``"state"`` or
    ``"placement"``), the *phase* field is ignored.
    """

    name: str = "nn-vs-greedy"
    games: int = 100
    ai: list[str] = field(default_factory=lambda: ["nn", "greedy"])
    phase: str = "both"  # "placement", "combined", or "both"
    search_time_ms: int | None = None
    server_prior_mode: str | None = None
    server_max_prior_depth: int | None = None


@dataclass
class PipelineConfig:
    """Top-level orchestrator configuration."""

    # Shared identifiers.
    map_config: str = "mini"
    game_config: str = "mini_2p"
    model_config: str = "small"
    placement_model_config: str | None = None  # defaults to model_config
    model_type: str = "state"  # "state", "placement", or "combined"

    # Reproducibility.
    seed: int | None = None

    # Directories (relative to project root).
    data_dir: str = "pipeline/data"
    model_dir: str = "pipeline/models"
    results_dir: str = "pipeline/results"

    # How many generations to run.
    generations: int = 10

    # Skip generations before this threshold (treat them as complete).
    skip_until_gen: int | None = None

    # Paths to the CLI tools (relative to project root).
    dotnet_project: str = "src/Gimbur.Cli"
    python_module: str = "gimbur_nn"

    # Section configs.
    simulate: SimulateConfig = field(default_factory=SimulateConfig)
    placement_simulate: SimulateConfig | None = None  # overrides simulate for placement phase
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    gimbur_server: GimburServerConfig = field(default_factory=GimburServerConfig)
    benchmarks: list[BenchmarkConfig] = field(
        default_factory=lambda: [
            BenchmarkConfig(name="nn-vs-greedy", games=100, ai=["nn", "greedy"]),
            BenchmarkConfig(name="nn-vs-random", games=100, ai=["nn", "random"]),
        ]
    )


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
    cfg = PipelineConfig()

    # Top-level scalars.
    for attr in (
        "map_config",
        "game_config",
        "model_config",
        "placement_model_config",
        "model_type",
        "seed",
        "data_dir",
        "model_dir",
        "results_dir",
        "generations",
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
    if "placementSimulate" in raw:
        cfg.placement_simulate = _load_section(SimulateConfig, raw["placementSimulate"])
    if "train" in raw:
        cfg.train = _load_section(TrainConfig, raw["train"])
    if "serve" in raw:
        cfg.serve = _load_section(ServeConfig, raw["serve"])
    if "gimburServer" in raw:
        cfg.gimbur_server = _load_section(GimburServerConfig, raw["gimburServer"])
    if "benchmarks" in raw:
        cfg.benchmarks = [_load_section(BenchmarkConfig, b) for b in raw["benchmarks"]]

    return cfg


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


def _data_path(cfg: PipelineConfig, gen: int, model_type: str | None = None) -> Path:
    """Return the data directory for a generation.

    When *model_type* is given (used by combined pipelines), a
    ``placement/`` or ``state/`` subdirectory is inserted:
    ``{data_dir}/placement/gen{N}`` or ``{data_dir}/state/gen{N}``.
    """
    base = Path(cfg.data_dir)
    if model_type is not None:
        base = base / model_type
    return base / f"gen{gen}"


def _model_path(cfg: PipelineConfig, gen: int, model_type: str | None = None) -> Path:
    """Return the model checkpoint path for a generation.

    When *model_type* is given, a subdirectory is inserted:
    ``{model_dir}/placement/gen{N}.pt`` or ``{model_dir}/state/gen{N}.pt``.
    """
    base = Path(cfg.model_dir)
    if model_type is not None:
        base = base / model_type
    return base / f"gen{gen}.pt"


def _results_path(cfg: PipelineConfig, gen: int, name: str) -> Path:
    return Path(cfg.results_dir) / f"gen{gen}" / f"{name}.json"


def _checkpoint_path(cfg: PipelineConfig, gen: int, model_type: str | None = None) -> Path:
    """Return the checkpoint directory for per-epoch checkpoints.

    When *model_type* is given, a subdirectory is inserted.
    """
    base = Path(cfg.model_dir)
    if model_type is not None:
        base = base / model_type
    return base / f"gen{gen}_checkpoints"


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


def _effective_simulate(
    cfg: PipelineConfig, model_type: str | None = None
) -> SimulateConfig:
    """Return the effective ``SimulateConfig`` for a pipeline step.

    For combined pipelines with ``model_type="placement"``, returns
    ``cfg.placement_simulate`` when set, otherwise ``cfg.simulate``.
    """
    if model_type == "placement" and cfg.placement_simulate is not None:
        return cfg.placement_simulate
    return cfg.simulate


def _simulation_complete(cfg: PipelineConfig, gen: int, model_type: str | None = None) -> bool:
    """True if the generation's data directory has enough game files.

    When ``cfg.skip_until_gen`` is set and *gen* is below that
    threshold, the generation is considered complete (i.e. skipped).
    """
    if cfg.skip_until_gen is not None and gen < cfg.skip_until_gen:
        return True
    sim = _effective_simulate(cfg, model_type)
    data_dir = _data_path(cfg, gen, model_type)
    if not data_dir.is_dir():
        return False
    return _count_json_files(data_dir) >= sim.games


def _training_complete(cfg: PipelineConfig, gen: int, model_type: str | None = None) -> bool:
    """True if the generation's model checkpoint exists."""
    return _model_path(cfg, gen, model_type).is_file()


def _benchmarks_for_phase(cfg: PipelineConfig, phase: str) -> list[BenchmarkConfig]:
    """Return benchmarks that should run for a given phase.

    For single-model pipelines, all benchmarks are returned regardless
    of their *phase* field.  For combined pipelines, only benchmarks
    whose *phase* matches (or is ``"both"``) are included.
    """
    if cfg.model_type != "combined":
        return list(cfg.benchmarks)
    return [b for b in cfg.benchmarks if b.phase in (phase, "both")]


def _benchmark_complete(cfg: PipelineConfig, gen: int, phase: str | None = None) -> bool:
    """True if all applicable benchmark result files exist.

    When *phase* is given, only benchmarks matching that phase are
    checked.  Otherwise all benchmarks are checked.
    """
    if phase is not None:
        benchmarks = _benchmarks_for_phase(cfg, phase)
    else:
        benchmarks = cfg.benchmarks
    return all(_results_path(cfg, gen, bench.name).is_file() for bench in benchmarks)


def _generation_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if simulate, train, and benchmark are all done for a generation."""
    if cfg.model_type == "combined":
        return (
            _simulation_complete(cfg, gen, "placement")
            and _training_complete(cfg, gen, "placement")
            and _benchmark_complete(cfg, gen, "placement")
            and _simulation_complete(cfg, gen, "state")
            and _training_complete(cfg, gen, "state")
            and _benchmark_complete(cfg, gen, "combined")
        )
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


def _run(args: list[str], *, label: str, cwd: Path | None = None) -> None:
    """Run a command, streaming stdout and capturing stderr. Raises on non-zero exit."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(args)}")
    print(f"{'=' * 60}\n", flush=True)

    result = subprocess.run(args, cwd=cwd, stderr=subprocess.PIPE, text=True)
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
    state_model_path: Path | None = None,
    state_model_config: str | None = None,
    placement_model_path: Path | None = None,
    placement_model_config: str | None = None,
) -> dict[str, Any]:
    """Build the serve config dict for the inference server.

    Supports loading a state model, a placement model, or both.
    """
    config: dict[str, Any] = {
        "port": serve_cfg.port,
        "host": serve_cfg.host,
        "logLevel": serve_cfg.log_level,
        "gameConfig": game_config,
    }

    if state_model_path is not None:
        config["model"] = str(state_model_path)
        config["modelConfig"] = state_model_config

    if placement_model_path is not None:
        config["placementModel"] = str(placement_model_path)
        config["placementModelConfig"] = placement_model_config

    return config


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
        state_model_path: Path | None = None,
        state_model_config: str | None = None,
        placement_model_path: Path | None = None,
        placement_model_config: str | None = None,
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
            state_model_path=state_model_path,
            state_model_config=state_model_config,
            placement_model_path=placement_model_path,
            placement_model_config=placement_model_config,
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
            ]
            if state_model_path is not None:
                args.extend(["--model", str(state_model_path)])
                if state_model_config is not None:
                    args.extend(["--model-config", state_model_config])
            if placement_model_path is not None:
                args.extend(["--placement-model", str(placement_model_path)])
                if placement_model_config is not None:
                    args.extend(["--placement-model-config", placement_model_config])

        print(f"\n--- Starting inference server: {' '.join(args)}")
        self._proc = subprocess.Popen(args, cwd=cwd)

        # Wait for health check.
        url = f"http://{serve_cfg.host}:{serve_cfg.port}/health"
        self._wait_for_health(url, timeout=60)

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


_SERVER_AI_KINDS = frozenset({
    "server-mcts",
    "server-mcts-nn",
    "nn-mcts-placement",
    "nn-mcts-placement-random",
})


def _benchmarks_need_game_server(benchmarks: list[BenchmarkConfig]) -> bool:
    """Return True if any benchmark uses server-mcts or server-mcts-nn AI kinds."""
    return any(
        ai in _SERVER_AI_KINDS for bench in benchmarks for ai in bench.ai
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _step_simulate(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    nn_url: str | None,
    model_type: str | None = None,
    sim_override: SimulateConfig | None = None,
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
    out_dir = _data_path(cfg, gen, model_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if this generation is below the skip threshold.
    if cfg.skip_until_gen is not None and gen < cfg.skip_until_gen:
        print(f"  Simulate: gen {gen} < skipUntilGen {cfg.skip_until_gen}, skipping.")
        return

    sim = sim_override or _effective_simulate(cfg, model_type)
    target_games = sim.games
    existing = _count_json_files(out_dir)

    if existing >= target_games:
        print(f"  Simulate: {existing}/{target_games} games already exist, skipping.")
        return

    remaining = target_games - existing
    requested_games = math.ceil(remaining * max(sim.oversample, 1.0))

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
    if sim.max_prior_depth is not None:
        sim_config["maxPriorDepth"] = sim.max_prior_depth
    if sim.simulations_per_action is not None:
        sim_config["simulationsPerAction"] = sim.simulations_per_action
    if not sim.symmetries:
        sim_config["noSymmetries"] = True
    if sim.verbosity:
        sim_config["verbosity"] = sim.verbosity
    if cfg.seed is not None:
        sim_config["seed"] = cfg.seed + gen
    if nn_url is not None:
        sim_config["prior"] = True
        sim_config["nnUrl"] = nn_url
    effective_type = model_type or cfg.model_type
    if effective_type == "placement":
        sim_config["exportType"] = "InitialPlacement"

    config_path = _write_config(cfg, f"simulate_gen{gen}", sim_config)

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
        result = subprocess.run(args, cwd=project_root, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            stderr_msg = result.stderr.strip() if result.stderr else ""
            detail = f"{label} failed with exit code {result.returncode}"
            if stderr_msg:
                detail += f"\nstderr:\n{stderr_msg}"
            raise RuntimeError(detail)
        return

    # Oversample mode: launch as a background process and monitor the folder.
    proc = subprocess.Popen(args, cwd=project_root, stderr=subprocess.PIPE)
    try:
        while proc.poll() is None:
            count = _count_json_files(out_dir)
            if count >= target_games:
                print(
                    f"  Target reached: {count}/{target_games} games. Terminating simulation early."
                )
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return
            time.sleep(1.0)

        # Process exited on its own — check exit code.
        if proc.returncode != 0:
            stderr_msg = ""
            if proc.stderr is not None:
                stderr_bytes = proc.stderr.read()
                if isinstance(stderr_bytes, bytes):
                    stderr_msg = stderr_bytes.decode(errors="replace").strip()
                else:
                    stderr_msg = stderr_bytes.strip()
            detail = f"{label} failed with exit code {proc.returncode}"
            if stderr_msg:
                detail += f"\nstderr:\n{stderr_msg}"
            raise RuntimeError(detail)

        # Verify we got enough games even though process exited normally.
        count = _count_json_files(out_dir)
        if count < target_games:
            print(f"  WARNING: Simulation finished but only produced {count}/{target_games} games.")
    except BaseException:
        # Ensure subprocess is cleaned up on any error (including KeyboardInterrupt).
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise


def _count_json_files(directory: Path) -> int:
    """Count ``.json`` files in *directory* (non-recursive)."""
    return sum(1 for _ in directory.glob("*.json"))


def _step_train(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    model_type: str | None = None,
    sim_override: SimulateConfig | None = None,
) -> None:
    """Train the model for a generation. Skips if the checkpoint already exists."""
    out_path = _model_path(cfg, gen, model_type)
    if out_path.is_file():
        print(f"  Train: Model already exists at {out_path}, skipping.")
        return

    data_path = _data_path(cfg, gen, model_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tr = cfg.train

    effective_type = model_type or cfg.model_type
    effective_model_config = (
        cfg.placement_model_config or cfg.model_config
        if effective_type == "placement"
        else cfg.model_config
    )

    # Build config JSON for training.
    train_config: dict[str, Any] = {
        "data": str(data_path),
        "gameConfig": cfg.game_config,
        "modelConfig": effective_model_config,
        "modelType": effective_type,
        "out": str(out_path),
        "epochs": tr.epochs,
        "patience": tr.patience,
        "batchSize": tr.batch_size,
        "lr": tr.lr,
        "valSplit": tr.val_split,
        "testSplit": tr.test_split,
        "logInterval": tr.log_interval,
        "loss": tr.loss,
        "lossSigma": tr.loss_sigma,
        "target": tr.target,
    }

    # Enable per-epoch checkpointing if configured.
    if tr.checkpoint_dir:
        ckpt_dir = _checkpoint_path(cfg, gen, model_type)
        train_config["checkpointDir"] = str(ckpt_dir)

    # Resume from previous generation's model if available.
    if tr.resume_from_previous and gen > 0:
        prev_model = _model_path(cfg, gen - 1, model_type)
        if prev_model.exists():
            train_config["resume"] = str(prev_model)

    type_suffix = f"_{model_type}" if model_type else ""
    config_path = _write_config(cfg, f"train_gen{gen}{type_suffix}", train_config)

    args = [
        sys.executable,
        "-m",
        f"{cfg.python_module}.train",
        "--config",
        str(config_path),
    ]

    type_label = f" ({model_type})" if model_type else ""
    _run(args, label=f"Gen {gen}: Train{type_label}", cwd=project_root)


def _step_benchmark(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    nn_url: str,
    phase: str | None = None,
    gimbur_server: _GimburServerProcess | None = None,
) -> dict[str, Any]:
    """Run benchmarks for a generation. Returns aggregated results.

    When *phase* is given (used by combined pipelines), only benchmarks
    whose ``phase`` field matches are executed.  For single-model
    pipelines, *phase* should be ``None`` so all benchmarks run.

    Skips individual benchmarks whose result files already exist (resume).
    """
    gen_results: dict[str, Any] = {}

    benchmarks = _benchmarks_for_phase(cfg, phase) if phase is not None else cfg.benchmarks

    # Start the Gimbur.Server if any benchmark needs it.
    started_game_server = False
    if gimbur_server is not None and _benchmarks_need_game_server(benchmarks):
        gimbur_server.start(
            gimbur_server_cfg=cfg.gimbur_server,
            cwd=project_root,
        )
        started_game_server = True

    try:
        for bench in benchmarks:
            out_path = _results_path(cfg, gen, bench.name)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # If results already exist, load and skip.
            if out_path.is_file():
                print(f"  Benchmark '{bench.name}': results already exist, skipping.")
                gen_results[bench.name] = json.loads(out_path.read_text())
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
            if bench.server_max_prior_depth is not None:
                bench_config["serverMaxPriorDepth"] = bench.server_max_prior_depth

            config_path = _write_config(cfg, f"benchmark_gen{gen}_{bench.name}", bench_config)

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
                label=f"Gen {gen}: Benchmark '{bench.name}' ({bench.games} games)",
                cwd=project_root,
            )

            # Parse results.
            if out_path.exists():
                results = json.loads(out_path.read_text())
                gen_results[bench.name] = results

    finally:
        # Always stop the Gimbur.Server if we started it, even on error.
        if started_game_server and gimbur_server is not None:
            gimbur_server.stop()

    return gen_results


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
        return {wr["ai"]: wr["rate"] for wr in raw}
    # Already a dict (loaded from summary.json).
    return dict(raw)


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
                "draws": data.get("draws", 0),
                "totalGames": data.get("totalGames", 0),
            }
        summary.append(entry)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary saved to {summary_path}")


def _save_progress_chart(cfg: PipelineConfig, all_results: dict[int, dict[str, Any]]) -> None:
    """Save a win-rate progress chart (PNG) covering all generations so far.

    Plots the subject (first AI in each benchmark config) win rate on the
    y-axis against generation on the x-axis.  All benchmark configs appear
    as separate series on the same chart.  The file is overwritten after
    every benchmark phase so it always reflects the latest state.

    Requires ``matplotlib`` (optional ``pipeline`` dependency).  If not
    installed, silently skips.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not all_results:
        return

    # Build a lookup from benchmark name -> subject AI name (C# lowercase).
    bench_subjects: dict[str, str] = {}
    for bench in cfg.benchmarks:
        subject = bench.ai[0].replace("-", "").lower()
        bench_subjects[bench.name] = subject

    # Collect series: bench_name -> [(gen, win_rate), ...]
    series: dict[str, list[tuple[int, float]]] = {}
    for gen in sorted(all_results):
        for bench_name, data in all_results[gen].items():
            rates = _normalize_win_rates(data)
            subject = bench_subjects.get(bench_name)
            if subject is None:
                # Benchmark not in current config (leftover from earlier run);
                # fall back to first key in rates dict.
                subject = next(iter(rates), None)
            rate = rates.get(subject)
            if rate is not None:
                series.setdefault(bench_name, []).append((gen, rate))

    if not series:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for bench_name, points in sorted(series.items()):
        gens = [g for g, _ in points]
        rates = [r * 100 for _, r in points]
        ax.plot(gens, rates, marker="o", markersize=4, linewidth=1.5, label=bench_name)

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
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Progress chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# Combined pipeline generation
# ---------------------------------------------------------------------------


def _run_combined_generation(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    server: _ServerProcess,
    nn_url: str,
    all_results: dict[int, dict[str, Any]],
    gimbur_server: _GimburServerProcess | None = None,
) -> None:
    """Run a single generation of the combined (placement + state) pipeline.

    The six-step flow:
      1. Simulate placement states.
      2. Train placement model.
      3. Benchmark placement model (placement-phase benchmarks).
      4. Simulate value states (with latest placement + prev value model).
      5. Train value model.
      6. Benchmark combined (combined-phase benchmarks).
    """
    effective_placement_model_config = cfg.placement_model_config or cfg.model_config

    # ---------------------------------------------------------------
    # Step 1: Simulate placement states
    # ---------------------------------------------------------------
    sim_placement_done = _simulation_complete(cfg, gen, "placement")
    gen_nn_url: str | None = None

    if gen > 0 and not sim_placement_done:
        # Serve previous placement model as prior for placement simulation.
        prev_placement = _model_path(cfg, gen - 1, "placement")
        if not prev_placement.exists():
            raise FileNotFoundError(
                f"Placement model for gen {gen - 1} not found at {prev_placement}."
            )
        server.start(
            serve_cfg=cfg.serve,
            game_config=cfg.game_config,
            python_module=cfg.python_module,
            placement_model_path=prev_placement,
            placement_model_config=effective_placement_model_config,
            pipeline_cfg=cfg,
            cwd=project_root,
        )
        gen_nn_url = nn_url

    _step_simulate(
        cfg, gen, project_root, gen_nn_url,
        model_type="placement",
    )

    if not sim_placement_done:
        server.stop()

    # ---------------------------------------------------------------
    # Step 2: Train placement model
    # ---------------------------------------------------------------
    _step_train(cfg, gen, project_root, model_type="placement")

    # ---------------------------------------------------------------
    # Step 3: Benchmark placement model (placement-phase benchmarks)
    # ---------------------------------------------------------------
    placement_benchmarks = _benchmarks_for_phase(cfg, "placement")
    placement_model = _model_path(cfg, gen, "placement")

    if placement_benchmarks and placement_model.exists():
        bench_done = _benchmark_complete(cfg, gen, "placement")
        if not bench_done:
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                placement_model_path=placement_model,
                placement_model_config=effective_placement_model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )

        gen_results = _step_benchmark(cfg, gen, project_root, nn_url, phase="placement", gimbur_server=gimbur_server)

        if not bench_done:
            server.stop()

        if gen_results:
            _print_generation_summary(gen, gen_results)
            all_results.setdefault(gen, {}).update(gen_results)
            _save_summary(cfg, all_results)
            _save_progress_chart(cfg, all_results)

    # ---------------------------------------------------------------
    # Step 4: Simulate value states
    # ---------------------------------------------------------------
    sim_state_done = _simulation_complete(cfg, gen, "state")
    gen_nn_url = None

    if gen > 0 and not sim_state_done:
        # Serve latest placement model + previous value model for
        # state simulation (dual-model prior).
        state_model_kwargs: dict[str, Any] = {}
        prev_state = _model_path(cfg, gen - 1, "state")
        has_state_model = prev_state.exists()
        if has_state_model:
            state_model_kwargs["state_model_path"] = prev_state
            state_model_kwargs["state_model_config"] = cfg.model_config

        server.start(
            serve_cfg=cfg.serve,
            game_config=cfg.game_config,
            python_module=cfg.python_module,
            placement_model_path=placement_model,
            placement_model_config=effective_placement_model_config,
            pipeline_cfg=cfg,
            cwd=project_root,
            **state_model_kwargs,
        )
        # Only enable state priors when a state model is actually loaded;
        # without one the /state/ endpoints don't exist on the server and
        # prior requests would silently 404.
        if has_state_model:
            gen_nn_url = nn_url
    _step_simulate(cfg, gen, project_root, gen_nn_url, model_type="state")

    if gen > 0 and not sim_state_done:
        server.stop()

    # ---------------------------------------------------------------
    # Step 5: Train value model
    # ---------------------------------------------------------------
    _step_train(cfg, gen, project_root, model_type="state")

    # ---------------------------------------------------------------
    # Step 6: Benchmark combined (combined-phase benchmarks)
    # ---------------------------------------------------------------
    combined_benchmarks = _benchmarks_for_phase(cfg, "combined")
    state_model = _model_path(cfg, gen, "state")

    if combined_benchmarks and state_model.exists():
        bench_done = _benchmark_complete(cfg, gen, "combined")
        if not bench_done:
            # Serve both models for combined benchmarks.
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                state_model_path=state_model,
                state_model_config=cfg.model_config,
                placement_model_path=placement_model,
                placement_model_config=effective_placement_model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )

        gen_results = _step_benchmark(cfg, gen, project_root, nn_url, phase="combined", gimbur_server=gimbur_server)

        if not bench_done:
            server.stop()

        if gen_results:
            _print_generation_summary(gen, gen_results)
            all_results.setdefault(gen, {}).update(gen_results)
            _save_summary(cfg, all_results)
            _save_progress_chart(cfg, all_results)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _run_single_generation(
    cfg: PipelineConfig,
    gen: int,
    project_root: Path,
    server: _ServerProcess,
    nn_url: str,
    all_results: dict[int, dict[str, Any]],
    gimbur_server: _GimburServerProcess | None = None,
) -> None:
    """Run a single generation of a single-model (state or placement) pipeline."""
    # Determine which model-type param to use for serve.
    is_placement = cfg.model_type == "placement"
    effective_model_config = (
        cfg.placement_model_config or cfg.model_config if is_placement else cfg.model_config
    )

    # --- Simulate ---
    # Gen 0: no NN prior (greedy rollouts only).
    # Gen N>0: start server with gen N-1 model for prior evaluation.
    sim_already_done = _simulation_complete(cfg, gen)
    gen_nn_url: str | None = None
    if gen > 0 and not sim_already_done:
        prev_model = _model_path(cfg, gen - 1)
        if not prev_model.exists():
            raise FileNotFoundError(
                f"Model for gen {gen - 1} not found at {prev_model}. "
                f"Cannot run gen {gen} with NN prior."
            )
        if is_placement:
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                placement_model_path=prev_model,
                placement_model_config=effective_model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )
        else:
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                state_model_path=prev_model,
                state_model_config=effective_model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )
        gen_nn_url = nn_url

    _step_simulate(cfg, gen, project_root, gen_nn_url)

    # Stop server after simulation (not needed for training).
    if not sim_already_done:
        server.stop()

    # --- Train ---
    _step_train(cfg, gen, project_root)

    # --- Benchmark ---
    # Start server with this generation's model (only if needed).
    bench_already_done = _benchmark_complete(cfg, gen)
    model_path = _model_path(cfg, gen)
    if not model_path.exists():
        print(f"WARNING: Model not found at {model_path}, skipping benchmarks.")
        return

    if not bench_already_done:
        if is_placement:
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                placement_model_path=model_path,
                placement_model_config=effective_model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )
        else:
            server.start(
                serve_cfg=cfg.serve,
                game_config=cfg.game_config,
                python_module=cfg.python_module,
                state_model_path=model_path,
                state_model_config=effective_model_config,
                pipeline_cfg=cfg,
                cwd=project_root,
            )

    gen_results = _step_benchmark(cfg, gen, project_root, nn_url, gimbur_server=gimbur_server)

    if not bench_already_done:
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

    Dispatches to ``_run_combined_generation`` when ``model_type`` is
    ``"combined"``, otherwise runs the simpler single-model flow.
    """
    server = _ServerProcess()
    gimbur_server = _GimburServerProcess()
    nn_url = f"http://{cfg.serve.host}:{cfg.serve.port}"
    all_results: dict[int, dict[str, Any]] = {}

    # Load any existing summary.
    summary_path = Path(cfg.results_dir) / "summary.json"
    if summary_path.exists():
        for entry in json.loads(summary_path.read_text()):
            all_results[entry["generation"]] = entry.get("benchmarks", {})

    is_combined = cfg.model_type == "combined"

    try:
        for gen in range(start_gen, cfg.generations):
            # Check if entire generation is already done.
            if _generation_complete(cfg, gen):
                print(f"\n--- Generation {gen} already complete, skipping.")
                # Ensure results are loaded for summary tracking.
                if gen not in all_results:
                    gen_results: dict[str, Any] = {}
                    for bench in cfg.benchmarks:
                        rp = _results_path(cfg, gen, bench.name)
                        if rp.is_file():
                            gen_results[bench.name] = json.loads(rp.read_text())
                    if gen_results:
                        all_results[gen] = gen_results
                continue

            print(f"\n{'#' * 60}")
            print(f"  GENERATION {gen}")
            print(f"{'#' * 60}\n")

            if is_combined:
                _run_combined_generation(cfg, gen, project_root, server, nn_url, all_results, gimbur_server=gimbur_server)
            else:
                _run_single_generation(cfg, gen, project_root, server, nn_url, all_results, gimbur_server=gimbur_server)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config(args.config)

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
            return

    print(f"Project root: {project_root}")
    print(f"Generations:  {start_gen} .. {cfg.generations - 1}")
    print(f"Data dir:     {cfg.data_dir}")
    print(f"Model dir:    {cfg.model_dir}")
    print(f"Results dir:  {cfg.results_dir}")

    run_pipeline(cfg, start_gen, project_root)


if __name__ == "__main__":
    main()
