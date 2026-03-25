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
    resume_from_previous: bool = True


@dataclass
class ServeConfig:
    """Parameters for ``python -m gimbur_nn.serve``."""

    port: int = 8000
    host: str = "127.0.0.1"
    log_level: str = "warning"


@dataclass
class BenchmarkConfig:
    """A single benchmark run."""

    name: str = "nn-vs-greedy"
    games: int = 100
    ai: list[str] = field(default_factory=lambda: ["nn", "greedy"])


@dataclass
class PipelineConfig:
    """Top-level orchestrator configuration."""

    # Shared identifiers.
    map_config: str = "mini"
    game_config: str = "mini_2p"
    model_config: str = "small"
    model_type: str = "state"

    # Reproducibility.
    seed: int | None = None

    # Directories (relative to project root).
    data_dir: str = "pipeline/data"
    model_dir: str = "pipeline/models"
    results_dir: str = "pipeline/results"

    # How many generations to run.
    generations: int = 10

    # Paths to the CLI tools (relative to project root).
    dotnet_project: str = "src/Gimbur.Cli"
    python_module: str = "gimbur_nn"

    # Section configs.
    simulate: SimulateConfig = field(default_factory=SimulateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
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
        "model_type",
        "seed",
        "data_dir",
        "model_dir",
        "results_dir",
        "generations",
        "dotnet_project",
        "python_module",
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
    if "benchmarks" in raw:
        cfg.benchmarks = [_load_section(BenchmarkConfig, b) for b in raw["benchmarks"]]

    return cfg


def _to_camel(snake: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _load_section(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a camelCase JSON dict."""
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


def _results_path(cfg: PipelineConfig, gen: int, name: str) -> Path:
    return Path(cfg.results_dir) / f"gen{gen}" / f"{name}.json"


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
    """True if the generation's data directory has enough game files."""
    data_dir = _data_path(cfg, gen)
    if not data_dir.is_dir():
        return False
    return _count_json_files(data_dir) >= cfg.simulate.games


def _training_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if the generation's model checkpoint exists."""
    return _model_path(cfg, gen).is_file()


def _benchmark_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if all configured benchmark result files exist."""
    return all(_results_path(cfg, gen, bench.name).is_file() for bench in cfg.benchmarks)


def _generation_complete(cfg: PipelineConfig, gen: int) -> bool:
    """True if simulate, train, and benchmark are all done for a generation."""
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


class _ServerProcess:
    """Manages the lifecycle of the inference server subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def start(
        self,
        *,
        model_path: Path,
        game_config: str,
        model_config: str,
        model_type: str = "state",
        serve_cfg: ServeConfig,
        python_module: str,
        pipeline_cfg: PipelineConfig | None = None,
        cwd: Path | None = None,
    ) -> None:
        if self._proc is not None:
            self.stop()

        # Fail fast if the port is already occupied by another process.
        self._check_port_available(serve_cfg.host, serve_cfg.port)

        # Build serve config JSON.
        serve_config: dict[str, Any] = {
            "model": str(model_path),
            "gameConfig": game_config,
            "modelConfig": model_config,
            "modelType": model_type,
            "port": serve_cfg.port,
            "host": serve_cfg.host,
            "logLevel": serve_cfg.log_level,
        }

        if pipeline_cfg is not None:
            config_path = _write_config(pipeline_cfg, "serve", serve_config)
            args = [
                sys.executable, "-m", f"{python_module}.serve",
                "--config", str(config_path),
            ]
        else:
            args = [
                sys.executable,
                "-m",
                f"{python_module}.serve",
                "--model",
                str(model_path),
                "--game-config",
                game_config,
                "--model-config",
                model_config,
                "--model-type",
                model_type,
                "--port",
                str(serve_cfg.port),
                "--host",
                serve_cfg.host,
                "--log-level",
                serve_cfg.log_level,
            ]

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
        """Raise if another process is already listening on host:port."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                raise RuntimeError(
                    f"Port {port} on {host} is already in use. "
                    f"Kill the existing process before running the pipeline."
                )

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


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _step_simulate(cfg: PipelineConfig, gen: int, project_root: Path, nn_url: str | None) -> None:
    """Run self-play simulation for a generation.

    Uses ``--export-format json`` so each game is written to its own
    ``.json`` file in the generation data directory.  When
    ``oversample > 1.0``, requests more games than needed and monitors
    the output folder, terminating the CLI once the target game count
    is reached.  This avoids blocking on long-tail slow games.

    On resume, counts existing ``.json`` files and only requests the
    remaining games needed to reach the target.
    """
    out_dir = _data_path(cfg, gen)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = cfg.simulate
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
    if not sim.symmetries:
        sim_config["noSymmetries"] = True
    if sim.verbosity:
        sim_config["verbosity"] = sim.verbosity
    if cfg.seed is not None:
        sim_config["seed"] = cfg.seed + gen
    if nn_url is not None:
        sim_config["prior"] = True
        sim_config["nnUrl"] = nn_url
    if cfg.model_type == "placement":
        sim_config["exportType"] = "InitialPlacement"

    config_path = _write_config(cfg, f"simulate_gen{gen}", sim_config)

    args = [
        "dotnet", "run", "--project", cfg.dotnet_project,
        "--", "simulate", "--config", str(config_path),
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


def _step_train(cfg: PipelineConfig, gen: int, project_root: Path) -> None:
    """Train the model for a generation. Skips if the checkpoint already exists."""
    out_path = _model_path(cfg, gen)
    if out_path.is_file():
        print(f"  Train: Model already exists at {out_path}, skipping.")
        return

    data_path = _data_path(cfg, gen)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tr = cfg.train

    # Build config JSON for training.
    train_config: dict[str, Any] = {
        "data": str(data_path),
        "gameConfig": cfg.game_config,
        "modelConfig": cfg.model_config,
        "modelType": cfg.model_type,
        "out": str(out_path),
        "epochs": tr.epochs,
        "patience": tr.patience,
        "batchSize": tr.batch_size,
        "lr": tr.lr,
        "valSplit": tr.val_split,
        "testSplit": tr.test_split,
        "logInterval": tr.log_interval,
    }

    # Enable per-epoch checkpointing if configured.
    if tr.checkpoint_dir:
        ckpt_dir = _checkpoint_path(cfg, gen)
        train_config["checkpointDir"] = str(ckpt_dir)

    # Resume from previous generation's model if available.
    if tr.resume_from_previous and gen > 0:
        prev_model = _model_path(cfg, gen - 1)
        if prev_model.exists():
            train_config["resume"] = str(prev_model)

    config_path = _write_config(cfg, f"train_gen{gen}", train_config)

    args = [
        sys.executable, "-m", f"{cfg.python_module}.train",
        "--config", str(config_path),
    ]

    _run(args, label=f"Gen {gen}: Train", cwd=project_root)


def _step_benchmark(
    cfg: PipelineConfig, gen: int, project_root: Path, nn_url: str
) -> dict[str, Any]:
    """Run all configured benchmarks for a generation. Returns aggregated results.

    Skips individual benchmarks whose result files already exist (resume).
    """
    gen_results: dict[str, Any] = {}

    for bench in cfg.benchmarks:
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

        config_path = _write_config(
            cfg, f"benchmark_gen{gen}_{bench.name}", bench_config
        )

        args = [
            "dotnet", "run", "--project", cfg.dotnet_project,
            "--", "benchmark", "--config", str(config_path),
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_pipeline(cfg: PipelineConfig, start_gen: int, project_root: Path) -> None:
    """Execute the full self-play training pipeline.

    Supports step-level resume: each step (simulate, train, benchmark)
    checks whether its artifacts already exist and skips if so.
    """
    server = _ServerProcess()
    nn_url = f"http://{cfg.serve.host}:{cfg.serve.port}"
    all_results: dict[int, dict[str, Any]] = {}

    # Load any existing summary.
    summary_path = Path(cfg.results_dir) / "summary.json"
    if summary_path.exists():
        for entry in json.loads(summary_path.read_text()):
            all_results[entry["generation"]] = entry.get("benchmarks", {})

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
                server.start(
                    model_path=prev_model,
                    game_config=cfg.game_config,
                    model_config=cfg.model_config,
                    model_type=cfg.model_type,
                    serve_cfg=cfg.serve,
                    python_module=cfg.python_module,
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
                continue

            if not bench_already_done:
                server.start(
                    model_path=model_path,
                    game_config=cfg.game_config,
                    model_config=cfg.model_config,
                    model_type=cfg.model_type,
                    serve_cfg=cfg.serve,
                    python_module=cfg.python_module,
                    pipeline_cfg=cfg,
                    cwd=project_root,
                )

            gen_results = _step_benchmark(cfg, gen, project_root, nn_url)

            if not bench_already_done:
                server.stop()

            # --- Report ---
            if gen_results:
                _print_generation_summary(gen, gen_results)
                all_results[gen] = gen_results
                _save_summary(cfg, all_results)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    finally:
        server.stop()
        if all_results:
            _save_summary(cfg, all_results)
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
