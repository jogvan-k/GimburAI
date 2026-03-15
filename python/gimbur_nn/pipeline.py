"""
Self-play training pipeline orchestrator.

Drives the AlphaZero-style loop: simulate → train → benchmark → repeat.
Each iteration is called a "generation". Generation 0 uses greedy rollouts
(no NN prior); subsequent generations use the previous generation's model
as the MCTS prior evaluator.

Usage:
    python -m gimbur_nn.pipeline --config pipeline.json
    python -m gimbur_nn.pipeline --config pipeline.json --start-gen 3  # resume
"""

from __future__ import annotations

import argparse
import json
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
    symmetries: bool = True
    verbosity: str = "quiet"


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


@dataclass
class ServeConfig:
    """Parameters for ``python -m gimbur_nn.serve``."""

    port: int = 8000
    host: str = "127.0.0.1"


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
    return Path(cfg.data_dir) / f"gen{gen}.jsonl"


def _model_path(cfg: PipelineConfig, gen: int) -> Path:
    return Path(cfg.model_dir) / f"gen{gen}.pt"


def _results_path(cfg: PipelineConfig, gen: int, name: str) -> Path:
    return Path(cfg.results_dir) / f"gen{gen}" / f"{name}.json"


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _run(args: list[str], *, label: str, cwd: Path | None = None) -> None:
    """Run a command, streaming output. Raises on non-zero exit."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(args)}")
    print(f"{'=' * 60}\n", flush=True)

    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


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
        serve_cfg: ServeConfig,
        python_module: str,
        cwd: Path | None = None,
    ) -> None:
        if self._proc is not None:
            self.stop()

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
            "--port",
            str(serve_cfg.port),
            "--host",
            serve_cfg.host,
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

    def _wait_for_health(self, url: str, timeout: float) -> None:
        """Poll the /health endpoint until the server is ready."""
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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
    """Run self-play simulation for a generation."""
    out_path = _data_path(cfg, gen)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim = cfg.simulate
    args = [
        "dotnet",
        "run",
        "--project",
        cfg.dotnet_project,
        "--",
        "simulate",
        "--games",
        str(sim.games),
        "--players",
        str(sim.players),
        "--map-config",
        cfg.map_config,
        "--export",
        str(out_path),
        "--search-time",
        str(sim.search_time_ms),
        "--max-simulations",
        str(sim.max_simulations),
        "--max-rollout-depth",
        str(sim.max_rollout_depth),
    ]

    if sim.action_rollout_limit is not None:
        args.extend(["--action-rollout-limit", str(sim.action_rollout_limit)])

    if not sim.symmetries:
        args.append("--no-symmetries")

    if sim.verbosity:
        args.extend(["--verbosity", sim.verbosity])

    if cfg.seed is not None:
        args.extend(["--seed", str(cfg.seed + gen)])

    if nn_url is not None:
        args.extend(["--prior", "--nn-url", nn_url])

    _run(args, label=f"Gen {gen}: Simulate ({sim.games} games)", cwd=project_root)


def _step_train(cfg: PipelineConfig, gen: int, project_root: Path) -> None:
    """Train the model for a generation."""
    data_path = _data_path(cfg, gen)
    out_path = _model_path(cfg, gen)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tr = cfg.train
    args = [
        sys.executable,
        "-m",
        f"{cfg.python_module}.train",
        "--data",
        str(data_path),
        "--game-config",
        cfg.game_config,
        "--model-config",
        cfg.model_config,
        "--out",
        str(out_path),
        "--epochs",
        str(tr.epochs),
        "--patience",
        str(tr.patience),
        "--batch-size",
        str(tr.batch_size),
        "--lr",
        str(tr.lr),
        "--val-split",
        str(tr.val_split),
        "--test-split",
        str(tr.test_split),
        "--log-interval",
        str(tr.log_interval),
    ]

    # Resume from previous generation's model if available.
    if gen > 0:
        prev_model = _model_path(cfg, gen - 1)
        if prev_model.exists():
            args.extend(["--resume", str(prev_model)])

    _run(args, label=f"Gen {gen}: Train", cwd=project_root)


def _step_benchmark(
    cfg: PipelineConfig, gen: int, project_root: Path, nn_url: str
) -> dict[str, Any]:
    """Run all configured benchmarks for a generation. Returns aggregated results."""
    gen_results: dict[str, Any] = {}

    for bench in cfg.benchmarks:
        out_path = _results_path(cfg, gen, bench.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "dotnet",
            "run",
            "--project",
            cfg.dotnet_project,
            "--",
            "benchmark",
            "--games",
            str(bench.games),
            "--ai",
            *bench.ai,
            "--map-config",
            cfg.map_config,
            "--output",
            str(out_path),
            "--nn-url",
            nn_url,
        ]

        if cfg.seed is not None:
            args.extend(["--seed", str(cfg.seed + gen * 1000)])

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
    """Print a summary of benchmark results for a generation."""
    print(f"\n{'=' * 60}")
    print(f"  Generation {gen} — Results Summary")
    print(f"{'=' * 60}")

    for name, data in results.items():
        print(f"\n  {name}:")
        for wr in data.get("winRates", []):
            pct = wr["rate"] * 100
            print(f"    {wr['ai']:>8s}: {pct:5.1f}% ({wr['wins']}/{data['totalGames']})")
        draws = data.get("draws", 0)
        if draws > 0:
            print(f"    {'draws':>8s}: {draws}/{data['totalGames']}")

    print()


def _save_summary(cfg: PipelineConfig, all_results: dict[int, dict[str, Any]]) -> None:
    """Save a summary JSON tracking results across all generations."""
    summary_path = Path(cfg.results_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for gen, results in sorted(all_results.items()):
        entry: dict[str, Any] = {"generation": gen, "benchmarks": {}}
        for name, data in results.items():
            win_rates = {wr["ai"]: wr["rate"] for wr in data.get("winRates", [])}
            entry["benchmarks"][name] = {
                "winRates": win_rates,
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
    """Execute the full self-play training pipeline."""
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
            print(f"\n{'#' * 60}")
            print(f"  GENERATION {gen}")
            print(f"{'#' * 60}\n")

            # --- Simulate ---
            # Gen 0: no NN prior (greedy rollouts only).
            # Gen N>0: start server with gen N-1 model for prior evaluation.
            gen_nn_url: str | None = None
            if gen > 0:
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
                    serve_cfg=cfg.serve,
                    python_module=cfg.python_module,
                    cwd=project_root,
                )
                gen_nn_url = nn_url

            _step_simulate(cfg, gen, project_root, gen_nn_url)

            # Stop server after simulation (not needed for training).
            server.stop()

            # --- Train ---
            _step_train(cfg, gen, project_root)

            # --- Benchmark ---
            # Start server with this generation's model.
            model_path = _model_path(cfg, gen)
            if not model_path.exists():
                print(f"WARNING: Model not found at {model_path}, skipping benchmarks.")
                continue

            server.start(
                model_path=model_path,
                game_config=cfg.game_config,
                model_config=cfg.model_config,
                serve_cfg=cfg.serve,
                python_module=cfg.python_module,
                cwd=project_root,
            )

            gen_results = _step_benchmark(cfg, gen, project_root, nn_url)

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
        default=0,
        help="Generation to start from (for resuming). Default: 0.",
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

    print(f"Project root: {project_root}")
    print(f"Generations:  {args.start_gen} .. {cfg.generations - 1}")
    print(f"Data dir:     {cfg.data_dir}")
    print(f"Model dir:    {cfg.model_dir}")
    print(f"Results dir:  {cfg.results_dir}")

    run_pipeline(cfg, args.start_gen, project_root)


if __name__ == "__main__":
    main()
