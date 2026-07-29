# Self-Play Training Pipeline

The pipeline orchestrator drives an AlphaZero-style training loop for
GimburAI: simulate self-play games, train a neural network on the
resulting data, benchmark the new model, and repeat. Each iteration of
this loop is called a **generation**.

## Quick Start

```bash
# From the project root:
python -m gimbur_nn.pipeline --config pipeline.json
```

Copy `pipeline.example.json` as a starting point and adjust parameters
to your hardware and goals.

## Generation Flow

```
Gen 0 (no NN prior — greedy rollouts)
  1. Simulate  →  pipeline/data/gen0/*.json
  2. Train     →  pipeline/models/gen0.pt
  3. Benchmark →  pipeline/results/gen0/{name}.json

Gen N (N > 0 — uses gen N-1 model as prior)
  1. Start inference server with gen N-1 model
  2. Simulate  →  pipeline/data/genN/*.json
  3. Stop server
  4. Train (resume from gen N-1 weights) →  pipeline/models/genN.pt
  5. Start inference server with gen N model
  6. Benchmark →  pipeline/results/genN/{name}.json
  7. Stop server
```

After each generation, a cumulative summary is saved to
`pipeline/results/summary.json`.

## Configuration

The pipeline reads a single JSON config file. Standard JSON with `//`
line comments is supported. Keys use **camelCase**; the orchestrator
maps them to Python `snake_case` internally.

### Top-Level Fields

| Key             | Type     | Default            | Description |
|-----------------|----------|--------------------|-------------|
| `mapConfig`     | string   | `"mini"`           | Board layout preset passed to the CLI. |
| `gameConfig`    | string   | `"mini_2p"`        | Game rules preset (players, VP target, etc.). |
| `modelConfig`   | string   | `"small"`          | Neural network architecture preset. |
| `placementModelConfig` | string/null | `null` | Optional placement architecture preset; defaults to `modelConfig`. |
| `modelType`     | string   | `"state"`          | Pipeline mode: `state`, `placement`, or `combined`. |
| `seed`          | int/null | `null`             | Base seed for reproducibility. Each gen offsets it. |
| `dataDir`       | string   | `"pipeline/data"`  | Root directory for per-generation game data. |
| `modelDir`      | string   | `"pipeline/models"`| Directory for model checkpoints. |
| `resultsDir`    | string   | `"pipeline/results"`| Directory for benchmark results. |
| `generations`   | int      | `10`               | Number of generations to run (0 .. N-1). |
| `dotnetProject` | string   | `"src/Gimbur.Cli"` | Path to the C# CLI project. |
| `pythonModule`  | string   | `"gimbur_nn"`      | Python package name for train/serve. |

### `simulate` Section

| Key                  | Type   | Default   | Description |
|----------------------|--------|-----------|-------------|
| `games`              | int    | `1000`    | Target number of self-play games per generation. |
| `players`            | int    | `2`       | Number of players per game. |
| `searchTimeMs`       | int    | `500`     | MCTS time budget per move (ms). |
| `maxSimulations`     | int    | `200`     | MCTS simulation cap per move. |
| `maxRolloutDepth`    | int    | `500`     | Max depth for random rollouts. |
| `actionRolloutLimit` | int    | *(none)*  | Cap per-action rollouts (omit to disable). |
| `symmetries`         | bool   | `true`    | Export rotational symmetries of each game state. |
| `verbosity`          | string | `"quiet"` | CLI verbosity level. |
| `oversample`         | float  | `1.0`     | Request `oversample * remaining` games and stop early. `1.1` = 10% extra. |

### `train` Section

| Key           | Type  | Default | Description |
|---------------|-------|---------|-------------|
| `epochs`      | int   | `0`     | Max epochs (`0` = unlimited, uses patience). |
| `patience`    | int   | `5`     | Early-stop after N epochs without improvement. |
| `batchSize`   | int   | `64`    | Training batch size. |
| `lr`          | float | `1e-4`  | Learning rate. |
| `valSplit`    | float | `0.1`   | Fraction of data for validation. |
| `testSplit`   | float | `0.0`   | Fraction of data for testing. |
| `logInterval` | int   | `50`    | Print training stats every N batches. |
| `outputMode`  | string | `"value"` | Placement head topology: `value` or `combined`. State pipelines use `value`. |
| `target`      | string | `"winrate"` | Placement data target; combined output uses dense visit-share policy targets. |
| `valueLossWeight` | float | `1.0` | Value-loss weight for combined placement training. |
| `policyLossWeight` | float | `1.0` | Masked policy-loss weight for combined placement training. |

## Placement Architecture

Placement checkpoints use `checkpoint_version: 2` and `architecture: "placement_state_v2"`. The model input is only the serialized placement state; the tokenizer action vocabulary supplies dense policy output indices, not input tokens. Placement models support a value-only head or combined heads: value logits are `[B,128]`, and combined policy logits are `[B,A]` with `A=60/82/144` for mini/small/standard.

The model emits raw logits. Combined training constructs a legal mask from every exported legal composite action and applies it to policy loss. Policy targets must come from standard shared-root MCTS visit shares; `simulationsPerAction` rollout counts are independent evaluation budgets and are unsuitable. Serving softmaxes the full vocabulary, while C# remains authoritative for legality and masks and normalizes policy probabilities before MCTS use.

Old action-conditioned placement checkpoints do not match this architecture and cannot be resumed or served; retrain them as version 2 checkpoints.

### `serve` Section

| Key        | Type   | Default       | Description |
|------------|--------|---------------|-------------|
| `port`     | int    | `8000`        | HTTP port for the inference server. |
| `host`     | string | `"127.0.0.1"` | Bind address. |
| `logLevel` | string | `"warning"`   | Uvicorn log level. Use `"warning"` to suppress HTTP 200/202 access logs; `"info"` to see them. |

### `benchmarks` Section

An array of benchmark configurations. Each entry runs independently
after training.

| Key     | Type     | Default             | Description |
|---------|----------|---------------------|-------------|
| `name`  | string   | `"nn-vs-greedy"`    | Name used for the result file. |
| `games` | int      | `100`               | Number of benchmark games. |
| `ai`    | string[] | `["nn", "greedy"]`  | AI player types for the matchup. |

### Example Config

See [`pipeline.example.json`](../pipeline.example.json) in the project
root for a fully commented example.

## Artifact Layout

```
pipeline/
├── data/
│   ├── gen0/          # One .json file per game
│   │   ├── a1b2c3.json
│   │   └── ...
│   ├── gen1/
│   └── ...
├── models/
│   ├── gen0.pt        # Versioned checkpoint with model weights and metadata
│   ├── gen1.pt
│   └── ...
└── results/
    ├── gen0/
    │   ├── nn-vs-greedy.json
    │   └── nn-vs-random.json
    ├── gen1/
    │   └── ...
    └── summary.json   # Cumulative win rates across all generations
```

## Resume-on-Interrupt

The pipeline is designed to survive interruptions (Ctrl+C, crashes,
OOM kills) and automatically resume from where it left off.

### How It Works

On startup, the pipeline scans artifact directories to determine
progress. Three **completion markers** define the state of each
generation:

| Step       | Completion Marker |
|------------|-------------------|
| Simulate   | `data/genN/` directory contains >= `simulate.games` `.json` files |
| Train      | `models/genN.pt` file exists |
| Benchmark  | All `results/genN/{name}.json` files exist (one per benchmark) |

A generation is fully complete when all three markers are present.

### Auto-Detection

When `--start-gen` is not specified, the pipeline calls
`_detect_resume_gen()` which scans from generation 0 upward and returns
the first incomplete generation. Within that generation, each step also
checks its own markers:

- **Simulate**: counts existing `.json` files and only requests the
  *remaining* games (with oversample applied to the remainder, not the
  total). If enough files already exist, the step is skipped entirely.
- **Train**: if the model checkpoint already exists, training is
  skipped.
- **Benchmark**: each benchmark is checked individually. Completed
  benchmarks are loaded from disk; only missing ones are executed. The
  inference server is only started if at least one benchmark needs
  running.

### Usage

```bash
# First run — starts from gen 0
python -m gimbur_nn.pipeline --config pipeline.json

# Interrupted! Restart — auto-detects where to resume
python -m gimbur_nn.pipeline --config pipeline.json

# Or explicitly override the start generation
python -m gimbur_nn.pipeline --config pipeline.json --start-gen 3
```

### Server Lifecycle on Resume

The pipeline avoids starting the inference server when it isn't needed:

- If simulation is already complete, no server is started for the
  simulate step (even for gen > 0).
- If all benchmarks are complete, no server is started for the
  benchmark step.
- The server is always stopped between simulate and train, and between
  train and benchmark, to free resources.

## Log Level

The inference server (uvicorn) logs every HTTP request by default,
producing many `200 OK` / `202 Accepted` lines during simulation and
benchmarking. To suppress these while keeping error logs visible, set
`logLevel` to `"warning"` in the `serve` config section (this is the
default).

| logLevel    | Access logs | Error logs |
|-------------|-------------|------------|
| `"debug"`   | Yes         | Yes        |
| `"info"`    | Yes         | Yes        |
| `"warning"` | No          | Yes        |
| `"error"`   | No          | Yes        |

The `--log-level` flag can also be passed directly when running the
inference server standalone:

```bash
python -m gimbur_nn.serve --model gen0.pt --game-config mini_2p \
    --model-config small --log-level warning
```

## CLI Reference

```
python -m gimbur_nn.pipeline --config PATH [--start-gen N] [--project-root DIR]
```

| Flag             | Required | Default      | Description |
|------------------|----------|--------------|-------------|
| `--config`       | Yes      | —            | Path to pipeline JSON config. |
| `--start-gen`    | No       | *(auto)*     | Generation to start from. Auto-detects if omitted. |
| `--project-root` | No       | *(auto)*     | Project root. Auto-detected from package location. |
