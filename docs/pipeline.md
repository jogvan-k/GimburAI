# Self-Play Training Pipeline

The current full-game policy/value contract is documented in
[complete-policy-value-model.md](complete-policy-value-model.md).

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
Gen 0 (no NN prior; greedy placement policy labels, stochastic rollout values)
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
| `trainingMode`  | string   | `"single"`         | Set to `"placement-and-state"` for one shared simulation corpus and two models. |
| `stateModelConfig` | string/null | `null` | State architecture preset; defaults to `modelConfig`. |
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
| `placementSearchTimeMs` | int | `16000` | Placement budget in `placement-and-state` mode. |
| `mainGameSearchTimeMs` | int | `8000` | Normal-play budget in `placement-and-state` mode. |
| `maxSimulations`     | int    | `200`     | MCTS simulation cap per move. |
| `maxRolloutDepth`    | int    | `500`     | Max depth for random rollouts. |
| `actionRolloutLimit` | int    | *(none)*  | Cap per-action rollouts (omit to disable). |
| `symmetries`         | bool   | `true`    | Export rotational symmetries of each game state. |
| `verbosity`          | string | `"quiet"` | CLI verbosity level. |
| `oversample`         | float  | `1.0`     | Request `oversample * remaining` games and stop early. `1.1` = 10% extra. |
| `parallelism`        | int    | *(auto)*  | Maximum concurrent games. Defaults to 4 with NN priors, otherwise all logical processors. |
| `maxPendingEvaluations` | int | `32` | Maximum outstanding neural leaf requests per search. |
| `leafEvaluationTimeoutMs` | int | `500` | Request timeout before rollout fallback. |
| `drainTimeoutMs` | int | `1000` | Maximum post-deadline response drain time. |
| `maxErrorsPerGame` | int | `5` | Discard a game when hard evaluation errors exceed this count. |
| `maxErrorRatePerGame` | float | `0.02` | Discard when hard errors/submitted requests exceeds this rate. |
| `minimumRequestsForRate` | int | `50` | Minimum submitted requests before applying the per-game rate. |
| `discardGamesWithFallbacks` | bool | `false` | Discard any game that used rollout fallback. |
| `maxDiscardedGames` | int | `20` | Stop generation after this many discarded games are exceeded. |
| `maxDiscardRate` | float | `0.05` | Stop when discarded/attempted exceeds this rate. |
| `minimumAttemptsForDiscardRate` | int | `50` | Minimum attempts before applying the discard rate. |
| `maxConsecutiveDiscards` | int | `5` | Stop after this completion-order discard streak is exceeded. |
| `greedyPrior` | bool | `true` | Generation 0 only: use the greedy AI as the local policy source for placement and state PUCT. |
| `greedyPriorUniformMix` | float | `0.25` | Uniform exploration mixed into the one-hot greedy policy before PUCT; the exported `modelPrior` remains raw `1/0`. |

Hard errors are leaf evaluation timeouts, invalid responses, and orphan responses.
Deadline cancellation and rollout fallback are reported separately. Accepted games are
written directly under `data/genN/`; compact rejected-game records are written under
`data/genN/discarded/`. The monitor counts only direct `.json` children toward the target,
observes discarded records during oversampling, and stops promptly when a generation gate
is exceeded. The CLI itself retries attempts until the requested accepted count is reached,
so `oversample` remains useful only for terminating long-tail in-flight work early.

### `train` Section

| Key           | Type  | Default | Description |
|---------------|-------|---------|-------------|
| `enabled`     | bool  | `true`  | Train this model. In `placement-and-state` mode, `false` freezes the corresponding model. |
| `epochs`      | int   | `0`     | Max epochs (`0` = unlimited, uses patience). |
| `patience`    | int   | `5`     | Early-stop after N epochs without improvement. |
| `batchSize`   | int   | `64`    | Training batch size. |
| `lr`          | float | `1e-4`  | Learning rate. |
| `valSplit`    | float | `0.1`   | Fraction of data for validation. |
| `testSplit`   | float | `0.0`   | Fraction of data for testing. |
| `logInterval` | int   | `50`    | Print training stats every N batches. |
| `outputMode`  | string | `"value"` | Placement head topology: `value` or `combined`. State pipelines use `value`. |
| `target`      | string | `"winrate"` | Placement data target; combined output uses dense visit-share policy targets. |
| `mctsValueWeightStart` | float | `0.9` | Initial weight of normalized MCTS wins in value targets. Placement states use this weight. |
| `mctsValueWeightEnd` | float | `0.1` | Final MCTS weight reached linearly at `turnNumber / game.turns = 1`. |
| `victoryPointSamplingStatistic` | string | `"median"` | Full-state only: bucket-size reference statistic, `median` or `average`. |
| `victoryPointSamplingUpperPercentage` | float | `0.10` | Full-state only: fraction above the reference used for the adaptive bucket cap. |
| `valueLossWeight` | float | `1.0` | Value-loss weight for combined placement training. |
| `policyLossWeight` | float | `1.0` | Masked policy-loss weight for combined placement training. |
| `policyTargetTemperature` | float | `1.0` | Placement only: sharpen (`<1`) or flatten (`>1`) positive legal visit-share targets; must be greater than zero. |

## Placement Architecture

Placement checkpoints use the current-only `architecture: "placement_stage_policy"`; state checkpoints require version 4 and `architecture: "state_player_value_v1"`. Version 4 adds exact development-deck and winner tokens. Placement models always emit value logits `[B,N]` and one policy head `[B,max(V,6)]`. Settlement stages use the first `V` coordinates and road stages use the first six in `N, NE, SE, S, SW, NW` order.

Value labels blend normalized per-player MCTS values with the game's one-hot final winner. For full states, the MCTS weight is `start + (end - start) * clamp(turnNumber / max(1, game.turns), 0, 1)`; placement states use `mctsValueWeightStart`. Placement values are built from root action value sums weighted by completed visits. If either source is invalid or has no positive evidence, the available source is used alone; samples with neither source are skipped.

Full-state sampling runs independently inside each dataset after the game-level train/validation/test split and before symmetry or player rotation. States from all games in that split are grouped by `floor(sum(player victory points))`. The median (default) or average bucket size is computed before augmentation, and the cap is `ceil(reference * (1 + victoryPointSamplingUpperPercentage))`. Buckets at or below the cap, including short high-VP tails, are retained fully; oversized buckets are sampled deterministically from dataset/game seeds and the total-VP bucket. The exact first normal-play state (`turnNumber: 1`, `stage: "r"`) and final exported root of every game are mandatory, even when they exceed the cap. This per-split policy means train, validation, and test can have different adaptive caps. Legacy games without per-state `scores` remain usable for placement replay but are excluded from state replay. Placement datasets do not use this sampling policy.

The loader emits one sample for each exported state and symmetry. It validates stage and pending markers and maps settlement vertex or road direction `policyIndex` values directly. Generation 0 exports the raw one-hot greedy `modelPrior`; MCTS mixes it with `greedyPriorUniformMix` uniform support so regular PUCT can still explore alternatives. Training uses the raw one-hot prior for the bootstrap policy target. Later generations export soft neural `modelPrior` values but train policy from improved root-edge visit shares. `policyTargetTemperature` transforms positive shares by exponent `1 / temperature` and renormalizes them. The model emits raw fixed-width stage-policy logits; legality masking remains external.

Old action-conditioned and bucket-value checkpoints cannot be resumed or served; retrain them as version 3 checkpoints.

### Asynchronous MCTS Values

State-model MCTS submits locally coalesced leaf batches through `/state/leaf-predict`,
which synchronously returns full `[B,N]` player distributions. The endpoint flattens
requests into one model invocation and then restores request boundaries; all outcomes
of one stochastic action therefore stay in one request and model batch. Searches
reserve pending tree edges and sleep on a client completion event when the tree is
blocked, allowing the shared evaluator to
spend that time batching other games. Completed evaluations, not submissions, are
the simulation and visit counts.

The C# transport uses a bounded local queue and a dedicated sender to coalesce leaves
from concurrent games into short-window batches without blocking MCTS threads. Each
direct response must contain every request ID exactly once; malformed or failed batches
become immediate invalid responses. Cancellation removes local ownership, omits work
not yet sent, and discards responses that arrive after an in-flight cancellation. The
older enqueue, collect, and cancel endpoints remain temporarily for compatibility.

In `placement-and-state` simulation, main-game MCTS uses uniform PUCT with the
state model as its leaf evaluator. It does not also request child-state value
"priors": that would duplicate state-model inference for every expansion and
duplicate state-model inference. Placement MCTS still uses the placement policy head
for PUCT and the state value model at the exact `PreRoll` horizon.

Placement prior responses already contain their node's full player distribution.
Kjarni consumes that value when the prior response is available at expansion, so a
placement policy/value prediction can serve both purposes without another HTTP
request. Late placement values are retained on the node for later use and are not
raced against an already-started rollout.

`summary.json` and `progress.png` are refreshed after every individual benchmark result, so partial generation progress is visible while later benchmarks are still running.

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
| `games` | int      | `10000`             | Number of benchmark games. At 10,000, the worst-case 95% Wald margin is 0.98 percentage points. |
| `ai`    | string[] | `["nn", "greedy"]`  | AI player types for the matchup. |
| `parallelism` | int/null | `null` | Override concurrent benchmark games; NN/server benchmarks otherwise use up to 4. |

For placement-and-state pipelines, the focused benchmark AIs are:

- `nn-placement`: placement model during setup, greedy afterward.
- `nn-state`: greedy setup, state-model one-step evaluation afterward.
- `nn-placement-state`: placement model during setup and state-model evaluation afterward, without MCTS.
- `nn-mcts-placement-state`: placement-prior MCTS during setup and state-prior MCTS with asynchronous state leaf evaluation afterward.

Each should be compared with `greedy`, which also supplies the fallback behavior for
the single-model variants. The pipeline's dual-model inference server loads both models,
and one NN URL routes calls to `/placement/predict` and `/state/predict`. Benchmark JSON,
`summary.json`, and progress-chart error bars preserve `confidence95Margin` at the
observed rate and `worstCaseConfidence95Margin`.

### `promotion` Section

Promotion is optional and requires `trainingMode: "placement-and-state"`. Generation 0
is promoted as the bootstrap champion. Every later generation trains an isolated
challenger from the current champion and must pass both direct and MCTS hybrid gates.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable champion/challenger lifecycle. |
| `additionalTrainingGames` | `500` | Accepted self-play games appended after each failed attempt. |
| `maxRetries` | `2` | Additional train-and-gate attempts after the first failure. |
| `direct.games` | `10000` | Games per direct challenger comparison. |
| `direct.ai` | `nn-placement-state` | Direct policy/value player. |
| `hybrid.games` | `1000` | Games per MCTS hybrid comparison. |
| `hybrid.ai` | `nn-mcts-placement-state` | MCTS-guided player. |
| `minimumImprovementVsGreedy` | `0.0` | Required score above 50% against greedy. |
| `minimumImprovementVsChampion` | `0.0` | Required score above 50% head-to-head against champion. |

Each enabled gate runs challenger-versus-greedy and challenger-versus-champion. Draws
count as half a win, and the benchmark must complete exactly the configured number of
games. All enabled comparisons must pass. A failed gate grows the same generation corpus,
restarts training from the unchanged champion, and retries. Exhausted retries write a
terminal `rejected` decision and leave the champion unchanged for the next generation.

Candidates live under `models/candidates/genN/attemptK/`; immutable promoted copies live
under `models/champions/genN/`; `models/champion.json` is the authoritative active model
pair. Resume uses attempt and generation decisions under `results/promotion/genN/`.

### Example Config

```json
{
  "trainingMode": "placement-and-state",
  "placementModelConfig": "small",
  "stateModelConfig": "small",
  "simulate": {
    "games": 1000,
    "placementSearchTimeMs": 16000,
    "mainGameSearchTimeMs": 8000
  },
  "placementTrain": { "outputMode": "combined", "batchSize": 64, "policyTargetTemperature": 0.5 },
  "stateTrain": { "outputMode": "value", "batchSize": 64 },
  "benchmarks": [
    { "name": "placement", "games": 10000, "ai": ["nn-placement", "greedy"] },
    { "name": "state", "games": 10000, "ai": ["nn-state", "greedy"] },
    { "name": "hybrid", "games": 10000, "ai": ["nn-placement-state", "greedy"] },
    { "name": "mcts-hybrid", "games": 1000, "ai": ["nn-mcts-placement-state", "greedy"] }
  ]
}
```

This mode writes one `data/genN/` directory. Generation N simulation serves both frozen
generation N-1 checkpoints; generation 0 uses rollout fallback. Training reads the same
files twice and writes `models/placement/genN.pt` and `models/state/genN.pt`. Resume treats
simulation and each checkpoint as separate DAG completion markers.

Set `enabled` to `false` in `placementTrain` or `stateTrain` to freeze that model while the
other continues training. For generation N > 0, the pipeline copies the disabled model's
`genN-1.pt` to `genN.pt` when the destination is absent; it does not invoke the trainer or
create training/checkpoint directories. To freeze a model from generation 0, seed its
`models/placement/gen0.pt` or `models/state/gen0.pt` destination before starting. Both model
files remain required for generation completion, serving, and resume.

See [`pipeline.example.json`](../pipeline.example.json) in the project
root for a fully commented example.

## Artifact Layout

```
pipeline/
├── data/
│   ├── gen0/          # One .json file per game
│   │   ├── a1b2c3.json
│   │   ├── discarded/     # Compact diagnostics for rejected attempts
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
