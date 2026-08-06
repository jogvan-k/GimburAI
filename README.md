# GimburAI

Settlers of Catan playing AI.

The goal is to train a neural network that evaluates board positions (state to win probability), using an AlphaZero-inspired self-play loop. The final AI picks the legal action that leads to the highest expected win probability — no deep search at game time.

## Project Structure

- **Kjarni** (`src/Kjarni/`) — F# Monte Carlo Tree Search engine. Game-agnostic.
- **Gimbur** (`src/Gimbur/`) — C# Catan game rules and state, integrates with Kjarni via `ICoreState`/`ICoreAction`.
- **Gimbur.Cli** (`src/Gimbur.Cli/`) — Command-line interface for simulation and interactive play.
- **Gimbur.Tui** (`src/Gimbur.Tui/`) — Terminal UI for manual board initialization and rendering.
- **Tests** (`tests/`) — NUnit test suites.
- **Docs** (`docs/`) — Board topology reference, state serialization spec, SVG diagrams.
- **Scripts** (`scripts/`) — Topology SVG generation.

## Roadmap

### Phase 0: Foundation & Topology Documentation

- [x] Board topology reference (coordinate system, vertex/edge/port identity)
- [x] Standard map topology tables (19 tiles, 54 vertices, 72 edges, 9 ports)
- [x] Mini map topology tables (7 tiles, 24 vertices, 30 edges, 6 ports)
- [x] SVG diagram generation script
- [x] State serialization format spec

### Phase 1: MCTS Engine (Kjarni)

- [x] Core MCTS algorithm (selection, expansion, simulation, backpropagation)
- [x] UCB1-based tree policy
- [x] Transposition table support
- [x] Async parallel search
- [x] Best-path extraction
- [x] `IGameAI` public API (`MonteCarloTreeSearch` class)
- [x] Test suite (selection, expansion, backprop, state tracking, benchmark)
- [x] Extend `Player` enum to support 3-4 players
- [x] Multiplayer backpropagation (per-player outcome vector, not just binary win/loss)
- [x] Support stochastic rollout EV via weighted outcomes (`IStochasticCoreAction`)
- [x] Neural network prior evaluator

### Phase 2: Catan Game Engine (Gimbur)

#### 2a: Board Representation

- [x] Board data structures (tiles, vertices, edges, ports) matching topology docs
- [x] Board setup: random tile/number/port placement with standard constraints
- [x] Number token spiral placement (alphabetical letter order per official rules, skip desert)
- [x] Mini map variant support (for faster training iterations)

#### 2b: Game State

- [x] Full game state implementing `ICoreState`
- [x] Player resources, dev cards, knights played
- [x] Vertex/edge occupancy, robber position, longest road, largest army
- [x] Turn stage state machine (initial placement, pre-roll, robber, build/trade)
- [x] State hashing for transposition table
- [x] State serialization/deserialization (per `docs/state-serialization.md`)

#### 2c: Rules & Action Generation

- [x] Initial settlement/road placement (with second-placement resource grant)
- [x] Dice roll and resource production
- [x] Robber placement (on 7 or knight) + stealing + discard-on-7
- [x] Building: roads, settlements, cities (cost, placement rules, supply limits)
- [x] Development cards: buying, playing (knight, road building, monopoly, year of plenty, VP)
- [x] Trading: bank trade (4:1 default, port-adjusted) — no player-to-player trade (adds branching factor without improving training label quality)
- [x] Longest road calculation
- [x] Largest army tracking
- [x] Victory point calculation and win detection (10 VP)
- [x] Legal action enumeration
- [x] Remove temporary `CatanActionType` enum and migrate all consumers to pure action-type polymorphism

#### 2d: Greedy AI

- [x] Heuristic evaluation function (weighted sum of VP, resources, board position, etc.)
- [x] Greedy action selection: evaluate each legal action's resulting state, pick highest score
- [x] Use as default simulation policy (replaces random legal moves for higher quality rollouts)

#### 2e: TUI for Manual Testing

- [x] Terminal-based game UI (board rendering, resource display, action selection)
- [x] Human-vs-AI mode
- [x] AI-vs-AI spectator mode

#### 2f: Simulation Harness

- [x] Game simulation using greedy AI, run to completion
- [x] Game result recording (winner, final scores, turn count)
- [x] Batch simulation runner (multi-core parallel via `Parallel.For`)
- [x] CLI `simulate` command wired to real game logic
- [x] Training data export (state + winner label, aligned with `docs/state-serialization.md`)

#### 2g: Testing

- [x] Unit tests for each rule subsystem
- [x] Integration tests: full random games complete without crashes
- [x] Regression tests for edge cases

### Phase 3: Training Data Generation & Model Training (Python)

#### 3a: Data Pipeline

- [x] Export game states as training examples (state, win probability label)
- [x] Random rollout labeling (play N random games from state, label = win%)
- [x] MCTS-informed labeling (MCTS search value as label)
- [x] Dihedral group D₆ expansion: augment each training example by applying all 12 symmetries (6 rotations + 6 reflections) of the hexagonal board, multiplying dataset size by 12
- [ ] Large-scale batch generation (parallel simulation, output to files)

#### 3b: Transformer Model (v1)

- [x] Python project setup (PyTorch, pyproject.toml)
- [x] Transformer architecture: serialized state tokens to win probability (scalar 0-1)
- [x] Training loop, validation, metrics (MSE, win/loss prediction accuracy)
- [x] Serve model using python endpoint
- [x] `placement_stage_policy`: five-section placement encoder with player-value and stage-policy heads

#### 3c: Self-Play Loop (AlphaZero-style)

- [x] Generation 0: train model on greedy-rollout-labeled data
- [x] Inference integration: load model into Kjarni as leaf evaluator
- [ ] Generation N: model-backed MCTS plays games, generate training data, retrain
- [x] Automated pipeline: generate, train, evaluate, promote best model, repeat
- [ ] Elo tracking per generation (each generation vs. previous + vs. random and greedy baseline)
- [ ] Win-rate plotting across generations (improvement curve)
- [ ] Loss / accuracy plots per training run
- [ ] Saturation detection: identify when win-rate gains plateau across generations

#### 3d: Scale to compute cluster

- [ ] Set up Azure resource templates using bicept, including storage, compute & inference VMs, and k8s
- [ ] Deployment scripts for resources
- [ ] Create orchestration node on local machine that starts and monitors jobs, and iterates between label generation, model training, and benchmarking

### Phase 4: Final AI (Model-Only, No Search)

- [ ] Pure model evaluator: apply each legal action, evaluate resulting state, pick highest
- [ ] Chance node handling: EV = sum of P(outcome) * model(resulting state)
- [ ] Performance optimization (batch inference, caching)
- [ ] Comparison: model-only vs. MCTS-backed vs. greedy baseline

### Phase 5: JSettlers Benchmark Adapter

- [ ] Research JSettlers API for plugging in custom AI players
- [ ] Adapter: translate JSettlers state/actions to GimburAI and back
- [ ] Tournament runner: GimburAI vs. JSettlers built-in AIs
- [ ] Elo / win-rate benchmarking

### Phase 6: Model Architecture Experiments

- [ ] CNN/ConvNet layers over spatial board representation (hex grid as 2D input)
- [ ] Hybrid architectures: convolutional feature extraction + transformer sequence modeling
- [ ] Architecture comparison: transformer-only vs. CNN vs. hybrid (same training data)
- [ ] Retrain best architecture through self-play loop, compare to transformer v1 Elo
- [ ] Ablation studies (layer count, embedding size, attention heads, kernel size)

### Phase 7: Colonist.io Autonomous Player

- [ ] Colonist.io client protocol integration (WebSocket/API)
- [ ] Game state extraction and mapping to GimburAI state
- [ ] Action translation: GimburAI decisions to client commands
- [ ] Autonomous game loop (join, play, handle timers)

## Build & Test

```bash
dotnet build                # build everything
dotnet test                 # run all tests
dotnet run --project src/Gimbur.Cli  # run CLI
```

## Gimbur CLI

The CLI (`gimbur`) is the main entry point for running simulations and benchmarks.

```bash
dotnet run --project src/Gimbur.Cli -- <command> [options]
```

### Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config <file>` | `-c` | Path to a configuration file |
| `--seed <int>` | | Random seed for reproducibility |
| `--map-config <preset>` | | Map layout: `mini`, `small`, or `standard` |
| `--verbosity <level>` | `-v` | Verbosity: `q[uiet]`, `m[inimal]`, `n[ormal]`, `d[etailed]`, `diag[nostic]` |
| `-q` | | Shorthand for `--verbosity quiet` |
| `--verbose` | | Shorthand for `--verbosity diagnostic` |

### `simulate` -- AI Self-Play

Run MCTS self-play games with optional JSONL training data export.

Use `--export-type PlacementAndState` to export placement policy roots and full-game state
roots from the same game. Combined runs default to 16000 ms placement search and 8000 ms
main-game search; override them with `--placement-search-time` and `--main-game-search-time`.

```bash
# Run 100 games on the mini map, export training data
dotnet run --project src/Gimbur.Cli -- simulate \
    --games 100 --map-config mini --export training.jsonl

# Quick run with limited search budget
dotnet run --project src/Gimbur.Cli -- simulate \
    --games 10 --search-time 500 --max-simulations 200

# Self-play with NN prior evaluation (requires running inference server)
dotnet run --project src/Gimbur.Cli -- simulate \
    --games 50 --prior --nn-url http://localhost:8000 --export training.jsonl
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--games <n>` | `-g` | `1` | Number of games to simulate |
| `--players <n>` | `-p` | map default | Player count (1-4) |
| `--export <file>` | | none | Path to export training data as JSONL |
| `--search-time <ms>` | | `1000` | MCTS search time limit in ms per decision |
| `--max-simulations <n>` | | unlimited | Max MCTS simulations per decision |
| `--max-rollout-depth <n>` | | `500` | Max rollout depth |
| `--action-rollout-limit <n>` | | unlimited | Stop MCTS when any action reaches this many rollouts |
| `--no-symmetries` | | `false` | Disable board symmetry permutations in export |
| `--prior` | | `false` | Enable async NN prior evaluation during MCTS search |
| `--nn-url <url>` | | `http://localhost:8000` | Base URL of the NN inference server (used with `--prior`) |

The exported JSONL file contains one record per game state, with the serialized board state and the game winner, suitable for training the neural network. Board symmetry augmentation (D6 group, 12x) is applied by default.

When `--prior` is enabled, the MCTS search uses asynchronous neural network priors to bias action selection via the PUCT formula. This requires a running inference server (see the Python `serve` command below). Prior requests are fire-and-forget on node expansion; the search loop is never blocked.

### `benchmark` -- AI vs AI

Run games between different AI players and compute win rates. Seat rotation is applied to eliminate positional bias.

```bash
# MCTS vs greedy, 50 games
dotnet run --project src/Gimbur.Cli -- benchmark \
    --games 50 --ai mcts greedy --map-config mini

# NN-backed player vs MCTS (requires running inference server)
dotnet run --project src/Gimbur.Cli -- benchmark \
    --ai nn mcts --nn-url http://localhost:8000

# Placement policy during setup, state evaluation afterward, versus greedy
dotnet run --project src/Gimbur.Cli -- benchmark \
    --games 10000 --ai nn-placement-state greedy --map-config mini
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--games <n>` | `-g` | `10000` | Number of games to run |
| `--ai <types...>` | | `random greedy` | AI for each player seat |
| `--output <file>` | `-o` | none | Path to write JSON results file |
| `--search-time <ms>` | | `1000` | MCTS search time limit in ms |
| `--max-simulations <n>` | | unlimited | Max MCTS simulations per decision |
| `--max-rollout-depth <n>` | | `500` | Max rollout depth |
| `--nn-url <url>` | | `http://localhost:8000` | Base URL of the NN inference server |

Focused model AI types are `nn-placement` (placement model, then greedy), `nn-state`
(greedy placement, then state model), and `nn-placement-state` (placement model, then
state model without MCTS). `nn-mcts-placement-state` uses placement-policy MCTS during
setup and state-prior MCTS with asynchronous state leaf evaluation afterward; it also
requires `Gimbur.Server`. The same `--nn-url` serves both model endpoints.

Use 10,000 games for reported comparisons. Its worst-case 95% Wald margin is 0.0098
(0.98 percentage points); result JSON includes observed and worst-case confidence margins.

## Python CLI (gimbur-nn)

The Python package provides model training and an inference server. Run from the `python/` directory.

### Setup

```bash
cd python
pip install -e ".[serve,dev]"
```

### `train` -- Train the Neural Network

Train a transformer model on JSONL data exported by `gimbur simulate --export`.

```bash
python -m gimbur_nn.train \
    --data exports/ \
    --game-config mini_2p \
    --model-config small \
    --out model.pt
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data <path>` | yes | | JSONL file or directory of JSONL files |
| `--game-config <preset>` | yes | | Game config preset |
| `--model-config <preset>` | yes | | Model size preset |
| `--out <path>` | no | `model.pt` | Output checkpoint path |
| `--resume <path>` | no | none | Resume training from existing checkpoint |
| `--epochs <n>` | no | `0` | Max epochs (0 = unlimited, stop via patience) |
| `--patience <n>` | no | `5` | Stop after N epochs with no val loss improvement |
| `--batch-size <n>` | no | `64` | Batch size |
| `--lr <float>` | no | `1e-4` | Learning rate |
| `--val-split <frac>` | no | `0.1` | Fraction of games for validation (0 to disable) |
| `--test-split <frac>` | no | `0.0` | Fraction of games for test (0 to disable) |
| `--log-interval <n>` | no | `50` | Print training loss every N batches (0 to disable) |

Game config presets: `mini_2p`, `small_2p`, `small_3p`, `standard_3p`, `standard_4p`.
Model config presets: `small`, `medium`, `large`.

### `serve` -- Inference Server

Serve a trained model over HTTP for use by the `nn` AI player and MCTS prior evaluation.

```bash
python -m gimbur_nn.serve \
    --model model.pt \
    --game-config mini_2p \
    --model-config small \
    --port 8000
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model <path>` | yes | | Path to trained model checkpoint |
| `--game-config <preset>` | yes | | Game config preset (must match training) |
| `--model-config <preset>` | yes | | Model size preset (must match training) |
| `--port <int>` | no | `8000` | HTTP port |
| `--host <addr>` | no | `127.0.0.1` | Bind address |

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/state/predict` | Batch predict win probabilities for full game states |
| `POST` | `/state/predict-player` | Predict win probability for a specific player |
| `POST` | `/placement/predict` | Predict placement values and optional dense policies from placement states |
| `POST` | `/state/prior-enqueue`, `/placement/prior-enqueue` | Enqueue state or placement MCTS prior requests |
| `POST` | `/state/prior-collect`, `/placement/prior-collect` | Collect completed prior results |
| `POST` | `/state/prior-flush`, `/placement/prior-flush` | Clear the corresponding server queue |

The `placement_stage_policy` model accepts canonical five-section placement states. It always emits player-value logits `[B,N]` and one fixed-width policy head `[B,max(V,6)]`: settlement stages use the first `V` vertex logits, while road stages use the first six logits in `N, NE, SE, S, SW, NW` order. Masking and interpretation are external to the model. `/placement/predict` softmaxes the fixed-width head and returns it as `policy_probabilities`.

State checkpoints use version 4 architecture `state_player_value_v1`. Placement checkpoints use the current-only `placement_stage_policy` architecture; older checkpoints are incompatible.

### End-to-End Workflow (Manual)

```bash
# 1. Generate training data (dotnet CLI)
dotnet run --project src/Gimbur.Cli -- simulate \
    --games 1000 --map-config mini --export data/training.jsonl -q

# 2. Train the model (Python)
cd python
python -m gimbur_nn.train \
    --data ../data/training.jsonl \
    --game-config mini_2p --model-config small \
    --out model.pt

# 3. Start inference server (Python)
python -m gimbur_nn.serve \
    --model model.pt \
    --game-config mini_2p --model-config small

# 4. Benchmark NN player vs baselines (dotnet CLI, in another terminal)
dotnet run --project src/Gimbur.Cli -- benchmark \
    --ai nn greedy --games 100 --map-config mini

# 5. Generate improved training data with NN prior self-play
dotnet run --project src/Gimbur.Cli -- simulate \
    --games 1000 --map-config mini --prior --export data/gen1.jsonl -q

# 6. Retrain the model on the new data and repeat from step 3
```

### `pipeline` -- Automated Self-Play Loop

The pipeline orchestrator automates the AlphaZero-style loop: simulate, train, benchmark, repeat. It manages the inference server lifecycle, passes the right arguments to each tool, and tracks results across generations.

```bash
cd python
python -m gimbur_nn.pipeline --config ../pipeline.example.json
```

**Per-generation flow:**
1. **Simulate** -- Gen 0 uses greedy placement policy labels and stochastic rollout values (no NN). Gen N>0 starts the inference server with gen N-1's model and enables `--prior`.
2. **Train** -- Trains on the new generation's data. Resumes from previous generation's checkpoint when available.
3. **Benchmark** -- Starts the inference server with this generation's model and runs all configured benchmarks.
4. **Report** -- Prints win rates and saves results.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config <file>` | yes | | Path to pipeline configuration JSON |
| `--start-gen <n>` | no | `0` | Generation to resume from |
| `--project-root <dir>` | no | auto-detect | Project root directory |

**Pipeline configuration** (see `pipeline.example.json`):

```json
{
  "mapConfig": "mini",
  "gameConfig": "mini_2p",
  "modelConfig": "small",
  "seed": 42,
  "dataDir": "pipeline/data",
  "modelDir": "pipeline/models",
  "resultsDir": "pipeline/results",
  "generations": 10,

  "simulate": {
    "games": 1000,
    "players": 2,
    "searchTimeMs": 500,
    "maxSimulations": 200,
    "maxRolloutDepth": 500,
    "symmetries": true,
    "verbosity": "quiet"
  },

  "train": {
    "epochs": 0,
    "patience": 5,
    "batchSize": 64,
    "lr": 1e-4,
    "valSplit": 0.1,
    "logInterval": 50
  },

  "serve": { "port": 8000, "host": "127.0.0.1" },

  "benchmarks": [
    { "name": "nn-vs-greedy", "games": 100, "ai": ["nn", "greedy"] },
    { "name": "nn-vs-random", "games": 100, "ai": ["nn", "random"] }
  ]
}
```

Per-generation artifacts are stored at deterministic paths:
- Training data: `{dataDir}/gen{N}.jsonl`
- Model checkpoint: `{modelDir}/gen{N}.pt`
- Benchmark results: `{resultsDir}/gen{N}/{name}.json`
- Cross-generation summary: `{resultsDir}/summary.json`

The pipeline supports `// comments` in the JSON config file. To resume after interruption, use `--start-gen N` to skip completed generations.

## TUI Manual Testing

Run the TUI:

```bash
dotnet run --project src/Gimbur.Tui
```

Startup flow:
- Choose map topology: `mini` or `standard`
- Enter player count (validated against selected map config)
- Choose controller per player: `human` or `greedy`

Rendering notes:
- ANSI colors are used for resources, ports, and markers
- Each tile shows resource name (top), number token (middle), and robber marker `*` (bottom)
- Vertices/edges are shown so settlements/cities/roads can be visualized
- Ports are labeled (`3:1`, `Wood`, `Brick`, `Sheep`, `Wheat`, `Ore`) with connector lines to their two vertices

Gameplay notes:
- Initial setup goes directly to settlement/road placement with legal spots highlighted
- Use arrow keys + Enter to pick settlement/road/robber tile targets
- During normal turns, legal actions are shown in a menu (Up/Down + Enter)
- Resource and VP summaries for all players are displayed under the board
- Chance actions (`RollDice`, `BuyDevCard`) are single actions; outcomes are resolved by the game engine
