# GimburAI

Settlers of Catan playing AI.

The goal is to train a neural network that evaluates board positions (state to win probability), using an AlphaZero-inspired self-play loop. The final AI picks the legal action that leads to the highest expected win probability — no deep search at game time.

## Project Structure

- **Kjarni** (`src/Kjarni/`) — F# Monte Carlo Tree Search engine. Game-agnostic.
- **Gimbur** (`src/Gimbur/`) — C# Catan game rules and state, integrates with Kjarni via `ICoreState`/`ICoreAction`.
- **Gimbur.Cli** (`src/Gimbur.Cli/`) — Command-line interface for simulation and interactive play.
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
- [ ] Extend `Player` enum to support 3-4 players
- [ ] Multiplayer backpropagation (win/loss per player, not just binary)
- [ ] Replace deterministic simulation with random rollouts
- [ ] Neural network leaf evaluator (replace rollouts with model inference)

### Phase 2: Catan Game Engine (Gimbur)

#### 2a: Board Representation

- [x] Board data structures (tiles, vertices, edges, ports) matching topology docs
- [x] Board setup: random tile/number/port placement with standard constraints
- [ ] Number token spiral placement (alphabetical letter order per official rules, skip desert)
- [x] Mini map variant support (for faster training iterations)

#### 2b: Game State

- [ ] Full game state implementing `ICoreState`
- [ ] Player resources, dev cards, knights played
- [ ] Vertex/edge occupancy, robber position, longest road, largest army
- [ ] Turn stage state machine (initial placement, pre-roll, robber, build/trade)
- [ ] State hashing for transposition table
- [ ] State serialization/deserialization (per `docs/state-serialization.md`)

#### 2c: Rules & Action Generation

- [ ] Initial settlement/road placement (with second-placement resource grant)
- [ ] Dice roll and resource production
- [ ] Robber placement (on 7 or knight) + stealing + discard-on-7
- [ ] Building: roads, settlements, cities (cost, placement rules, supply limits)
- [ ] Development cards: buying, playing (knight, road building, monopoly, year of plenty, VP)
- [ ] Trading: bank trade (4:1 default, port-adjusted) — no player-to-player trade (adds branching factor without improving training label quality)
- [ ] Longest road calculation
- [ ] Largest army tracking
- [ ] Victory point calculation and win detection (10 VP)
- [ ] Legal action enumeration

#### 2d: Greedy AI

- [ ] Heuristic evaluation function (weighted sum of VP, resources, board position, etc.)
- [ ] Greedy action selection: evaluate each legal action's resulting state, pick highest score
- [ ] Use as default simulation policy (replaces random legal moves for higher quality rollouts)

#### 2e: TUI for Manual Testing

- [ ] Terminal-based game UI (board rendering, resource display, action selection)
- [ ] Human-vs-AI mode
- [ ] AI-vs-AI spectator mode
- [ ] CLI `play` command wired to TUI

#### 2f: Simulation Harness

- [ ] Game simulation using greedy AI, run to completion
- [ ] Game result recording (winner, final scores, turn count)
- [ ] Batch simulation runner
- [ ] CLI `simulate` command wired to real game logic

#### 2g: Testing

- [ ] Unit tests for each rule subsystem
- [ ] Integration tests: full random games complete without crashes
- [ ] Regression tests for edge cases

### Phase 3: Training Data Generation & Model Training (Python)

#### 3a: Data Pipeline

- [ ] Export game states as training examples (state, win probability label)
- [ ] Random rollout labeling (play N random games from state, label = win%)
- [ ] MCTS-informed labeling (MCTS search value as label)
- [ ] Dihedral group D₆ expansion: augment each training example by applying all 12 symmetries (6 rotations + 6 reflections) of the hexagonal board, multiplying dataset size by 12
- [ ] Large-scale batch generation (parallel simulation, output to files)

#### 3b: Transformer Model (v1)

- [ ] Python project setup (PyTorch, pyproject.toml)
- [ ] Transformer architecture: serialized state tokens to win probability (scalar 0-1)
- [ ] Training loop, validation, metrics (MSE, win/loss prediction accuracy)
- [ ] Model export for .NET inference (ONNX)

#### 3c: Self-Play Loop (AlphaZero-style)

- [ ] Generation 0: train model on greedy-rollout-labeled data
- [ ] Inference integration: load model into Kjarni as leaf evaluator
- [ ] Generation N: model-backed MCTS plays games, generate training data, retrain
- [ ] Automated pipeline: generate, train, evaluate, promote best model, repeat
- [ ] Elo tracking per generation (each generation vs. previous + vs. greedy baseline)
- [ ] Win-rate plotting across generations (improvement curve)
- [ ] Loss / accuracy plots per training run
- [ ] Saturation detection: identify when win-rate gains plateau across generations

### Phase 4: Final AI (Model-Only, No Search)

- [ ] Pure model evaluator: apply each legal action, evaluate resulting state, pick highest
- [ ] Chance node handling: EV = sum of P(outcome) * model(resulting state)
- [ ] Performance optimization (batch inference, caching)
- [ ] Comparison: model-only vs. MCTS-backed vs. greedy baseline

### Phase 5: Catanatron Benchmark Adapter

- [ ] Research Catanatron API for plugging in custom AI players
- [ ] Adapter: translate Catanatron state/actions to GimburAI and back
- [ ] Tournament runner: GimburAI vs. Catanatron built-in AIs
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
