# AGENTS NOTES

## Repository Overview
- `src/` holds production code:
  - `Kjarni` — F# library implementing the core Monte Carlo Tree Search (MCTS) game-playing engine for multiplayer board games.
  - `Gimbur` (assembly name `Gimbur.Core`) — C# library containing Catan game rules, board topology, and the `ICoreState`/`ICoreAction` adapter layer that bridges to the Kjarni MCTS engine. Rules types live under the `Gimbur.Rules` namespace.
  - `Gimbur.Cli` — C# CLI application for simulation and interactive play.
- `tests/` contains automated tests:
  - `Kjarni.Tests` — exercises the MCTS engine.
  - `Gimbur.Rules.Tests` — exercises board topology, setup, and game rules.

## Kjarni Project
- Project file: `src/Kjarni/Kjarni.fsproj`.
- Key areas:
  - `DomainTypes.fs` defines shared domain types exposed by the engine.
  - `Algorithms/` contains helper algorithms such as utility functions used by the search.
  - `MCTS/` contains the Monte Carlo Tree Search implementation (`Algorithm.fs`, `MonteCarloTreeSearch.fs`, `Types.fs`).
- Build artifacts land under `src/Kjarni/bin/` and `src/Kjarni/obj/`.

## Gimbur Project
- Project file: `src/Gimbur/Gimbur.csproj` (assembly name: `Gimbur.Core`).
- Contains both the Catan rules (under `Gimbur.Rules` namespace) and the Kjarni adapter layer (under `Gimbur` namespace).
- Directory structure:
  - `Rules/Types/` — small enum and struct types: `ResourceType.cs`, `PortType.cs`, `DevCardType.cs`, `BuildingType.cs`, `TurnStage.cs`, `HexCoord.cs`, `VertexOccupancy.cs`, `EdgeOccupancy.cs`.
  - `Rules/Board/` — board-related classes: `BoardTopology.cs`, `Board.cs`, `BoardSetup.cs`, `MapConfig.cs`.
  - `Rules/GameConfig.cs` — game-level configuration (supply limits, victory conditions, costs, dev card pool). Defines `Standard` and `Mini` presets.
  - `GameState.cs` — Kjarni adapter layer (namespace `Gimbur`), implements `ICoreState`/`ICoreAction`.
- All rules files share the `Gimbur.Rules` namespace regardless of subfolder.
- Build artifacts land under `src/Gimbur/bin/` and `src/Gimbur/obj/`.

## Tests
- Project file: `tests/Kjarni.Tests/Kjarni.Test.fsproj`.
- Test modules live under `tests/Kjarni.Tests/MCTS/` and cover selection, expansion, back propagation, benchmarking, and state behavior of the engine.
- `tests/Kjarni.Tests/TestTypes.fs` provides support types shared across the test modules.
- `tests/Kjarni.Tests/program.fs` is the entry point for running the test project.
- Project file: `tests/Gimbur.Rules.Tests/Gimbur.Rules.Tests.csproj`.
- Test files: `BoardTopologyTests.cs`, `BoardSetupTests.cs`, `BoardTests.cs`, `TypeTests.cs`, `GameConfigTests.cs`.

## Future Projects
- A future `Gimbur` project will provide Settlers of Catan rules and a CLI for simulating games powered by the `Kjarni` engine. Expect corresponding tests under `tests/Gimbur.Tests/` when more test areas are introduced.

## Python Project (gimbur-nn)
- Lives under `python/` with package `gimbur_nn`.
- Project config: `python/pyproject.toml` (uses ruff for linting/formatting).
- Key modules:
  - `game_config.py` — map dimension constants and tensor size formulas for mini/small/standard maps.
  - `tokenizer.py` — legacy game-state tokenizer; parses 10-section serialized strings into PyTorch tensors.
  - `state_tokenizer.py` — game-state tokenizer class (`StateTokenizer`).
  - `placement_tokenizer.py` — placement-phase tokenizer class (`PlacementTokenizer`); handles canonical 5-section placement states and exposes vertex/direction policy indices.
  - `data_loader.py` — loads JSONL training data exported by `gimbur simulate --export`, builds `SimulationDataset` for PyTorch `DataLoader`.
  - `model_config.py` — model hyperparameter configuration.
  - `transformer_model.py` — transformer-based models. `GimburPlacementTransformer` always emits player-value `[B,N]` and stage-policy `[B,max(V,6)]` logits for `placement_stage_policy`.
  - `pipeline.py` — training/evaluation pipeline utilities.
  - `train.py` — training loop; reads JSONL data exported by `gimbur simulate --export`.
  - `serve.py` — HTTP inference server. `/placement/predict` accepts `states` and returns `player_win_probabilities` plus fixed-width stage `policy_probabilities`. Placement prior requests contain `id`, `state`, and `priority`; collect responses contain dense `priors` and per-player values. `/state/leaf-*` batches asynchronous MCTS leaf evaluations.
- Placement models emit raw logits. Combined training masks policy loss using legal actions exported by C#; serving softmaxes the full vocabulary, and C# is authoritative for legality and masks/normalizes before MCTS use.
- Placement stage policy uses vertex indices at settlement stages and direction indices in `N, NE, SE, S, SW, NW` order at road stages. Shared-root visits are policy targets; `simulations-per-action` rollout counts are not.
- State checkpoints require `checkpoint_version: 3` and `state_player_value_v1`; placement checkpoints use current-only architecture `placement_stage_policy`.
- Tests live under `python/tests/` using pytest:
  - `test_tokenizer.py` — tests for all tokenizer classes (game state, placement state, action vocab).
  - `test_data_loader.py` — tests for data loading, sample expansion, and dataset construction.
- Style: Python 3.11+, `from __future__ import annotations`, ruff for formatting/linting.

## Build/Test Commands

### Build Commands
```bash
# Build the entire solution
dotnet build

# Build specific project
dotnet build src/Kjarni/Kjarni.fsproj
dotnet build tests/Kjarni.Tests/Kjarni.Test.fsproj

# Build in Release configuration
dotnet build -c Release

# Clean build artifacts
dotnet clean
```

### Test Commands
```bash
# Run all tests
dotnet test

# Run tests in a specific project
dotnet test tests/Kjarni.Tests/Kjarni.Test.fsproj

# Run a single test by full name (use FullyQualifiedName)
dotnet test --filter "FullyQualifiedName=KjarniTest.MCTS.AlgorithmTest.selectionTests.TerminalNode"

# Run all tests in a specific test class/fixture
dotnet test --filter "FullyQualifiedName~KjarniTest.MCTS.AlgorithmTest.selectionTests"

# Run tests matching a name pattern
dotnet test --filter "FullyQualifiedName~Selection"

# Run tests with detailed output
dotnet test -v detailed

# List all available tests without running them
dotnet test -t
```

### Python Commands
```bash
# Run all Python tests (from repo root)
python -m pytest python/tests/ -v

# Run a specific Python test file
python -m pytest python/tests/test_tokenizer.py -v

# Run a specific test class or test
python -m pytest python/tests/test_tokenizer.py::TestPlacementTokenizer -v
python -m pytest python/tests/test_tokenizer.py::TestPlacementTokenizer::test_policy_size_matches_config -v

# Lint with ruff (from python/ directory)
python -m ruff check python/

# Format with ruff
python -m ruff format python/

# Run training
python -m gimbur_nn.train

# Run inference server
python -m gimbur_nn.serve
```

### Other Commands
```bash
# Restore NuGet packages
dotnet restore

# Clean, restore, and build
dotnet clean && dotnet restore && dotnet build
```

## Code Style Guidelines

### General Conventions
- Target framework: .NET 10.0
- Use F# idiomatic patterns and functional programming style
- Prefer immutability; use mutable state sparingly and only when necessary (e.g., for performance-critical MCTS tree nodes)

### File Organization
- File order in `.fsproj` files matters in F# - dependencies must come before files that use them
- Group related files in subdirectories (`MCTS/`, `Algorithms/`)
- Test files mirror the structure of source files with `Test` suffix (e.g., `SelectionTest.fs`)

### Imports/Open Statements
- System imports first: `open System`, `open System.Diagnostics`
- Project namespace imports second: `open Kjarni`, `open Kjarni.MCTS.Types`
- Test framework imports in tests: `open NUnit.Framework`, `open FsUnit`
- Keep imports minimal and specific to what's needed in each file

### Naming Conventions
- **Types**: PascalCase for types, interfaces, classes (e.g., `ICoreState`, `Player`, `MonteCarloTreeSearch`)
- **Functions/Methods**: camelCase for functions and methods (e.g., `leafEvaluator`, `registerWin`, `extractBestPath`)
- **Parameters**: camelCase for parameters (e.g., `state`, `playerTurn`, `maxSimulationCount`)
- **Private fields**: _prefixed with underscore (e.g., `_leaves`, `_visitCount`, `_logInfos`)
- **Discriminated unions**: PascalCase for cases (e.g., `Unexplored`, `Leaf`, `Terminal`)
- **Module-level constants**: camelCase (e.g., `explorationConstant`)

### Types and Type Annotations
- Use explicit type annotations for interface implementations and public APIs
- Infer types for local bindings when obvious
- Define discriminated unions for domain modeling (e.g., `Leaf`, `SelectionResult`, `searchTime`)
- Use records and classes appropriately - classes for mutable state, records for data
- Use `struct` for small, performance-critical types (e.g., `LogInfo`)

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Keep lines readable; break long function calls/pipelines across multiple lines
- **Pattern matching**: Align match cases consistently, one per line
- **Function parameters**: When multiple, use clear spacing and break across lines if needed
- **Operators**: Space around binary operators (`=`, `+`, `-`, `<-`, etc.)

### Functions and Methods
- Use `let` bindings for module-level functions
- Use `member` for class/type methods
- Use recursive functions with `rec` keyword when needed (e.g., `recSelection`, `recComplexTree`)
- Prefer piping (`|>`) for data transformation chains
- Keep functions focused and single-purpose

### Pattern Matching
- Use exhaustive pattern matching with discriminated unions
- Use `match` expressions for complex branching
- Active patterns are acceptable (e.g., `ofCase <@ Exhausted @>` in tests)

### Error Handling
- Use exceptions for truly exceptional cases (e.g., `raise (Exception "Target leaf is already expanded")`)
- Use `Option` types for values that may or may not exist (e.g., `Option<TranspositionTable>`)
- Use discriminated unions for domain errors when appropriate
- Include descriptive error messages

### Mutable State
- Mark mutable bindings explicitly with `mutable` keyword
- Limit mutable state to classes where performance requires it (MCTS tree nodes)
- Use immutable data structures by default
- Document why mutability is necessary when used

### Comments and Documentation
- Use XML doc comments (`///`) for public APIs
- Keep comments concise and meaningful
- Prefer self-documenting code over comments where possible
- Document complex algorithms (e.g., MCTS selection, expansion)

### Testing Conventions
- Use NUnit with `[<TestFixture>]` and `[<Test>]` attributes
- Use FsUnit for fluent assertions (e.g., `should equal`, `should haveLength`)
- Use `[<TestCase>]` for parameterized tests with specific values
- Use `[<Values>]` and `[<Range>]` for data-driven tests
- Test names should clearly describe what is being tested
- Use helper functions in test files to reduce duplication (e.g., `constructSut`, `stateHash`)
- Test types and helpers live in `TestTypes.fs` and are shared across test modules

### Module Structure
- Use `namespace` for types that are part of the public API (e.g., `namespace Kjarni`)
- Use `module` for groupings of functions (e.g., `module Kjarni.MCTS.Algorithm`)
- Keep module-level functions in logical groups

## Python Style Guidelines

### General Conventions
- Target Python 3.11+ (`requires-python = ">=3.11"` in `pyproject.toml`).
- Always include `from __future__ import annotations` at the top of every module.
- Use ruff for both linting and formatting (`line-length = 100`, rules: E, F, I, W, UP).
- PyTorch is the only required runtime dependency; `fastapi`/`uvicorn` are optional for serving.

### File Organization
- Production code lives in `python/gimbur_nn/` (the package).
- Tests live in `python/tests/` with `test_` prefix (pytest convention).
- Configuration constants (map sizes, vocab sizes, tensor formulas) belong in `game_config.py`.
- Tokenizer classes are split by concern: `state_tokenizer.py` (game state), `placement_tokenizer.py` (placement phase), `tokenizer.py` (legacy/shared).

### Naming Conventions
- **Modules**: snake_case (`game_config.py`, `placement_tokenizer.py`).
- **Classes**: PascalCase (`PlacementTokenizer`, `StateTokenizer`, `SimulationDataset`).
- **Functions/methods**: snake_case (`tokenize_state`, `decode_action`, `load_games`).
- **Constants**: UPPER_SNAKE_CASE for module-level constants (`MINI_TILES`, `RESOURCE_CHARS`), or lowercase for config dict keys.
- **Type aliases**: PascalCase when using `TypeAlias` or similar.

### Testing Conventions
- Use pytest (no unittest subclassing required, but grouping with classes is fine).
- Test classes use `Test` prefix: `TestPlacementTokenizer`, `TestVocab`, `TestMiniMap`.
- Test methods use `test_` prefix with descriptive snake_case names.
- Use `pytest.raises` for expected exceptions.
- Keep test data inline when small; use helper functions for repeated setup.

# Dotnet CLI
Use dotnet cli when adding or modifying the solution, e.g. when setting up new projects, use `dotnet new classlib --name <project_name>`. Don't forget to update the .slnx file, e.g. `dotnet sln add <project_path>`.
