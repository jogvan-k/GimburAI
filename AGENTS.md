# AGENTS NOTES

## Repository Overview
- `src/` holds production code. Currently it contains the `Kjarni` project, an F# library that implements the core Monte Carlo Tree Search (MCTS) game-playing engine for multiplayer board games.
- `tests/` contains automated tests. At the moment there is a single test project, `Kjarni.Tests`, which exercises the MCTS engine.

## Kjarni Project
- Project file: `src/Kjarni/Kjarni.fsproj`.
- Key areas:
  - `DomainTypes.fs` defines shared domain types exposed by the engine.
  - `Algorithms/` contains helper algorithms such as utility functions used by the search.
  - `MCTS/` contains the Monte Carlo Tree Search implementation (`Algorithm.fs`, `MonteCarloTreeSearch.fs`, `Types.fs`).
- Build artifacts land under `src/Kjarni/bin/` and `src/Kjarni/obj/`.

## Tests
- Project file: `tests/Kjarni.Tests/Kjarni.Test.fsproj`.
- Test modules live under `tests/Kjarni.Tests/MCTS/` and cover selection, expansion, back propagation, benchmarking, and state behavior of the engine.
- `tests/Kjarni.Tests/TestTypes.fs` provides support types shared across the test modules.
- `tests/Kjarni.Tests/program.fs` is the entry point for running the test project.

## Future Projects
- A future `Gimbur` project will provide Settlers of Catan rules and a CLI for simulating games powered by the `Kjarni` engine. Expect it to live under `src/Gimbur/` with corresponding tests under `tests/Gimbur.Tests/` when introduced.

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
