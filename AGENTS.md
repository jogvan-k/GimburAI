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
