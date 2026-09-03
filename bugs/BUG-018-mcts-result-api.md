# The public MCTS result API is incomplete

**Severity:** Medium

`DomainTypes.fs:112-126` advertises `IGameAI.RunSimulation(ICoreState)` and a `SimulationResult` with rollouts, action values, and elapsed time. `MonteCarloTreeSearch.fs:9-16,63-65` accepts `MCTSState`, does not implement `IGameAI`, populates only rollouts, and discards the computed path. `ActionValues` remains null.

## Recommended fix

Choose one coherent public API, implement it, and return an immutable fully initialized result containing elapsed time and selected/path statistics. Add a non-explicit contract test.
