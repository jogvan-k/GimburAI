# MCTS tree objects have an unsafe concurrency contract

**Severity:** Medium

`MCTS/Types.fs:8-65` and backpropagation/expansion in `Algorithm.fs` mutate counters, arrays, and action states without synchronization. `MonteCarloTreeSearch.fs:9-67` also mutates logging state. Concurrent runs sharing a root or search instance can lose visits, double-expand, or throw.

## Recommended fix

Either document and enforce single-run ownership with an interlocked guard, or add proper node/edge synchronization for parallel tree search. Test concurrent calls for safe operation or fail-fast rejection.
