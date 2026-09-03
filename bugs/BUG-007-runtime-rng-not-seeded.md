# Runtime chance ignores the per-game seed

**Severity:** High

The seeded `Random` passed to `CatanState` is used only for board setup (`GameState.cs:52-65`). Dice, development-card draws, robber steals, and MCTS sampling use process-global RNGs (`GameState.cs:812-821,853-906,1668-1706`; `Kjarni/MCTS/Algorithm.fs:59-67`). CLI simulation applies stochastic actions through `DoCoreAction` (`SimulationRunner.cs:1454-1488`).

Recorded seeds do not reproduce trajectories, and parallel scheduling changes outcomes.

## Recommended fix

Resolve stochastic actions with a caller-owned per-game RNG by sampling their explicit weighted outcomes. Give search randomness an explicit seed as well. Test identical trajectories across repeated runs and parallelism settings.
