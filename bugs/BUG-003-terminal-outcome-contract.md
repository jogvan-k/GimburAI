# MCTS infers terminal outcomes from PlayerTurn

**Severity:** High

`src/Kjarni/DomainTypes.fs:12-19` exposes no terminal result. Empty-action paths throughout `MCTS/Algorithm.fs` create a one-hot winner from `state.PlayerTurn` (including lines 70-72, 163-178, 294-296, 502-504, and 1027-1030).

This undocumented convention cannot represent draws/shared outcomes and declares the next player the winner in games that advance the turn before becoming terminal. Existing tests encode the same assumption, so they cannot detect it.

## Recommended fix

Add an explicit finite per-player terminal outcome to `ICoreState` and use it at every terminal site. Test a winner different from `PlayerTurn`, draws, shared outcomes, and invalid vector dimensions.
