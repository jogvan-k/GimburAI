# winRate returns NaN for an unvisited node

**Severity:** Low

`Algorithm.fs:25-26` divides by `state.Rollouts` without a zero guard. A fresh node returns `0/0 = NaN`; future callers can contaminate ordering or diagnostics.

## Recommended fix

Return zero or an option for unvisited states, consistently with edge-Q handling. Add a zero-rollout test.
