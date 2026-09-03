# Stochastic outcome weights are not validated

**Severity:** Medium

`Kjarni/MCTS/Algorithm.fs:59-67,123-127,175-182,289-316,381-419` assumes non-empty outcomes, positive weights, and an `int` sum that does not overflow. Empty, zero, negative, or overflowing inputs can throw, sample incorrectly, or produce non-finite values.

## Recommended fix

Validate at the adapter boundary, require each weight greater than zero, accumulate and sample with checked 64-bit arithmetic, and validate player-vector dimensions. Add boundary tests for every invalid shape.
