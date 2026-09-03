# Greedy evaluation samples chance actions once

**Severity:** Medium

`GreedyActionSelector.cs:51-68` calls `DoCoreAction()` once for every candidate, including dice/card/steal actions, instead of evaluating weighted outcomes. This makes rankings noisy and consumes global randomness merely by considering alternatives. `GreedyPriorClient.cs:34-38` propagates this noise into teacher priors.

## Recommended fix

For `CatanStochasticAction`, compute the weighted expected evaluation over `Outcomes()`. Sample only the selected action during actual play. Add deterministic expectation tests.
