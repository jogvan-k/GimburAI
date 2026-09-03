# Placement export reads dense priors in the wrong index space

**Severity:** Low

This affects the legacy placement-only diagnostic schema, not current full-state policy training.

`GreedyPriorClient.cs:34-59` stores `DensePriors` in the complete global policy vocabulary: settlement indices start at `T`, and road indices start at `T + V`. Placement export in `SimulationRunner.cs:1139-1143` reads that vector using `actionIndex`, which is only the ordinal position in `state.Actions()`.

Whenever a dense prior exists, `modelPrior` can therefore be attached to the wrong placement action or exported as zero. The correct action-order `mctsRoot.Priors[actionIndex]` fallback is bypassed.

## Recommended fix

For placement records, export the action-order prior directly, or map the placement action through the complete serializer before indexing a dense vector. Add a generation-zero one-hot test proving that the selected settlement and road export `modelPrior: 1`.
