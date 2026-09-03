# MCTS configuration accepts invalid values

**Severity:** Medium

`DomainTypes.fs:128-161` permits negative limits/timeouts, NaN exploration, and overflowing time conversions without validation. `Algorithms/Utility.fs:7-19` performs unchecked intermediate arithmetic. Invalid inputs can throw, skip search, create arbitrary ordering, or violate evaluator timeout contracts.

## Recommended fix

Validate `MCTSConfig` once at construction/search entry, require finite non-negative constants and sensible positive timeouts, and use checked 64-bit time arithmetic. Add parameterized invalid-boundary tests.
