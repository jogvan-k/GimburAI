# Prior generation rematerializes all successors

**Severity:** Low / Improvement

`Algorithm.fs:328-371,711-723` materializes every deterministic successor and stochastic outcome for a prior request. Expansion later executes the selected transition again. This increases allocations and assumes undocumented transition purity/idempotence.

## Recommended fix

Document pure stable transition methods and cache materialized successors for reuse during expansion. Benchmark allocation and latency on wide Catan states.
