# Horizon prior fallback skips child statistics

**Severity:** Low

When first expanding a `HorizonAction`, a failed prior enqueue backpropagates `evaluationPath` (`Algorithm.fs:962-977`), which excludes the retained horizon child. Other fallback and successful-evaluation paths update that child. The parent edge records a visit while the child remains unvisited.

## Recommended fix

Use `expandedPath` or explicitly update the horizon state. Add a rejecting-prior integration test at a leaf boundary.
