# Search duration can substantially exceed its budget

**Severity:** Medium

`Algorithm.fs:846-866,1094-1112` extends the deadline for a root-prior wait of up to 250 ms and then drains evaluations for up to `DrainTimeoutMs` (default 1000 ms). A nominal 10 ms search can therefore take over a second, excluding client overhead.

## Recommended fix

Distinguish tree budget from total wall-clock deadline in the API. For latency-sensitive callers, cap every phase against one absolute deadline. Add elapsed-time tests with scheduling tolerance.
