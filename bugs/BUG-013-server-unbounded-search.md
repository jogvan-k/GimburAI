# The action endpoint permits unbounded uncancellable searches

**Severity:** High

`Gimbur.Server/Program.cs:18-194` performs synchronous MCTS with caller-controlled `searchTimeMs` and rollout depth, `int.MaxValue` simulation/tree limits, no cancellation token, no concurrency bound, and no rate limit. Disconnected requests keep consuming CPU and inference capacity.

## Recommended fix

Enforce server-side maxima, finite node/simulation budgets, rate limits, and a bounded search semaphore/queue. Add cancellation support to MCTS and link it to `RequestAborted` plus a hard deadline.
