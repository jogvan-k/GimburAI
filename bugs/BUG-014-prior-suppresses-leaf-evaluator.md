# Policy-only priors suppress the configured leaf evaluator

**Severity:** Medium

Evaluation branches in `Kjarni/MCTS/Algorithm.fs:913-1080` choose `PriorClient` with `if/elif`, so `LeafEvaluator` is never enqueued when a prior client exists. If the documented optional `PriorResponse.ValueEstimates` is absent, lines 778-807 use rollout fallback rather than the configured evaluator.

The production combined client masks this, but the public interfaces promise composable policy and value providers.

## Recommended fix

Request policy and value independently. Use a combined value when present; otherwise enqueue the leaf evaluator. Test policy-only, combined, invalid-value, and delayed responses.
