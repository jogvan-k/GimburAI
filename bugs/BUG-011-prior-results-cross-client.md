# Inference results are not isolated by client

**Severity:** High

`python/gimbur_nn/serve.py:314-319,790-794` stores all completed priors in one list and destructively returns all of them to any collector. Each `PriorClient` keeps only locally known IDs (`PriorClient.cs:321-359`) and silently discards the rest.

Two clients/processes sharing a server can consume each other's responses; process-local IDs may also collide.

## Recommended fix

Add a globally unique client/session ID and partition queues/results by owner. Collection must return only that owner's results. Add a two-client concurrency test, including separate processes.
