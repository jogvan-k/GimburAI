# A malformed prior request can kill the worker

**Severity:** Medium

`serve.py:528-599` does not isolate failures from `_legal_policy_softmax` or malformed model input per request. Exceptions escape the sole `_prior_worker`; lifecycle cleanup removes completed tasks but does not supervise or restart them (`serve.py:687-691`). Subsequent accepted work then remains queued indefinitely.

## Recommended fix

Validate state lengths and legal indices on enqueue, convert request-local failures into error results, and supervise/restart workers after unexpected exceptions. Test a malformed request followed by a valid request.
