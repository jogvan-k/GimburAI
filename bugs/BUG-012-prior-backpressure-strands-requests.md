# Inference backpressure silently strands requests

**Severity:** High

The prior queue can reject or evict requests (`serve.py:203-262`) but `/state/prior-enqueue` reports only counts and discards evicted IDs (`serve.py:775-788`). `PriorClient.SendEnqueueBatch` (`PriorClient.cs:197-217`) treats any success status as full acceptance and never reads the body.

Dropped IDs remain pending forever, causing timeouts, fallback, leaks, and misleading diagnostics under load.

## Recommended fix

Return accepted, rejected, and evicted IDs. Clear/fail those IDs client-side or emit terminal failure responses. Prefer atomic enqueue semantics if ownership cannot be reported. Test full-queue rejection and priority eviction end to end.
