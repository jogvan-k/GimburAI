# Dataset and leaf batching have avoidable memory spikes

**Severity:** Low / Improvement

`data_loader.py:278-319,408-442` eagerly retains parsed games, symmetry-expanded samples, and many individual tensors. `serve.py:601-640` caps leaf requests but not total states, so one request can create an oversized GPU batch.

## Recommended fix

Use a lazy/indexed dataset with compact storage and cap inference by total states/tokens while preserving response boundaries. Add representative peak-memory and oversized-request tests.
