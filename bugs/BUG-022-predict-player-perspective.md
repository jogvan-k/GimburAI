# predict-player violates the trained perspective contract

**Severity:** Medium

Training rotates the acting player to canonical slot one (`data_loader.py:364-372`), as do normal inference paths. `/state/predict-player` (`serve.py:728-771`) tokenizes the unrotated state and indexes `player - 1`, contradicting `NnClient.cs:62-82` and feeding layouts absent from training.

The checked-in test `python/tests/test_serve.py:107-135` explicitly asserts that no rotation occurs, so it preserves the defect rather than validating the documented contract. No current production caller uses this endpoint; primary direct, prior, and leaf inference use `/state/predict` or canonicalize internally and are correct.

## Recommended fix

Define whether the endpoint canonicalizes by acting player or requested player, perform the corresponding rotation and slot translation, and update docs/tests. Include cases where current and requested players differ.
