# Replay generation window is truncated to the newest generation

**Severity:** High

`python/gimbur_nn/pipeline.py:1276-1304` supplies replay paths newest-first but sets one global `maxGames` equal to the newest generation size. `data_loader.py:195-196` concatenates in that order and `train.py:173-183,570-582` keeps only the first `maxGames` records. Under normal generation sizes, no older replay game survives.

This defeats `replay_generations`, increasing forgetting and target instability.

## Reproduction

Load three tagged generation files with 75 games each and `maxGames=75`; all retained games come from the newest file.

## Recommended fix

Apply quotas per generation or sample from the complete replay window. Add an integration test asserting that each configured generation contributes records.
