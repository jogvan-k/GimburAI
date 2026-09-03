# Exact value samples with zero policy visits are discarded

**Severity:** High

`python/gimbur_nn/data_loader.py:337-363` recognizes `valueTarget` as exact, then drops any state with non-empty actions whose total visits are zero. `SimulationRunner.cs:1251-1274,1299-1323` deliberately exports unsearched forced/terminal records in exactly that shape.

Clean endgame value labels are lost because policy supervision is absent.

## Recommended fix

Separate value-target validity from policy-target validity. Retain exact value rows and mark policy loss invalid with an explicit flag or empty legal mask. Add a test matching `CreateUnsearchedStateRecord` output.
