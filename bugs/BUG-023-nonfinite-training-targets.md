# Training accepts non-finite targets

**Severity:** Medium

`data_loader.py:53-61,337-356` clamps negative values but does not reject NaN or infinity. `loss_config.py:56-58` and `train.py:465-478` have no finite-loss guard, so one corrupt row can produce NaN gradients and permanently poison a checkpoint.

## Recommended fix

Require finite target values, visits, weights, and sums. Abort with game/state identity if the final loss is non-finite. Add NaN and infinity tests.
