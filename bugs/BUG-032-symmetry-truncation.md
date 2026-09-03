# Symmetry mismatches are silently truncated

**Severity:** Low

`data_loader.py:333-380` combines board/state variants with `zip`, silently dropping extras when counts differ, while action permutation indexing assumes alignment. Corrupt or migrated exports can lose augmentation without diagnosis.

## Recommended fix

Validate equal board, state, and action permutation counts before expansion and report game seed/state index. Add malformed-export tests.
