# Score cutoff mishandles negative and invalid scores

**Severity:** Low

`Algorithm.fs:132-161` substitutes zero for missing scores, accepts NaN/infinity, treats any non-positive maximum as a draw, and uses exact floating equality. Scores `[-1,-2]` incorrectly produce a draw rather than player one leading.

## Recommended fix

Define the score contract, validate finite exact dimensions, select maxima regardless of sign for utility scores, and define tie tolerance if scores are floating estimates. Add malformed and negative-score tests.
