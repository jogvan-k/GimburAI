# State equality ignores rule configuration

**Severity:** Medium

`CatanState.Equals/GetHashCode` (`GameState.cs:560-669`) omit `GameConfig`, although custom configuration changes costs, supplies, thresholds, and legal transitions (`GameConfig.cs:93-130`). Equal states can therefore have different futures, violating transposition/cache equivalence.

## Recommended fix

Give immutable configurations stable structural identity and include every transition/reward-affecting field in state equality and hashing. Test otherwise identical states with different costs and victory thresholds.
