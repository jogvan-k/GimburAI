# Seven discards are automatic rather than player decisions

**Severity:** High

`src/Gimbur/GameState.cs:1397-1436,1708-1721` discards half of each affected hand using a fixed largest-stack heuristic. No turn stage or policy action allows players to choose the discarded multiset. `CatanStateTests.cs:220-246` checks only the resulting count.

This removes a strategically important decision from the game and training vocabulary.

## Recommended fix

Model required discards explicitly, sequentially if necessary, including the choosing player, remaining count, return stage, serialization, and policy vocabulary. Test arbitrary legal compositions and multiple affected players.
