# The resource bank is not modeled

**Severity:** High

`GameConfig.ResourceCardsPerType` defines finite supply, but `src/Gimbur/GameState.cs:979-997,1145-1163,1372-1395,1522-1533` creates and destroys resources directly in hands. Production shortages, unavailable bank trades, and Year of Plenty availability cannot be enforced.

Games can exceed physical supply, omitting scarcity strategy and producing invalid self-play data.

## Recommended fix

Store per-resource bank counts in state, cloning, equality, hashing, and serialization. Route every gain/cost/discard through the bank and implement production shortage rules. Add conservation tests asserting `bank + all hands == configured supply` after every action.
