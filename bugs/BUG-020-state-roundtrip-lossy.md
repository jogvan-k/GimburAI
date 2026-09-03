# State serialization is not lossless

**Severity:** Medium

`CatanStateSerializer.cs:14-148,322-372` omits `TurnNumber` and `LastDiceRoll`; deserialization assigns turn 0 or 1. `TurnNumber` participates in equality/hash (`GameState.cs:560-669`), so a late-game state does not round-trip.

The omission does not currently make neural inputs non-Markov: neither field affects legal actions or rewards in the implemented rules. `LastDiceRoll` is diagnostic after the transition, and the model receives the resulting resources/stage. The defect is instead that `SerializeHumanReadable` is described as reversible and is used by server/state restoration paths where callers can reasonably expect an exact engine state.

`PendingSettlementVertex` is inferred during initial road stages and `_vertexPlacementRound` is omitted. Those values are sufficient for current full-state rules, but placement-display serialization after a full-state round trip cannot recover original placement-round labels.

## Recommended fix

Either add the fields to a versioned lossless protocol or explicitly define this format as model/information-state encoding and create a separate persistence format. Add normal-play round-trip tests at multiple turns and stages, and distinguish behavioral equivalence from object equality.
