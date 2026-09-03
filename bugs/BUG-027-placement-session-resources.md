# InitialPlacementSession omits setup resources

**Severity:** Medium

`Rules/InitialPlacementSession.cs:100-141` models setup board placement but has no hands and never grants final-round adjacent resources. `CatanState.ApplyInitialSettlement` does grant them (`GameState.cs:737-758,1522-1533`). Two public setup implementations therefore produce different rule outcomes.

## Recommended fix

Use one authoritative setup implementation, or make the session return a complete result including hands and round metadata. Add parity tests for identical action sequences.
