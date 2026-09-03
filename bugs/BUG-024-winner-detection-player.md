# Winner detection can select a non-active player

**Severity:** Medium

`GameState.cs:1727-1745` scans players numerically and selects the first above threshold. Catan victory is declared by the active player on that player's turn. Restored/custom states with multiple qualifying players can therefore award the wrong seat.

## Recommended fix

During normal play, evaluate `CurrentPlayer` only. Specify any setup/restore exception separately and test active-player precedence.
