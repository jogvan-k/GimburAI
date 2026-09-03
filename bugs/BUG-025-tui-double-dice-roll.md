# TUI rolls dice twice

**Severity:** Medium

`Gimbur.Tui/Program.cs:1018-1030` calls `action.DoCoreAction()` and discards its result, then passes the action to `ApplyActionAndLog`, which executes it again. Two random rolls are consumed and only the second is shown.

## Recommended fix

Remove the unused first execution and apply/log the action exactly once. Add a test with an instrumented stochastic action or deterministic RNG.
