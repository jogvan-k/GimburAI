# TUI inference can hang and bypass cleanup

**Severity:** Medium

The TUI blocks on async inference with `GetAwaiter().GetResult()` (`Program.cs:403-404,572-573,630`) while `NnClient` uses long default HTTP timeouts and retries without cancellation (`NnClient.cs:23-26,144-174`). Exceptions bypass normal cursor restoration and client disposal (`Program.cs:82-121,193-210`).

## Recommended fix

Use an async cancellable game loop with one short overall deadline, graceful retry/fallback, and `try/finally` for HTTP disposal and terminal restoration. Test an inference server that accepts but never responds.
