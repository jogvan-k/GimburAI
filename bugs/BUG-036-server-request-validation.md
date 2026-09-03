# Server request validation silently changes behavior

**Severity:** Low

`Gimbur.Server/Program.cs:31-37,84-126` maps every unknown/whitespace config to Standard and accepts negative search time/depth and invalid player counts. Mistakes become confusing deserialization errors, empty searches, generic 500s, or silently altered AI behavior.

## Recommended fix

Explicitly validate normalized config names, player count, bounded positive search time, non-negative bounded depth, and state size. Return structured HTTP 400 errors.
