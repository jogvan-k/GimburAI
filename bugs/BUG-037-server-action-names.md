# Server action names are incomplete

**Severity:** Low

`Gimbur.Server/Program.cs:209-225` names only action tags 0-12, while `CatanActions.cs:143-241` defines tags through 19. Normal staged actions are returned as `Unknown(...)`, degrading diagnostics and integrations.

## Recommended fix

Use one authoritative action tag/name registry or derive names from action types. Test that every legal concrete action has a known response name.
