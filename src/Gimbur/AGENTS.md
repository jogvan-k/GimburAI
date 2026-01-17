# Gimbur Directory Guidelines

- Files here belong to the C# CLI extension of Kjarni.
- Target `net10.0` unless explicitly noted otherwise.
- Use `System.CommandLine` for CLI plumbing.
- Mark new types as `internal` unless external use is required.
- Command handlers should not throw; log TODO messages instead.
