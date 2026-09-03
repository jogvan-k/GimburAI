# Training ignores the export schema version

**Severity:** High

Current C# full-state exports set `version = 1` in `SimulationRunner.cs:1906-1909,2087-2090`, but `python/gimbur_nn/data_loader.py` never reads or validates that field. It assumes the latest required state/action shape directly.

The checked-in `trainingdata/` demonstrates the risk: these files also declare version 1 but use a historical 169-token Small 2P state, omit `stage`, `scores`, and action records, and cannot feed the current 184-token model. The same version number therefore identifies incompatible schemas.

Current old data usually fails later with a missing key, no selected samples, or positional-embedding shape error rather than a precise compatibility message. A future format that happens to retain lengths/field names could be silently misinterpreted.

## Recommended fix

Increment the export version whenever serialized state/action semantics change. Validate version, map, player count, exact state length, required fields, action identifiers, symmetry counts, and policy width before splitting or sampling. Fail fast with file, line, seed, and supported versions. Add fixtures proving that historical version-1 files are rejected clearly.
