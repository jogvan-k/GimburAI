# State serialization fails for counts above 31

**Severity:** High

`CrockfordBase32.cs:11-21` encodes each number as one character and throws above 31. `CatanStateSerializer.cs:72-142,609-647` uses it for resource/card counts, while the current rules do not cap hands and Monopoly can concentrate resources.

Inference, export, or hashing can crash late in a valid in-memory game.

## Recommended fix

Version the protocol and use a representation covering every reachable exact count. Update Python size formulas/tokenizers atomically. A finite bank reduces frequency but does not make one-character encoding safe for all configured supplies.
