# State and action protocol documentation is obsolete

**Severity:** Medium

The production C#, training loader, and primary inference paths agree on the current protocol, but the documents claimed to define that protocol describe an older layout.

## State mismatches

`docs/state-action-serialization.md:209-218,395-410` defines section 12 as two tokens and reports lengths 138/174/185/279/290/301. Production serializes five staged-state tokens in `CatanStateSerializer.cs:123-137,660-672`, and Python allocates the resulting lengths 148/184/195/289/300/311 in `game_config.py:104-112`.

The three documented examples at lines 252-302 contain only 12 sections and 139/175/291 compact tokens. Production requires 14 sections and 148/184/300 tokens for those configurations. The examples omit three staged fields, the development deck, and winner.

The documented turn-stage alphabet at lines 30 and 117-127 omits `u v z c g h m j k`. The count alphabet at line 31 stops at 19, while C# and Python support the full one-character Crockford range 0-31.

`game_config.py:9-18,104-110` repeats stale formulas in comments even though its expression is currently correct. `data_loader.py:4-13` claims all-player augmentation, while the implementation creates only acting-player-canonical samples.

## Action mismatches

`docs/simulation-export.md:157-167` documents obsolete identifiers such as `BuildCity:5` and `BankTrade:Wood->Brick`, and says full-state actions contain `policyIndex`. Current export emits staged descriptive actions and no integer index (`SimulationRunner.cs:1939-1958`); Python intentionally reconstructs it with `StateTokenizer.action_policy_index`.

`docs/complete-policy-value-model.md:55-56` omits the supported Standard 2P width 163. Lines 76-77 say serving always returns an unmasked full policy, but the MCTS prior endpoint performs legal-only softmax server-side; only `/state/predict` returns the complete distribution.

## Impact

An implementation built from the documentation produces incompatible model tensors and action records. Existing tokenizer tests claim their strings are copied verbatim from the docs (`python/tests/test_tokenizer.py:18-20`) but silently append the missing production sections, so they do not detect documentation drift.

## Recommended fix

Make one versioned protocol specification authoritative and generate size/offset tables where possible. Replace all examples with actual serializer output and test them verbatim in both languages. Document the exact endpoint-specific policy contract and current JSON field names.
