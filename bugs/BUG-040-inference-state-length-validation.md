# Inference state length validation is inconsistent

**Severity:** Low

`StateTokenizer.tokenize` (`state_tokenizer.py:240-257`) validates characters but not `cfg.state_token_size`; `tokenize_batch` checks only equality within the batch. `rotate_player_state` validates exact length only after returning early for target player 1 (`state_tokenizer.py:335-346`).

As a result, `/state/predict` and prior inference (`serve.py:528-573,702-726`) can pass malformed player-1 states to the model, where positional embedding lookup fails as an internal error or the prior worker terminates. Leaf inference separately checks exact length at lines 617-624.

## Recommended fix

Validate exact configured length in `tokenize` or `canonicalize` before any identity-rotation return. Apply the same validation to every endpoint and return a request-local 400/error result. Add too-short and too-long player-1 and non-player-1 tests for direct, prior, leaf, and predict-player calls.
