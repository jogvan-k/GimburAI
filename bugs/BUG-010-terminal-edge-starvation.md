# Solved terminal edges can starve unresolved actions

**Severity:** High

In `Kjarni/MCTS/Algorithm.fs:41-57`, terminal actions always score their exact value and ignore visits. An unresolved zero-prior action always has PUCT score zero. A known draw/partial terminal value can therefore be selected forever, while parent solving at lines 267-284 requires a guaranteed win or all-terminal actions.

Search can miss a superior unresolved move and waste its entire budget revisiting solved information.

## Recommended fix

Exclude solved edges while unresolved alternatives remain, or implement solved-node bounds. Ensure every unresolved legal action remains discoverable. Test a partial terminal edge against a zero-prior forced win.
