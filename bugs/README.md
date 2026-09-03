# Repository Review Findings

This directory contains findings from a static review of Kjarni, Gimbur, the CLI/server/TUI, and the Python training and inference stack. `jsettlers` was excluded.

## Critical

- [BUG-001: Caller-controlled inference URL enables SSRF and resource exhaustion](BUG-001-server-nn-url-ssrf.md)

## High

- [BUG-002: Replay generation window is truncated to the newest generation](BUG-002-replay-window-truncated.md)
- [BUG-003: MCTS infers terminal outcomes from PlayerTurn](BUG-003-terminal-outcome-contract.md)
- [BUG-004: The resource bank is not modeled](BUG-004-resource-bank-missing.md)
- [BUG-005: Seven discards are automatic rather than player decisions](BUG-005-seven-discard-choice.md)
- [BUG-006: Settlements do not recalculate Longest Road](BUG-006-longest-road-settlement-split.md)
- [BUG-007: Runtime chance ignores the per-game seed](BUG-007-runtime-rng-not-seeded.md)
- [BUG-008: State serialization fails for counts above 31](BUG-008-count-serialization-overflow.md)
- [BUG-009: Exact value samples with zero policy visits are discarded](BUG-009-exact-value-samples-dropped.md)
- [BUG-010: Solved terminal edges can starve unresolved actions](BUG-010-terminal-edge-starvation.md)
- [BUG-011: Inference results are not isolated by client](BUG-011-prior-results-cross-client.md)
- [BUG-012: Inference backpressure silently strands requests](BUG-012-prior-backpressure-strands-requests.md)
- [BUG-013: The action endpoint permits unbounded uncancellable searches](BUG-013-server-unbounded-search.md)
- [BUG-039: Training ignores the export schema version](BUG-039-export-version-not-validated.md)

## Medium

- [BUG-014: Policy-only priors suppress the configured leaf evaluator](BUG-014-prior-suppresses-leaf-evaluator.md)
- [BUG-015: Greedy evaluation samples chance actions once](BUG-015-greedy-stochastic-sampling.md)
- [BUG-016: A malformed prior request can kill the worker](BUG-016-prior-worker-crash.md)
- [BUG-017: Stochastic outcome weights are not validated](BUG-017-stochastic-weight-validation.md)
- [BUG-018: The public MCTS result API is incomplete](BUG-018-mcts-result-api.md)
- [BUG-019: MCTS configuration accepts invalid values](BUG-019-mcts-config-validation.md)
- [BUG-020: State serialization is not lossless](BUG-020-state-roundtrip-lossy.md)
- [BUG-021: State equality ignores rule configuration](BUG-021-state-equality-ignores-config.md)
- [BUG-022: predict-player violates the trained perspective contract](BUG-022-predict-player-perspective.md)
- [BUG-023: Training accepts non-finite targets](BUG-023-nonfinite-training-targets.md)
- [BUG-024: Winner detection can select a non-active player](BUG-024-winner-detection-player.md)
- [BUG-025: TUI rolls dice twice](BUG-025-tui-double-dice-roll.md)
- [BUG-026: TUI inference can hang and bypass cleanup](BUG-026-tui-inference-lifecycle.md)
- [BUG-027: InitialPlacementSession omits setup resources](BUG-027-placement-session-resources.md)
- [BUG-028: MCTS tree objects have an unsafe concurrency contract](BUG-028-mcts-concurrent-use.md)
- [BUG-029: Search duration can substantially exceed its budget](BUG-029-search-budget-overrun.md)
- [BUG-038: State and action protocol documentation is obsolete](BUG-038-protocol-documentation-obsolete.md)

## Low And Improvements

- [BUG-030: Horizon prior fallback skips child statistics](BUG-030-horizon-fallback-statistics.md)
- [BUG-031: Score cutoff mishandles negative and invalid scores](BUG-031-score-cutoff-validation.md)
- [BUG-032: Symmetry mismatches are silently truncated](BUG-032-symmetry-truncation.md)
- [BUG-033: Dataset and leaf batching have avoidable memory spikes](BUG-033-memory-scaling.md)
- [BUG-034: Prior generation rematerializes all successors](BUG-034-successor-rematerialization.md)
- [BUG-035: winRate returns NaN for an unvisited node](BUG-035-winrate-zero-rollouts.md)
- [BUG-036: Server request validation silently changes behavior](BUG-036-server-request-validation.md)
- [BUG-037: Server action names are incomplete](BUG-037-server-action-names.md)
- [BUG-040: Inference state length validation is inconsistent](BUG-040-inference-state-length-validation.md)
- [BUG-041: Placement export reads dense priors in the wrong index space](BUG-041-placement-dense-prior-index.md)
