# Simulation Export Data Schema

This document defines the JSON output schemas for the `gimbur simulate --export` command. The `--export-type` argument controls which schema is used.

## CLI Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--export` | `<path>` | *(none)* | File path for JSONL or directory path for JSON export. Required for any export. |
| `--export-format` | `jsonl`, `json` | `jsonl` | `jsonl`: all games in one file, one JSON object per line. `json`: one file per game in the directory. |
| `--export-type` | `GameState`, `InitialPlacement`, `PlacementAndState` | `GameState` | Controls the output schema. See below. |
| `--no-symmetries` | *(flag)* | *(off)* | Disable board symmetry permutations in exported data. |

When `--export-type InitialPlacement` is specified, `--placement-only` is automatically enabled.

`PlacementAndState` runs placement and normal play once and exports one object with shared
`winner`, `board`, and metadata. Its `placementStates` array uses the InitialPlacement state
schema, while `states` uses the GameState schema and includes placement, the exact
`turnNumber: 1`/`stage: "r"` boundary, and normal play. Placement search uses
`placementSearchTimeMs` (default `16000`); normal play uses `mainGameSearchTimeMs` (default
`8000`). The placement tree has the exact PreRoll leaf boundary and is discarded before the
main-game search starts.

## Evaluation Diagnostics And Discards

Both game schemas contain a game-level `evaluationDiagnostics` object aggregated across
every MCTS decision. It reports `submitted`, `applied`, `timeouts`, `invalidResponses`,
`cancelled`, `fallbacks`, `orphans`, `batches`, `states`, `latencyMs`,
`priorResponsesOrphaned`, and derived `hardErrors`. Latency is the sum of completed batch
latencies. Asynchronous leaf HTTP enqueue failures surface as invalid responses; prior
response orphans are reported separately.

Simulation config supports `maxErrorsPerGame` (5), `maxErrorRatePerGame` (0.02),
`minimumRequestsForRate` (50), and `discardGamesWithFallbacks` (false). Hard errors are
timeout + invalid response + leaf orphan. A rejected game is omitted from normal training
output and a compact reproducible record is written to the adjacent `discarded/` directory
with seed, map, export type, outcome metadata, reason, and diagnostics. Full states are not
duplicated there.

The accepted target excludes discarded attempts. Generation safety settings are
`maxDiscardedGames` (20), `maxDiscardRate` (0.05), `minimumAttemptsForDiscardRate` (50),
and `maxConsecutiveDiscards` (5). Exceeding one stops generation and produces a nonzero CLI
exit code, preventing endless retries when inference is unhealthy.

---

## GameState Export Schema

*Default export type. Produces training data for the **GimburStateEvaluator** model.*

Each game is exported as a single JSON object. In JSONL format, each line is one game.

### Game Object

```json
{
  "seed": 42,
  "map": "mini",
  "players": 2,
  "winner": 1,
  "turns": 47,
  "constraints": {
    "searchTimeMs": 1000,
    "maxSimulations": 2147483647,
    "maxRolloutDepth": 500,
    "actionRolloutLimit": 2147483647
  },
  "board": {
    "serialized": "w5lb3ls4lW3hd0nW4ho2l|gsgbgw",
    "permutations": [
      "b3lW3hd0nW4ho2lw5l|sgbgwg",
      "..."
    ]
  },
  "states": ["...see State Object below..."],
  "priorsCalculated": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Deterministic random seed for this game. |
| `map` | string | Map configuration identifier (`mini`, `small`, `standard`). |
| `players` | int | Number of players in the game. |
| `winner` | int | 1-based player index of the winner, or 0 if no winner (e.g. game aborted). |
| `turns` | int | Total number of completed turns. |
| `constraints` | object | MCTS search parameters used for this game. |
| `board.serialized` | string | Board serialization (tiles and ports only): `tiles\|ports`. See [state-action-serialization.md](state-action-serialization.md) Part I sections 1-2. |
| `board.permutations` | string[] | Board string under each non-trivial symmetry permutation. Empty array when symmetries are disabled or unavailable. |
| `states` | State[] | Array of states evaluated by MCTS, including states with one forced action. Reused roots inherit existing rollouts and are topped up toward the configured search limit before export. |
| `priorsCalculated` | object? | Per-depth count of NN prior states evaluated across all decisions. `null` when priors were not used. Keys are depth strings (`"0"`, `"1"`, ...), values are counts. |

### State Object (GameState)

```json
{
  "playerTurn": 1,
  "turnNumber": 1,
  "stage": "r",
  "serializedState": "...",
  "simulations": 5000,
  "elapsedMs": 1000,
  "winRate": 0.64,
  "wins": [3200.0, 1800.0],
  "valueTarget": null,
  "scores": [2.0, 3.0],
  "actions": [
    {
      "action": "2:0:0",
      "wins": [120.0, 80.0],
      "visits": 200,
      "winRate": 0.6,
      "modelPrior": 0.4,
      "selected": true
    }
  ],
  "reachedTerminal": false,
  "priorsRequested": 0,
  "priorsApplied": 0,
  "priorStatesEvaluated": 0,
  "permutations": [
    "...",
    "..."
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `playerTurn` | int | 1-based player index of the acting player. |
| `turnNumber` | int | Current game turn number. Initial placement is turn 0; normal play starts at turn 1. |
| `stage` | string | Encoded turn stage. `r` is `PreRoll`; `turnNumber: 1` with `stage: "r"` identifies the exact post-placement state. |
| `serializedState` | string | State-only serialization (12 pipe-delimited sections: robber through winner). See [state-action-serialization.md](state-action-serialization.md) Part I sections 3-14. |
| `simulations` | int | Total MCTS rollouts performed for this decision. |
| `elapsedMs` | int | Wall-clock time spent on MCTS search (milliseconds). |
| `winRate` | float | Acting player's win rate at the MCTS root (wins / rollouts). |
| `wins` | float[] | Raw MCTS win counts at the root, 0-indexed (index 0 = player 1). |
| `valueTarget` | float[]? | Exact resolved value distribution when available. Training prefers this over accumulated `wins`. |
| `scores` | float[] | Authoritative victory-point scores from `CatanState.Scores()`, indexed by player. |
| `actions` | Action[] | Every legal action at the root with MCTS edge diagnostics. Forced states contain their sole selected action with zero visits; terminal states are empty. |
| `reachedTerminal` | bool | Whether MCTS fully resolved the tree (all root actions are Terminal). |
| `priorsRequested` | int | Number of NN prior requests sent during this search. |
| `priorsApplied` | int | Number of NN prior responses applied to tree nodes. |
| `priorStatesEvaluated` | int | Number of individual states evaluated by the NN server. |
| `permutations` | string[] | `serializedState` under each non-trivial symmetry permutation. Same order as `board.permutations`. |

Each `actions` entry contains `action` (stable `typeTag:arg1:arg2` identity), `wins`, `visits`,
`winRate`, nullable `modelPrior`, and `selected`, which identifies the action played in the
recorded game.

State-value training blends each root's normalized per-player MCTS wins with the
one-hot final game winner. The MCTS weight decreases linearly from
`mctsValueWeightStart` to `mctsValueWeightEnd` using `turnNumber / game.turns`.
If one target is unavailable, the other is used alone; if both are unavailable,
the state is skipped.
After the game-level split, full-state sampling groups all roots in each split by
`floor(sum(scores))`. It derives an adaptive cap from the median (default) or average bucket
size plus a configurable upper percentage. Short buckets are retained fully; oversized
buckets are sampled deterministically, with every game's exact post-placement and final
roots mandatory. Sampling before symmetry and player augmentation means each split derives
its cap independently. Placement datasets are unaffected.
Candidate result states are not exported or trained unless they naturally become a
later searched root.

States with exactly one legal action are exported for value-model coverage and advanced
without MCTS. Their sole action is included for diagnostics with zero visits. Terminal
states are also exported with an exact one-hot `valueTarget`; forced stochastic roots whose
outcomes are all terminal carry the probability-weighted resolved distribution.

The iterative pipeline replays the current and recent generations (three by default)
when training a state model. Configure `train.replayGenerations` to bound this window;
replay reduces forgetting and prevents the latest guided search distribution from
entirely replacing earlier state coverage.

State training reports decoded-probability MAE, Brier score, and expected calibration
error in addition to bucket loss. `gimbur_nn.metrics.candidate_ranking_accuracy` is
provided for grouped candidate evaluation against an independent deeper-search corpus.

State training defaults to ordinal CDF loss, so nearby bucket errors cost less than
distant errors. MCTS rollout counts remain available as diagnostics.

### Symmetry Permutations (GameState)

Board symmetry permutations rearrange position-dependent data (tile indices, vertex indices, edge indices) while leaving player-identity and per-player sections unchanged.

- `board.permutations[i]` is the board string under symmetry `i`.
- `states[j].permutations[i]` is the state-only string under the same symmetry `i`.
- Both arrays have the same length and use the same permutation order.

---

## InitialPlacement Export Schema

*Produces tree-level settlement and road training data for `placement_stage_policy`.*

When `--export-type InitialPlacement` is specified, the game loop runs in placement-only mode. Each non-forced settlement and road decision is searched and exported as its own MCTS root. Forced roots are applied without search or export. The MCTS leaf boundary is the exact deterministic result of the final placement road (`turnNumber: 1`, `stage: "r"`/`PreRoll`).

Player 1 is always the player that placed the first settlement. No player rotation is applied.

### Game Object

```json
{
  "seed": 42,
  "map": "mini",
  "players": 2,
  "winner": 0,
  "exportType": "initialPlacement",
  "constraints": {
    "searchTimeMs": 1000,
    "maxSimulations": 2147483647,
    "maxRolloutDepth": 500,
    "actionRolloutLimit": 2147483647
  },
  "board": {
    "serialized": "w5lb3ls4lW3hd0nW4ho2l|gsgbgw",
    "permutations": [
      "b3lW3hd0nW4ho2lw5l|sgbgwg",
      "..."
    ]
  },
  "states": ["...see Placement State Object below..."]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Deterministic random seed for this game. |
| `map` | string | Map configuration identifier. |
| `players` | int | Number of players. |
| `winner` | int | 1-based final winner, or 0 if the seeded continuation does not finish. |
| `exportType` | string | Always `"initialPlacement"`. |
| `constraints` | object | MCTS search parameters (same structure as GameState). |
| `board.serialized` | string | Board serialization (tiles and ports): `tiles\|ports`. |
| `board.permutations` | string[] | Board string under each symmetry permutation. |
| `states` | PlacementState[] | One record per non-forced settlement or road MCTS root. |

The `turns` field is omitted. After recording placement decisions, simulation continues with seeded random legal play solely to obtain `winner`; no post-placement states are exported. If that continuation cannot finish within the action safety limit, `winner` is 0 and training uses only the MCTS target.

### Placement State Object

```json
{
  "playerTurn": 1,
  "stage": "a",
  "serializedState": "w5lb3ls4lW3hd0nW4ho2l|gsgbgw|a|._._._._._._._._._._._._._._._._._._._._._._._._|______________________________",
  "simulations": 5000,
  "elapsedMs": 1000,
  "modelValue": [0.61, 0.39],
  "valueTarget": [0.62, 0.38],
  "actions": [
    {
      "policyIndex": 6,
      "wins": [320.0, 180.0],
      "visits": 500,
      "winRate": 0.64,
      "modelPrior": 0.18,
      "permutations": [8, 10, 14, 16]
    },
    {
      "policyIndex": 12,
      "wins": [280.0, 220.0],
      "visits": 480,
      "winRate": 0.583,
      "permutations": [4, 18, 20, 22]
    }
  ],
  "permutations": [
    "b3lW3hd0nW4ho2lw5l|sgbgwg|._._._._._._._._._._._._._._._._._._._._._._._._|______________________________|",
    "..."
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `playerTurn` | int | 1-based player index of the acting player. |
| `stage` | string | `a`/`f` for settlement or `e`/`i` for road. It matches the serialized stage marker. |
| `serializedState` | string | Canonical 5-section placement state: `tiles\|ports\|stage\|placementVertices\|edges`. Owner IDs put the acting player first. |
| `simulations` | int | Total MCTS rollouts performed for this decision. |
| `elapsedMs` | int | Wall-clock time spent on MCTS search (milliseconds). |
| `modelValue` | float[]? | Placement model's per-player value estimate when a prior response was applied; otherwise `null`. |
| `valueTarget` | float[]? | Visit-weighted per-player root value target; `null` if no action visits are available. |
| `actions` | Action[] | Every legal root action, including actions with zero visits. This list defines the legal mask. |
| `permutations` | string[] | `serializedState` under each non-trivial symmetry permutation. Same order as `board.permutations`. |

### Action Object

`policyIndex` is the settlement vertex at stages `a`/`f`, and the road direction index `0..5` from the pending settlement at stages `e`/`i`. Direction order is `N, NE, SE, S, SW, NW`.

Generation-0 MCTS uses a local greedy one-hot prior. The selected action exports
`modelPrior: 1`; every other action exports `modelPrior: 0`. The same field holds
soft neural priors in later generations. Before PUCT, the greedy prior is mixed with
configurable uniform support, so other actions remain explorable. MCTS `visits` and
`wins` remain stochastic rollout evidence; later-generation policy targets use the
improved visit shares.

```json
{
  "policyIndex": 6,
  "wins": [320.0, 180.0],
  "visits": 500,
  "winRate": 0.64,
  "permutations": [8, 10, 14, 16]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `policyIndex` | int | Vertex index for settlement or direction index `0..5` for road. |
| `wins` | float[] | Per-player value sums from `root.ActionStats[actionIndex]`; empty when unvisited. |
| `visits` | int | Completed visits from the same root action edge. |
| `winRate` | float | Acting player's value sum divided by visits, or zero when unvisited. |
| `modelPrior` | float? | `root.Priors[actionIndex]`, or `null` when the root has no model prior. |
| `permutations` | int[] | Transformed policy index under each state symmetry. |

For unexplored actions, the action remains in the export with `wins: []`, `visits: 0`, and `winRate: 0`. Visit shares form the policy target; `policyTargetTemperature` transforms only positive shares and preserves the legal mask.

### Symmetry Permutations (InitialPlacement)

Board symmetry permutations transform vertex and edge indices. For placement data, both the state and the actions must be permuted:

1. **State permutation**: The 5-section placement state is permuted via `BoardSymmetry.PermutePlacementState`, which rearranges tiles, ports, vertex pairs (including `p`), and edges while retaining stage.
2. **Action permutation**: Settlement indices use `permutation.Vertices`. Road indices use `TransformDirectionIndex(pendingVertex, edge, permutation)`.

The `wins`, `visits`, and `winRate` are unchanged; only `policyIndex` is transformed.

Example with mini map (5 symmetry permutations):

```json
{
  "policyIndex": 6,
  "wins": [320.0, 180.0],
  "visits": 500,
  "winRate": 0.64,
  "permutations": [8, 10, 14, 16]
}
```

Here `6N` under rotational symmetries maps to `8SE`, `10SW`, `14N`, and `16SE` respectively. The training pipeline can expand each sample into 1 + len(permutations) training examples by pairing each permuted state with its permuted action list.

---

## Export Formats

### JSONL (default)

All games are written to a single file, one JSON object per line. Each line is a complete game object as described above. Thread-safe: games may appear in any order.

```
gimbur simulate --games 10 --export data/training.jsonl --export-type GameState
```

### JSON

Each game is written to its own file with a random GUID filename inside the specified directory.

```
gimbur simulate --games 10 --export data/games/ --export-format json --export-type InitialPlacement
```
