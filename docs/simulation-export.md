# Simulation Export Data Schema

This document defines the JSON output schemas for the `gimbur simulate --export` command. The `--export-type` argument controls which schema is used.

## CLI Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--export` | `<path>` | *(none)* | File path for JSONL or directory path for JSON export. Required for any export. |
| `--export-format` | `jsonl`, `json` | `jsonl` | `jsonl`: all games in one file, one JSON object per line. `json`: one file per game in the directory. |
| `--export-type` | `GameState`, `InitialPlacement` | `GameState` | Controls the output schema. See below. |
| `--no-symmetries` | *(flag)* | *(off)* | Disable board symmetry permutations in exported data. |

When `--export-type InitialPlacement` is specified, `--placement-only` is automatically enabled.

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
  "serializedState": "4|-t|__|._._._._._._v-._._._._v+._._._._._._._._._|_____-_______+________________|21010/00130|0/0|00000/00000",
  "simulations": 5000,
  "elapsedMs": 1000,
  "winRate": 0.64,
  "wins": [3200.0, 1800.0],
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
| `serializedState` | string | State-only serialization (8 pipe-delimited sections: robber through devCards). See [state-action-serialization.md](state-action-serialization.md) Part I sections 3-10. |
| `simulations` | int | Total MCTS rollouts performed for this decision. |
| `elapsedMs` | int | Wall-clock time spent on MCTS search (milliseconds). |
| `winRate` | float | Acting player's win rate at the MCTS root (wins / rollouts). |
| `wins` | float[] | Raw MCTS win counts at the root, 0-indexed (index 0 = player 1). |
| `reachedTerminal` | bool | Whether MCTS fully resolved the tree (all root actions are Terminal). |
| `priorsRequested` | int | Number of NN prior requests sent during this search. |
| `priorsApplied` | int | Number of NN prior responses applied to tree nodes. |
| `priorStatesEvaluated` | int | Number of individual states evaluated by the NN server. |
| `permutations` | string[] | `serializedState` under each non-trivial symmetry permutation. Same order as `board.permutations`. |

State-value training blends each root's normalized per-player MCTS wins with the
one-hot final game winner. The blend weight is configurable as `mctsValueWeight`.
If one target is unavailable, the other is used alone; if both are unavailable,
the state is skipped.
Candidate result states are not exported or trained unless they naturally become a
later searched root.

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

*Produces training data for the state-only `placement_state_v3` model. Records every legal composite settlement-road action, including legal composites with zero rollouts.*

When `--export-type InitialPlacement` is specified, the game loop runs in placement-only mode. MCTS search is performed at settlement placement steps (`PlaceFirstSettlement` and `PlaceSecondSettlement`). Each MCTS root action is a `PlaceSettlementAction`, and its child state has `PlaceRoadAction` choices. The export combines these into composite actions serialized as `<vertex><direction>` strings (see [state-action-serialization.md](state-action-serialization.md) Part III). The MCTS leaf boundary is the exact deterministic result of the final placement road (`turnNumber: 1`, `stage: "r"`/`PreRoll`); it does not block or sample the subsequent stochastic `RollDiceAction`.

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
| `states` | PlacementState[] | Array of placement state records, one per settlement decision. |

The `turns` field is omitted. After recording placement decisions, simulation continues with seeded random legal play solely to obtain `winner`; no post-placement states are exported. If that continuation cannot finish within the action safety limit, `winner` is 0 and training uses only the MCTS target.

### Placement State Object

```json
{
  "playerTurn": 1,
  "stage": "a",
  "serializedState": "w5lb3ls4lW3hd0nW4ho2l|gsgbgw|._._._._._._._._._._._._._._._._._._._._._._._._|______________________________|",
  "simulations": 5000,
  "elapsedMs": 1000,
  "modelValue": 0.61,
  "valueTarget": 0.62,
  "actions": [
    {
      "action": "6N",
      "wins": [320.0, 180.0],
      "rollouts": 500,
      "winRate": 0.64,
      "modelPrior": 0.18,
      "permutations": ["8SE", "10SW", "14N", "16SE"]
    },
    {
      "action": "6SW",
      "wins": [280.0, 220.0],
      "rollouts": 480,
      "winRate": 0.583,
      "permutations": ["8N", "10SE", "14SW", "16N"]
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
| `stage` | string | Turn stage character: `a` (place 1st settlement) or `f` (place 2nd settlement). See [state-action-serialization.md](state-action-serialization.md) Part I section 4. |
| `serializedState` | string | 4-section placement phase state: `tiles\|ports\|placementVertices\|edges`. See [state-action-serialization.md](state-action-serialization.md) Part II. |
| `simulations` | int | Total MCTS rollouts performed for this decision. |
| `elapsedMs` | int | Wall-clock time spent on MCTS search (milliseconds). |
| `modelValue` | float? | Placement model's scalar value estimate when a combined prior response was applied; otherwise `null`. |
| `valueTarget` | float? | Rollout-weighted acting-player value target across legal composites; `null` if no rollouts are available. |
| `actions` | Action[] | Every C#-legal composite action with per-action MCTS statistics, including zero-rollout actions. This list also defines the exported legal mask. |
| `permutations` | string[] | `serializedState` under each non-trivial symmetry permutation. Same order as `board.permutations`. |

### Action Object

Each action represents a composite settlement + road placement. The action string encodes the settlement vertex and road direction as defined in [state-action-serialization.md](state-action-serialization.md) Part III.

```json
{
  "action": "6N",
  "wins": [320.0, 180.0],
  "rollouts": 500,
  "winRate": 0.64,
  "permutations": ["8SE", "10SW", "14N", "16SE"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `action` | string | Composite action string: `<vertex_index><road_direction>` (e.g. `6N`, `12NW`, `53NE`). |
| `wins` | float[] | MCTS win counts at the road grandchild node, 0-indexed (index 0 = player 1). See [Composite Action Stats](#composite-action-stats). |
| `rollouts` | int | Total rollouts at the road grandchild node. |
| `winRate` | float | Acting player's win rate (wins[playerIndex] / rollouts). |
| `modelPrior` | float? | Masked, globally normalized NN probability for this legal composite. The root inference retains the complete legal dense policy, so unexpanded actions receive priors too. `null` only when no valid root model response was applied. |
| `permutations` | string[] | Action string under each symmetry permutation. Same order as `board.permutations`. The wins, rollouts, and winRate are identical under permutation and are not repeated. |

### Composite Action Stats

During initial placement, the MCTS tree has this structure at each settlement decision:

```
Root (settlement choices)
 +-- PlaceSettlement(v=6)
 |    +-- PlaceRoad(e=5)   -> road grandchild: wins, rollouts
 |    +-- PlaceRoad(e=7)   -> road grandchild: wins, rollouts
 |    +-- PlaceRoad(e=8)   -> road grandchild: wins, rollouts
 +-- PlaceSettlement(v=10)
 |    +-- PlaceRoad(e=12)  -> road grandchild: wins, rollouts
 |    +-- PlaceRoad(e=14)  -> road grandchild: wins, rollouts
 ...etc
```

Each composite action maps to a specific road grandchild. The `wins` and `rollouts` are read directly from that grandchild MCTS node. This provides per-(vertex, road) granularity, allowing the model to learn directional road preferences.

For unexplored actions, the action remains in the export with `wins: []`, `rollouts: 0`, and `winRate: 0`. Python maps all listed actions to the dense legal mask. In combined training, visit shares from a normal shared-root MCTS search form the policy target and policy loss is masked to these legal indices.

`--simulations-per-action` runs separate rollouts from each post-composite state. Those rollout counts measure independent evaluation budgets, not shared-root action preference, and are therefore unsuitable as policy targets. Use standard shared-root placement search when training a combined value/policy model. `valueTarget` remains the rollout-weighted value label.

### Symmetry Permutations (InitialPlacement)

Board symmetry permutations transform vertex and edge indices. For placement data, both the state and the actions must be permuted:

1. **State permutation**: The 4-section placement state is permuted via `BoardSymmetry.PermutePlacementState`, which rearranges tiles, ports, vertices, and edges according to the symmetry.
2. **Action permutation**: Each `(vertex, edge)` pair is mapped through the symmetry's vertex and edge permutation arrays to produce a new `(vertex', edge')`, which is then re-serialized as a new action string via `PlacementActionSerializer`.

The `wins`, `rollouts`, and `winRate` for a permuted action are identical to the original — only the action identity changes. This avoids redundant data in the export.

Example with mini map (5 symmetry permutations):

```json
{
  "action": "6N",
  "wins": [320.0, 180.0],
  "rollouts": 500,
  "winRate": 0.64,
  "permutations": ["8SE", "10SW", "14N", "16SE"]
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
