# State Serialization

This document defines a fixed-length, human-readable board state serialization for 2-4 players, plus a reversible compact form for transformer ingestion.

## Encoding Overview

- **Human-readable form**: fixed-width decimal tokens separated by `|`.
- **Compact form**: remove separators, parse fixed-width decimal tokens, then convert each token to **Crockford base-32**.
- **Indexing**: all indices are 0-based and refer to the topology in `docs/board-topology.md`.

### Crockford Base-32 Alphabet

`0123456789ABCDEFGHJKMNPQRSTVWXYZ`

## Field Widths

Use **2-digit decimal** for all fields unless otherwise specified. If a value can exceed 99, use **3 digits** for that field and note it explicitly. In this schema all fields fit in 2 digits.

## Enumerations

### Resource Types (tile and ports)
- `00` = desert
- `01` = wood
- `02` = brick
- `03` = sheep
- `04` = wheat
- `05` = ore

### Number Tokens (tile)
- `00` = no token (desert)
- `02`..`12` = usual pip values (with `07` allowed but unused)

### Player Ids
- `00` = none/unassigned
- `01`..`04` = player 1..4

### Turn Stage
- `00` = initial placement: place 1st settlement
- `01` = initial placement: place 1st road
- `02` = initial placement: place 2nd settlement
- `03` = initial placement: place 2nd road
- `04` = pre-roll
- `05` = choose robber location
- `06` = build/trade

### Vertex Occupancy
- `00` = empty
- `01`..`04` = settlement by player 1..4
- `05`..`08` = city by player 1..4 (city = player id + 4)

### Edge Occupancy
- `00` = empty
- `01`..`04` = road by player 1..4

### Port Types (Harbors)
Each port is a coastal edge connecting two vertices on the board perimeter (see `docs/topology-reference.md` §5.6). Only the **type** is variable and needs serialization:
- `01` = 3:1 generic
- `02` = wood
- `03` = brick
- `04` = sheep
- `05` = wheat
- `06` = ore

## Serialization Layout

Tokens are emitted in the order below. All counts are fixed-length.

### 1) Tiles (19 tiles)
For each tile index `t` in order:
- `tile[t].resource` (00..05)
- `tile[t].number` (00..12)

**Total**: `19 * 2 = 38` tokens.

### 2) Robber
- `robber.tileIndex` (00..18)

**Total**: 1 token.

### 3) Current Turn
- `currentPlayer` (00..04)
- `turnStage` (00..06)

**Total**: 2 tokens.

### 4) Longest Road / Largest Army
- `longestRoadOwner` (00..04)
- `largestArmyOwner` (00..04)

**Total**: 2 tokens.

### 5) Vertices (54)
For each vertex index `v`:
- `vertex[v].occupancy` (00..08)

**Total**: 54 tokens.

### 6) Edges (72)
For each edge index `e`:
- `edge[e].occupancy` (00..04)

**Total**: 72 tokens.

### 7) Ports / Harbors (9 ports)
For each **port index** `p` (0..8), in the fixed order defined by the topology (see `docs/topology-reference.md` §5.6):
- `port[p].type` (01..06)

Port positions (which two boundary vertices each port connects) are fixed by the board topology and do not need to be serialized. Only the randomly assigned type is stored.

**Total**: 9 tokens.

### 8) Per-Player Resources (5 per player)
For each player in order (1..N):
- `resources[wood, brick, sheep, wheat, ore]` (00..99)

**Total**: `5 * N` tokens.

### 9) Per-Player Knights Played
For each player (1..N):
- `knightsPlayed` (00..99)

**Total**: `1 * N` tokens.

### 10) Per-Player Dev Cards in Hand (5 per player)
Order: `[knight, victoryPoint, roadBuilding, monopoly, yearOfPlenty]`

For each player (1..N):
- `devCards[5]` (00..99)

**Total**: `5 * N` tokens.

## Total Token Count

Let `N` be the number of players.

```
38 + 1 + 2 + 2 + 54 + 72 + 9 + (5*N) + (1*N) + (5*N)
= 178 + 11*N
```

Totals:
- 2 players: `178 + 22 = 200` tokens
- 3 players: `178 + 33 = 211` tokens
- 4 players: `178 + 44 = 222` tokens

## Human-Readable Example (partial)

```
01|06|05|08|...|18|03|02|04|00|00|...
```

## Compact Form (Transformer Ingestion)

1) Split by `|` into fixed-width decimal tokens. Each token is 2 digits.
2) Convert each token (0-99) to **Crockford base-32**, 1 or 2 chars. Recommended: pad to **2 chars** for fixed length.
3) Concatenate with no separators.

Example conversion:
- Decimal `00` -> base-32 `00`
- Decimal `31` -> base-32 `0Z`
- Decimal `99` -> base-32 `33`

This yields a compact, fixed-length string that is reversible to the human-readable form.

## Notes and Constraints

- All indices and ordering are defined in `docs/board-topology.md` and `docs/topology-reference.md`.
- Port positions are fixed by the topology (9 ports, each connecting two boundary vertices). Only the port type is serialized.
- Standard map: 4 generic (3:1) ports + 5 resource-specific (2:1) ports = 9 ports total.
- Desert tiles must have number token `00`.
- If `currentPlayer = 00`, the turn stage must be `00` (initial placement), otherwise it must be `04..06`.
