# State Serialization

This document defines a fixed-length, human-readable board state serialization for 2-4 players, plus a reversible compact form for transformer ingestion. Counts are given for the **mini map** (radius 1), the **small map** (10 tiles, non-circular), and the **standard map** (radius 2).

## Encoding Overview

- **Human-readable form**: single-character Crockford base-32 tokens. `/` separates tokens within tiles and players within per-player sections; `|` separates major sections.
- **Compact form**: strip all `/` and `|` separators to produce a fixed-length string of Crockford base-32 characters.
- **Indexing**: all indices are 0-based and refer to the topology in `docs/board-topology.md`.

### Crockford Base-32 Alphabet

`0123456789ABCDEFGHJKMNPQRSTVWXYZ`

Each character encodes a value 0–31. All fields in this schema fit within that range.

## Enumerations

All values below are shown in decimal. In serialized form each value is a **single** Crockford base-32 character.

### Resource Types (tile and ports)
- `0` = desert
- `1` = wood
- `2` = brick
- `3` = sheep
- `4` = wheat
- `5` = ore

### Number Tokens (tile)
- `0` = no token (desert)
- `2`..`12` = usual pip values (with `7` unused in standard play)

### Player Ids
- `0` = none/unassigned
- `1`..`4` = player 1..4

### Turn Stage
- `0` = initial placement: place 1st settlement
- `1` = initial placement: place 1st road
- `2` = initial placement: place 2nd settlement
- `3` = initial placement: place 2nd road
- `4` = pre-roll
- `5` = choose robber location
- `6` = choose player to rob from
- `7` = build/trade

### Vertex Occupancy
- `0` = empty
- `1`..`4` = settlement by player 1..4
- `5`..`8` = city by player 1..4 (city = player id + 4)

### Edge Occupancy
- `0` = empty
- `1`..`4` = road by player 1..4

### Port Types (Harbors)
Each port is a coastal edge connecting two vertices on the board perimeter (see `docs/topology-reference.md` §4.6, §5.6, §6.6). Only the **type** is variable and needs serialization:
- `1` = 3:1 generic
- `2` = wood
- `3` = brick
- `4` = sheep
- `5` = wheat
- `6` = ore

## Serialization Layout

Tokens are emitted in the order below. All counts are fixed-length. Major sections are separated by `|`. Within tiles each token is separated by `/`. Within per-player sections (resources, knights, dev cards) individual players are separated by `/`.

Let `T` = number of tiles, `V` = number of vertices, `E` = number of edges, `P` = number of ports, `N` = number of players.

| Parameter | Mini | Small | Standard |
|-----------|------|-------|----------|
| T (tiles) | 7 | 10 | 19 |
| V (vertices) | 24 | 32 | 54 |
| E (edges) | 30 | 41 | 72 |
| P (ports) | 6 | 7 | 9 |
| N (players) | 2 | 2–3 | 3–4 |

### 1) Tiles

For each tile index `t` in order:
- `tile[t].resource` (0..5)
- `tile[t].number` (0..C, i.e. 0..12)

Tokens within this section are separated by `/`.

**Tokens**: `T * 2` — mini: 14, small: 20, standard: 38.

### 2) Robber

- `robber.tileIndex` (0..T-1)

**Tokens**: 1.

### 3) Current Turn

- `currentPlayer` (0..4)
- `turnStage` (0..7)

**Tokens**: 2.

### 4) Longest Road / Largest Army

- `longestRoadOwner` (0..4)
- `largestArmyOwner` (0..4)

**Tokens**: 2.

### 5) Vertices

For each vertex index `v`:
- `vertex[v].occupancy` (0..8)

**Tokens**: `V` — mini: 24, small: 32, standard: 54.

### 6) Edges

For each edge index `e`:
- `edge[e].occupancy` (0..4)

**Tokens**: `E` — mini: 30, small: 41, standard: 72.

### 7) Ports / Harbors

For each **port index** `p` (0..P-1), in the fixed order defined by the topology (see `docs/topology-reference.md`):
- `port[p].type` (1..6)

Port positions (which two boundary vertices each port connects) are fixed by the board topology and do not need to be serialized. Only the randomly assigned type is stored.

**Tokens**: `P` — mini: 6, small: 7, standard: 9.

### 8) Per-Player Resources (5 per player)

For each player in order (1..N):
- `resources[wood, brick, sheep, wheat, ore]` (0..19 standard, 0..14 small, 0..10 mini; all fit in single base-32 char)

Each player's 5 resource tokens are concatenated; players are separated by `/`.

**Tokens**: `5 * N` — mini: 10, small: 10–15, standard: 15–20.

### 9) Per-Player Knights Played

For each player (1..N):
- `knightsPlayed` (0..14 standard, 0..10 small, 0..7 mini)

Players are separated by `/`.

**Tokens**: `1 * N` — mini: 2, small: 2–3, standard: 3–4.

### 10) Per-Player Dev Cards in Hand (5 per player)

Order: `[knight, victoryPoint, roadBuilding, monopoly, yearOfPlenty]`

For each player (1..N):
- `devCards[5]` (0..14 standard, 0..10 small, 0..7 mini; per card type)

Each player's 5 dev-card tokens are concatenated; players are separated by `/`.

**Tokens**: `5 * N` — mini: 10, small: 10–15, standard: 15–20.

## Total Token Count

Let `T`, `V`, `E`, `P` be map-dependent and `N` be the number of players.

```
(2*T) + 1 + 2 + 2 + V + E + P + (5*N) + (1*N) + (5*N)
= (2*T + V + E + P + 5) + 11*N
```

### Mini Map (T=7, V=24, E=30, P=6, N=2)

```
(14 + 24 + 30 + 6 + 5) + 11*2
= 79 + 22 = 101 tokens
```

### Small Map (T=10, V=32, E=41, P=7)

```
(20 + 32 + 41 + 7 + 5) + 11*N
= 105 + 11*N
```

- 2 players: `105 + 22 = 127` tokens
- 3 players: `105 + 33 = 138` tokens

### Standard Map (T=19, V=54, E=72, P=9)

```
(38 + 54 + 72 + 9 + 5) + 11*N
= 178 + 11*N
```

- 3 players: `178 + 33 = 211` tokens
- 4 players: `178 + 44 = 222` tokens

## Human-Readable Example

### Mini Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. A few turns have passed.

Tile layout (7 tiles): wood/6, brick/4, sheep/5, wheat/A, desert/0, wheat/9, ore/3.

Tiles (resource/pip pairs separated by `/`):
```
1/6/2/4/3/5/4/A/0/0/4/9/5/3
```

Board state:
- **Robber**: tile 4 (desert)
- **Current turn**: player 1, build/trade (stage 7)
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlement on vertex 6, player 2 settlement on vertex 14
- **Edges**: player 1 road on edge 5, player 2 road on edge 13
- **Ports**: generic, sheep, generic, brick, generic, wood
- **Player 1 resources**: wood=2, brick=1, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=3, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none

Full serialized (sections separated by `|`):
```
1/6/2/4/3/5/4/A/0/0/4/9/5/3|4|17|00|000001000000000200000000|000001000000000000000000000000|134132|21010/00130|0/0|00000/00000
```

### Small Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. One turn has begun (player 2 is about to roll).

Tile layout (10 tiles): wheat/3, brick/4, sheep/5, wood/A, brick/B, wood/C, ore/6, wheat/9, sheep/8, desert/0.

Tiles (resource/pip pairs separated by `/`):
```
4/3/2/4/3/5/1/A/2/B/1/C/5/6/4/9/3/8/0/0
```

Board state:
- **Robber**: tile 9 (desert)
- **Current turn**: player 2, pre-roll (stage 4)
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlements on v0 and v7; player 2 settlements on v1 and v2
- **Edges**: player 1 roads on e0 and e6; player 2 roads on e2 and e4
- **Ports**: generic, wood, wheat, generic, sheep, generic, brick
- **Player 1 resources**: wood=1, brick=0, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=0, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none

Full serialized (sections separated by `|`):
```
4/3/2/4/3/5/1/A/2/B/1/C/5/6/4/9/3/8/0/0|9|24|00|12200001000000000000000000000000|10202010000000000000000000000000000000000|1251413|10010/00100|0/0|00000/00000
```

### Standard Map (3 players, mid-game)

Several turns in: players have built additional roads, player 2 has upgraded a settlement to a city, robber has been moved. Player 1 has played 2 knights.

Tile layout (19 tiles): wood/5, ore/2, brick/6, wheat/3, wood/8, sheep/A, wheat/9, ore/C, sheep/B, wood/4, brick/8, sheep/A, wheat/9, ore/4, sheep/5, brick/6, wood/3, wheat/B, desert/0.

Tiles (resource/pip pairs separated by `/`):
```
1/5/5/2/2/6/4/3/1/8/3/A/4/9/5/C/3/B/1/4/2/8/3/A/4/9/5/4/3/5/2/6/1/3/4/B/0/0
```

Board state:
- **Robber**: tile 5 (sheep/A — blocking a productive hex)
- **Current turn**: player 2, build/trade (stage 7)
- **Longest road**: none; **Largest army**: player 1
- **Vertices**: player 1 settlements on v8 and v35; player 2 city on v18, settlement on v44; player 3 settlements on v24 and v31
- **Edges**: player 1 roads on e6, e16, e49, e41; player 2 roads on e25, e32, e55, e61; player 3 roads on e34, e35, e42, e48
- **Ports**: generic, generic, wood, generic, brick, sheep, wheat, ore, generic
- **Player 1 resources**: wood=3, brick=1, sheep=2, wheat=0, ore=1
- **Player 2 resources**: wood=0, brick=2, sheep=1, wheat=4, ore=3
- **Player 3 resources**: wood=1, brick=0, sheep=3, wheat=2, ore=0
- **Knights played**: 2 / 0 / 1
- **Dev cards (knight/vp/roadBuild/monopoly/yearOfPlenty)**: 1/0/0/0/0 / 0/1/0/1/0 / 0/0/1/0/0

Full serialized:
```
1/5/5/2/2/6/4/3/1/8/3/A/4/9/5/C/3/B/1/4/2/8/3/A/4/9/5/4/3/5/2/6/1/3/4/B/0/0|5|27|01|000000001000000000600000000000000200000000000000030003|000000100000000000200000010020000100300000020010000000001000200000000000|112134561|31201/02143/10320|2/0/1|10000/01010/00100
```

## Compact Form (Transformer Ingestion)

Remove all `/` and `|` separators. The result is a fixed-length string of Crockford base-32 characters, one character per token.

### Mini Map Example

From the human-readable example above:
```
1624354A004953 4 17 00 000001000000000200000000 000001000000000000000000000000 134132 2101000130 00 0000000000
```
(spaces added for clarity — actual compact form has no spaces)

Compact string (`101` characters):
```
1624354A004953417000000010000000002000000000000010000000000000000000000001341322101000130000000000000
```

### Small Map Example

From the human-readable example above:
```
4324351A2B1C56493800 9 24 00 12200001000000000000000000000000 10202010000000000000000000000000000000000 1251413 1001000100 00 0000000000
```
(spaces added for clarity — actual compact form has no spaces)

Compact string (`127` characters):
```
4324351A2B1C5649380092400122000010000000000000000000000001020201000000000000000000000000000000000012514131001000100000000000000
```

### Standard Map Example

Compact string (`211` characters):
```
15522643183A495C3B14283A49543526134B0052701000000001000000000600000000000000200000000000000030003000000100000000000200000010020000100300000020010000000001000200000000000112134561312010214310320201100000101000100
```

Compact string lengths:
- **Mini map** (2 players): `101` characters
- **Small map** (2 players): `127` characters
- **Small map** (3 players): `138` characters
- **Standard map** (3 players): `211` characters
- **Standard map** (4 players): `222` characters

This yields a compact, fixed-length string that is reversible to the human-readable form by re-inserting separators at the known fixed positions.

## Notes and Constraints

- All indices and ordering are defined in `docs/board-topology.md` and `docs/topology-reference.md`.
- Port positions are fixed by the topology. Standard map: 4 generic (3:1) + 5 resource-specific (2:1) = 9 ports. Small map: 3 generic (3:1) + 4 resource-specific (2:1) = 7 ports. Mini map: 3 generic (3:1) + 3 resource-specific (2:1) = 6 ports.
- Desert tiles must have number token `0`.
- Turn stage `0`–`3` applies during initial placement. Turn stages `4`–`7` apply during normal play.
- If `currentPlayer = 0`, the game has not started or is between rounds.
