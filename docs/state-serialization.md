# State Serialization

This document defines a fixed-length, human-readable board state serialization for 2-4 players, plus a reversible compact form for transformer ingestion. Counts are given for the **mini map** (radius 1), the **small map** (10 tiles, non-circular), and the **standard map** (radius 2).

## Encoding Overview

- **Human-readable form**: single-character tokens. Most fields use Crockford base-32 (uppercase + digits). Tile number tokens use a separate lowercase alphabet (`a`–`k`) encoding pip count and side. `/` separates players within per-player sections; `|` separates major sections. Tiles are concatenated directly without separators.
- **Compact form**: strip all `/` and `|` separators to produce a fixed-length string, one character per token.
- **Indexing**: all indices are 0-based and refer to the topology in `docs/board-topology.md`.

### Crockford Base-32 Alphabet

`0123456789ABCDEFGHJKMNPQRSTVWXYZ`

Each character encodes a value 0–31. All fields except tile numbers use this alphabet.

## Serialization Layout

Tokens are emitted in the order below. All counts are fixed-length. Major sections are separated by `|`. Within per-player sections (resources, knights, dev cards) individual players are separated by `/`. Tile tokens are concatenated directly without separators.

Let `T` = number of tiles, `V` = number of vertices, `E` = number of edges, `P` = number of ports, `N` = number of players.

| Parameter | Mini | Small | Standard |
|-----------|------|-------|----------|
| T (tiles) | 7 | 10 | 19 |
| V (vertices) | 24 | 32 | 54 |
| E (edges) | 30 | 41 | 72 |
| P (ports) | 6 | 7 | 9 |
| N (players) | 2 | 2–3 | 3–4 |

### 1) Tiles

For each tile index `t` in order, two tokens:
- `tile[t].resource` — Crockford base-32, using these **resource type** values:
  - `0` = desert, `1` = wood, `2` = brick, `3` = sheep, `4` = wheat, `5` = ore
- `tile[t].number` — lowercase pip letter (`a`..`k`), using a separate **tile number alphabet** intentionally disjoint from Crockford base-32 so a tokenizer can distinguish tile likelihood tokens from all other fields. Characters are ordered by ascending pip count; within each pip level, "low" (below 7) comes before "high" (above 7):

| Char | Pips | Side | Number |
|------|------|------|--------|
| `a`  | 0    | —    | 0 (desert) |
| `b`  | 1    | low  | 2  |
| `c`  | 1    | high | 12 |
| `d`  | 2    | low  | 3  |
| `e`  | 2    | high | 11 |
| `f`  | 3    | low  | 4  |
| `g`  | 3    | high | 10 |
| `h`  | 4    | low  | 5  |
| `i`  | 4    | high | 9  |
| `j`  | 5    | low  | 6  |
| `k`  | 5    | high | 8  |

Desert tiles must have pip letter `a` (= number 0). Tokens are concatenated directly (no separators).

**Tokens**: `T * 2` — mini: 14, small: 20, standard: 38.

### 2) Ports / Harbors

Each port is a coastal edge connecting two vertices on the board perimeter (see `docs/topology-reference.md` §4.6, §5.6, §6.6). Port positions are fixed by the board topology and do not need to be serialized. Only the randomly assigned type is stored.

For each **port index** `p` (0..P-1), in the fixed order defined by the topology:
- `port[p].type` — **port type** value:
  - `1` = 3:1 generic
  - `2` = wood
  - `3` = brick
  - `4` = sheep
  - `5` = wheat
  - `6` = ore

Standard map: 4 generic (3:1) + 5 resource-specific (2:1) = 9 ports. Small map: 3 generic (3:1) + 4 resource-specific (2:1) = 7 ports. Mini map: 3 generic (3:1) + 3 resource-specific (2:1) = 6 ports.

**Tokens**: `P` — mini: 6, small: 7, standard: 9.

### 3) Robber

- `robber.tileIndex` (0..T-1)

**Tokens**: 1.

### 4) Current Turn

- `currentPlayer` — **player id**: `0` = none/unassigned, `1`..`4` = player 1..4
- `turnStage` — **turn stage** value:
  - `0` = initial placement: place 1st settlement
  - `1` = initial placement: place 1st road
  - `2` = initial placement: place 2nd settlement
  - `3` = initial placement: place 2nd road
  - `4` = pre-roll
  - `5` = choose robber location
  - `6` = choose player to rob from
  - `7` = build/trade

Turn stages `0`–`3` apply during initial placement. Turn stages `4`–`7` apply during normal play. If `currentPlayer = 0`, the game has not started or is between rounds.

**Tokens**: 2.

### 5) Longest Road / Largest Army

- `longestRoadOwner` — player id (0..4, where 0 = none)
- `largestArmyOwner` — player id (0..4, where 0 = none)

**Tokens**: 2.

### 6) Vertices

For each vertex index `v`:
- `vertex[v].occupancy` — **vertex occupancy** value:
  - `0` = empty
  - `1`..`4` = settlement by player 1..4
  - `5`..`8` = city by player 1..4 (city = player id + 4)

**Tokens**: `V` — mini: 24, small: 32, standard: 54.

### 7) Edges

For each edge index `e`:
- `edge[e].occupancy` — **edge occupancy** value:
  - `0` = empty
  - `1`..`4` = road by player 1..4

**Tokens**: `E` — mini: 30, small: 41, standard: 72.

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
(2*T) + P + 1 + 2 + 2 + V + E + (5*N) + (1*N) + (5*N)
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

## Human-Readable Examples

### Mini Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. A few turns have passed.

Board state:
- **Tiles** (7 tiles): wood/6, brick/4, sheep/5, wheat/10, desert/0, wheat/9, ore/3
- **Ports**: generic, sheep, generic, brick, generic, wood
- **Robber**: tile 4 (desert)
- **Current turn**: player 1, build/trade (stage 7)
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlement on vertex 6, player 2 settlement on vertex 14
- **Edges**: player 1 road on edge 5, player 2 road on edge 13
- **Player 1 resources**: wood=2, brick=1, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=3, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none

Full serialized (sections separated by `|`):
```
1j2f3h4g0a4i5d|134132|4|17|00|000001000000000200000000|000001000000000000000000000000|21010/00130|0/0|00000/00000
├── 1:tiles ─┤ │2:   ││ │  │  └──── 6:vertices ──────┤ ├──────── 7:edges ───────────┤ ├─ 8:res ─┤ │   ├ 10:dev ─┤
               ├Ports┤│ │  │                                                                      └── 9: Knights played
                      │ │  └── 5:longest-road/largest-army
                      │ └── 4:current-turn (player=1, stage=7)
                      └── 3:robber (tile 4)
```

### Small Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. One turn has begun (player 2 is about to roll).

Board state:
- **Tiles** (10 tiles): wheat/3, brick/4, sheep/5, wood/10, brick/11, wood/12, ore/6, wheat/9, sheep/8, desert/0
- **Ports**: generic, wood, wheat, generic, sheep, generic, brick
- **Robber**: tile 9 (desert)
- **Current turn**: player 2, pre-roll (stage 4)
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlements on v0 and v7; player 2 settlements on v1 and v2
- **Edges**: player 1 roads on e0 and e6; player 2 roads on e2 and e4
- **Player 1 resources**: wood=1, brick=0, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=0, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none

Full serialized (sections separated by `|`):
```
4d2f3h1g2e1c5j4i3k0a|1251413|9|24|00|12200001000000000000000000000000|10202010000000000000000000000000000000000|10010/00100|0/0|00000/00000
├──── 1:tiles ─────┤ │2:    ││ │  │  ├───────── 6:vertices ─────────┤ ├──────────── 7:edges ──────────────────┤ ├─ 8:res ──┤│   ├ 10:dev ─┤
                     ├ports ┤│ │  │                                                                                         └── 9: Knights played
                             │ │  └── 5:longest-road/largest-army
                             │ └── 4:current-turn (player=2, stage=4)
                             └── 3:robber (tile 9)
```

### Standard Map (3 players, mid-game)

Several turns in: players have built additional roads, player 2 has upgraded a settlement to a city, robber has been moved. Player 1 has played 2 knights.

Board state:
- **Tiles** (19 tiles): wood/5, ore/2, brick/6, wheat/3, wood/8, sheep/10, wheat/9, ore/12, sheep/11, wood/4, brick/8, sheep/10, wheat/9, ore/4, sheep/5, brick/6, wood/3, wheat/11, desert/0
- **Ports**: generic, generic, wood, generic, brick, sheep, wheat, ore, generic
- **Robber**: tile 5 (sheep/10 — blocking a productive hex)
- **Current turn**: player 2, build/trade (stage 7)
- **Longest road**: none; **Largest army**: player 1
- **Vertices**: player 1 settlements on v8 and v35; player 2 city on v18, settlement on v44; player 3 settlements on v24 and v31
- **Edges**: player 1 roads on e6, e16, e49, e41; player 2 roads on e25, e32, e55, e61; player 3 roads on e34, e35, e42, e48
- **Player 1 resources**: wood=3, brick=1, sheep=2, wheat=0, ore=1
- **Player 2 resources**: wood=0, brick=2, sheep=1, wheat=4, ore=3
- **Player 3 resources**: wood=1, brick=0, sheep=3, wheat=2, ore=0
- **Knights played**: 2 / 0 / 1
- **Dev cards (knight/vp/roadBuild/monopoly/yearOfPlenty)**: 1/0/0/0/0 / 0/1/0/1/0 / 0/0/1/0/0

Full serialized:
```
1h5b2j4d1k3g4i5c3e1f2k3g4i5f3h2j1d4e0a|112134561|5|27|01|000000001000000000600000000000000200000000000000030003 ...
├──────────── 1:tiles ───────────────┤ ├2:ports┤ │ │  │  ├────────────────────── 6:vertices ─────────────────── ...
                                                 │ │  │
                                                 │ │  └── 5:longest-road=0/largest-army=1 (player 1)
                                                 │ └── 4:current-turn (player=2, stage=7)
                                                 └── 3:robber (tile 5)

... 0003|000000100000000000200000010020000100300000020010000000001000200000000000|31201/02143/10320|2/0/1|10000/01010/00100
───────┤ ├─────────────────────────────── 7:edges ──────────────────────────────┤ ├─8:resources ──┤ │     ├ 10:dev-cards ─┤
                                                                                                    └── 9: Knights played
```

## Compact Form (Transformer Ingestion)

Remove all `/` and `|` separators. The result is a fixed-length string, one character per token (Crockford base-32 uppercase/digits for most fields, lowercase `a`–`k` for tile numbers). Since tiles are already concatenated without separators, only `|` (section boundaries) and `/` (player separators in sections 8–10) are stripped.

Compact string lengths:
- **Mini map** (2 players): `101` characters
- **Small map** (2 players): `127` characters
- **Small map** (3 players): `138` characters
- **Standard map** (3 players): `211` characters
- **Standard map** (4 players): `222` characters

## Notes and Constraints

- All indices and ordering are defined in `docs/board-topology.md`.
