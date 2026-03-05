# State Serialization

This document defines a fixed-length, human-readable board state serialization for 2-4 players, plus a reversible compact form for transformer ingestion. Counts are given for the **mini map** (radius 1), the **small map** (10 tiles, non-circular), and the **standard map** (radius 2).

## Encoding Overview

- **Human-readable form**: single-character tokens drawn from semantically disjoint alphabets (see §Token Alphabets below). `/` separates players within per-player sections; `|` separates major sections. Tiles are concatenated directly without separators.
- **Compact form**: strip all `/` and `|` separators to produce a fixed-length string, one character per token.
- **Indexing**: all indices are 0-based and refer to the topology in `docs/board-topology.md`.

### Token Alphabets

Each semantic category uses its own disjoint character set so that a tokenizer can learn category-specific embeddings. Categories that share the same underlying concept reuse the same alphabet — positional embeddings disambiguate context.

| Category | Characters | Notes |
|----------|-----------|-------|
| Resource type | `d w b s W o` | Desert, wood, brick, sheep, wheat, ore. Shared by tiles and ports. |
| Port generic | `g` | 3:1 generic harbor (extends the resource alphabet for ports) |
| Pip count | `0 1 2 3 4 5` | 0 = desert, 1–5 = pip count. Shares digit chars with count alphabet. |
| Side | `l h n` | Low (below 7), high (above 7), none (desert) |
| Building type | `. v c` | Empty, village (settlement), city |
| Player ID | `_ - + * ^` | None, player 1–4 |
| Turn stage | `a e f i r x y t` | See §4 for mapping |
| Count | `0 1 2 3 4 5 6 7 8 9 A B C D E F G H J K` | Crockford base-32 values 0–19 |

**Shared digits**: pip count (`0`–`5`) and count (`0`–`9`) share digit characters. These appear at known fixed positions, so positional embeddings disambiguate.

**Full token vocabulary** (46 unique characters):
- Lowercase: `a b c d e f g h i l n o r s t v w x y`
- Uppercase: `A B C D E F G H J K W`
- Digits: `0 1 2 3 4 5 6 7 8 9`
- Punctuation: `. _ - + * ^`
- Separators (human-readable only): `| /`

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

For each tile index `t` in order, three tokens:
- `tile[t].resource` — **resource type**:
  - `d` = desert, `w` = wood, `b` = brick, `s` = sheep, `W` = wheat, `o` = ore
- `tile[t].pips` — **pip count**:
  - `0` = desert (no pips), `1`–`5` = number of pips on the token
- `tile[t].side` — **side** (which side of 7 the number falls on):
  - `l` = low (number < 7), `h` = high (number > 7), `n` = none (desert)

The three tokens together encode the tile number:

| Pips | Side | Number |
|------|------|--------|
| `0`  | `n`  | 0 (desert) |
| `1`  | `l`  | 2  |
| `1`  | `h`  | 12 |
| `2`  | `l`  | 3  |
| `2`  | `h`  | 11 |
| `3`  | `l`  | 4  |
| `3`  | `h`  | 10 |
| `4`  | `l`  | 5  |
| `4`  | `h`  | 9  |
| `5`  | `l`  | 6  |
| `5`  | `h`  | 8  |

Desert tiles must have pips `0` and side `n`. Tokens are concatenated directly (no separators).

**Tokens**: `T * 3` — mini: 21, small: 30, standard: 57.

### 2) Ports / Harbors

Each port is a coastal edge connecting two vertices on the board perimeter (see `docs/topology-reference.md` §4.6, §5.6, §6.6). Port positions are fixed by the board topology and do not need to be serialized. Only the randomly assigned type is stored.

For each **port index** `p` (0..P-1), in the fixed order defined by the topology:
- `port[p].type` — same **resource type** characters as tiles, plus `g` for generic:
  - `g` = 3:1 generic
  - `w` = wood
  - `b` = brick
  - `s` = sheep
  - `W` = wheat
  - `o` = ore

Standard map: 4 generic (3:1) + 5 resource-specific (2:1) = 9 ports. Small map: 2 generic (3:1) + 4 resource-specific (2:1) = 6 ports. Mini map: 3 generic (3:1) + 3 resource-specific (2:1) = 6 ports.

**Tokens**: `P` — mini: 6, small: 6, standard: 9.

### 3) Robber

- `robber.tileIndex` — **count** character (Crockford base-32, value 0..18)

**Tokens**: 1.

### 4) Current Turn

- `currentPlayer` — **player id**: `_` = none/unassigned, `-` = player 1, `+` = player 2, `*` = player 3, `^` = player 4
- `turnStage` — **turn stage**:
  - `a` = initial placement: place 1st settlement
  - `e` = initial placement: place 1st road
  - `f` = initial placement: place 2nd settlement
  - `i` = initial placement: place 2nd road
  - `r` = pre-roll
  - `x` = choose robber location
  - `y` = choose player to rob from
  - `t` = build/trade

Turn stages `a`, `e`, `f`, `i` apply during initial placement. Turn stages `r`, `x`, `y`, `t` apply during normal play. If `currentPlayer = _`, the game has not started or is between rounds.

**Tokens**: 2.

### 5) Longest Road / Largest Army

- `longestRoadOwner` — **player id** (`_` = none, `-` `+` `*` `^` = player 1..4)
- `largestArmyOwner` — **player id** (`_` = none, `-` `+` `*` `^` = player 1..4)

**Tokens**: 2.

### 6) Vertices

For each vertex index `v`, two tokens:
- `vertex[v].building` — **building type**:
  - `.` = empty
  - `v` = village (settlement)
  - `c` = city
- `vertex[v].owner` — **player id**:
  - `_` = none (must pair with `.`)
  - `-` `+` `*` `^` = player 1..4

Empty vertices are encoded as `._`. Settlements as `v-`, `v+`, `v*`, `v^`. Cities as `c-`, `c+`, `c*`, `c^`.

**Tokens**: `V * 2` — mini: 48, small: 64, standard: 108.

### 7) Edges

For each edge index `e`:
- `edge[e].occupancy` — **player id**:
  - `_` = empty (no road)
  - `-` `+` `*` `^` = road by player 1..4

**Tokens**: `E` — mini: 30, small: 41, standard: 72.

### 8) Per-Player Resources (5 per player)

For each player in order (1..N):
- `resources[wood, brick, sheep, wheat, ore]` — **count** (Crockford base-32; `0`=0 .. `K`=19)

Each player's 5 resource tokens are concatenated; players are separated by `/`.

**Tokens**: `5 * N` — mini: 10, small: 10–15, standard: 15–20.

### 9) Per-Player Knights Played

For each player (1..N):
- `knightsPlayed` — **count** (Crockford base-32; `0`=0 .. `E`=14)

Players are separated by `/`.

**Tokens**: `1 * N` — mini: 2, small: 2–3, standard: 3–4.

### 10) Per-Player Dev Cards in Hand (5 per player)

Order: `[knight, victoryPoint, roadBuilding, monopoly, yearOfPlenty]`

For each player (1..N):
- `devCards[5]` — **count** (Crockford base-32; `0`=0 .. `E`=14)

Each player's 5 dev-card tokens are concatenated; players are separated by `/`.

**Tokens**: `5 * N` — mini: 10, small: 10–15, standard: 15–20.

## Total Token Count

Let `T`, `V`, `E`, `P` be map-dependent and `N` be the number of players.

```
(3*T) + P + 1 + 2 + 2 + (2*V) + E + (5*N) + (1*N) + (5*N)
= (3*T + 2*V + E + P + 5) + 11*N
```

### Mini Map (T=7, V=24, E=30, P=6, N=2)

```
(21 + 48 + 30 + 6 + 5) + 11*2
= 110 + 22 = 132 tokens
```

### Small Map (T=10, V=32, E=41, P=7)

```
(30 + 64 + 41 + 7 + 5) + 11*N
= 147 + 11*N
```

- 2 players: `147 + 22 = 169` tokens
- 3 players: `147 + 33 = 180` tokens

### Standard Map (T=19, V=54, E=72, P=9)

```
(57 + 108 + 72 + 9 + 5) + 11*N
= 251 + 11*N
```

- 3 players: `251 + 33 = 284` tokens
- 4 players: `251 + 44 = 295` tokens

## Human-Readable Examples

### Mini Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. A few turns have passed.

Board state:
- **Tiles** (7 tiles): wood/6, brick/4, sheep/5, wheat/10, desert/0, wheat/9, ore/3
- **Ports**: generic, sheep, generic, brick, generic, wood
- **Robber**: tile 4 (desert)
- **Current turn**: player 1, build/trade
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlement on vertex 6, player 2 settlement on vertex 14
- **Edges**: player 1 road on edge 5, player 2 road on edge 13
- **Player 1 resources**: wood=2, brick=1, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=3, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none

Full serialized (sections separated by `|`):
```
w5lb3ls4lW3hd0nW4ho2l|gsgbgw|4|-t|__|._._._._._._v-._._._._._._._v+._._._._._._._._._|_____-_______+________________|21010/00130|0/0|00000/00000
├─── 1:tiles ────────┤│2:   ││ │  │  ├──────────── 6:vertices ──────────────────────┤ ├─────── 7:edges ────────────┤ ├─ 8:res ─┤ │   ├ 10:dev ─┤
                      ├ports┤│ │  │                                                                                              └── 9: Knights
                             │ │  └── 5:longest-road/largest-army (__ = none/none)
                             │ └── 4:current-turn (-=player 1, t=build/trade)
                             └── 3:robber (4 = tile 4)
```

### Small Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. One turn has begun (player 2 is about to roll).

Board state:
- **Tiles** (10 tiles): wheat/3, brick/4, sheep/5, wood/10, brick/11, wood/12, ore/6, wheat/9, sheep/8, desert/0
- **Ports**: generic, wood, wheat, generic, sheep, generic, brick
- **Robber**: tile 9 (desert)
- **Current turn**: player 2, pre-roll
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlements on v0 and v7; player 2 settlements on v1 and v2
- **Edges**: player 1 roads on e0 and e6; player 2 roads on e2 and e4
- **Player 1 resources**: wood=1, brick=0, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=0, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none

Full serialized (sections separated by `|`):
```
W2lb3ls4lw3hb2hw1ho5lW4hs5hd0n|gwWgsgb|9|+r|__|v-v+v+._._._._v-._._._._._._._._._._._._._._._._._._._._._._._._|-_+_+_-__________________________________|10010/00100|0/0|00000/00000
├──── 1:tiles ───────────────┤ │2:   │ │ │  │  ├──────────────── 6:vertices ──────────────────────────────────┤ ├────────────── 7:edges ────────────────┤ ├─ 8:res ─┤ │   ├ 10:dev ─┤
                               ├ports┤ │ │  │                                                                                                                         └── 9: Knights
                                       │ │  └── 5:longest-road/largest-army (__ = none/none)
                                       │ └── 4:current-turn (+=player 2, r=pre-roll)
                                       └── 3:robber (9 = tile 9)
```

### Standard Map (3 players, mid-game)

Several turns in: players have built additional roads, player 2 has upgraded a settlement to a city, robber has been moved. Player 1 has played 2 knights.

Board state:
- **Tiles** (19 tiles): wood/5, ore/2, brick/6, wheat/3, wood/8, sheep/10, wheat/9, ore/12, sheep/11, wood/4, brick/8, sheep/10, wheat/9, ore/4, sheep/5, brick/6, wood/3, wheat/11, desert/0
- **Ports**: generic, generic, wood, generic, brick, sheep, wheat, ore, generic
- **Robber**: tile 5 (sheep/10 — blocking a productive hex)
- **Current turn**: player 2, build/trade
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
w4lo1lb5lW2lw5hs3hW4ho1hs2hw3lb5hs3hW4ho3ls4lb5lw2lW2hd0n|ggwgbsWog|5|+t|_-|._._._._._._._._v-._._._._._._._._._c+._._._._._v*._._._._._._v*._._._v-._._._._._._._._v+._._._._._._._._._ ...
├──────────────── 1:tiles ──────────────────────────────┤ ├2:ports┤ │ │  │  ├────────────────────────────────── 6:vertices ──────────────────────────────────────────────────────────────── ...
                                                                    │ │  │
                                                                    │ │  └── 5:longest-road=_(none)/largest-army=-(player 1)
                                                                    │ └── 4:current-turn (+=player 2, t=build/trade)
                                                                    └── 3:robber (5 = tile 5)

... |______-_________-________+______+_**_____-*_____*-_____+_____+__________|31201/02143/10320|2/0/1|10000/01010/00100
     ├──────────────────────────── 7:edges ─────────────────────────────────┤ ├─8:resources ──┤ │     ├ 10:dev-cards ─┤
                                                                                                └── 9: Knights played
```

## Compact Form (Transformer Ingestion)

Remove all `/` and `|` separators. The result is a fixed-length string, one character per token. Each character belongs to one of the disjoint alphabets defined in §Token Alphabets (digits and a few others are shared across categories, disambiguated by position). Since tiles are already concatenated without separators, only `|` (section boundaries) and `/` (player separators in sections 8–10) are stripped.

Compact string lengths:
- **Mini map** (2 players): `132` characters
- **Small map** (2 players): `169` characters
- **Small map** (3 players): `180` characters
- **Standard map** (3 players): `284` characters
- **Standard map** (4 players): `295` characters

## Notes and Constraints

- All indices and ordering are defined in `docs/board-topology.md`.
- Tile numbers are decomposed into pip count + side (3 tokens per tile) rather than a single combined token. This gives the model direct access to the probability dimension (pip count) as a learnable feature.
- Vertex occupancy is decomposed into building type + player owner (2 tokens per vertex). The player ID alphabet is shared across all player-identity fields (current turn, longest road, largest army, vertex owner, edge occupancy), enabling the model to learn a unified player embedding.
- Resource type characters are shared between tiles and ports, since they refer to the same underlying concept.
- Pip count digits and count digits share the same characters (`0`–`5`). These appear at structurally distinct positions, so positional embeddings disambiguate.
