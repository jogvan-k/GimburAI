# State & Action Serialization

This document defines the full-state serialization used by the complete GimburAI
policy/value model. It uses fixed-length, human-readable token strings with a
reversible compact form for transformer ingestion.

| Model | Input | Output | Used during |
|-------|-------|--------|-------------|
| **`catan_policy_value_v1`** | Full game state | Per-player value and complete policy | All decision stages |

## Encoding Overview

- **Human-readable form** (state): single-character tokens drawn from semantically disjoint alphabets (see [Token Alphabets](#token-alphabets) below). `/` separates players within per-player sections; `|` separates major sections. Tiles are concatenated directly without separators.
- **Compact form** (state): strip all `/` and `|` separators to produce a fixed-length string, one character per token.
- **Indexing**: all indices are 0-based and refer to the topology in [topology-reference.md](topology-reference.md).

### Token Alphabets

These alphabets apply to the **state** tokenizer (Parts I and II). Placement actions are never model inputs. Each semantic category uses its own character set so that the state tokenizer can learn category-specific embeddings.

| Category | Characters | Notes |
|----------|-----------|-------|
| Resource type | `d w b s W o` | Desert, wood, brick, sheep, wheat, ore. Shared by tiles and ports. |
| Port generic | `g` | 3:1 generic harbor (extends the resource alphabet for ports) |
| Pip count | `0 1 2 3 4 5` | 0 = desert, 1–5 = pip count. Shares digit chars with count alphabet. |
| Side | `l h n` | Low (below 7), high (above 7), none (desert) |
| Building type | `. v c` | Empty, village (settlement), city. [Game state vertices](#6-vertices) only. |
| Placement number | `. a b p` | Empty, 1st settlement, 2nd settlement, pending settlement awaiting its road. Placement vertices only. |
| Player ID | `- + * ^` | Player 1–4. `_` denotes “none” in vertex/edge/award contexts. |
| Turn stage | `a e f i r x y t` | See [current turn](#4-current-turn) for mapping. |
| Count | `0 1 2 3 4 5 6 7 8 9 A B C D E F G H J K` | Crockford base-32 values 0–19 |

**Shared characters**: pip count (`0`–`5`) and count (`0`–`9`) share digit characters. Placement number `a`/`b` shares characters with turn stage `a` (place 1st settlement) and resource type `b` (brick), but these appear in different serialization formats or at structurally distinct positions — positional embeddings disambiguate. Resource type characters are shared between tiles and ports since they refer to the same underlying concept.

---

# Part I — Game State Serialization

The staged normal-play action graph and full policy vocabulary are specified in
[complete-policy-value-model.md](complete-policy-value-model.md).

*Consumed by the **GimburStateEvaluator** model during normal play (after initial placement is complete).*

## Serialization Layout

Tokens are emitted in the order below. All counts are fixed-length. Major sections are separated by `|`. Within per-player sections (resources, knights, dev cards) individual players are separated by `/`. Tile tokens are concatenated directly without separators. Sections 13 and 14 preserve the exact development deck and terminal winner so the serialized state remains Markov and terminal values are explicit.

Let `T` = number of tiles, `V` = number of vertices, `E` = number of edges, `P` = number of ports, `N` = number of players.

| Parameter | Mini | Small | Standard |
|-----------|------|-------|----------|
| T (tiles) | 7 | 10 | 19 |
| V (vertices) | 24 | 32 | 54 |
| E (edges) | 30 | 41 | 72 |
| P (ports) | 6 | 6 | 9 |
| N (players) | 2 | 2–3 | 3–4 |

### 1) Tiles

For each tile index `t` in order, three tokens:
- `tile[t].resource` — **resource type**:
  - `d` = desert, `w` = wood, `b` = brick, `s` = sheep, `W` = wheat, `o` = ore
- `tile[t].pips` — **pip count**:
  - `0` = desert (no pips), `1`–`5` = number of pips on the token
- `tile[t].side` — **side** (which side of 7 the number falls on):
  - `l` = low (number < 7), `h` = high (number > 7), `n` = none (desert)

Tile numbers are decomposed into pip count + side (3 tokens per tile) rather than a single combined token. This gives the model direct access to the probability dimension (pip count) as a learnable feature.

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

Each port is a coastal edge connecting two vertices on the board perimeter (see [topology-reference.md](topology-reference.md)). Port positions are fixed by the board topology and do not need to be serialized. Only the randomly assigned type is stored.

For each **port index** `p` (0..P-1), in the fixed order defined by the topology:
- `port[p].type` — same **resource type** characters as tiles, plus `g` for generic:
  - `g` = 3:1 generic
  - `w` = wood
  - `b` = brick
  - `s` = sheep
  - `W` = wheat
  - `o` = ore

Standard map: 4 generic (3:1) + 5 resource-specific (2:1) = 9 ports. Small map: 2 generic (3:1) + 4 resource-specific (2:1, no ore) = 6 ports. Mini map: 3 generic (3:1) + 3 resource-specific (2:1) = 6 ports.

**Tokens**: `P` — mini: 6, small: 6, standard: 9.

### 3) Robber

- `robber.tileIndex` — **count** character (Crockford base-32, value 0..18)

**Tokens**: 1.

### 4) Current Turn

Two tokens:
- `currentPlayer` — **player id**: `-` = player 1, `+` = player 2, `*` = player 3, `^` = player 4
- `turnStage` — **turn stage**:
  - `a` = initial placement: place 1st settlement
  - `e` = initial placement: place 1st road
  - `f` = initial placement: place 2nd settlement
  - `i` = initial placement: place 2nd road
  - `r` = pre-roll
  - `x` = choose robber location
  - `y` = choose player to rob from
  - `t` = build/trade

Turn stages `a`, `e`, `f`, `i` apply during initial placement. Turn stages `r`, `x`, `y`, `t` apply during normal play.

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

Vertex occupancy is decomposed into building type + player owner (2 tokens per vertex). The player ID alphabet is shared across all player-identity fields (current turn, longest road, largest army, vertex owner, edge occupancy), enabling the model to learn a unified player embedding.

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

For the **current player**, cards purchased this turn are excluded from these counts — they appear in [section 11](#11-new-dev-cards-this-turn) instead. This lets the model distinguish playable cards from newly purchased ones.

Each player's 5 dev-card tokens are concatenated; players are separated by `/`.

Pip count digits and count digits share the same characters (`0`–`5`). These appear at structurally distinct positions, so positional embeddings disambiguate.

**Tokens**: `5 * N` — mini: 10, small: 10–15, standard: 15–20.

### 11) New Dev Cards This Turn (active player only)

Order: `[knight, roadBuilding, monopoly, yearOfPlenty, victoryPoint]`

- `newDevCards[5]` — **count** (Crockford base-32; typically `0` or `1`)

These are the development cards purchased by the **current player** during their current turn. Per Catan rules, non-VP dev cards bought on the current turn cannot be played until the next turn. Victory-point cards are included so their immediate score contribution is not lost from the serialized state.

This section is not per-player — it always represents the active player's newly purchased cards. During [player rotation](#player-rotation-invariance), these tokens are left unchanged (they are always relative to the current player).

**Tokens**: 5 (fixed, player-count-independent).

### 12) Dev Card Resolution State

Two tokens capturing the state of an in-progress Road Building or Knight dev card play:

- `pendingRoadBuildingPlacements` — **count** (Crockford base-32; `0`=0, `1`=1, `2`=2): how many free road placements the current player still owes after playing a Road Building card. When > 0, the only legal actions are `PlaceRoad`.
- `postDevCardStage` — **turn stage** or `_`: the turn stage to return to after the dev card effect resolves. Set when a Knight or Road Building card is played during `PreRoll` (value `r`) or `BuildTrade` (value `t`). `_` = null (no dev card effect in progress).

This section is not per-player — it always represents the current player's state. During [player rotation](#player-rotation-invariance), these tokens are left unchanged.

**Tokens**: 2 (fixed, player-count-independent).

### 13) Remaining Development Deck

Five count tokens in `DevCardType` order: `[knight, victoryPoint, roadBuilding, monopoly, yearOfPlenty]`. These are the exact cards remaining in the deck and are unchanged by geometric or player rotation.

**Tokens**: 5.

### 14) Winner

One player-id token: `_` while the game is active, otherwise the winning player. Player rotation remaps this token with every other player identifier.

**Tokens**: 1.

## Game State Examples

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
- **New dev cards this turn**: none

Full serialized (sections separated by `|`):
```
w5lb3ls4lW3hd0nW4ho2l|gsgbgw|4|-t|__|._._._._._._v-._._._._._._._v+._._._._._._._._._|_____-_______+________________|21010/00130|0/0|00000/00000|00000|0_
```

### Small Map (2 players, early game)

After initial placement (1 round): each player has placed 1 settlement and 1 road. One turn has begun (player 2 is about to roll).

Board state:
- **Tiles** (10 tiles): wheat/3, brick/4, sheep/5, wood/10, brick/11, wood/12, ore/6, wheat/9, sheep/8, desert/0
- **Ports**: generic, wood, wheat, generic, sheep, brick
- **Robber**: tile 9 (desert)
- **Current turn**: player 2, pre-roll
- **Longest road / largest army**: none / none
- **Vertices**: player 1 settlements on v0 and v7; player 2 settlements on v1 and v2
- **Edges**: player 1 roads on e0 and e6; player 2 roads on e2 and e4
- **Player 1 resources**: wood=1, brick=0, sheep=0, wheat=1, ore=0
- **Player 2 resources**: wood=0, brick=0, sheep=1, wheat=0, ore=0
- **Knights played**: 0 / 0
- **Dev cards**: none / none
- **New dev cards this turn**: none

Full serialized (sections separated by `|`):
```
W2lb3ls4lw3hb2hw1ho5lW4hs5hd0n|gwWgsb|9|+r|__|v-v+v+._._._._v-._._._._._._._._._._._._._._._._._._._._._._._._|-_+_+_-__________________________________|10010/00100|0/0|00000/00000|00000|0_
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
- **New dev cards this turn**: none (player 2 bought no cards this turn)

Full serialized:
```
w4lo1lb5lW2lw5hs3hW4ho1hs2hw3lb5hs3hW4ho3ls4lb5lw2lW2hd0n|ggwgbsWog|5|+t|_-|._._._._._._._._v-._._._._._._._._._c+._._._._._v*._._._._._._v*._._._v-._._._._._._._._v+._._._._._._._._._|______-_________-________+______+_**_____-*_____*-_____+_____+__________|31201/02143/10320|2/0/1|10000/01010/00100|00000|0_
```

## Player Rotation Invariance

The neural network always predicts from **player 1's** perspective. To obtain predictions for an arbitrary target player, the serialized state is **rotated** so that the target player occupies the player-1 slot before inference. The rotation is a cyclic shift of player identities; the board itself (tiles, ports, robber) is unchanged.

### Rotation Definition

Given `N` players and a target player `T` (1-based), the rotation amount is `R = T − 1`. A player with original 1-based index `P` is mapped to new index:

```
new_index = ((P − 1 − R) mod N) + 1
```

This makes the target player `T` become player 1, and all other players shift down accordingly (wrapping around).

**Example** (3-player game, target = player 2, so R = 1):

| Original | New |
|----------|-----|
| Player 1 | Player 3 |
| Player 2 | Player 1 |
| Player 3 | Player 2 |

### What Gets Rotated

The rotation affects two kinds of data in the compact form:

#### 1. Player-ID tokens (character-level remapping)

Each player-ID character is remapped according to the rotation formula. The character `_` (none) is unchanged.

| Char | Meaning | Remapped to |
|------|---------|-------------|
| `_`  | None    | `_` (unchanged) |
| `-`  | Player 1 | new player ID char |
| `+`  | Player 2 | new player ID char |
| `*`  | Player 3 | new player ID char |
| `^`  | Player 4 | new player ID char |

Affected positions:
- **Current player** ([section 4](#4-current-turn)): 1 token at offset `3T + P + 1`.
- **Longest road owner** ([section 5](#5-longest-road--largest-army)): 1 token at offset `3T + P + 3`.
- **Largest army owner** ([section 5](#5-longest-road--largest-army)): 1 token at offset `3T + P + 4`.
- **Vertex owners** ([section 6](#6-vertices)): every 2nd character (the owner field) starting at offset `3T + P + 5` (i.e. positions `3T + P + 5 + 2v + 1` for each vertex `v`).
- **Edge occupancy** ([section 7](#7-edges)): every character starting at offset `3T + P + 5 + 2V`.

#### 2. Per-player data blocks (block-level reordering)

Contiguous blocks of tokens belonging to each player are cyclically shifted so that the target player's block comes first. With rotation `R`, the block originally at player index `i` (0-based) moves to position `(i − R) mod N`.

Affected sections:
- **Resources** ([section 8](#8-per-player-resources-5-per-player)): `N` blocks of 5 tokens each, starting at offset `3T + P + 5 + 2V + E`.
- **Knights played** ([section 9](#9-per-player-knights-played)): `N` blocks of 1 token each.
- **Dev cards** ([section 10](#10-per-player-dev-cards-in-hand-5-per-player)): `N` blocks of 5 tokens each.

**Not affected**: Sections 11-13 are relative/global and remain unchanged. Section 14 winner is remapped as a player ID.

### Rotation Example

Using the mini map game state example (2 players, player 1's turn):

**Original (human-readable):**
```
w5lb3ls4lW3hd0nW4ho2l|gsgbgw|4|-t|__|._._._._._._v-._._._._._._._v+._._._._._._._._._|_____-_______+________________|21010/00130|0/0|00000/00000|00000|0_
```

**Rotated for player 2** (R = 1, N = 2 — player 1 and player 2 swap):
```
w5lb3ls4lW3hd0nW4ho2l|gsgbgw|4|+t|__|._._._._._._v+._._._._._._._v-._._._._._._._._._|_____+_______-________________|00130/21010|0/0|00000/00000|00000|0_
```

Changes:
- Current player: `-` becomes `+`
- Vertex owners: `v-` becomes `v+` and `v+` becomes `v-`
- Edge occupancy: `-` becomes `+` and `+` becomes `-`
- Resources: `21010/00130` reordered to `00130/21010`
- Knights: `0/0` reordered (no visible change since both are `0`)
- Dev cards: `00000/00000` reordered (no visible change)
- New dev cards this turn: `00000` unchanged (always current-player relative)
- Dev card resolution state: `0_` unchanged (always current-player relative)

### Implementation

The rotation is implemented in `python/gimbur_nn/tokenizer.py` as `rotate_player_state()`. The inference server's `/state/predict-player` endpoint applies rotation automatically: callers send the original (unrotated) compact state together with the target player number, and the server handles the rest.

# Token Counts

Compact form: strip all `/` and `|` separators from the human-readable form. The result is a fixed-length string, one character per token.

## Game State

```
(3*T) + P + 1 + 2 + 2 + (2*V) + E + (5*N) + (1*N) + (5*N) + 5 + 2 + 5 + 1
= (3*T + 2*V + E + P + 5) + 11*N + 13
```

The constant 5 = robber(1) + current-turn(2) + awards(2). The trailing 13 = new dev cards (5), resolution state (2), remaining deck (5), and winner (1).

## Summary

| Map | Players | Game State |
|-----|---------|------------|
| Mini (T=7, V=24, E=30, P=6) | 2 | 138 |
| Small (T=10, V=32, E=41, P=6) | 2 | 174 |
| Small (T=10, V=32, E=41, P=6) | 3 | 185 |
| Standard (T=19, V=54, E=72, P=9) | 2 | 279 |
| Standard (T=19, V=54, E=72, P=9) | 3 | 290 |
| Standard (T=19, V=54, E=72, P=9) | 4 | 301 |
