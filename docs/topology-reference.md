# Topology Reference

This document defines the coordinate system, element identities, and adjacency
relationships for hex-based Catan board layouts. It is the canonical reference
for the GimburAI engine. For the serialization format, see
[state-action-serialization.md](state-action-serialization.md).

---

## 1. Coordinate System

### 1.1 Tile Coordinates (Axial)

Tiles use **axial coordinates** `(q, r)` with the center tile at `(0, 0)`.

**Circular boards**: A hex grid of radius `R` contains all tiles where
`|q| <= R`, `|r| <= R`, and `|q + r| <= R`. The Mini (radius 1) and Standard
(radius 2) maps use this layout.

**Non-circular boards**: Some maps are defined by an explicit set of tile
coordinates rather than a radius formula. The Small map, for example, uses two
central hexes `(0,0)` and `(1,0)` with one layer of hexes around them,
forming a 10-tile oval shape. All other topology rules (vertex triplets, edge
identity, adjacency derivation) apply identically to non-circular boards.

The 6 axial neighbor directions, clockwise from east:

| Direction | dq | dr |
| --- | --- | --- |
| d0 (E)  | +1 |  0 |
| d1 (SE) | +1 | -1 |
| d2 (SW) |  0 | -1 |
| d3 (W)  | -1 |  0 |
| d4 (NW) | -1 | +1 |
| d5 (NE) |  0 | +1 |

Tile `(q, r)` has neighbor in direction `i` at `(q + di.q, r + di.r)`.
Neighbors outside the board radius are **virtual tiles** -- they don't exist on
the board but are used to define boundary vertex identity.

### 1.2 Hex Orientation and Pixel Mapping

**Pointy-top** hexagons viewed top-down. Pixel position:

```
x = size * sqrt(3) * (q + r / 2)
y = -size * 1.5 * r
```

**Tile index**: Tiles are sorted by screen position -- ascending `y` (top-to-
bottom), then ascending `x` (left-to-right).

Corner angles: `60 * i - 30` degrees (i = 0..5).

### 1.3 Vertex Identity (Triplet)

Every vertex sits at the meeting point of exactly 3 hex cells (some of which may
be virtual tiles off the board). A vertex is **uniquely identified** by the
sorted triplet of axial coordinates of those 3 cells.

For a tile at `(q, r)`, corner `i` (0..5) is shared with neighbors in
directions `di` and `d(i-1 mod 6)`:

```
corner_i(q, r) = sorted( (q, r),  (q, r) + d[i],  (q, r) + d[(i-1) % 6] )
```

| Corner | Angle | Shared with directions |
| --- | --- | --- |
| 0 | 30 deg  | d0 (E), d5 (NE) |
| 1 | 90 deg  | d1 (SE), d0 (E) |
| 2 | 150 deg | d2 (SW), d1 (SE) |
| 3 | 210 deg | d3 (W), d2 (SW) |
| 4 | 270 deg | d4 (NW), d3 (W) |
| 5 | 330 deg | d5 (NE), d4 (NW) |

**Vertex index**: Vertices are sorted by screen position of their pixel
coordinates (ascending `y`, then ascending `x`). The canonical indices preserve
the computed IEEE 754 double ordering used by `BoardTopology`; mathematically
equal rows are not merged with an epsilon. This matters for compatibility with
existing serialized states and checkpoints.

### 1.4 Edge Identity

An edge connects two adjacent vertices on a hex boundary. It is uniquely
identified by the **sorted pair of vertex triplets** of its two endpoints.

**Edge index**: Endpoint indices are first normalized so
`vertex_A_index < vertex_B_index`, then edges are sorted by
`(vertex_A_index, vertex_B_index)`.

### 1.5 Port Identity

A port is a fixed position on the board perimeter where a player with a
settlement or city on one of its two vertices can trade at a favorable rate.
Each port is a **coastal edge** — an edge on the board perimeter connecting
two vertices.

Port positions are determined by walking the ring of coastal edges clockwise
from the top of the board and selecting evenly-spaced edges. For circular hex
boards of radius `R`, the port count is `3 × (R + 1)`. For non-circular boards,
the port count is specified explicitly when constructing the topology.

**Port index**: Ports are numbered 0..N-1 in clockwise order from the top.

Port **type** (3:1 generic, or 2:1 for a specific resource) is assigned during
game setup and is part of the game state, not the topology.

### 1.6 Summary of Identity

| Element | Identity | Index ordering |
| --- | --- | --- |
| Tile | axial `(q, r)` | screen position (y asc, x asc) |
| Vertex | sorted triplet of 3 tile `(q, r)` | screen position (y asc, x asc) |
| Edge | sorted pair of vertex triplets | `(min vertex index, max vertex index)` |
| Port | pair of vertex indices (coastal edge) | clockwise from top |

---

## 2. How Topology Defines a Game State

The board topology is a graph `G = (V, E)` where `V` is the set of vertices
and `E` is the set of edges, embedded in the plane with hex faces. The game
state is fully determined by assigning values to every element of this graph
plus some global/per-player scalars:

| What | Attached to | Values |
| --- | --- | --- |
| Resource type | each tile | desert, wood, brick, sheep, wheat, ore |
| Number token | each tile | 0 (desert), 2-6, 8-12 |
| Robber location | 1 tile index | 0..N-1 |
| Settlement/city | each vertex | empty, settlement(player), city(player) |
| Road | each edge | empty, road(player) |
| Port type | each port | 3:1, wood, brick, sheep, wheat, ore |
| Current player | global | 0..P-1 |
| Turn stage | global | placement / pre-roll / robber / build |
| Longest road owner | global | 0..P-1 |
| Largest army owner | global | 0..P-1 |
| Resources in hand | per player (5 types) | 0..99 |
| Knights played | per player | 0..99 |
| Dev cards in hand | per player (5 types) | 0..99 |

Because every tile, vertex, and edge has a **fixed index**, the entire state can
be serialized as a fixed-length vector of tokens (see
[state-action-serialization.md](state-action-serialization.md)). Two states are identical if
and only if their serialized vectors are identical.

### What is NOT stored (derivable)

- Remaining buildable pieces (settlements, cities, roads) -- derivable from
  placed pieces.
- Victory point totals -- derivable from vertex occupancy, dev cards, longest
  road, largest army.
- Available development card pool -- derivable from total cards minus all
  players' hands and played knights.

---

## 3. Computing Adjacency from First Principles

All adjacency relationships are derivable from two primitives:

1. **Vertex triplet**: for each vertex, the sorted set of 3 tile `(q, r)`.
2. **Edge endpoints**: for each edge, its two vertex indices.

### 3.1 Vertex -> Tiles

```
tiles_of(vertex) = { t in triplet(vertex) | t is on the board }
```

### 3.2 Tile -> Vertices

```
vertices_of(tile) = { v | tile's (q,r) in triplet(v) }
```

### 3.3 Edge -> Tiles

```
tiles_of(edge) = { t | t in triplet(endpointA) AND t in triplet(endpointB) AND t is on board }
```

### 3.4 Tile -> Edges

```
edges_of(tile) = { e | tile in tiles_of(e) }
```

### 3.5 Vertex -> Edges

```
edges_of(vertex) = { e | vertex in endpoints(e) }
```

### 3.6 Vertex -> Adjacent Vertices

```
neighbors(vertex) = { other_endpoint(e, vertex) | e in edges_of(vertex) }
```

### 3.7 Tile -> Adjacent Tiles

```
neighbors(tile) = { t2 | exists edge e where tiles_of(e) = {tile, t2} }
```

### 3.8 Coastal Test

```
is_coastal(edge) = |tiles_of(edge)| == 1
```

---

## 4. Mini Map (Radius 1)

![Mini Map (Radius 1) topology indices](mini-board-topology.svg)

### 4.1 Counts

| Element | Count |
| --- | --- |
| Tiles | 7 |
| Vertices | 24 |
| Edges | 30 |
| Ports | 6 |
| Coastal edges | 18 |
| Interior edges | 12 |
| Boundary vertices (degree 2) | 12 |
| Interior vertices (degree 3) | 12 |

Vertex tile-count distribution: 12 touch 1 tile, 6 touch 2 tiles, 6 touch 3 tiles.

Tile rows (top-to-bottom): 2, 3, 2.

### 4.2 Tile Table

| Tile | q | r |
| --- | --- | --- |
| 0 | -1 | +1 |
| 1 | +0 | +1 |
| 2 | -1 | +0 |
| 3 | +0 | +0 |
| 4 | +1 | +0 |
| 5 | +0 | -1 |
| 6 | +1 | -1 |

### 4.3 Vertex Table

| Vertex | Triplet (tile axial coords) |
| --- | --- |
| 0 | (-2, +2) (-1, +1) (-1, +2) |
| 1 | (-1, +2) (+0, +1) (+0, +2) |
| 2 | (-2, +1) (-2, +2) (-1, +1) |
| 3 | (-1, +1) (-1, +2) (+0, +1) |
| 4 | (+0, +1) (+0, +2) (+1, +1) |
| 5 | (-2, +1) (-1, +0) (-1, +1) |
| 6 | (-1, +1) (+0, +0) (+0, +1) |
| 7 | (+0, +1) (+1, +0) (+1, +1) |
| 8 | (-2, +0) (-2, +1) (-1, +0) |
| 9 | (-1, +0) (-1, +1) (+0, +0) |
| 10 | (+0, +0) (+0, +1) (+1, +0) |
| 11 | (+1, +0) (+1, +1) (+2, +0) |
| 12 | (-2, +0) (-1, -1) (-1, +0) |
| 13 | (+1, +0) (+2, -1) (+2, +0) |
| 14 | (-1, +0) (+0, -1) (+0, +0) |
| 15 | (+0, +0) (+1, -1) (+1, +0) |
| 16 | (-1, -1) (-1, +0) (+0, -1) |
| 17 | (+0, -1) (+0, +0) (+1, -1) |
| 18 | (+1, -1) (+1, +0) (+2, -1) |
| 19 | (-1, -1) (+0, -2) (+0, -1) |
| 20 | (+0, -1) (+1, -2) (+1, -1) |
| 21 | (+1, -1) (+2, -2) (+2, -1) |
| 22 | (+0, -2) (+0, -1) (+1, -2) |
| 23 | (+1, -2) (+1, -1) (+2, -2) |

### 4.4 Edge Table

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 2 |
| 1 | 0 | 3 |
| 2 | 1 | 3 |
| 3 | 1 | 4 |
| 4 | 2 | 5 |
| 5 | 3 | 6 |
| 6 | 4 | 7 |
| 7 | 5 | 8 |
| 8 | 5 | 9 |
| 9 | 6 | 9 |
| 10 | 6 | 10 |
| 11 | 7 | 10 |
| 12 | 7 | 11 |
| 13 | 8 | 12 |
| 14 | 9 | 14 |
| 15 | 10 | 15 |
| 16 | 11 | 13 |
| 17 | 12 | 16 |
| 18 | 13 | 18 |
| 19 | 14 | 16 |
| 20 | 14 | 17 |
| 21 | 15 | 17 |
| 22 | 15 | 18 |
| 23 | 16 | 19 |
| 24 | 17 | 20 |
| 25 | 18 | 21 |
| 26 | 19 | 22 |
| 27 | 20 | 22 |
| 28 | 20 | 23 |
| 29 | 21 | 23 |

### 4.5 Coastal Edges (18 total)

`0, 1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 18, 23, 25, 26, 27, 28, 29`

### 4.6 Port Table (6 ports)

Ports are ordered clockwise from the top of the board.

| Port | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 3 | 1 |
| 1 | 7 | 11 |
| 2 | 18 | 21 |
| 3 | 20 | 22 |
| 4 | 16 | 12 |
| 5 | 5 | 2 |

### 4.7 Tile -> Vertices

| Tile | Vertices |
| --- | --- |
| 0 | 0, 2, 3, 5, 6, 9 |
| 1 | 1, 3, 4, 6, 7, 10 |
| 2 | 5, 8, 9, 12, 14, 16 |
| 3 | 6, 9, 10, 14, 15, 17 |
| 4 | 7, 10, 11, 13, 15, 18 |
| 5 | 14, 16, 17, 19, 20, 22 |
| 6 | 15, 17, 18, 20, 21, 23 |

### 4.8 Tile -> Edges

| Tile | Edges |
| --- | --- |
| 0 | 0, 1, 4, 5, 8, 9 |
| 1 | 2, 3, 5, 6, 10, 11 |
| 2 | 7, 8, 13, 14, 17, 19 |
| 3 | 9, 10, 14, 15, 20, 21 |
| 4 | 11, 12, 15, 16, 18, 22 |
| 5 | 19, 20, 23, 24, 26, 27 |
| 6 | 21, 22, 24, 25, 28, 29 |

### 4.9 Tile -> Adjacent Tiles

| Tile | Neighbors |
| --- | --- |
| 0 | 1, 2, 3 |
| 1 | 0, 3, 4 |
| 2 | 0, 3, 5 |
| 3 | 0, 1, 2, 4, 5, 6 |
| 4 | 1, 3, 6 |
| 5 | 2, 3, 6 |
| 6 | 3, 4, 5 |

### 4.10 Vertex -> Tiles

| Vertex | Tiles |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 0 |
| 3 | 0, 1 |
| 4 | 1 |
| 5 | 0, 2 |
| 6 | 0, 1, 3 |
| 7 | 1, 4 |
| 8 | 2 |
| 9 | 0, 2, 3 |
| 10 | 1, 3, 4 |
| 11 | 4 |
| 12 | 2 |
| 13 | 4 |
| 14 | 2, 3, 5 |
| 15 | 3, 4, 6 |
| 16 | 2, 5 |
| 17 | 3, 5, 6 |
| 18 | 4, 6 |
| 19 | 5 |
| 20 | 5, 6 |
| 21 | 6 |
| 22 | 5 |
| 23 | 6 |

### 4.11 Vertex -> Edges

| Vertex | Edges |
| --- | --- |
| 0 | 0, 1 |
| 1 | 2, 3 |
| 2 | 0, 4 |
| 3 | 1, 2, 5 |
| 4 | 3, 6 |
| 5 | 4, 7, 8 |
| 6 | 5, 9, 10 |
| 7 | 6, 11, 12 |
| 8 | 7, 13 |
| 9 | 8, 9, 14 |
| 10 | 10, 11, 15 |
| 11 | 12, 16 |
| 12 | 13, 17 |
| 13 | 16, 18 |
| 14 | 14, 19, 20 |
| 15 | 15, 21, 22 |
| 16 | 17, 19, 23 |
| 17 | 20, 21, 24 |
| 18 | 18, 22, 25 |
| 19 | 23, 26 |
| 20 | 24, 27, 28 |
| 21 | 25, 29 |
| 22 | 26, 27 |
| 23 | 28, 29 |

### 4.12 Vertex -> Adjacent Vertices

| Vertex | Neighbors |
| --- | --- |
| 0 | 2, 3 |
| 1 | 3, 4 |
| 2 | 0, 5 |
| 3 | 0, 1, 6 |
| 4 | 1, 7 |
| 5 | 2, 8, 9 |
| 6 | 3, 9, 10 |
| 7 | 4, 10, 11 |
| 8 | 5, 12 |
| 9 | 5, 6, 14 |
| 10 | 6, 7, 15 |
| 11 | 7, 13 |
| 12 | 8, 16 |
| 13 | 11, 18 |
| 14 | 9, 16, 17 |
| 15 | 10, 17, 18 |
| 16 | 12, 14, 19 |
| 17 | 14, 15, 20 |
| 18 | 13, 15, 21 |
| 19 | 16, 22 |
| 20 | 17, 22, 23 |
| 21 | 18, 23 |
| 22 | 19, 20 |
| 23 | 20, 21 |

### 4.13 Edge -> Vertices

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 2 |
| 1 | 0 | 3 |
| 2 | 1 | 3 |
| 3 | 1 | 4 |
| 4 | 2 | 5 |
| 5 | 3 | 6 |
| 6 | 4 | 7 |
| 7 | 5 | 8 |
| 8 | 5 | 9 |
| 9 | 6 | 9 |
| 10 | 6 | 10 |
| 11 | 7 | 10 |
| 12 | 7 | 11 |
| 13 | 8 | 12 |
| 14 | 9 | 14 |
| 15 | 10 | 15 |
| 16 | 11 | 13 |
| 17 | 12 | 16 |
| 18 | 13 | 18 |
| 19 | 14 | 16 |
| 20 | 14 | 17 |
| 21 | 15 | 17 |
| 22 | 15 | 18 |
| 23 | 16 | 19 |
| 24 | 17 | 20 |
| 25 | 18 | 21 |
| 26 | 19 | 22 |
| 27 | 20 | 22 |
| 28 | 20 | 23 |
| 29 | 21 | 23 |

### 4.14 Edge -> Tiles

| Edge | Tiles |
| --- | --- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 0 |
| 5 | 0, 1 |
| 6 | 1 |
| 7 | 2 |
| 8 | 0, 2 |
| 9 | 0, 3 |
| 10 | 1, 3 |
| 11 | 1, 4 |
| 12 | 4 |
| 13 | 2 |
| 14 | 2, 3 |
| 15 | 3, 4 |
| 16 | 4 |
| 17 | 2 |
| 18 | 4 |
| 19 | 2, 5 |
| 20 | 3, 5 |
| 21 | 3, 6 |
| 22 | 4, 6 |
| 23 | 5 |
| 24 | 5, 6 |
| 25 | 6 |
| 26 | 5 |
| 27 | 5 |
| 28 | 6 |
| 29 | 6 |

### 4.15 Action Table (60 actions)

Each action represents a settlement vertex plus road direction. Entries are
sorted by vertex index, then direction string.

| Token | Action | Vertex | Direction | Edge |
| --- | --- | --- | --- | --- |
| 0 | 0SE | 0 | SE | 1 |
| 1 | 0SW | 0 | SW | 0 |
| 2 | 1SE | 1 | SE | 3 |
| 3 | 1SW | 1 | SW | 2 |
| 4 | 2NE | 2 | NE | 0 |
| 5 | 2S | 2 | S | 4 |
| 6 | 3NE | 3 | NE | 2 |
| 7 | 3NW | 3 | NW | 1 |
| 8 | 3S | 3 | S | 5 |
| 9 | 4NW | 4 | NW | 3 |
| 10 | 4S | 4 | S | 6 |
| 11 | 5N | 5 | N | 4 |
| 12 | 5SE | 5 | SE | 8 |
| 13 | 5SW | 5 | SW | 7 |
| 14 | 6N | 6 | N | 5 |
| 15 | 6SE | 6 | SE | 10 |
| 16 | 6SW | 6 | SW | 9 |
| 17 | 7N | 7 | N | 6 |
| 18 | 7SE | 7 | SE | 12 |
| 19 | 7SW | 7 | SW | 11 |
| 20 | 8NE | 8 | NE | 7 |
| 21 | 8S | 8 | S | 13 |
| 22 | 9NE | 9 | NE | 9 |
| 23 | 9NW | 9 | NW | 8 |
| 24 | 9S | 9 | S | 14 |
| 25 | 10NE | 10 | NE | 11 |
| 26 | 10NW | 10 | NW | 10 |
| 27 | 10S | 10 | S | 15 |
| 28 | 11NW | 11 | NW | 12 |
| 29 | 11S | 11 | S | 16 |
| 30 | 12N | 12 | N | 13 |
| 31 | 12SE | 12 | SE | 17 |
| 32 | 13N | 13 | N | 16 |
| 33 | 13SW | 13 | SW | 18 |
| 34 | 14N | 14 | N | 14 |
| 35 | 14SE | 14 | SE | 20 |
| 36 | 14SW | 14 | SW | 19 |
| 37 | 15N | 15 | N | 15 |
| 38 | 15SE | 15 | SE | 22 |
| 39 | 15SW | 15 | SW | 21 |
| 40 | 16NE | 16 | NE | 19 |
| 41 | 16NW | 16 | NW | 17 |
| 42 | 16S | 16 | S | 23 |
| 43 | 17NE | 17 | NE | 21 |
| 44 | 17NW | 17 | NW | 20 |
| 45 | 17S | 17 | S | 24 |
| 46 | 18NE | 18 | NE | 18 |
| 47 | 18NW | 18 | NW | 22 |
| 48 | 18S | 18 | S | 25 |
| 49 | 19N | 19 | N | 23 |
| 50 | 19SE | 19 | SE | 26 |
| 51 | 20N | 20 | N | 24 |
| 52 | 20SE | 20 | SE | 28 |
| 53 | 20SW | 20 | SW | 27 |
| 54 | 21N | 21 | N | 25 |
| 55 | 21SW | 21 | SW | 29 |
| 56 | 22NE | 22 | NE | 27 |
| 57 | 22NW | 22 | NW | 26 |
| 58 | 23NE | 23 | NE | 29 |
| 59 | 23NW | 23 | NW | 28 |

---

## 5. Small Map (10 tiles, non-circular)

![Small Map (10 tiles, non-circular) topology indices](small-board-topology.svg)

The Small map is a non-circular oval board built from two central hexes
`(0,0)` and `(1,0)` plus one layer of hexes around them.

### 5.1 Counts

| Element | Count |
| --- | --- |
| Tiles | 10 |
| Vertices | 32 |
| Edges | 41 |
| Ports | 6 |
| Coastal edges | 22 |
| Interior edges | 19 |
| Boundary vertices (degree 2) | 14 |
| Interior vertices (degree 3) | 18 |

Vertex tile-count distribution: 14 touch 1 tile, 8 touch 2 tiles, 10 touch 3 tiles.

Tile rows (top-to-bottom): 3, 4, 3.

### 5.2 Tile Table

| Tile | q | r |
| --- | --- | --- |
| 0 | -1 | +1 |
| 1 | +0 | +1 |
| 2 | +1 | +1 |
| 3 | -1 | +0 |
| 4 | +0 | +0 |
| 5 | +1 | +0 |
| 6 | +2 | +0 |
| 7 | +0 | -1 |
| 8 | +1 | -1 |
| 9 | +2 | -1 |

### 5.3 Vertex Table

| Vertex | Triplet (tile axial coords) |
| --- | --- |
| 0 | (-2, +2) (-1, +1) (-1, +2) |
| 1 | (-1, +2) (+0, +1) (+0, +2) |
| 2 | (+0, +2) (+1, +1) (+1, +2) |
| 3 | (-2, +1) (-2, +2) (-1, +1) |
| 4 | (-1, +1) (-1, +2) (+0, +1) |
| 5 | (+0, +1) (+0, +2) (+1, +1) |
| 6 | (+1, +1) (+1, +2) (+2, +1) |
| 7 | (-2, +1) (-1, +0) (-1, +1) |
| 8 | (-1, +1) (+0, +0) (+0, +1) |
| 9 | (+0, +1) (+1, +0) (+1, +1) |
| 10 | (+1, +1) (+2, +0) (+2, +1) |
| 11 | (-2, +0) (-2, +1) (-1, +0) |
| 12 | (-1, +0) (-1, +1) (+0, +0) |
| 13 | (+0, +0) (+0, +1) (+1, +0) |
| 14 | (+1, +0) (+1, +1) (+2, +0) |
| 15 | (+2, +0) (+2, +1) (+3, +0) |
| 16 | (-2, +0) (-1, -1) (-1, +0) |
| 17 | (-1, +0) (+0, -1) (+0, +0) |
| 18 | (+0, +0) (+1, -1) (+1, +0) |
| 19 | (+1, +0) (+2, -1) (+2, +0) |
| 20 | (+2, +0) (+3, -1) (+3, +0) |
| 21 | (-1, -1) (-1, +0) (+0, -1) |
| 22 | (+0, -1) (+0, +0) (+1, -1) |
| 23 | (+1, -1) (+1, +0) (+2, -1) |
| 24 | (+2, -1) (+2, +0) (+3, -1) |
| 25 | (-1, -1) (+0, -2) (+0, -1) |
| 26 | (+0, -1) (+1, -2) (+1, -1) |
| 27 | (+1, -1) (+2, -2) (+2, -1) |
| 28 | (+2, -1) (+3, -2) (+3, -1) |
| 29 | (+0, -2) (+0, -1) (+1, -2) |
| 30 | (+1, -2) (+1, -1) (+2, -2) |
| 31 | (+2, -2) (+2, -1) (+3, -2) |

### 5.4 Edge Table

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 3 |
| 1 | 0 | 4 |
| 2 | 1 | 4 |
| 3 | 1 | 5 |
| 4 | 2 | 5 |
| 5 | 2 | 6 |
| 6 | 3 | 7 |
| 7 | 4 | 8 |
| 8 | 5 | 9 |
| 9 | 6 | 10 |
| 10 | 7 | 11 |
| 11 | 7 | 12 |
| 12 | 8 | 12 |
| 13 | 8 | 13 |
| 14 | 9 | 13 |
| 15 | 9 | 14 |
| 16 | 10 | 14 |
| 17 | 10 | 15 |
| 18 | 11 | 16 |
| 19 | 12 | 17 |
| 20 | 13 | 18 |
| 21 | 14 | 19 |
| 22 | 15 | 20 |
| 23 | 16 | 21 |
| 24 | 17 | 21 |
| 25 | 17 | 22 |
| 26 | 18 | 22 |
| 27 | 18 | 23 |
| 28 | 19 | 23 |
| 29 | 19 | 24 |
| 30 | 20 | 24 |
| 31 | 21 | 25 |
| 32 | 22 | 26 |
| 33 | 23 | 27 |
| 34 | 24 | 28 |
| 35 | 25 | 29 |
| 36 | 26 | 29 |
| 37 | 26 | 30 |
| 38 | 27 | 30 |
| 39 | 27 | 31 |
| 40 | 28 | 31 |

### 5.5 Coastal Edges (22 total)

`0, 1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 22, 23, 30, 31, 34, 35, 36, 37, 38, 39, 40`

### 5.6 Port Table (6 ports)

Ports are ordered clockwise from the top of the board.

| Port | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 4 | 1 |
| 1 | 2 | 6 |
| 2 | 20 | 24 |
| 3 | 27 | 30 |
| 4 | 29 | 25 |
| 5 | 11 | 7 |

### 5.7 Tile -> Vertices

| Tile | Vertices |
| --- | --- |
| 0 | 0, 3, 4, 7, 8, 12 |
| 1 | 1, 4, 5, 8, 9, 13 |
| 2 | 2, 5, 6, 9, 10, 14 |
| 3 | 7, 11, 12, 16, 17, 21 |
| 4 | 8, 12, 13, 17, 18, 22 |
| 5 | 9, 13, 14, 18, 19, 23 |
| 6 | 10, 14, 15, 19, 20, 24 |
| 7 | 17, 21, 22, 25, 26, 29 |
| 8 | 18, 22, 23, 26, 27, 30 |
| 9 | 19, 23, 24, 27, 28, 31 |

### 5.8 Tile -> Edges

| Tile | Edges |
| --- | --- |
| 0 | 0, 1, 6, 7, 11, 12 |
| 1 | 2, 3, 7, 8, 13, 14 |
| 2 | 4, 5, 8, 9, 15, 16 |
| 3 | 10, 11, 18, 19, 23, 24 |
| 4 | 12, 13, 19, 20, 25, 26 |
| 5 | 14, 15, 20, 21, 27, 28 |
| 6 | 16, 17, 21, 22, 29, 30 |
| 7 | 24, 25, 31, 32, 35, 36 |
| 8 | 26, 27, 32, 33, 37, 38 |
| 9 | 28, 29, 33, 34, 39, 40 |

### 5.9 Tile -> Adjacent Tiles

| Tile | Neighbors |
| --- | --- |
| 0 | 1, 3, 4 |
| 1 | 0, 2, 4, 5 |
| 2 | 1, 5, 6 |
| 3 | 0, 4, 7 |
| 4 | 0, 1, 3, 5, 7, 8 |
| 5 | 1, 2, 4, 6, 8, 9 |
| 6 | 2, 5, 9 |
| 7 | 3, 4, 8 |
| 8 | 4, 5, 7, 9 |
| 9 | 5, 6, 8 |

### 5.10 Vertex -> Tiles

| Vertex | Tiles |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 0 |
| 4 | 0, 1 |
| 5 | 1, 2 |
| 6 | 2 |
| 7 | 0, 3 |
| 8 | 0, 1, 4 |
| 9 | 1, 2, 5 |
| 10 | 2, 6 |
| 11 | 3 |
| 12 | 0, 3, 4 |
| 13 | 1, 4, 5 |
| 14 | 2, 5, 6 |
| 15 | 6 |
| 16 | 3 |
| 17 | 3, 4, 7 |
| 18 | 4, 5, 8 |
| 19 | 5, 6, 9 |
| 20 | 6 |
| 21 | 3, 7 |
| 22 | 4, 7, 8 |
| 23 | 5, 8, 9 |
| 24 | 6, 9 |
| 25 | 7 |
| 26 | 7, 8 |
| 27 | 8, 9 |
| 28 | 9 |
| 29 | 7 |
| 30 | 8 |
| 31 | 9 |

### 5.11 Vertex -> Edges

| Vertex | Edges |
| --- | --- |
| 0 | 0, 1 |
| 1 | 2, 3 |
| 2 | 4, 5 |
| 3 | 0, 6 |
| 4 | 1, 2, 7 |
| 5 | 3, 4, 8 |
| 6 | 5, 9 |
| 7 | 6, 10, 11 |
| 8 | 7, 12, 13 |
| 9 | 8, 14, 15 |
| 10 | 9, 16, 17 |
| 11 | 10, 18 |
| 12 | 11, 12, 19 |
| 13 | 13, 14, 20 |
| 14 | 15, 16, 21 |
| 15 | 17, 22 |
| 16 | 18, 23 |
| 17 | 19, 24, 25 |
| 18 | 20, 26, 27 |
| 19 | 21, 28, 29 |
| 20 | 22, 30 |
| 21 | 23, 24, 31 |
| 22 | 25, 26, 32 |
| 23 | 27, 28, 33 |
| 24 | 29, 30, 34 |
| 25 | 31, 35 |
| 26 | 32, 36, 37 |
| 27 | 33, 38, 39 |
| 28 | 34, 40 |
| 29 | 35, 36 |
| 30 | 37, 38 |
| 31 | 39, 40 |

### 5.12 Vertex -> Adjacent Vertices

| Vertex | Neighbors |
| --- | --- |
| 0 | 3, 4 |
| 1 | 4, 5 |
| 2 | 5, 6 |
| 3 | 0, 7 |
| 4 | 0, 1, 8 |
| 5 | 1, 2, 9 |
| 6 | 2, 10 |
| 7 | 3, 11, 12 |
| 8 | 4, 12, 13 |
| 9 | 5, 13, 14 |
| 10 | 6, 14, 15 |
| 11 | 7, 16 |
| 12 | 7, 8, 17 |
| 13 | 8, 9, 18 |
| 14 | 9, 10, 19 |
| 15 | 10, 20 |
| 16 | 11, 21 |
| 17 | 12, 21, 22 |
| 18 | 13, 22, 23 |
| 19 | 14, 23, 24 |
| 20 | 15, 24 |
| 21 | 16, 17, 25 |
| 22 | 17, 18, 26 |
| 23 | 18, 19, 27 |
| 24 | 19, 20, 28 |
| 25 | 21, 29 |
| 26 | 22, 29, 30 |
| 27 | 23, 30, 31 |
| 28 | 24, 31 |
| 29 | 25, 26 |
| 30 | 26, 27 |
| 31 | 27, 28 |

### 5.13 Edge -> Vertices

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 3 |
| 1 | 0 | 4 |
| 2 | 1 | 4 |
| 3 | 1 | 5 |
| 4 | 2 | 5 |
| 5 | 2 | 6 |
| 6 | 3 | 7 |
| 7 | 4 | 8 |
| 8 | 5 | 9 |
| 9 | 6 | 10 |
| 10 | 7 | 11 |
| 11 | 7 | 12 |
| 12 | 8 | 12 |
| 13 | 8 | 13 |
| 14 | 9 | 13 |
| 15 | 9 | 14 |
| 16 | 10 | 14 |
| 17 | 10 | 15 |
| 18 | 11 | 16 |
| 19 | 12 | 17 |
| 20 | 13 | 18 |
| 21 | 14 | 19 |
| 22 | 15 | 20 |
| 23 | 16 | 21 |
| 24 | 17 | 21 |
| 25 | 17 | 22 |
| 26 | 18 | 22 |
| 27 | 18 | 23 |
| 28 | 19 | 23 |
| 29 | 19 | 24 |
| 30 | 20 | 24 |
| 31 | 21 | 25 |
| 32 | 22 | 26 |
| 33 | 23 | 27 |
| 34 | 24 | 28 |
| 35 | 25 | 29 |
| 36 | 26 | 29 |
| 37 | 26 | 30 |
| 38 | 27 | 30 |
| 39 | 27 | 31 |
| 40 | 28 | 31 |

### 5.14 Edge -> Tiles

| Edge | Tiles |
| --- | --- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 2 |
| 5 | 2 |
| 6 | 0 |
| 7 | 0, 1 |
| 8 | 1, 2 |
| 9 | 2 |
| 10 | 3 |
| 11 | 0, 3 |
| 12 | 0, 4 |
| 13 | 1, 4 |
| 14 | 1, 5 |
| 15 | 2, 5 |
| 16 | 2, 6 |
| 17 | 6 |
| 18 | 3 |
| 19 | 3, 4 |
| 20 | 4, 5 |
| 21 | 5, 6 |
| 22 | 6 |
| 23 | 3 |
| 24 | 3, 7 |
| 25 | 4, 7 |
| 26 | 4, 8 |
| 27 | 5, 8 |
| 28 | 5, 9 |
| 29 | 6, 9 |
| 30 | 6 |
| 31 | 7 |
| 32 | 7, 8 |
| 33 | 8, 9 |
| 34 | 9 |
| 35 | 7 |
| 36 | 7 |
| 37 | 8 |
| 38 | 8 |
| 39 | 9 |
| 40 | 9 |

### 5.15 Action Table (82 actions)

Each action represents a settlement vertex plus road direction. Entries are
sorted by vertex index, then direction string.

| Token | Action | Vertex | Direction | Edge |
| --- | --- | --- | --- | --- |
| 0 | 0SE | 0 | SE | 1 |
| 1 | 0SW | 0 | SW | 0 |
| 2 | 1SE | 1 | SE | 3 |
| 3 | 1SW | 1 | SW | 2 |
| 4 | 2SE | 2 | SE | 5 |
| 5 | 2SW | 2 | SW | 4 |
| 6 | 3NE | 3 | NE | 0 |
| 7 | 3S | 3 | S | 6 |
| 8 | 4NE | 4 | NE | 2 |
| 9 | 4NW | 4 | NW | 1 |
| 10 | 4S | 4 | S | 7 |
| 11 | 5NE | 5 | NE | 4 |
| 12 | 5NW | 5 | NW | 3 |
| 13 | 5S | 5 | S | 8 |
| 14 | 6NW | 6 | NW | 5 |
| 15 | 6S | 6 | S | 9 |
| 16 | 7N | 7 | N | 6 |
| 17 | 7SE | 7 | SE | 11 |
| 18 | 7SW | 7 | SW | 10 |
| 19 | 8N | 8 | N | 7 |
| 20 | 8SE | 8 | SE | 13 |
| 21 | 8SW | 8 | SW | 12 |
| 22 | 9N | 9 | N | 8 |
| 23 | 9SE | 9 | SE | 15 |
| 24 | 9SW | 9 | SW | 14 |
| 25 | 10N | 10 | N | 9 |
| 26 | 10SE | 10 | SE | 17 |
| 27 | 10SW | 10 | SW | 16 |
| 28 | 11NE | 11 | NE | 10 |
| 29 | 11S | 11 | S | 18 |
| 30 | 12NE | 12 | NE | 12 |
| 31 | 12NW | 12 | NW | 11 |
| 32 | 12S | 12 | S | 19 |
| 33 | 13NE | 13 | NE | 14 |
| 34 | 13NW | 13 | NW | 13 |
| 35 | 13S | 13 | S | 20 |
| 36 | 14NE | 14 | NE | 16 |
| 37 | 14NW | 14 | NW | 15 |
| 38 | 14S | 14 | S | 21 |
| 39 | 15NW | 15 | NW | 17 |
| 40 | 15S | 15 | S | 22 |
| 41 | 16N | 16 | N | 18 |
| 42 | 16SE | 16 | SE | 23 |
| 43 | 17N | 17 | N | 19 |
| 44 | 17SE | 17 | SE | 25 |
| 45 | 17SW | 17 | SW | 24 |
| 46 | 18N | 18 | N | 20 |
| 47 | 18SE | 18 | SE | 27 |
| 48 | 18SW | 18 | SW | 26 |
| 49 | 19N | 19 | N | 21 |
| 50 | 19SE | 19 | SE | 29 |
| 51 | 19SW | 19 | SW | 28 |
| 52 | 20N | 20 | N | 22 |
| 53 | 20SW | 20 | SW | 30 |
| 54 | 21NE | 21 | NE | 24 |
| 55 | 21NW | 21 | NW | 23 |
| 56 | 21S | 21 | S | 31 |
| 57 | 22NE | 22 | NE | 26 |
| 58 | 22NW | 22 | NW | 25 |
| 59 | 22S | 22 | S | 32 |
| 60 | 23NE | 23 | NE | 28 |
| 61 | 23NW | 23 | NW | 27 |
| 62 | 23S | 23 | S | 33 |
| 63 | 24NE | 24 | NE | 30 |
| 64 | 24NW | 24 | NW | 29 |
| 65 | 24S | 24 | S | 34 |
| 66 | 25N | 25 | N | 31 |
| 67 | 25SE | 25 | SE | 35 |
| 68 | 26N | 26 | N | 32 |
| 69 | 26SE | 26 | SE | 37 |
| 70 | 26SW | 26 | SW | 36 |
| 71 | 27N | 27 | N | 33 |
| 72 | 27SE | 27 | SE | 39 |
| 73 | 27SW | 27 | SW | 38 |
| 74 | 28N | 28 | N | 34 |
| 75 | 28SW | 28 | SW | 40 |
| 76 | 29NE | 29 | NE | 36 |
| 77 | 29NW | 29 | NW | 35 |
| 78 | 30NE | 30 | NE | 38 |
| 79 | 30NW | 30 | NW | 37 |
| 80 | 31NE | 31 | NE | 40 |
| 81 | 31NW | 31 | NW | 39 |

---

## 6. Standard Map (Radius 2)

![Standard Map (Radius 2) topology indices](board-topology.svg)

### 6.1 Counts

| Element | Count |
| --- | --- |
| Tiles | 19 |
| Vertices | 54 |
| Edges | 72 |
| Ports | 9 |
| Coastal edges | 30 |
| Interior edges | 42 |
| Boundary vertices (degree 2) | 18 |
| Interior vertices (degree 3) | 36 |

Vertex tile-count distribution: 18 touch 1 tile, 12 touch 2 tiles, 24 touch 3 tiles.

Tile rows (top-to-bottom): 3, 4, 5, 4, 3.

### 6.2 Tile Table

| Tile | q | r |
| --- | --- | --- |
| 0 | -2 | +2 |
| 1 | -1 | +2 |
| 2 | +0 | +2 |
| 3 | -2 | +1 |
| 4 | -1 | +1 |
| 5 | +0 | +1 |
| 6 | +1 | +1 |
| 7 | -2 | +0 |
| 8 | -1 | +0 |
| 9 | +0 | +0 |
| 10 | +1 | +0 |
| 11 | +2 | +0 |
| 12 | -1 | -1 |
| 13 | +0 | -1 |
| 14 | +1 | -1 |
| 15 | +2 | -1 |
| 16 | +0 | -2 |
| 17 | +1 | -2 |
| 18 | +2 | -2 |

### 6.3 Vertex Table

| Vertex | Triplet (tile axial coords) |
| --- | --- |
| 0 | (-3, +3) (-2, +2) (-2, +3) |
| 1 | (-2, +3) (-1, +2) (-1, +3) |
| 2 | (-1, +3) (+0, +2) (+0, +3) |
| 3 | (-3, +2) (-3, +3) (-2, +2) |
| 4 | (-2, +2) (-2, +3) (-1, +2) |
| 5 | (-1, +2) (-1, +3) (+0, +2) |
| 6 | (+0, +2) (+0, +3) (+1, +2) |
| 7 | (-3, +2) (-2, +1) (-2, +2) |
| 8 | (-2, +2) (-1, +1) (-1, +2) |
| 9 | (-1, +2) (+0, +1) (+0, +2) |
| 10 | (+0, +2) (+1, +1) (+1, +2) |
| 11 | (-3, +1) (-3, +2) (-2, +1) |
| 12 | (-2, +1) (-2, +2) (-1, +1) |
| 13 | (-1, +1) (-1, +2) (+0, +1) |
| 14 | (+0, +1) (+0, +2) (+1, +1) |
| 15 | (+1, +1) (+1, +2) (+2, +1) |
| 16 | (-3, +1) (-2, +0) (-2, +1) |
| 17 | (-2, +1) (-1, +0) (-1, +1) |
| 18 | (-1, +1) (+0, +0) (+0, +1) |
| 19 | (+0, +1) (+1, +0) (+1, +1) |
| 20 | (+1, +1) (+2, +0) (+2, +1) |
| 21 | (-3, +0) (-3, +1) (-2, +0) |
| 22 | (-2, +0) (-2, +1) (-1, +0) |
| 23 | (-1, +0) (-1, +1) (+0, +0) |
| 24 | (+0, +0) (+0, +1) (+1, +0) |
| 25 | (+1, +0) (+1, +1) (+2, +0) |
| 26 | (+2, +0) (+2, +1) (+3, +0) |
| 27 | (-3, +0) (-2, -1) (-2, +0) |
| 28 | (+2, +0) (+3, -1) (+3, +0) |
| 29 | (-2, +0) (-1, -1) (-1, +0) |
| 30 | (-1, +0) (+0, -1) (+0, +0) |
| 31 | (+0, +0) (+1, -1) (+1, +0) |
| 32 | (+1, +0) (+2, -1) (+2, +0) |
| 33 | (-2, -1) (-2, +0) (-1, -1) |
| 34 | (-1, -1) (-1, +0) (+0, -1) |
| 35 | (+0, -1) (+0, +0) (+1, -1) |
| 36 | (+1, -1) (+1, +0) (+2, -1) |
| 37 | (+2, -1) (+2, +0) (+3, -1) |
| 38 | (-2, -1) (-1, -2) (-1, -1) |
| 39 | (-1, -1) (+0, -2) (+0, -1) |
| 40 | (+0, -1) (+1, -2) (+1, -1) |
| 41 | (+1, -1) (+2, -2) (+2, -1) |
| 42 | (+2, -1) (+3, -2) (+3, -1) |
| 43 | (-1, -2) (-1, -1) (+0, -2) |
| 44 | (+0, -2) (+0, -1) (+1, -2) |
| 45 | (+1, -2) (+1, -1) (+2, -2) |
| 46 | (+2, -2) (+2, -1) (+3, -2) |
| 47 | (-1, -2) (+0, -3) (+0, -2) |
| 48 | (+0, -2) (+1, -3) (+1, -2) |
| 49 | (+1, -2) (+2, -3) (+2, -2) |
| 50 | (+2, -2) (+3, -3) (+3, -2) |
| 51 | (+0, -3) (+0, -2) (+1, -3) |
| 52 | (+1, -3) (+1, -2) (+2, -3) |
| 53 | (+2, -3) (+2, -2) (+3, -3) |

### 6.4 Edge Table

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 3 |
| 1 | 0 | 4 |
| 2 | 1 | 4 |
| 3 | 1 | 5 |
| 4 | 2 | 5 |
| 5 | 2 | 6 |
| 6 | 3 | 7 |
| 7 | 4 | 8 |
| 8 | 5 | 9 |
| 9 | 6 | 10 |
| 10 | 7 | 11 |
| 11 | 7 | 12 |
| 12 | 8 | 12 |
| 13 | 8 | 13 |
| 14 | 9 | 13 |
| 15 | 9 | 14 |
| 16 | 10 | 14 |
| 17 | 10 | 15 |
| 18 | 11 | 16 |
| 19 | 12 | 17 |
| 20 | 13 | 18 |
| 21 | 14 | 19 |
| 22 | 15 | 20 |
| 23 | 16 | 21 |
| 24 | 16 | 22 |
| 25 | 17 | 22 |
| 26 | 17 | 23 |
| 27 | 18 | 23 |
| 28 | 18 | 24 |
| 29 | 19 | 24 |
| 30 | 19 | 25 |
| 31 | 20 | 25 |
| 32 | 20 | 26 |
| 33 | 21 | 27 |
| 34 | 22 | 29 |
| 35 | 23 | 30 |
| 36 | 24 | 31 |
| 37 | 25 | 32 |
| 38 | 26 | 28 |
| 39 | 27 | 33 |
| 40 | 28 | 37 |
| 41 | 29 | 33 |
| 42 | 29 | 34 |
| 43 | 30 | 34 |
| 44 | 30 | 35 |
| 45 | 31 | 35 |
| 46 | 31 | 36 |
| 47 | 32 | 36 |
| 48 | 32 | 37 |
| 49 | 33 | 38 |
| 50 | 34 | 39 |
| 51 | 35 | 40 |
| 52 | 36 | 41 |
| 53 | 37 | 42 |
| 54 | 38 | 43 |
| 55 | 39 | 43 |
| 56 | 39 | 44 |
| 57 | 40 | 44 |
| 58 | 40 | 45 |
| 59 | 41 | 45 |
| 60 | 41 | 46 |
| 61 | 42 | 46 |
| 62 | 43 | 47 |
| 63 | 44 | 48 |
| 64 | 45 | 49 |
| 65 | 46 | 50 |
| 66 | 47 | 51 |
| 67 | 48 | 51 |
| 68 | 48 | 52 |
| 69 | 49 | 52 |
| 70 | 49 | 53 |
| 71 | 50 | 53 |

### 6.5 Coastal Edges (30 total)

`0, 1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 22, 23, 32, 33, 38, 39, 40, 49, 53, 54, 61, 62, 65, 66, 67, 68, 69, 70, 71`

### 6.6 Port Table (9 ports)

Ports are ordered clockwise from the top of the board.

| Port | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 4 | 1 |
| 1 | 2 | 6 |
| 2 | 15 | 20 |
| 3 | 37 | 42 |
| 4 | 50 | 53 |
| 5 | 52 | 48 |
| 6 | 43 | 38 |
| 7 | 27 | 21 |
| 8 | 11 | 7 |

### 6.7 Tile -> Vertices

| Tile | Vertices |
| --- | --- |
| 0 | 0, 3, 4, 7, 8, 12 |
| 1 | 1, 4, 5, 8, 9, 13 |
| 2 | 2, 5, 6, 9, 10, 14 |
| 3 | 7, 11, 12, 16, 17, 22 |
| 4 | 8, 12, 13, 17, 18, 23 |
| 5 | 9, 13, 14, 18, 19, 24 |
| 6 | 10, 14, 15, 19, 20, 25 |
| 7 | 16, 21, 22, 27, 29, 33 |
| 8 | 17, 22, 23, 29, 30, 34 |
| 9 | 18, 23, 24, 30, 31, 35 |
| 10 | 19, 24, 25, 31, 32, 36 |
| 11 | 20, 25, 26, 28, 32, 37 |
| 12 | 29, 33, 34, 38, 39, 43 |
| 13 | 30, 34, 35, 39, 40, 44 |
| 14 | 31, 35, 36, 40, 41, 45 |
| 15 | 32, 36, 37, 41, 42, 46 |
| 16 | 39, 43, 44, 47, 48, 51 |
| 17 | 40, 44, 45, 48, 49, 52 |
| 18 | 41, 45, 46, 49, 50, 53 |

### 6.8 Tile -> Edges

| Tile | Edges |
| --- | --- |
| 0 | 0, 1, 6, 7, 11, 12 |
| 1 | 2, 3, 7, 8, 13, 14 |
| 2 | 4, 5, 8, 9, 15, 16 |
| 3 | 10, 11, 18, 19, 24, 25 |
| 4 | 12, 13, 19, 20, 26, 27 |
| 5 | 14, 15, 20, 21, 28, 29 |
| 6 | 16, 17, 21, 22, 30, 31 |
| 7 | 23, 24, 33, 34, 39, 41 |
| 8 | 25, 26, 34, 35, 42, 43 |
| 9 | 27, 28, 35, 36, 44, 45 |
| 10 | 29, 30, 36, 37, 46, 47 |
| 11 | 31, 32, 37, 38, 40, 48 |
| 12 | 41, 42, 49, 50, 54, 55 |
| 13 | 43, 44, 50, 51, 56, 57 |
| 14 | 45, 46, 51, 52, 58, 59 |
| 15 | 47, 48, 52, 53, 60, 61 |
| 16 | 55, 56, 62, 63, 66, 67 |
| 17 | 57, 58, 63, 64, 68, 69 |
| 18 | 59, 60, 64, 65, 70, 71 |

### 6.9 Tile -> Adjacent Tiles

| Tile | Neighbors |
| --- | --- |
| 0 | 1, 3, 4 |
| 1 | 0, 2, 4, 5 |
| 2 | 1, 5, 6 |
| 3 | 0, 4, 7, 8 |
| 4 | 0, 1, 3, 5, 8, 9 |
| 5 | 1, 2, 4, 6, 9, 10 |
| 6 | 2, 5, 10, 11 |
| 7 | 3, 8, 12 |
| 8 | 3, 4, 7, 9, 12, 13 |
| 9 | 4, 5, 8, 10, 13, 14 |
| 10 | 5, 6, 9, 11, 14, 15 |
| 11 | 6, 10, 15 |
| 12 | 7, 8, 13, 16 |
| 13 | 8, 9, 12, 14, 16, 17 |
| 14 | 9, 10, 13, 15, 17, 18 |
| 15 | 10, 11, 14, 18 |
| 16 | 12, 13, 17 |
| 17 | 13, 14, 16, 18 |
| 18 | 14, 15, 17 |

### 6.10 Vertex -> Tiles

| Vertex | Tiles |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 0 |
| 4 | 0, 1 |
| 5 | 1, 2 |
| 6 | 2 |
| 7 | 0, 3 |
| 8 | 0, 1, 4 |
| 9 | 1, 2, 5 |
| 10 | 2, 6 |
| 11 | 3 |
| 12 | 0, 3, 4 |
| 13 | 1, 4, 5 |
| 14 | 2, 5, 6 |
| 15 | 6 |
| 16 | 3, 7 |
| 17 | 3, 4, 8 |
| 18 | 4, 5, 9 |
| 19 | 5, 6, 10 |
| 20 | 6, 11 |
| 21 | 7 |
| 22 | 3, 7, 8 |
| 23 | 4, 8, 9 |
| 24 | 5, 9, 10 |
| 25 | 6, 10, 11 |
| 26 | 11 |
| 27 | 7 |
| 28 | 11 |
| 29 | 7, 8, 12 |
| 30 | 8, 9, 13 |
| 31 | 9, 10, 14 |
| 32 | 10, 11, 15 |
| 33 | 7, 12 |
| 34 | 8, 12, 13 |
| 35 | 9, 13, 14 |
| 36 | 10, 14, 15 |
| 37 | 11, 15 |
| 38 | 12 |
| 39 | 12, 13, 16 |
| 40 | 13, 14, 17 |
| 41 | 14, 15, 18 |
| 42 | 15 |
| 43 | 12, 16 |
| 44 | 13, 16, 17 |
| 45 | 14, 17, 18 |
| 46 | 15, 18 |
| 47 | 16 |
| 48 | 16, 17 |
| 49 | 17, 18 |
| 50 | 18 |
| 51 | 16 |
| 52 | 17 |
| 53 | 18 |

### 6.11 Vertex -> Edges

| Vertex | Edges |
| --- | --- |
| 0 | 0, 1 |
| 1 | 2, 3 |
| 2 | 4, 5 |
| 3 | 0, 6 |
| 4 | 1, 2, 7 |
| 5 | 3, 4, 8 |
| 6 | 5, 9 |
| 7 | 6, 10, 11 |
| 8 | 7, 12, 13 |
| 9 | 8, 14, 15 |
| 10 | 9, 16, 17 |
| 11 | 10, 18 |
| 12 | 11, 12, 19 |
| 13 | 13, 14, 20 |
| 14 | 15, 16, 21 |
| 15 | 17, 22 |
| 16 | 18, 23, 24 |
| 17 | 19, 25, 26 |
| 18 | 20, 27, 28 |
| 19 | 21, 29, 30 |
| 20 | 22, 31, 32 |
| 21 | 23, 33 |
| 22 | 24, 25, 34 |
| 23 | 26, 27, 35 |
| 24 | 28, 29, 36 |
| 25 | 30, 31, 37 |
| 26 | 32, 38 |
| 27 | 33, 39 |
| 28 | 38, 40 |
| 29 | 34, 41, 42 |
| 30 | 35, 43, 44 |
| 31 | 36, 45, 46 |
| 32 | 37, 47, 48 |
| 33 | 39, 41, 49 |
| 34 | 42, 43, 50 |
| 35 | 44, 45, 51 |
| 36 | 46, 47, 52 |
| 37 | 40, 48, 53 |
| 38 | 49, 54 |
| 39 | 50, 55, 56 |
| 40 | 51, 57, 58 |
| 41 | 52, 59, 60 |
| 42 | 53, 61 |
| 43 | 54, 55, 62 |
| 44 | 56, 57, 63 |
| 45 | 58, 59, 64 |
| 46 | 60, 61, 65 |
| 47 | 62, 66 |
| 48 | 63, 67, 68 |
| 49 | 64, 69, 70 |
| 50 | 65, 71 |
| 51 | 66, 67 |
| 52 | 68, 69 |
| 53 | 70, 71 |

### 6.12 Vertex -> Adjacent Vertices

| Vertex | Neighbors |
| --- | --- |
| 0 | 3, 4 |
| 1 | 4, 5 |
| 2 | 5, 6 |
| 3 | 0, 7 |
| 4 | 0, 1, 8 |
| 5 | 1, 2, 9 |
| 6 | 2, 10 |
| 7 | 3, 11, 12 |
| 8 | 4, 12, 13 |
| 9 | 5, 13, 14 |
| 10 | 6, 14, 15 |
| 11 | 7, 16 |
| 12 | 7, 8, 17 |
| 13 | 8, 9, 18 |
| 14 | 9, 10, 19 |
| 15 | 10, 20 |
| 16 | 11, 21, 22 |
| 17 | 12, 22, 23 |
| 18 | 13, 23, 24 |
| 19 | 14, 24, 25 |
| 20 | 15, 25, 26 |
| 21 | 16, 27 |
| 22 | 16, 17, 29 |
| 23 | 17, 18, 30 |
| 24 | 18, 19, 31 |
| 25 | 19, 20, 32 |
| 26 | 20, 28 |
| 27 | 21, 33 |
| 28 | 26, 37 |
| 29 | 22, 33, 34 |
| 30 | 23, 34, 35 |
| 31 | 24, 35, 36 |
| 32 | 25, 36, 37 |
| 33 | 27, 29, 38 |
| 34 | 29, 30, 39 |
| 35 | 30, 31, 40 |
| 36 | 31, 32, 41 |
| 37 | 28, 32, 42 |
| 38 | 33, 43 |
| 39 | 34, 43, 44 |
| 40 | 35, 44, 45 |
| 41 | 36, 45, 46 |
| 42 | 37, 46 |
| 43 | 38, 39, 47 |
| 44 | 39, 40, 48 |
| 45 | 40, 41, 49 |
| 46 | 41, 42, 50 |
| 47 | 43, 51 |
| 48 | 44, 51, 52 |
| 49 | 45, 52, 53 |
| 50 | 46, 53 |
| 51 | 47, 48 |
| 52 | 48, 49 |
| 53 | 49, 50 |

### 6.13 Edge -> Vertices

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 3 |
| 1 | 0 | 4 |
| 2 | 1 | 4 |
| 3 | 1 | 5 |
| 4 | 2 | 5 |
| 5 | 2 | 6 |
| 6 | 3 | 7 |
| 7 | 4 | 8 |
| 8 | 5 | 9 |
| 9 | 6 | 10 |
| 10 | 7 | 11 |
| 11 | 7 | 12 |
| 12 | 8 | 12 |
| 13 | 8 | 13 |
| 14 | 9 | 13 |
| 15 | 9 | 14 |
| 16 | 10 | 14 |
| 17 | 10 | 15 |
| 18 | 11 | 16 |
| 19 | 12 | 17 |
| 20 | 13 | 18 |
| 21 | 14 | 19 |
| 22 | 15 | 20 |
| 23 | 16 | 21 |
| 24 | 16 | 22 |
| 25 | 17 | 22 |
| 26 | 17 | 23 |
| 27 | 18 | 23 |
| 28 | 18 | 24 |
| 29 | 19 | 24 |
| 30 | 19 | 25 |
| 31 | 20 | 25 |
| 32 | 20 | 26 |
| 33 | 21 | 27 |
| 34 | 22 | 29 |
| 35 | 23 | 30 |
| 36 | 24 | 31 |
| 37 | 25 | 32 |
| 38 | 26 | 28 |
| 39 | 27 | 33 |
| 40 | 28 | 37 |
| 41 | 29 | 33 |
| 42 | 29 | 34 |
| 43 | 30 | 34 |
| 44 | 30 | 35 |
| 45 | 31 | 35 |
| 46 | 31 | 36 |
| 47 | 32 | 36 |
| 48 | 32 | 37 |
| 49 | 33 | 38 |
| 50 | 34 | 39 |
| 51 | 35 | 40 |
| 52 | 36 | 41 |
| 53 | 37 | 42 |
| 54 | 38 | 43 |
| 55 | 39 | 43 |
| 56 | 39 | 44 |
| 57 | 40 | 44 |
| 58 | 40 | 45 |
| 59 | 41 | 45 |
| 60 | 41 | 46 |
| 61 | 42 | 46 |
| 62 | 43 | 47 |
| 63 | 44 | 48 |
| 64 | 45 | 49 |
| 65 | 46 | 50 |
| 66 | 47 | 51 |
| 67 | 48 | 51 |
| 68 | 48 | 52 |
| 69 | 49 | 52 |
| 70 | 49 | 53 |
| 71 | 50 | 53 |

### 6.14 Edge -> Tiles

| Edge | Tiles |
| --- | --- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 2 |
| 5 | 2 |
| 6 | 0 |
| 7 | 0, 1 |
| 8 | 1, 2 |
| 9 | 2 |
| 10 | 3 |
| 11 | 0, 3 |
| 12 | 0, 4 |
| 13 | 1, 4 |
| 14 | 1, 5 |
| 15 | 2, 5 |
| 16 | 2, 6 |
| 17 | 6 |
| 18 | 3 |
| 19 | 3, 4 |
| 20 | 4, 5 |
| 21 | 5, 6 |
| 22 | 6 |
| 23 | 7 |
| 24 | 3, 7 |
| 25 | 3, 8 |
| 26 | 4, 8 |
| 27 | 4, 9 |
| 28 | 5, 9 |
| 29 | 5, 10 |
| 30 | 6, 10 |
| 31 | 6, 11 |
| 32 | 11 |
| 33 | 7 |
| 34 | 7, 8 |
| 35 | 8, 9 |
| 36 | 9, 10 |
| 37 | 10, 11 |
| 38 | 11 |
| 39 | 7 |
| 40 | 11 |
| 41 | 7, 12 |
| 42 | 8, 12 |
| 43 | 8, 13 |
| 44 | 9, 13 |
| 45 | 9, 14 |
| 46 | 10, 14 |
| 47 | 10, 15 |
| 48 | 11, 15 |
| 49 | 12 |
| 50 | 12, 13 |
| 51 | 13, 14 |
| 52 | 14, 15 |
| 53 | 15 |
| 54 | 12 |
| 55 | 12, 16 |
| 56 | 13, 16 |
| 57 | 13, 17 |
| 58 | 14, 17 |
| 59 | 14, 18 |
| 60 | 15, 18 |
| 61 | 15 |
| 62 | 16 |
| 63 | 16, 17 |
| 64 | 17, 18 |
| 65 | 18 |
| 66 | 16 |
| 67 | 16 |
| 68 | 17 |
| 69 | 17 |
| 70 | 18 |
| 71 | 18 |

### 6.15 Action Table (144 actions)

Each action represents a settlement vertex plus road direction. Entries are
sorted by vertex index, then direction string.

| Token | Action | Vertex | Direction | Edge |
| --- | --- | --- | --- | --- |
| 0 | 0SE | 0 | SE | 1 |
| 1 | 0SW | 0 | SW | 0 |
| 2 | 1SE | 1 | SE | 3 |
| 3 | 1SW | 1 | SW | 2 |
| 4 | 2SE | 2 | SE | 5 |
| 5 | 2SW | 2 | SW | 4 |
| 6 | 3NE | 3 | NE | 0 |
| 7 | 3S | 3 | S | 6 |
| 8 | 4NE | 4 | NE | 2 |
| 9 | 4NW | 4 | NW | 1 |
| 10 | 4S | 4 | S | 7 |
| 11 | 5NE | 5 | NE | 4 |
| 12 | 5NW | 5 | NW | 3 |
| 13 | 5S | 5 | S | 8 |
| 14 | 6NW | 6 | NW | 5 |
| 15 | 6S | 6 | S | 9 |
| 16 | 7N | 7 | N | 6 |
| 17 | 7SE | 7 | SE | 11 |
| 18 | 7SW | 7 | SW | 10 |
| 19 | 8N | 8 | N | 7 |
| 20 | 8SE | 8 | SE | 13 |
| 21 | 8SW | 8 | SW | 12 |
| 22 | 9N | 9 | N | 8 |
| 23 | 9SE | 9 | SE | 15 |
| 24 | 9SW | 9 | SW | 14 |
| 25 | 10N | 10 | N | 9 |
| 26 | 10SE | 10 | SE | 17 |
| 27 | 10SW | 10 | SW | 16 |
| 28 | 11NE | 11 | NE | 10 |
| 29 | 11S | 11 | S | 18 |
| 30 | 12NE | 12 | NE | 12 |
| 31 | 12NW | 12 | NW | 11 |
| 32 | 12S | 12 | S | 19 |
| 33 | 13NE | 13 | NE | 14 |
| 34 | 13NW | 13 | NW | 13 |
| 35 | 13S | 13 | S | 20 |
| 36 | 14NE | 14 | NE | 16 |
| 37 | 14NW | 14 | NW | 15 |
| 38 | 14S | 14 | S | 21 |
| 39 | 15NW | 15 | NW | 17 |
| 40 | 15S | 15 | S | 22 |
| 41 | 16N | 16 | N | 18 |
| 42 | 16SE | 16 | SE | 24 |
| 43 | 16SW | 16 | SW | 23 |
| 44 | 17N | 17 | N | 19 |
| 45 | 17SE | 17 | SE | 26 |
| 46 | 17SW | 17 | SW | 25 |
| 47 | 18N | 18 | N | 20 |
| 48 | 18SE | 18 | SE | 28 |
| 49 | 18SW | 18 | SW | 27 |
| 50 | 19N | 19 | N | 21 |
| 51 | 19SE | 19 | SE | 30 |
| 52 | 19SW | 19 | SW | 29 |
| 53 | 20N | 20 | N | 22 |
| 54 | 20SE | 20 | SE | 32 |
| 55 | 20SW | 20 | SW | 31 |
| 56 | 21NE | 21 | NE | 23 |
| 57 | 21S | 21 | S | 33 |
| 58 | 22NE | 22 | NE | 25 |
| 59 | 22NW | 22 | NW | 24 |
| 60 | 22S | 22 | S | 34 |
| 61 | 23NE | 23 | NE | 27 |
| 62 | 23NW | 23 | NW | 26 |
| 63 | 23S | 23 | S | 35 |
| 64 | 24NE | 24 | NE | 29 |
| 65 | 24NW | 24 | NW | 28 |
| 66 | 24S | 24 | S | 36 |
| 67 | 25NE | 25 | NE | 31 |
| 68 | 25NW | 25 | NW | 30 |
| 69 | 25S | 25 | S | 37 |
| 70 | 26NW | 26 | NW | 32 |
| 71 | 26S | 26 | S | 38 |
| 72 | 27N | 27 | N | 33 |
| 73 | 27SE | 27 | SE | 39 |
| 74 | 28N | 28 | N | 38 |
| 75 | 28SW | 28 | SW | 40 |
| 76 | 29N | 29 | N | 34 |
| 77 | 29SE | 29 | SE | 42 |
| 78 | 29SW | 29 | SW | 41 |
| 79 | 30N | 30 | N | 35 |
| 80 | 30SE | 30 | SE | 44 |
| 81 | 30SW | 30 | SW | 43 |
| 82 | 31N | 31 | N | 36 |
| 83 | 31SE | 31 | SE | 46 |
| 84 | 31SW | 31 | SW | 45 |
| 85 | 32N | 32 | N | 37 |
| 86 | 32SE | 32 | SE | 48 |
| 87 | 32SW | 32 | SW | 47 |
| 88 | 33NE | 33 | NE | 41 |
| 89 | 33NW | 33 | NW | 39 |
| 90 | 33S | 33 | S | 49 |
| 91 | 34NE | 34 | NE | 43 |
| 92 | 34NW | 34 | NW | 42 |
| 93 | 34S | 34 | S | 50 |
| 94 | 35NE | 35 | NE | 45 |
| 95 | 35NW | 35 | NW | 44 |
| 96 | 35S | 35 | S | 51 |
| 97 | 36NE | 36 | NE | 47 |
| 98 | 36NW | 36 | NW | 46 |
| 99 | 36S | 36 | S | 52 |
| 100 | 37NE | 37 | NE | 40 |
| 101 | 37NW | 37 | NW | 48 |
| 102 | 37S | 37 | S | 53 |
| 103 | 38N | 38 | N | 49 |
| 104 | 38SE | 38 | SE | 54 |
| 105 | 39N | 39 | N | 50 |
| 106 | 39SE | 39 | SE | 56 |
| 107 | 39SW | 39 | SW | 55 |
| 108 | 40N | 40 | N | 51 |
| 109 | 40SE | 40 | SE | 58 |
| 110 | 40SW | 40 | SW | 57 |
| 111 | 41N | 41 | N | 52 |
| 112 | 41SE | 41 | SE | 60 |
| 113 | 41SW | 41 | SW | 59 |
| 114 | 42N | 42 | N | 53 |
| 115 | 42SW | 42 | SW | 61 |
| 116 | 43NE | 43 | NE | 55 |
| 117 | 43NW | 43 | NW | 54 |
| 118 | 43S | 43 | S | 62 |
| 119 | 44NE | 44 | NE | 57 |
| 120 | 44NW | 44 | NW | 56 |
| 121 | 44S | 44 | S | 63 |
| 122 | 45NE | 45 | NE | 59 |
| 123 | 45NW | 45 | NW | 58 |
| 124 | 45S | 45 | S | 64 |
| 125 | 46NE | 46 | NE | 61 |
| 126 | 46NW | 46 | NW | 60 |
| 127 | 46S | 46 | S | 65 |
| 128 | 47N | 47 | N | 62 |
| 129 | 47SE | 47 | SE | 66 |
| 130 | 48N | 48 | N | 63 |
| 131 | 48SE | 48 | SE | 68 |
| 132 | 48SW | 48 | SW | 67 |
| 133 | 49N | 49 | N | 64 |
| 134 | 49SE | 49 | SE | 70 |
| 135 | 49SW | 49 | SW | 69 |
| 136 | 50N | 50 | N | 65 |
| 137 | 50SW | 50 | SW | 71 |
| 138 | 51NE | 51 | NE | 67 |
| 139 | 51NW | 51 | NW | 66 |
| 140 | 52NE | 52 | NE | 69 |
| 141 | 52NW | 52 | NW | 68 |
| 142 | 53NE | 53 | NE | 71 |
| 143 | 53NW | 53 | NW | 70 |


---

## 7. Regeneration

All map tables and diagrams in this document are generated by
`scripts/generate_board_topology_svg.py`. `BoardTopology` is the runtime
authority; the generator mirrors its tile, vertex, edge, adjacency, and port
indexing rules.

```bash
# Generate SVG diagram
python3 scripts/generate_board_topology_svg.py standard
python3 scripts/generate_board_topology_svg.py mini
python3 scripts/generate_board_topology_svg.py small

# Dump index tables (tile, vertex, edge, coastal)
python3 scripts/generate_board_topology_svg.py standard --dump-tables
python3 scripts/generate_board_topology_svg.py mini --dump-tables
python3 scripts/generate_board_topology_svg.py small --dump-tables

# Dump adjacency tables
python3 scripts/generate_board_topology_svg.py standard --dump-adjacency
python3 scripts/generate_board_topology_svg.py mini --dump-adjacency
python3 scripts/generate_board_topology_svg.py small --dump-adjacency

# Rewrite the Mini, Small, and Standard sections in this document
python3 scripts/generate_board_topology_svg.py --write-reference
```
