# Board Topology

This document defines the standard 19-hex Catan board index tables. For the
full topology reference (coordinate system, adjacency relationships, multiple
map layouts, and derivation formulas), see
[topology-reference.md](topology-reference.md).

Indices are 0-based and ordered top-to-bottom, left-to-right by screen position,
using pointy-top hexagons orientation.

## Hex Orientation

**Pointy-top** hexagons viewed top-down.

- Pixel position: `x = size * sqrt(3) * (q + r/2)`, `y = -size * 1.5 * r`
- Corner angles: `60 * i - 30` degrees (i = 0..5)

## Tile Indexing

Axial coordinates use `(q, r)` with the board center at `(0, 0)`. Tiles are ordered by screen position: top-to-bottom (ascending y), left-to-right (ascending x).

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

## Vertex Indexing

Vertices are defined by the 3 tiles that meet at a point. Each vertex is represented as a sorted triplet of axial tile coordinates (including virtual tiles outside the board for boundary vertices). Vertices are ordered by screen position: top-to-bottom (ascending y), left-to-right (ascending x).

Axial neighbor directions (clockwise from east):
- `d0 = (1, 0)`
- `d1 = (1, -1)`
- `d2 = (0, -1)`
- `d3 = (-1, 0)`
- `d4 = (-1, 1)`
- `d5 = (0, 1)`

Corner `ci` of tile `(q, r)` is the vertex shared with neighbors `di` and `d(i-1 mod 6)`.

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

## Edge Indexing

Edges connect two adjacent corners on a tile. Each edge is represented by its two endpoint vertices (by vertex index). Edges are ordered by `(vertexA, vertexB)`.

| Edge | Vertex A | Vertex B |
| --- | --- | --- |
| 0 | 0 | 4 |
| 1 | 1 | 5 |
| 2 | 2 | 6 |
| 3 | 3 | 0 |
| 4 | 3 | 7 |
| 5 | 4 | 1 |
| 6 | 4 | 8 |
| 7 | 5 | 2 |
| 8 | 5 | 9 |
| 9 | 6 | 10 |
| 10 | 7 | 12 |
| 11 | 8 | 13 |
| 12 | 9 | 14 |
| 13 | 10 | 15 |
| 14 | 11 | 7 |
| 15 | 11 | 16 |
| 16 | 12 | 8 |
| 17 | 12 | 17 |
| 18 | 13 | 9 |
| 19 | 13 | 18 |
| 20 | 14 | 10 |
| 21 | 14 | 19 |
| 22 | 15 | 20 |
| 23 | 16 | 22 |
| 24 | 17 | 23 |
| 25 | 18 | 24 |
| 26 | 19 | 25 |
| 27 | 20 | 26 |
| 28 | 21 | 16 |
| 29 | 21 | 27 |
| 30 | 22 | 17 |
| 31 | 22 | 29 |
| 32 | 23 | 18 |
| 33 | 23 | 30 |
| 34 | 24 | 19 |
| 35 | 24 | 31 |
| 36 | 25 | 20 |
| 37 | 25 | 32 |
| 38 | 26 | 28 |
| 39 | 27 | 33 |
| 40 | 29 | 34 |
| 41 | 30 | 35 |
| 42 | 31 | 36 |
| 43 | 32 | 37 |
| 44 | 33 | 29 |
| 45 | 33 | 38 |
| 46 | 34 | 30 |
| 47 | 34 | 39 |
| 48 | 35 | 31 |
| 49 | 35 | 40 |
| 50 | 36 | 32 |
| 51 | 36 | 41 |
| 52 | 37 | 28 |
| 53 | 37 | 42 |
| 54 | 38 | 43 |
| 55 | 39 | 44 |
| 56 | 40 | 45 |
| 57 | 41 | 46 |
| 58 | 43 | 39 |
| 59 | 43 | 47 |
| 60 | 44 | 40 |
| 61 | 44 | 48 |
| 62 | 45 | 41 |
| 63 | 45 | 49 |
| 64 | 46 | 42 |
| 65 | 46 | 50 |
| 66 | 47 | 51 |
| 67 | 48 | 52 |
| 68 | 49 | 53 |
| 69 | 51 | 48 |
| 70 | 52 | 49 |
| 71 | 53 | 50 |

## Coastal Edges

An edge is **coastal** if the two vertex triplets share exactly 1 on-board tile. There are 30 coastal edges:

`0, 1, 2, 3, 4, 5, 7, 9, 13, 14, 15, 22, 27, 28, 29, 38, 39, 45, 52, 53, 54, 59, 64, 65, 66, 67, 68, 69, 70, 71`

## Ports (9 ports)

Each port is a coastal edge connecting two vertices on the board perimeter. Port **positions** are part of the topology; port **types** (3:1 generic or 2:1 resource-specific) are assigned randomly during game setup.

Pairs are listed clockwise from the top of the board:

| Port | Vertex A | Vertex B |
| --- | --- | --- |
| P0 | 4 | 1 |
| P1 | 2 | 6 |
| P2 | 15 | 20 |
| P3 | 37 | 42 |
| P4 | 50 | 53 |
| P5 | 52 | 48 |
| P6 | 43 | 38 |
| P7 | 27 | 21 |
| P8 | 11 | 7 |

Standard distribution: 4 generic (3:1) + 5 resource-specific (2:1, one per resource).

For full details, see [topology-reference.md §5.6](topology-reference.md).

## Topology Figure

The figure below shows tile, vertex, and edge indices for the fixed ordering.

![Board topology indices](board-topology.svg)

## Regeneration

The topology tables and SVG are generated by `scripts/generate_board_topology_svg.py`.

```bash
# Regenerate the SVG diagram
python3 scripts/generate_board_topology_svg.py

# Dump the tables to stdout (for updating this document)
python3 scripts/generate_board_topology_svg.py --dump-tables
```
