#!/usr/bin/env python3
"""
Generate SVG diagrams and topology data for Catan board maps.

Supports multiple map layouts:
  standard  -- 19-tile radius-2 hex grid (rows of 3,4,5,4,3)
  mini      -- 7-tile radius-1 hex grid (rows of 2,3,2)
  small     -- 10-tile non-circular oval (rows of 3,4,3)

Pointy-top hexagons, board viewed top-down.
Tiles numbered top-to-bottom, left-to-right (by screen position).
Vertices and edges use canonical indexing (top-to-bottom, left-to-right).
"""

import math
import os
import re
import sys

SQRT3 = math.sqrt(3.0)
SIZE = 40.0  # hex radius (center to corner)


# ── Pointy-top hex geometry ──────────────────────────────────────────


def axial_to_pixel(q, r):
    """Axial (q, r) -> screen (x, y), pointy-top. +r = up, so negate y."""
    x = SIZE * SQRT3 * (q + r / 2.0)
    y = -SIZE * 1.5 * r
    return (x, y)


def hex_corner_xy(cx, cy, i):
    """Corner i (0..5) of a pointy-top hex centered at (cx, cy)."""
    angle = math.radians(60 * i - 30)
    return (cx + SIZE * math.cos(angle), cy + SIZE * math.sin(angle))


# ── Map layouts ──────────────────────────────────────────────────────

MAP_LAYOUTS = {
    "standard": {"radius": 2, "expected_tiles": 19},
    "mini": {"radius": 1, "expected_tiles": 7},
    "small": {
        "tiles": [
            (-1, 1),
            (0, 1),
            (1, 1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (0, -1),
            (1, -1),
            (2, -1),
        ],
        "port_count": 6,
        "expected_tiles": 10,
    },
}


def make_tile_coords(radius):
    """Tiles of a hex grid with the given radius."""
    tiles = []
    for r in range(-radius, radius + 1):
        qmin = max(-radius, -r - radius)
        qmax = min(radius, -r + radius)
        for q in range(qmin, qmax + 1):
            tiles.append((q, r))
    return tiles


# ── Vertex & edge generation ────────────────────────────────────────

# Axial neighbor directions (clockwise from east)
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def make_vertices_and_edges(tiles):
    """
    Vertex identity: sorted triplet of 3 tile-axial-coords meeting at it.
    Edge identity: sorted pair of vertex keys for two adjacent corners.
    Pixel positions computed from hex_corner_xy of the first tile that creates it.
    """
    vertex_pos = {}  # vkey -> (px, py)
    edge_set = set()

    for q, r in tiles:
        cx, cy = axial_to_pixel(q, r)
        for i in range(6):
            n1 = add((q, r), DIRS[i])
            n2 = add((q, r), DIRS[(i - 1) % 6])
            vkey = tuple(sorted([(q, r), n1, n2]))
            if vkey not in vertex_pos:
                vertex_pos[vkey] = hex_corner_xy(cx, cy, i)

            n3 = add((q, r), DIRS[(i + 1) % 6])
            vkey2 = tuple(sorted([(q, r), n3, n1]))
            ekey = tuple(sorted([vkey, vkey2]))
            edge_set.add(ekey)

    # Order: top-to-bottom (y ascending), left-to-right (x ascending)
    vertices = sorted(vertex_pos.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    v_index = {key: i for i, (key, _) in enumerate(vertices)}

    # Match BoardTopology: normalize endpoint indices, then sort numerically.
    edges = [
        edge if v_index[edge[0]] < v_index[edge[1]] else (edge[1], edge[0])
        for edge in edge_set
    ]
    edges.sort(key=lambda edge: (v_index[edge[0]], v_index[edge[1]]))

    return vertices, v_index, edges


# ── Adjacency ────────────────────────────────────────────────────────


def build_adjacency(tiles, vertices, v_index, edges):
    """Build all adjacency lookups."""
    tile_set = set(tiles)
    centers = {t: axial_to_pixel(*t) for t in tiles}
    tiles_sorted = sorted(tiles, key=lambda t: (centers[t][1], centers[t][0]))
    tile_index = {t: i for i, t in enumerate(tiles_sorted)}
    num_tiles = len(tiles_sorted)
    num_vertices = len(vertices)
    num_edges = len(edges)

    # vertex -> adjacent on-board tiles (from vertex triplet)
    vertex_tiles = {}
    for vi, (key, pos) in enumerate(vertices):
        vertex_tiles[vi] = sorted(tile_index[t] for t in key if t in tile_set)

    # tile -> vertices (corner indices)
    tile_vertices = {i: [] for i in range(num_tiles)}
    for vi, (key, pos) in enumerate(vertices):
        for t in key:
            if t in tile_set:
                tile_vertices[tile_index[t]].append(vi)
    for ti in tile_vertices:
        tile_vertices[ti] = sorted(tile_vertices[ti])

    # edge -> adjacent on-board tiles
    edge_tiles = {}
    for ei, (k1, k2) in enumerate(edges):
        shared = set(k1) & set(k2)
        on_board = sorted(tile_index[t] for t in shared if t in tile_set)
        edge_tiles[ei] = on_board

    # tile -> edges
    tile_edges = {i: [] for i in range(num_tiles)}
    for ei in range(num_edges):
        for ti in edge_tiles[ei]:
            tile_edges[ti].append(ei)
    for ti in tile_edges:
        tile_edges[ti] = sorted(tile_edges[ti])

    # vertex -> edges
    vertex_edges = {vi: [] for vi in range(num_vertices)}
    for ei, (k1, k2) in enumerate(edges):
        a, b = v_index[k1], v_index[k2]
        vertex_edges[a].append(ei)
        vertex_edges[b].append(ei)
    for vi in vertex_edges:
        vertex_edges[vi] = sorted(vertex_edges[vi])

    # vertex -> adjacent vertices
    vertex_neighbors = {vi: [] for vi in range(num_vertices)}
    for ei, (k1, k2) in enumerate(edges):
        a, b = v_index[k1], v_index[k2]
        vertex_neighbors[a].append(b)
        vertex_neighbors[b].append(a)
    for vi in vertex_neighbors:
        vertex_neighbors[vi] = sorted(vertex_neighbors[vi])

    # edge -> endpoints
    edge_vertices = {}
    for ei, (k1, k2) in enumerate(edges):
        edge_vertices[ei] = (v_index[k1], v_index[k2])

    # tile -> adjacent tiles
    tile_neighbors = {i: set() for i in range(num_tiles)}
    for ei in edge_tiles:
        ts = edge_tiles[ei]
        if len(ts) == 2:
            tile_neighbors[ts[0]].add(ts[1])
            tile_neighbors[ts[1]].add(ts[0])
    for ti in tile_neighbors:
        tile_neighbors[ti] = sorted(tile_neighbors[ti])  # type: ignore[assignment]

    return {
        "tiles_sorted": tiles_sorted,
        "tile_index": tile_index,
        "vertex_tiles": vertex_tiles,
        "tile_vertices": tile_vertices,
        "edge_tiles": edge_tiles,
        "tile_edges": tile_edges,
        "vertex_edges": vertex_edges,
        "vertex_neighbors": vertex_neighbors,
        "edge_vertices": edge_vertices,
        "tile_neighbors": tile_neighbors,
    }


def find_coastal_edges(tiles, edges):
    """Return indices of edges that border exactly 1 on-board tile."""
    tile_set = set(tiles)
    coastal = []
    for ei, (k1, k2) in enumerate(edges):
        shared = set(k1) & set(k2)
        on_board = shared & tile_set
        if len(on_board) == 1:
            coastal.append(ei)
    return coastal


# ── Port generation ──────────────────────────────────────────────────


def make_ports(tiles, vertices, v_index, edges, port_count):
    """
    Compute port positions from the coastal edge ring.

    Walks the ring of coastal edges clockwise from the topmost vertex,
    then selects port edges using evenly-spaced positions along the ring.
    Each port is a coastal edge connecting two vertices (one or both may
    be boundary degree-2 vertices).

    Parameters:
      port_count -- number of ports to place around the board.

    Returns list of (vertex_a, vertex_b) pairs (by vertex index), ordered
    clockwise from the top of the board.
    """
    adj = build_adjacency(tiles, vertices, v_index, edges)

    # Build coastal-edge graph for walking the boundary ring
    coastal_indices = find_coastal_edges(tiles, edges)
    boundary_adj = {}  # vertex -> [(neighbor, edge_index), ...]
    for ei in coastal_indices:
        a, b = adj["edge_vertices"][ei]
        boundary_adj.setdefault(a, []).append((b, ei))
        boundary_adj.setdefault(b, []).append((a, ei))

    vpos = {i: pos for i, (key, pos) in enumerate(vertices)}

    # Start from the topmost (then leftmost) vertex on the boundary
    start = min(boundary_adj, key=lambda v: (vpos[v][1], vpos[v][0]))

    # Walk clockwise: pick the rightmost (most positive x) neighbor first
    visited_edges = set()
    ring = [start]
    current = start
    neighbors = sorted(boundary_adj[current], key=lambda ne: -vpos[ne[0]][0])
    next_v, next_e = neighbors[0]
    visited_edges.add(next_e)
    ring.append(next_v)
    current = next_v

    while current != start:
        for nv, ne in boundary_adj[current]:
            if ne not in visited_edges:
                visited_edges.add(ne)
                ring.append(nv)
                current = nv
                break

    ring = ring[:-1]  # remove duplicate start

    # Build ordered list of ring edges as (vertex_a, vertex_b) pairs
    ring_edges = []
    for i in range(len(ring)):
        ring_edges.append((ring[i], ring[(i + 1) % len(ring)]))

    # Select port positions: evenly spaced along the ring
    total = len(ring_edges)
    nports = port_count
    ports = []
    for i in range(nports):
        pos = (1 + (i * total) // nports) % total
        ports.append(ring_edges[pos])

    return ports


# ── SVG rendering ────────────────────────────────────────────────────


def generate_svg(tiles, vertices, v_index, edges, ports):
    centers = {t: axial_to_pixel(*t) for t in tiles}

    # Label tiles by screen order: top-to-bottom, left-to-right
    tiles_sorted = sorted(tiles, key=lambda t: (centers[t][1], centers[t][0]))
    tile_label = {t: i for i, t in enumerate(tiles_sorted)}

    vpos = {key: pos for key, pos in vertices}
    vpos_by_idx = {i: pos for i, (key, pos) in enumerate(vertices)}

    # Bounds
    all_x = [p[0] for p in centers.values()] + [p[0] for p in vpos.values()]
    all_y = [p[1] for p in centers.values()] + [p[1] for p in vpos.values()]
    minx, maxx = min(all_x), max(all_x)
    miny, maxy = min(all_y), max(all_y)

    pad = 50
    w = maxx - minx + pad * 2
    h = maxy - miny + pad * 2

    def tx(x):
        return x - minx + pad

    def ty(y):
        return y - miny + pad

    s = []
    s.append(
        f"<svg xmlns='http://www.w3.org/2000/svg'"
        f" width='{w:.0f}' height='{h:.0f}'"
        f" viewBox='0 0 {w:.0f} {h:.0f}'"
        f" style='background:#fff'>"
    )
    s.append("<style>")
    s.append("  .hex  { fill:#f5f0e1; stroke:#333; stroke-width:1.5; }")
    s.append(
        "  .tlbl { font:bold 13px sans-serif; fill:#333;"
        " text-anchor:middle; dominant-baseline:central; }"
    )
    s.append("  .vert { fill:#1a73e8; }")
    s.append(
        "  .vlbl { font:9px monospace; fill:#0b3d91;"
        " text-anchor:middle; dominant-baseline:central; }"
    )
    s.append("  .edge { stroke:#aaa; stroke-width:1.2; }")
    s.append(
        "  .elbl { font:8px monospace; fill:#888;"
        " text-anchor:middle; dominant-baseline:central; }"
    )
    s.append(
        "  .port { stroke:#d4a017; stroke-width:2.5; stroke-linecap:round; fill:none; }"
    )
    s.append(
        "  .plbl { font:bold 9px sans-serif; fill:#d4a017;"
        " text-anchor:middle; dominant-baseline:central; }"
    )
    s.append("</style>")

    # Layer 1: hex tiles
    for t in tiles_sorted:
        cx, cy = centers[t]
        pts = " ".join(
            f"{tx(hex_corner_xy(cx, cy, i)[0]):.1f},"
            f"{ty(hex_corner_xy(cx, cy, i)[1]):.1f}"
            for i in range(6)
        )
        s.append(f"<polygon class='hex' points='{pts}'/>")
        s.append(
            f"<text class='tlbl' x='{tx(cx):.1f}' y='{ty(cy):.1f}'>"
            f"{tile_label[t]}</text>"
        )

    # Layer 2: edges with offset labels
    for ei, (k1, k2) in enumerate(edges):
        x1, y1 = vpos[k1]
        x2, y2 = vpos[k2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        # perpendicular offset so label doesn't overlap the line
        nx, ny = -dy / length * 9, dx / length * 9
        s.append(
            f"<line class='edge'"
            f" x1='{tx(x1):.1f}' y1='{ty(y1):.1f}'"
            f" x2='{tx(x2):.1f}' y2='{ty(y2):.1f}'/>"
        )
        s.append(
            f"<text class='elbl'"
            f" x='{tx(mx) + nx:.1f}' y='{ty(my) + ny:.1f}'>"
            f"{ei}</text>"
        )

    # Layer 3: ports (bracket outside the board pointing at two vertices)
    for pi, (va, vb) in enumerate(ports):
        xa, ya = vpos_by_idx[va]
        xb, yb = vpos_by_idx[vb]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        # Edge perpendicular (two candidates, pick the one pointing outward)
        dx, dy = xb - xa, yb - ya
        length = math.hypot(dx, dy) or 1
        # Two perpendicular unit vectors
        nx1, ny1 = -dy / length, dx / length
        # Pick the one pointing away from the board center (dot with midpoint > 0)
        if nx1 * mx + ny1 * my < 0:
            nx1, ny1 = -nx1, -ny1
        # Port anchor: midpoint pushed outward along the perpendicular
        px, py = mx + nx1 * 22, my + ny1 * 22
        # Draw lines from port anchor to each vertex
        s.append(
            f"<line class='port'"
            f" x1='{tx(px):.1f}' y1='{ty(py):.1f}'"
            f" x2='{tx(xa):.1f}' y2='{ty(ya):.1f}'/>"
        )
        s.append(
            f"<line class='port'"
            f" x1='{tx(px):.1f}' y1='{ty(py):.1f}'"
            f" x2='{tx(xb):.1f}' y2='{ty(yb):.1f}'/>"
        )
        # Label: further out along the same perpendicular
        lx, ly = mx + nx1 * 35, my + ny1 * 35
        s.append(f"<text class='plbl' x='{tx(lx):.1f}' y='{ty(ly):.1f}'>P{pi}</text>")

    # Layer 4: vertices on top
    for vi, (key, pos) in enumerate(vertices):
        x, y = pos
        s.append(f"<circle class='vert' cx='{tx(x):.1f}' cy='{ty(y):.1f}' r='4'/>")
        s.append(f"<text class='vlbl' x='{tx(x):.1f}' y='{ty(y) - 10:.1f}'>{vi}</text>")

    s.append("</svg>")
    return "\n".join(s)


# ── Output helpers ───────────────────────────────────────────────────


def fmt_triplet(triplet):
    """Format a sorted triplet of axial coords for display."""
    return " ".join(f"({q:+d}, {r:+d})" for q, r in triplet)


def dump_tables(tiles, vertices, v_index, edges, ports):
    """Print markdown tables for tiles, vertices, edges, coastal edges, and ports."""
    centers = {t: axial_to_pixel(*t) for t in tiles}
    tiles_sorted = sorted(tiles, key=lambda t: (centers[t][1], centers[t][0]))

    print("## Tile Table\n")
    print("| Tile | q | r |")
    print("| --- | --- | --- |")
    for i, (q, r) in enumerate(tiles_sorted):
        print(f"| {i} | {q:+d} | {r:+d} |")

    print("\n## Vertex Table\n")
    print("| Vertex | Triplet (tile axial coords) |")
    print("| --- | --- |")
    for vi, (key, pos) in enumerate(vertices):
        print(f"| {vi} | {fmt_triplet(key)} |")

    print("\n## Edge Table\n")
    print("| Edge | Vertex A | Vertex B |")
    print("| --- | --- | --- |")
    for ei, (k1, k2) in enumerate(edges):
        a, b = v_index[k1], v_index[k2]
        print(f"| {ei} | {a} | {b} |")

    coastal = find_coastal_edges(tiles, edges)
    print(f"\n## Coastal Edges ({len(coastal)} total)\n")
    print(", ".join(str(e) for e in coastal))

    print(f"\n## Port Table ({len(ports)} ports)\n")
    print("| Port | Vertex A | Vertex B |")
    print("| --- | --- | --- |")
    for pi, (va, vb) in enumerate(ports):
        print(f"| {pi} | {va} | {vb} |")


def dump_adjacency(tiles, vertices, v_index, edges):
    """Print all adjacency tables."""
    adj = build_adjacency(tiles, vertices, v_index, edges)
    num_tiles = len(adj["tiles_sorted"])
    num_vertices = len(vertices)
    num_edges = len(edges)

    print("## Tile -> Vertices\n")
    print("| Tile | Vertices |")
    print("| --- | --- |")
    for ti in range(num_tiles):
        print(f"| {ti} | {', '.join(str(v) for v in adj['tile_vertices'][ti])} |")

    print("\n## Tile -> Edges\n")
    print("| Tile | Edges |")
    print("| --- | --- |")
    for ti in range(num_tiles):
        print(f"| {ti} | {', '.join(str(e) for e in adj['tile_edges'][ti])} |")

    print("\n## Tile -> Adjacent Tiles\n")
    print("| Tile | Neighbors |")
    print("| --- | --- |")
    for ti in range(num_tiles):
        print(f"| {ti} | {', '.join(str(t) for t in adj['tile_neighbors'][ti])} |")

    print("\n## Vertex -> Tiles\n")
    print("| Vertex | Tiles |")
    print("| --- | --- |")
    for vi in range(num_vertices):
        print(f"| {vi} | {', '.join(str(t) for t in adj['vertex_tiles'][vi])} |")

    print("\n## Vertex -> Edges\n")
    print("| Vertex | Edges |")
    print("| --- | --- |")
    for vi in range(num_vertices):
        print(f"| {vi} | {', '.join(str(e) for e in adj['vertex_edges'][vi])} |")

    print("\n## Vertex -> Adjacent Vertices\n")
    print("| Vertex | Neighbors |")
    print("| --- | --- |")
    for vi in range(num_vertices):
        print(f"| {vi} | {', '.join(str(v) for v in adj['vertex_neighbors'][vi])} |")

    print("\n## Edge -> Vertices\n")
    print("| Edge | Vertex A | Vertex B |")
    print("| --- | --- | --- |")
    for ei in range(num_edges):
        a, b = adj["edge_vertices"][ei]
        print(f"| {ei} | {a} | {b} |")

    print("\n## Edge -> Tiles\n")
    print("| Edge | Tiles |")
    print("| --- | --- |")
    for ei in range(num_edges):
        print(f"| {ei} | {', '.join(str(t) for t in adj['edge_tiles'][ei])} |")


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def placement_actions(vertices, v_index, edges):
    positions = {index: pos for index, (_, pos) in enumerate(vertices)}
    entries = []
    for edge_index, (key_a, key_b) in enumerate(edges):
        a, b = v_index[key_a], v_index[key_b]
        for source, target in ((a, b), (b, a)):
            sx, sy = positions[source]
            tx, ty = positions[target]
            dx, dy = tx - sx, ty - sy
            if abs(dx) < 1e-9:
                direction = "N" if dy < 0 else "S"
            elif dx > 0:
                direction = "NE" if dy < 0 else "SE"
            else:
                direction = "NW" if dy < 0 else "SW"
            entries.append((source, direction, edge_index))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return entries


def render_map_section(layout_name, section_number, tiles, vertices, v_index, edges, ports):
    adj = build_adjacency(tiles, vertices, v_index, edges)
    coastal = find_coastal_edges(tiles, edges)
    titles = {
        "mini": "Mini Map (Radius 1)",
        "small": "Small Map (10 tiles, non-circular)",
        "standard": "Standard Map (Radius 2)",
    }
    svg_names = {
        "mini": "mini-board-topology.svg",
        "small": "small-board-topology.svg",
        "standard": "board-topology.svg",
    }
    tile_rows = {"mini": "2, 3, 2", "small": "3, 4, 3", "standard": "3, 4, 5, 4, 3"}
    degree_counts = {
        degree: sum(1 for edges_for_vertex in adj["vertex_edges"].values() if len(edges_for_vertex) == degree)
        for degree in (2, 3)
    }
    tile_touch_counts = {
        count: sum(1 for vertex_tiles in adj["vertex_tiles"].values() if len(vertex_tiles) == count)
        for count in (1, 2, 3)
    }
    action_entries = placement_actions(vertices, v_index, edges)

    lines = [
        f"## {section_number}. {titles[layout_name]}",
        "",
        f"![{titles[layout_name]} topology indices]({svg_names[layout_name]})",
        "",
    ]
    if layout_name == "small":
        lines.extend([
            "The Small map is a non-circular oval board built from two central hexes",
            "`(0,0)` and `(1,0)` plus one layer of hexes around them.",
            "",
        ])
    lines.extend([
        f"### {section_number}.1 Counts",
        "",
        markdown_table(
            ["Element", "Count"],
            [
                ("Tiles", len(adj["tiles_sorted"])),
                ("Vertices", len(vertices)),
                ("Edges", len(edges)),
                ("Ports", len(ports)),
                ("Coastal edges", len(coastal)),
                ("Interior edges", len(edges) - len(coastal)),
                ("Boundary vertices (degree 2)", degree_counts[2]),
                ("Interior vertices (degree 3)", degree_counts[3]),
            ],
        ),
        "",
        "Vertex tile-count distribution: "
        + ", ".join(f"{tile_touch_counts[count]} touch {count} tile{'s' if count != 1 else ''}" for count in (1, 2, 3))
        + ".",
        "",
        f"Tile rows (top-to-bottom): {tile_rows[layout_name]}.",
        "",
        f"### {section_number}.2 Tile Table",
        "",
        markdown_table(["Tile", "q", "r"], [(i, f"{q:+d}", f"{r:+d}") for i, (q, r) in enumerate(adj["tiles_sorted"])]),
        "",
        f"### {section_number}.3 Vertex Table",
        "",
        markdown_table(["Vertex", "Triplet (tile axial coords)"], [(i, fmt_triplet(key)) for i, (key, _) in enumerate(vertices)]),
        "",
        f"### {section_number}.4 Edge Table",
        "",
        markdown_table(["Edge", "Vertex A", "Vertex B"], [(i, v_index[a], v_index[b]) for i, (a, b) in enumerate(edges)]),
        "",
        f"### {section_number}.5 Coastal Edges ({len(coastal)} total)",
        "",
        "`" + ", ".join(map(str, coastal)) + "`",
        "",
        f"### {section_number}.6 Port Table ({len(ports)} ports)",
        "",
        "Ports are ordered clockwise from the top of the board.",
        "",
        markdown_table(["Port", "Vertex A", "Vertex B"], [(i, a, b) for i, (a, b) in enumerate(ports)]),
    ])

    sections = [
        ("Tile -> Vertices", "Tile", "Vertices", adj["tile_vertices"]),
        ("Tile -> Edges", "Tile", "Edges", adj["tile_edges"]),
        ("Tile -> Adjacent Tiles", "Tile", "Neighbors", adj["tile_neighbors"]),
        ("Vertex -> Tiles", "Vertex", "Tiles", adj["vertex_tiles"]),
        ("Vertex -> Edges", "Vertex", "Edges", adj["vertex_edges"]),
        ("Vertex -> Adjacent Vertices", "Vertex", "Neighbors", adj["vertex_neighbors"]),
        ("Edge -> Vertices", "Edge", "Vertex A | Vertex B", adj["edge_vertices"]),
        ("Edge -> Tiles", "Edge", "Tiles", adj["edge_tiles"]),
    ]
    subsection = 7
    for title, key_header, value_header, values in sections:
        lines.extend(["", f"### {section_number}.{subsection} {title}", ""])
        if title == "Edge -> Vertices":
            rows = [(index, pair[0], pair[1]) for index, pair in values.items()]
            lines.append(markdown_table([key_header, "Vertex A", "Vertex B"], rows))
        else:
            rows = [(index, ", ".join(map(str, entries))) for index, entries in values.items()]
            lines.append(markdown_table([key_header, value_header], rows))
        subsection += 1

    lines.extend([
        "",
        f"### {section_number}.15 Action Table ({len(action_entries)} actions)",
        "",
        "Each action represents a settlement vertex plus road direction. Entries are",
        "sorted by vertex index, then direction string.",
        "",
        markdown_table(
            ["Token", "Action", "Vertex", "Direction", "Edge"],
            [(i, f"{vertex}{direction}", vertex, direction, edge) for i, (vertex, direction, edge) in enumerate(action_entries)],
        ),
        "",
    ])
    return "\n".join(lines)


def write_reference_document():
    document_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "docs", "topology-reference.md"))
    document = open(document_path, encoding="utf-8").read()
    rendered = []
    for section_number, layout_name in ((4, "mini"), (5, "small"), (6, "standard")):
        layout = MAP_LAYOUTS[layout_name]
        tiles = make_tile_coords(layout["radius"]) if "radius" in layout else list(layout["tiles"])
        port_count = 3 * (layout["radius"] + 1) if "radius" in layout else layout["port_count"]
        vertices, v_index, edges = make_vertices_and_edges(tiles)
        ports = make_ports(tiles, vertices, v_index, edges, port_count)
        rendered.append(render_map_section(layout_name, section_number, tiles, vertices, v_index, edges, ports))
    replacement = "\n---\n\n".join(rendered) + "\n\n---\n\n## 7. Regeneration"
    document = re.sub(r"## 4\. Mini Map.*?## 7\. Regeneration", replacement, document, flags=re.DOTALL)
    with open(document_path, "w", encoding="utf-8") as output:
        output.write(document)
    print(f"Wrote {document_path}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    if "--write-reference" in sys.argv:
        write_reference_document()
        return
    # Parse map layout
    layout_name = "standard"
    for arg in sys.argv[1:]:
        if not arg.startswith("--") and arg in MAP_LAYOUTS:
            layout_name = arg

    layout = MAP_LAYOUTS[layout_name]
    if "radius" in layout:
        tiles = make_tile_coords(layout["radius"])
        port_count = 3 * (layout["radius"] + 1)
    else:
        tiles = list(layout["tiles"])
        port_count = layout["port_count"]
    expected = layout["expected_tiles"]
    assert len(tiles) == expected, f"Expected {expected} tiles, got {len(tiles)}"

    vertices, v_index, edges = make_vertices_and_edges(tiles)
    ports = make_ports(tiles, vertices, v_index, edges, port_count)

    if "--dump-tables" in sys.argv:
        dump_tables(tiles, vertices, v_index, edges, ports)
        return

    if "--dump-adjacency" in sys.argv:
        dump_adjacency(tiles, vertices, v_index, edges)
        return

    svg = generate_svg(tiles, vertices, v_index, edges, ports)

    # Determine output filename
    if layout_name == "standard":
        svg_name = "board-topology.svg"
    else:
        svg_name = f"{layout_name}-board-topology.svg"

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "docs",
        svg_name,
    )
    out_path = os.path.normpath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Wrote {out_path}")

    # Print summary
    centers = {t: axial_to_pixel(*t) for t in tiles}
    tiles_sorted = sorted(tiles, key=lambda t: (centers[t][1], centers[t][0]))
    print(
        f"\n{layout_name} map: {len(tiles)} tiles, "
        f"{len(vertices)} vertices, {len(edges)} edges, {len(ports)} ports"
    )
    print("\nTile order (top-to-bottom, left-to-right):")
    for i, (q, r) in enumerate(tiles_sorted):
        print(f"  {i:2d}  q={q:+d}  r={r:+d}")


if __name__ == "__main__":
    main()
