from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def generator():
    path = Path(__file__).parents[2] / "scripts" / "generate_board_topology_svg.py"
    spec = importlib.util.spec_from_file_location("generate_board_topology_svg", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout_name", ["mini", "small", "standard"])
def test_edges_use_canonical_numeric_endpoint_order(generator, layout_name: str) -> None:
    layout = generator.MAP_LAYOUTS[layout_name]
    tiles = (
        generator.make_tile_coords(layout["radius"])
        if "radius" in layout
        else list(layout["tiles"])
    )
    _, vertex_index, edges = generator.make_vertices_and_edges(tiles)
    indexed = [(vertex_index[a], vertex_index[b]) for a, b in edges]

    assert all(a < b for a, b in indexed)
    assert indexed == sorted(indexed)


@pytest.mark.parametrize(
    ("layout_name", "counts"),
    [
        ("mini", (7, 24, 30)),
        ("small", (10, 32, 41)),
        ("standard", (19, 54, 72)),
    ],
)
def test_generated_topology_counts(
    generator, layout_name: str, counts: tuple[int, int, int]
) -> None:
    layout = generator.MAP_LAYOUTS[layout_name]
    tiles = (
        generator.make_tile_coords(layout["radius"])
        if "radius" in layout
        else list(layout["tiles"])
    )
    vertices, _, edges = generator.make_vertices_and_edges(tiles)

    assert (len(tiles), len(vertices), len(edges)) == counts


@pytest.mark.parametrize("layout_name", ["mini", "small", "standard"])
def test_checked_in_svg_matches_generator(generator, layout_name: str) -> None:
    layout = generator.MAP_LAYOUTS[layout_name]
    tiles = (
        generator.make_tile_coords(layout["radius"])
        if "radius" in layout
        else list(layout["tiles"])
    )
    port_count = 3 * (layout["radius"] + 1) if "radius" in layout else layout["port_count"]
    vertices, vertex_index, edges = generator.make_vertices_and_edges(tiles)
    ports = generator.make_ports(tiles, vertices, vertex_index, edges, port_count)
    filename = (
        "board-topology.svg"
        if layout_name == "standard"
        else f"{layout_name}-board-topology.svg"
    )
    checked_in = Path(__file__).parents[2] / "docs" / filename

    assert checked_in.read_text() == generator.generate_svg(
        tiles, vertices, vertex_index, edges, ports
    )
