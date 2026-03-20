"""
Game configuration for each valid map + player-count combination.

Mirrors the C# ``GameConfig`` presets (Mini, Small, Standard) and
expands them into one config per valid player count.  Each config
carries the topology sizes and derived ``state_token_size`` needed by
the neural-network layer (see ``transformer_model.py``).

Token sequence length formula (compact form, no separators)::

    (3*T + 2*V + E + P + 5) + 11*N

where T = tiles, V = vertices, E = edges, P = ports, N = players.
The constant 5 = robber(1) + current-turn(2) + awards(2).
"""

from __future__ import annotations


class GameConfig:
    """Game configuration for a specific map and player count."""

    # ── Identifiers ──────────────────────────────────────────────────
    name: str
    """Human-readable preset name (e.g. ``"mini_2p"``)."""

    map_name: str
    """Map variant: ``"mini"``, ``"small"``, or ``"standard"``."""

    player_count: int
    """Number of players for this configuration."""

    # ── Topology sizes ───────────────────────────────────────────────
    tile_count: int
    """Number of hex tiles on the board (T)."""

    vertex_count: int
    """Number of vertices on the board (V)."""

    edge_count: int
    """Number of edges on the board (E)."""

    port_count: int
    """Number of ports / harbors (P)."""

    # ── Derived ──────────────────────────────────────────────────────
    state_token_size: int
    """Compact-form token sequence length for this map + player count."""

    # ── Victory / thresholds ─────────────────────────────────────────
    victory_points_to_win: int

    longest_road_minimum: int

    largest_army_minimum: int

    discard_threshold: int

    # ── Supply limits (per player) ───────────────────────────────────
    max_settlements: int

    max_cities: int

    max_roads: int

    # ── Bank / deck ──────────────────────────────────────────────────
    resource_cards_per_type: int

    initial_placement_rounds: int


def _token_size(t: int, v: int, e: int, p: int, n: int) -> int:
    """Compute compact-form token sequence length."""
    return (3 * t + 2 * v + e + p + 5) + 11 * n


def _make_config(
    *,
    name: str,
    map_name: str,
    player_count: int,
    tile_count: int,
    vertex_count: int,
    edge_count: int,
    port_count: int,
    victory_points_to_win: int,
    longest_road_minimum: int,
    largest_army_minimum: int,
    discard_threshold: int,
    max_settlements: int,
    max_cities: int,
    max_roads: int,
    resource_cards_per_type: int,
    initial_placement_rounds: int,
) -> GameConfig:
    cfg = GameConfig()
    cfg.name = name
    cfg.map_name = map_name
    cfg.player_count = player_count
    cfg.tile_count = tile_count
    cfg.vertex_count = vertex_count
    cfg.edge_count = edge_count
    cfg.port_count = port_count
    cfg.state_token_size = _token_size(
        tile_count, vertex_count, edge_count, port_count, player_count
    )
    cfg.victory_points_to_win = victory_points_to_win
    cfg.longest_road_minimum = longest_road_minimum
    cfg.largest_army_minimum = largest_army_minimum
    cfg.discard_threshold = discard_threshold
    cfg.max_settlements = max_settlements
    cfg.max_cities = max_cities
    cfg.max_roads = max_roads
    cfg.resource_cards_per_type = resource_cards_per_type
    cfg.initial_placement_rounds = initial_placement_rounds
    return cfg


# ── Topology constants ───────────────────────────────────────────────
_MINI_T, _MINI_V, _MINI_E, _MINI_P = 7, 24, 30, 6
_SMALL_T, _SMALL_V, _SMALL_E, _SMALL_P = 10, 32, 41, 6
_STD_T, _STD_V, _STD_E, _STD_P = 19, 54, 72, 9

# ── Predefined configurations ────────────────────────────────────────

MINI_2P = _make_config(
    name="mini_2p",
    map_name="mini",
    player_count=2,
    tile_count=_MINI_T,
    vertex_count=_MINI_V,
    edge_count=_MINI_E,
    port_count=_MINI_P,
    victory_points_to_win=5,
    longest_road_minimum=4,
    largest_army_minimum=2,
    discard_threshold=5,
    max_settlements=4,
    max_cities=3,
    max_roads=10,
    resource_cards_per_type=10,
    initial_placement_rounds=1,
)

SMALL_2P = _make_config(
    name="small_2p",
    map_name="small",
    player_count=2,
    tile_count=_SMALL_T,
    vertex_count=_SMALL_V,
    edge_count=_SMALL_E,
    port_count=_SMALL_P,
    victory_points_to_win=7,
    longest_road_minimum=5,
    largest_army_minimum=3,
    discard_threshold=6,
    max_settlements=5,
    max_cities=3,
    max_roads=12,
    resource_cards_per_type=14,
    initial_placement_rounds=2,
)

SMALL_3P = _make_config(
    name="small_3p",
    map_name="small",
    player_count=3,
    tile_count=_SMALL_T,
    vertex_count=_SMALL_V,
    edge_count=_SMALL_E,
    port_count=_SMALL_P,
    victory_points_to_win=7,
    longest_road_minimum=5,
    largest_army_minimum=3,
    discard_threshold=6,
    max_settlements=5,
    max_cities=3,
    max_roads=12,
    resource_cards_per_type=14,
    initial_placement_rounds=2,
)

STANDARD_3P = _make_config(
    name="standard_3p",
    map_name="standard",
    player_count=3,
    tile_count=_STD_T,
    vertex_count=_STD_V,
    edge_count=_STD_E,
    port_count=_STD_P,
    victory_points_to_win=10,
    longest_road_minimum=5,
    largest_army_minimum=3,
    discard_threshold=7,
    max_settlements=5,
    max_cities=4,
    max_roads=15,
    resource_cards_per_type=19,
    initial_placement_rounds=2,
)

STANDARD_4P = _make_config(
    name="standard_4p",
    map_name="standard",
    player_count=4,
    tile_count=_STD_T,
    vertex_count=_STD_V,
    edge_count=_STD_E,
    port_count=_STD_P,
    victory_points_to_win=10,
    longest_road_minimum=5,
    largest_army_minimum=3,
    discard_threshold=7,
    max_settlements=5,
    max_cities=4,
    max_roads=15,
    resource_cards_per_type=19,
    initial_placement_rounds=2,
)

ALL_CONFIGS: tuple[GameConfig, ...] = (
    MINI_2P,
    SMALL_2P,
    SMALL_3P,
    STANDARD_3P,
    STANDARD_4P,
)
"""All valid game configurations, ordered by map size then player count."""

CONFIGS_BY_NAME: dict[str, GameConfig] = {cfg.name: cfg for cfg in ALL_CONFIGS}
"""Lookup table keyed by config name (e.g. ``"standard_4p"``)."""
