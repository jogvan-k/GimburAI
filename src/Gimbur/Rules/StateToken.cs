namespace Gimbur.Rules;

/// <summary>
/// Encodes and decodes serialization tokens using semantically disjoint
/// character alphabets as defined in <c>docs/state-serialization.md</c>.
/// <para>
/// Each category uses its own character set so that a tokenizer can learn
/// category-specific embeddings. Categories sharing the same underlying
/// concept reuse the same alphabet — positional embeddings disambiguate.
/// </para>
/// </summary>
public static class StateToken
{
    // ── Resource Type ───────────────────────────────────────────────
    // d=desert, w=wood, b=brick, s=sheep, W=wheat, o=ore
    // Shared by tiles and ports (ports add 'g' for generic).

    /// <summary>Encodes a <see cref="ResourceType"/> as a single character.</summary>
    public static char EncodeResource(ResourceType resource) => resource switch
    {
        ResourceType.Desert => 'd',
        ResourceType.Wood => 'w',
        ResourceType.Brick => 'b',
        ResourceType.Sheep => 's',
        ResourceType.Wheat => 'W',
        ResourceType.Ore => 'o',
        _ => throw new ArgumentOutOfRangeException(nameof(resource), resource,
            "Unknown resource type."),
    };

    /// <summary>Decodes a resource character back to <see cref="ResourceType"/>.</summary>
    public static ResourceType DecodeResource(char c) => c switch
    {
        'd' => ResourceType.Desert,
        'w' => ResourceType.Wood,
        'b' => ResourceType.Brick,
        's' => ResourceType.Sheep,
        'W' => ResourceType.Wheat,
        'o' => ResourceType.Ore,
        _ => throw new ArgumentOutOfRangeException(nameof(c), c,
            "Not a valid resource character. Expected one of: d w b s W o"),
    };

    // ── Port Type ───────────────────────────────────────────────────
    // g=generic, plus the resource alphabet (w b s W o).

    /// <summary>Encodes a <see cref="PortType"/> as a single character.</summary>
    public static char EncodePort(PortType port) => port switch
    {
        PortType.Generic => 'g',
        PortType.Wood => 'w',
        PortType.Brick => 'b',
        PortType.Sheep => 's',
        PortType.Wheat => 'W',
        PortType.Ore => 'o',
        _ => throw new ArgumentOutOfRangeException(nameof(port), port,
            "Unknown port type."),
    };

    /// <summary>Decodes a port character back to <see cref="PortType"/>.</summary>
    public static PortType DecodePort(char c) => c switch
    {
        'g' => PortType.Generic,
        'w' => PortType.Wood,
        'b' => PortType.Brick,
        's' => PortType.Sheep,
        'W' => PortType.Wheat,
        'o' => PortType.Ore,
        _ => throw new ArgumentOutOfRangeException(nameof(c), c,
            "Not a valid port character. Expected one of: g w b s W o"),
    };

    // ── Tile Number → Pip Count + Side ──────────────────────────────
    // Tile number is decomposed into 2 tokens: pip count (0-5) and side (l/h/n).
    //   0 → 0,n (desert)   2 → 1,l   12 → 1,h   3 → 2,l   11 → 2,h
    //   4 → 3,l   10 → 3,h   5 → 4,l   9 → 4,h   6 → 5,l   8 → 5,h

    /// <summary>
    /// Encodes a tile number (0, 2–6, 8–12) as a pip-count digit character ('0'–'5').
    /// </summary>
    public static char EncodeTilePips(int tileNumber) => tileNumber switch
    {
        0 => '0',
        2 or 12 => '1',
        3 or 11 => '2',
        4 or 10 => '3',
        5 or 9 => '4',
        6 or 8 => '5',
        _ => throw new ArgumentOutOfRangeException(nameof(tileNumber), tileNumber,
            "Must be 0 or 2–6 or 8–12."),
    };

    /// <summary>
    /// Encodes a tile number (0, 2–6, 8–12) as a side character:
    /// 'l' = low (&lt;7), 'h' = high (&gt;7), 'n' = none (desert).
    /// </summary>
    public static char EncodeTileSide(int tileNumber) => tileNumber switch
    {
        0 => 'n',
        2 or 3 or 4 or 5 or 6 => 'l',
        8 or 9 or 10 or 11 or 12 => 'h',
        _ => throw new ArgumentOutOfRangeException(nameof(tileNumber), tileNumber,
            "Must be 0 or 2–6 or 8–12."),
    };

    /// <summary>
    /// Decodes pip-count digit ('0'–'5') and side ('l','h','n') back to the tile number.
    /// </summary>
    public static int DecodeTileNumber(char pips, char side) => (pips, side) switch
    {
        ('0', 'n') => 0,
        ('1', 'l') => 2,
        ('1', 'h') => 12,
        ('2', 'l') => 3,
        ('2', 'h') => 11,
        ('3', 'l') => 4,
        ('3', 'h') => 10,
        ('4', 'l') => 5,
        ('4', 'h') => 9,
        ('5', 'l') => 6,
        ('5', 'h') => 8,
        _ => throw new ArgumentOutOfRangeException(
            $"Invalid pip/side combination: pips='{pips}', side='{side}'."),
    };

    // ── Player ID ───────────────────────────────────────────────────
    // _ = none (0), - = player 1, + = player 2, * = player 3, ^ = player 4

    /// <summary>Encodes a player number (0=none, 1–4) as a player ID character.</summary>
    public static char EncodePlayer(int player) => player switch
    {
        0 => '_',
        1 => '-',
        2 => '+',
        3 => '*',
        4 => '^',
        _ => throw new ArgumentOutOfRangeException(nameof(player), player,
            "Must be 0–4."),
    };

    /// <summary>Decodes a player ID character back to a player number (0=none, 1–4).</summary>
    public static int DecodePlayer(char c) => c switch
    {
        '_' => 0,
        '-' => 1,
        '+' => 2,
        '*' => 3,
        '^' => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(c), c,
            "Not a valid player character. Expected one of: _ - + * ^"),
    };

    // ── Building Type ───────────────────────────────────────────────
    // . = empty (None), v = village (Settlement), c = city

    /// <summary>Encodes a <see cref="BuildingType"/> as a single character.</summary>
    public static char EncodeBuilding(BuildingType building) => building switch
    {
        BuildingType.None => '.',
        BuildingType.Settlement => 'v',
        BuildingType.City => 'c',
        _ => throw new ArgumentOutOfRangeException(nameof(building), building,
            "Unknown building type."),
    };

    /// <summary>Decodes a building character back to <see cref="BuildingType"/>.</summary>
    public static BuildingType DecodeBuilding(char c) => c switch
    {
        '.' => BuildingType.None,
        'v' => BuildingType.Settlement,
        'c' => BuildingType.City,
        _ => throw new ArgumentOutOfRangeException(nameof(c), c,
            "Not a valid building character. Expected one of: . v c"),
    };

    // ── Turn Stage ──────────────────────────────────────────────────
    // a=PlaceFirstSettlement, e=PlaceFirstRoad, f=PlaceSecondSettlement,
    // i=PlaceSecondRoad, r=PreRoll, x=ChooseRobberLocation,
    // y=ChooseRobberVictim, t=BuildTrade

    /// <summary>Encodes a <see cref="TurnStage"/> as a single character.</summary>
    public static char EncodeTurnStage(TurnStage stage) => stage switch
    {
        TurnStage.PlaceFirstSettlement => 'a',
        TurnStage.PlaceFirstRoad => 'e',
        TurnStage.PlaceSecondSettlement => 'f',
        TurnStage.PlaceSecondRoad => 'i',
        TurnStage.PreRoll => 'r',
        TurnStage.ChooseRobberLocation => 'x',
        TurnStage.ChooseRobberVictim => 'y',
        TurnStage.BuildTrade => 't',
        _ => throw new ArgumentOutOfRangeException(nameof(stage), stage,
            "Unknown turn stage."),
    };

    /// <summary>Decodes a turn stage character back to <see cref="TurnStage"/>.</summary>
    public static TurnStage DecodeTurnStage(char c) => c switch
    {
        'a' => TurnStage.PlaceFirstSettlement,
        'e' => TurnStage.PlaceFirstRoad,
        'f' => TurnStage.PlaceSecondSettlement,
        'i' => TurnStage.PlaceSecondRoad,
        'r' => TurnStage.PreRoll,
        'x' => TurnStage.ChooseRobberLocation,
        'y' => TurnStage.ChooseRobberVictim,
        't' => TurnStage.BuildTrade,
        _ => throw new ArgumentOutOfRangeException(nameof(c), c,
            "Not a valid turn stage character. Expected one of: a e f i r x y t"),
    };
}
