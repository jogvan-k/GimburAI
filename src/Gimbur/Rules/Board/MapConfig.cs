using System.Collections.Immutable;

namespace Gimbur.Rules;

/// <summary>
/// Defines the tile resource distribution, number token distribution, and port type
/// distribution for a specific map variant.
/// </summary>
public sealed class MapConfig
{
    /// <summary>Resource types for each tile, to be shuffled during setup.</summary>
    public ImmutableArray<ResourceType> TileResources { get; }

    /// <summary>Number tokens for non-desert tiles, to be shuffled during setup.</summary>
    public ImmutableArray<int> NumberTokens { get; }

    /// <summary>Port types to be shuffled and assigned to port positions during setup.</summary>
    public ImmutableArray<PortType> PortTypes { get; }

    /// <summary>The board topology this config applies to.</summary>
    public BoardTopology Topology { get; }

    public MapConfig(
        BoardTopology topology,
        ImmutableArray<ResourceType> tileResources,
        ImmutableArray<int> numberTokens,
        ImmutableArray<PortType> portTypes)
    {
        if (tileResources.Length != topology.TileCount)
            throw new ArgumentException(
                $"Expected {topology.TileCount} tile resources, got {tileResources.Length}",
                nameof(tileResources));

        var desertCount = tileResources.Count(r => r == ResourceType.Desert);
        if (numberTokens.Length != topology.TileCount - desertCount)
            throw new ArgumentException(
                $"Expected {topology.TileCount - desertCount} number tokens, got {numberTokens.Length}",
                nameof(numberTokens));

        if (portTypes.Length != topology.PortCount)
            throw new ArgumentException(
                $"Expected {topology.PortCount} port types, got {portTypes.Length}",
                nameof(portTypes));

        Topology = topology;
        TileResources = tileResources;
        NumberTokens = numberTokens;
        PortTypes = portTypes;
    }

    /// <summary>
    /// Standard Catan map: 19 tiles (radius 2).
    /// 1 desert, 4 wood, 3 brick, 4 sheep, 4 wheat, 3 ore.
    /// Number tokens are in official alphabetical token order (A..R), to be placed in spiral order.
    /// Distribution remains: 2(x1) 3(x2) 4(x2) 5(x2) 6(x2) 8(x2) 9(x2) 10(x2) 11(x2) 12(x1).
    /// Ports: 4 generic (3:1) + 5 resource-specific (2:1, one per resource).
    /// </summary>
    public static MapConfig Standard { get; } = new(
        BoardTopology.Standard,
        [
            ResourceType.Desert,
            ResourceType.Wood, ResourceType.Wood, ResourceType.Wood, ResourceType.Wood,
            ResourceType.Brick, ResourceType.Brick, ResourceType.Brick,
            ResourceType.Sheep, ResourceType.Sheep, ResourceType.Sheep, ResourceType.Sheep,
            ResourceType.Wheat, ResourceType.Wheat, ResourceType.Wheat, ResourceType.Wheat,
            ResourceType.Ore, ResourceType.Ore, ResourceType.Ore,
        ],
        [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11],
        [
            PortType.Generic, PortType.Generic, PortType.Generic, PortType.Generic,
            PortType.Wood, PortType.Brick, PortType.Sheep, PortType.Wheat, PortType.Ore,
        ]);

    /// <summary>
    /// Mini Catan map: 7 tiles (radius 1).
    /// 1 desert, 1 wood, 1 brick, 1 sheep, 1 wheat, 1 ore, 1 extra (wheat).
    /// Number tokens: 3, 4, 5, 6, 9, 10.
    /// Ports: 3 generic (3:1) + 3 resource-specific (2:1).
    /// </summary>
    public static MapConfig Mini { get; } = new(
        BoardTopology.Mini,
        [
            ResourceType.Desert,
            ResourceType.Wood, ResourceType.Brick, ResourceType.Sheep,
            ResourceType.Wheat, ResourceType.Wheat, ResourceType.Ore,
        ],
        [3, 4, 5, 6, 9, 10],
        [
            PortType.Generic, PortType.Generic, PortType.Generic,
            PortType.Wood, PortType.Brick, PortType.Sheep,
        ]);
}
