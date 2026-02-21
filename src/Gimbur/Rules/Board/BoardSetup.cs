using System.Collections.Immutable;

namespace Gimbur.Rules;

/// <summary>
/// The result of setting up a board: assigned resource types, number tokens, and port types.
/// All positions are fixed by topology; this contains the randomized assignments.
/// </summary>
public sealed class BoardSetup
{
    /// <summary>Resource type for each tile, indexed by tile index.</summary>
    public ImmutableArray<ResourceType> TileResources { get; }

    /// <summary>Number token for each tile (0 for desert), indexed by tile index.</summary>
    public ImmutableArray<int> TileNumbers { get; }

    /// <summary>Port type for each port position, indexed by port index.</summary>
    public ImmutableArray<PortType> PortTypes { get; }

    /// <summary>The tile index where the robber starts (the desert tile).</summary>
    public int InitialRobberTile { get; }

    /// <summary>The topology this setup applies to.</summary>
    public BoardTopology Topology { get; }

    internal BoardSetup(
        BoardTopology topology,
        ImmutableArray<ResourceType> tileResources,
        ImmutableArray<int> tileNumbers,
        ImmutableArray<PortType> portTypes,
        int initialRobberTile)
    {
        Topology = topology;
        TileResources = tileResources;
        TileNumbers = tileNumbers;
        PortTypes = portTypes;
        InitialRobberTile = initialRobberTile;
    }

    /// <summary>
    /// Generates a randomized board setup from a map configuration.
    /// Shuffles tile resources, assigns number tokens in spiral order while
    /// skipping the desert tile, and shuffles port types. Optionally enforces the constraint that
    /// no two high-probability numbers (6 and 8) are on adjacent tiles.
    /// </summary>
    /// <param name="config">The map configuration defining distributions.</param>
    /// <param name="rng">Random number generator.</param>
    /// <param name="noAdjacentRedNumbers">
    /// If true, rejects placements where two tiles with numbers 6 or 8 are adjacent.
    /// Retries up to <paramref name="maxRetries"/> times.
    /// </param>
    /// <param name="maxRetries">Maximum shuffle attempts when enforcing constraints.</param>
    public static BoardSetup Generate(
        MapConfig config,
        Random rng,
        bool noAdjacentRedNumbers = true,
        int maxRetries = 1000)
    {
        var topology = config.Topology;
        var tileCount = topology.TileCount;

        // Shuffle tile resources.
        var resources = config.TileResources.ToArray();

        // Number tokens are consumed in declared order and assigned by spiral position.
        var numberPool = config.NumberTokens.ToArray();
        var spiral = BuildSpiralTileOrder(topology);

        // Shuffle port types.
        var ports = config.PortTypes.ToArray();
        Shuffle(ports, rng);

        for (var attempt = 0; attempt < maxRetries; attempt++)
        {
            Shuffle(resources, rng);

            // Assign number tokens in spiral order: desert gets 0 and is skipped.
            var numbers = new int[tileCount];
            var poolIndex = 0;
            foreach (var ti in spiral)
            {
                if (resources[ti] == ResourceType.Desert)
                {
                    numbers[ti] = 0;
                    continue;
                }

                numbers[ti] = numberPool[poolIndex++];
            }

            if (poolIndex != numberPool.Length)
                throw new InvalidOperationException(
                    $"Assigned {poolIndex} number tokens, expected {numberPool.Length}.");

            // Check adjacent red numbers constraint.
            if (noAdjacentRedNumbers && HasAdjacentRedNumbers(topology, numbers))
                continue;

            // Find the desert tile for initial robber placement.
            var robberTile = Array.IndexOf(resources, ResourceType.Desert);

            return new BoardSetup(
                topology,
                [.. resources],
                [.. numbers],
                [.. ports],
                robberTile);
        }

        throw new InvalidOperationException(
            $"Failed to generate a valid board setup after {maxRetries} attempts. " +
            "Consider relaxing constraints or increasing maxRetries.");
    }

    private static bool HasAdjacentRedNumbers(BoardTopology topology, int[] numbers)
    {
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            if (numbers[ti] is not (6 or 8))
                continue;

            foreach (var neighbor in topology.TileNeighbors[ti])
            {
                if (numbers[neighbor] is 6 or 8)
                    return true;
            }
        }
        return false;
    }

    private static void Shuffle<T>(T[] array, Random rng)
    {
        for (var i = array.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    private static int[] BuildSpiralTileOrder(BoardTopology topology)
    {
        var byCoord = new Dictionary<HexCoord, int>(topology.TileCount);
        for (var ti = 0; ti < topology.TileCount; ti++)
            byCoord[topology.Tiles[ti]] = ti;

        var dirs = HexCoord.Directions;
        var ringDirs = new[] { dirs[2], dirs[1], dirs[0], dirs[5], dirs[4], dirs[3] };
        var order = new List<int>(topology.TileCount);

        for (var radius = topology.Radius; radius >= 1; radius--)
        {
            var current = new HexCoord(-radius, radius);
            foreach (var dir in ringDirs)
            {
                for (var step = 0; step < radius; step++)
                {
                    order.Add(byCoord[current]);
                    current += dir;
                }
            }
        }

        // Center tile for odd-count boards (both standard and mini).
        if (byCoord.TryGetValue(new HexCoord(0, 0), out var center))
            order.Add(center);

        if (order.Count != topology.TileCount)
            throw new InvalidOperationException(
                $"Spiral order built {order.Count} tiles for topology with {topology.TileCount} tiles.");

        return [.. order];
    }
}
