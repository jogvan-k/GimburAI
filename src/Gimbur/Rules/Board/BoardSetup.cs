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

        var remaining = new HashSet<HexCoord>(topology.Tiles);
        var order = new List<int>(topology.TileCount);

        // Peel rings from outside in, walking each boundary ring clockwise.
        while (remaining.Count > 0)
        {
            // Find boundary tiles: tiles with at least one neighbor missing from remaining.
            var boundary = new HashSet<HexCoord>();
            foreach (var coord in remaining)
            {
                foreach (var dir in HexCoord.Directions)
                {
                    if (!remaining.Contains(coord + dir))
                    {
                        boundary.Add(coord);
                        break;
                    }
                }
            }

            // If all remaining tiles are interior (single tile or fully surrounded group),
            // all remaining tiles are boundary.
            if (boundary.Count == 0)
                boundary = new HashSet<HexCoord>(remaining);

            // Walk the boundary ring clockwise starting from the topmost-leftmost tile.
            var ring = WalkBoundaryRingClockwise(boundary);
            foreach (var coord in ring)
            {
                order.Add(byCoord[coord]);
                remaining.Remove(coord);
            }
        }

        if (order.Count != topology.TileCount)
            throw new InvalidOperationException(
                $"Spiral order built {order.Count} tiles for topology with {topology.TileCount} tiles.");

        return [.. order];
    }

    /// <summary>
    /// Walks a set of boundary hex coordinates in clockwise order,
    /// starting from the topmost (then leftmost by screen position) tile.
    /// </summary>
    private static List<HexCoord> WalkBoundaryRingClockwise(HashSet<HexCoord> boundary)
    {
        if (boundary.Count <= 1)
            return [.. boundary];

        // Sort by screen position to find the start tile (topmost-leftmost).
        var sorted = boundary.OrderBy(c =>
        {
            var (x, y) = BoardTopology.AxialToPixel(c);
            return (y, x);
        }).ToList();

        var start = sorted[0];
        var result = new List<HexCoord> { start };
        var visited = new HashSet<HexCoord> { start };
        var current = start;

        // Direction order for clockwise traversal: SW, W, NW, NE, E, SE
        // This is dirs[2], dirs[3], dirs[4], dirs[5], dirs[0], dirs[1]
        // Starting from the top, we want to go right first, then spiral clockwise.
        // Use direction priority: E, SE, SW, W, NW, NE (clockwise from east).
        var cwDirs = new[] { HexCoord.Directions[0], HexCoord.Directions[1],
                             HexCoord.Directions[2], HexCoord.Directions[3],
                             HexCoord.Directions[4], HexCoord.Directions[5] };

        // For each step, try to go to the next clockwise unvisited neighbor in the boundary.
        // Use the direction we came from to determine the preferred search order.
        var prevDir = 2; // Assume we "came from" the SW direction (so we prefer going right/E first)

        while (result.Count < boundary.Count)
        {
            var found = false;
            // Try directions starting from the one 120 degrees CW from the incoming direction
            // (turn back and sweep clockwise). This ensures clockwise traversal.
            for (var d = 0; d < 6; d++)
            {
                // Start searching from the direction opposite to prevDir, rotated CW by 2
                var dirIdx = (prevDir + 4 + d) % 6;
                var neighbor = current + HexCoord.Directions[dirIdx];
                if (boundary.Contains(neighbor) && !visited.Contains(neighbor))
                {
                    visited.Add(neighbor);
                    result.Add(neighbor);
                    current = neighbor;
                    prevDir = dirIdx;
                    found = true;
                    break;
                }
            }
            if (!found)
                break; // Disconnected boundary; add remaining tiles sorted by screen position.
        }

        // If some boundary tiles weren't reached (disconnected), append them sorted.
        if (result.Count < boundary.Count)
        {
            foreach (var coord in sorted)
            {
                if (!visited.Contains(coord))
                    result.Add(coord);
            }
        }

        return result;
    }
}
