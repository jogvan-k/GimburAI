using System.Collections.Immutable;

namespace Gimbur.Rules;

/// <summary>
/// A vertex key: the sorted triplet of three tile coordinates that meet at this vertex.
/// Includes virtual off-board tiles.
/// </summary>
public readonly record struct VertexKey(HexCoord A, HexCoord B, HexCoord C) : IComparable<VertexKey>
{
    /// <summary>
    /// Creates a VertexKey from three tile coordinates, sorting them into canonical order.
    /// </summary>
    public static VertexKey Create(HexCoord c0, HexCoord c1, HexCoord c2)
    {
        // Sort the three coords into canonical order.
        if (c0.CompareTo(c1) > 0) (c0, c1) = (c1, c0);
        if (c1.CompareTo(c2) > 0) (c1, c2) = (c2, c1);
        if (c0.CompareTo(c1) > 0) (c0, c1) = (c1, c0);
        return new VertexKey(c0, c1, c2);
    }

    /// <summary>
    /// Returns the three tile coords as an array.
    /// </summary>
    public HexCoord[] ToArray() => [A, B, C];

    public int CompareTo(VertexKey other)
    {
        var cmp = A.CompareTo(other.A);
        if (cmp != 0) return cmp;
        cmp = B.CompareTo(other.B);
        return cmp != 0 ? cmp : C.CompareTo(other.C);
    }
}

/// <summary>
/// Immutable board topology for a hex grid of a given radius.
/// Computes tile positions, vertices, edges, ports, and all adjacency lookups.
/// </summary>
/// <remarks>
/// All indices follow the ordering defined in docs/topology-reference.md:
/// tiles, vertices sorted by screen position (ascending y, then ascending x);
/// edges sorted by (min vertex index, max vertex index);
/// ports ordered clockwise from the top.
/// </remarks>
public sealed class BoardTopology
{
    /// <summary>Board radius (1 = mini 7-tile, 2 = standard 19-tile).</summary>
    public int Radius { get; }

    // ── Constants (must be initialized before static instances) ────

    private static readonly double Sqrt3 = Math.Sqrt(3.0);

    // ── Element arrays ──────────────────────────────────────────────

    /// <summary>Tile axial coordinates, indexed by tile index.</summary>
    public ImmutableArray<HexCoord> Tiles { get; }

    /// <summary>Vertex keys (sorted triplets), indexed by vertex index.</summary>
    public ImmutableArray<VertexKey> Vertices { get; }

    /// <summary>Edge endpoint pairs (vertexA, vertexB) where A &lt; B, indexed by edge index.</summary>
    public ImmutableArray<(int VertexA, int VertexB)> Edges { get; }

    /// <summary>Port vertex pairs, ordered clockwise from top. Each pair is a coastal edge.</summary>
    public ImmutableArray<(int VertexA, int VertexB)> Ports { get; }

    /// <summary>Indices of coastal edges (edges bordering exactly 1 on-board tile).</summary>
    public ImmutableArray<int> CoastalEdges { get; }

    // ── Counts ──────────────────────────────────────────────────────

    public int TileCount => Tiles.Length;
    public int VertexCount => Vertices.Length;
    public int EdgeCount => Edges.Length;
    public int PortCount => Ports.Length;

    // ── Adjacency lookups ───────────────────────────────────────────

    /// <summary>Tile -> its 6 vertex indices (sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> TileVertices { get; }

    /// <summary>Tile -> its 6 edge indices (sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> TileEdges { get; }

    /// <summary>Tile -> adjacent tile indices (sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> TileNeighbors { get; }

    /// <summary>Vertex -> on-board tile indices (1-3 tiles, sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> VertexTiles { get; }

    /// <summary>Vertex -> edge indices (2-3 edges, sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> VertexEdges { get; }

    /// <summary>Vertex -> adjacent vertex indices (2-3 vertices, sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> VertexNeighbors { get; }

    /// <summary>Edge -> on-board tile indices (1-2 tiles, sorted).</summary>
    public ImmutableArray<ImmutableArray<int>> EdgeTiles { get; }

    // ── Precomputed instances ───────────────────────────────────────

    /// <summary>Standard 19-tile board (radius 2).</summary>
    public static BoardTopology Standard { get; } = new(2);

    /// <summary>Mini 7-tile board (radius 1).</summary>
    public static BoardTopology Mini { get; } = new(1);

    // ── Constructor ─────────────────────────────────────────────────

    public BoardTopology(int radius)
    {
        if (radius < 1)
            throw new ArgumentOutOfRangeException(nameof(radius), radius, "Radius must be >= 1");

        Radius = radius;

        // 1) Generate tile coordinates, sorted by screen position.
        var tileCoords = GenerateTileCoords(radius);
        var tileSet = tileCoords.ToHashSet();
        Tiles = [.. SortByScreenPosition(tileCoords)];

        var tileIndex = new Dictionary<HexCoord, int>(Tiles.Length);
        for (var i = 0; i < Tiles.Length; i++)
            tileIndex[Tiles[i]] = i;

        // 2) Generate vertices and edges.
        var (vertexList, vertexIndex, edgeList) = GenerateVerticesAndEdges(tileCoords, tileSet);
        Vertices = [.. vertexList];
        Edges = [.. edgeList];

        // 3) Build adjacency.
        var (tileVerts, tileEdgs, tileNeighbors, vertTiles, vertEdgs, vertNeighbors, edgeTiles) =
            BuildAdjacency(Tiles, tileSet, tileIndex, vertexList, vertexIndex, edgeList);

        TileVertices = tileVerts;
        TileEdges = tileEdgs;
        TileNeighbors = tileNeighbors;
        VertexTiles = vertTiles;
        VertexEdges = vertEdgs;
        VertexNeighbors = vertNeighbors;
        EdgeTiles = edgeTiles;

        // 4) Find coastal edges.
        CoastalEdges = [.. FindCoastalEdges(edgeList, edgeTiles)];

        // 5) Generate ports.
        Ports = [.. GeneratePorts(radius, vertexList, vertexIndex, edgeList, edgeTiles)];
    }

    // ── Tile generation ─────────────────────────────────────────────

    private static List<HexCoord> GenerateTileCoords(int radius)
    {
        var tiles = new List<HexCoord>();
        for (var r = -radius; r <= radius; r++)
        {
            var qMin = Math.Max(-radius, -r - radius);
            var qMax = Math.Min(radius, -r + radius);
            for (var q = qMin; q <= qMax; q++)
                tiles.Add(new HexCoord(q, r));
        }
        return tiles;
    }

    // ── Screen-position sorting ─────────────────────────────────────

    internal static (double X, double Y) AxialToPixel(HexCoord c)
    {
        var x = Sqrt3 * (c.Q + c.R / 2.0);
        var y = -1.5 * c.R;
        return (x, y);
    }

    private static List<HexCoord> SortByScreenPosition(List<HexCoord> coords)
    {
        var sorted = new List<HexCoord>(coords);
        sorted.Sort((a, b) =>
        {
            var (ax, ay) = AxialToPixel(a);
            var (bx, by) = AxialToPixel(b);
            var cmp = ay.CompareTo(by);
            return cmp != 0 ? cmp : ax.CompareTo(bx);
        });
        return sorted;
    }

    // ── Vertex & edge generation ────────────────────────────────────

    private static (List<VertexKey>, Dictionary<VertexKey, int>, List<(int, int)>)
        GenerateVerticesAndEdges(List<HexCoord> tiles, HashSet<HexCoord> tileSet)
    {
        var dirs = HexCoord.Directions;
        // Collect unique vertices with pixel positions for sorting.
        var vertexPositions = new Dictionary<VertexKey, (double X, double Y)>();
        var edgeKeySet = new HashSet<(VertexKey, VertexKey)>();

        foreach (var tile in tiles)
        {
            var (cx, cy) = AxialToPixel(tile);
            for (var i = 0; i < 6; i++)
            {
                var n1 = tile + dirs[i];
                var n2 = tile + dirs[(i + 5) % 6]; // (i - 1 + 6) % 6
                var vkey = VertexKey.Create(tile, n1, n2);
                if (!vertexPositions.ContainsKey(vkey))
                {
                    var pos = HexCornerPixel(cx, cy, i);
                    vertexPositions[vkey] = pos;
                }

                // Edge: between corner i and corner (i+1)%6 of this tile.
                var n3 = tile + dirs[(i + 1) % 6];
                var vkey2 = VertexKey.Create(tile, n3, n1);
                var ekey = vkey.CompareTo(vkey2) <= 0 ? (vkey, vkey2) : (vkey2, vkey);
                edgeKeySet.Add(ekey);
            }
        }

        // Sort vertices by screen position (ascending y, then ascending x).
        var sortedVertices = vertexPositions
            .OrderBy(kv => kv.Value.Y)
            .ThenBy(kv => kv.Value.X)
            .Select(kv => kv.Key)
            .ToList();

        var vertexIndex = new Dictionary<VertexKey, int>(sortedVertices.Count);
        for (var i = 0; i < sortedVertices.Count; i++)
            vertexIndex[sortedVertices[i]] = i;

        // Sort edges by (min vertex index, max vertex index).
        var sortedEdges = edgeKeySet
            .Select(e => (A: vertexIndex[e.Item1], B: vertexIndex[e.Item2]))
            .Select(e => e.A <= e.B ? e : (e.B, e.A))
            .OrderBy(e => e.Item1)
            .ThenBy(e => e.Item2)
            .ToList();

        return (sortedVertices, vertexIndex, sortedEdges);
    }

    private static (double X, double Y) HexCornerPixel(double cx, double cy, int corner)
    {
        var angle = Math.PI / 180.0 * (60 * corner - 30);
        return (cx + Math.Cos(angle), cy + Math.Sin(angle));
    }

    // ── Adjacency building ──────────────────────────────────────────

    private static (
        ImmutableArray<ImmutableArray<int>> TileVertices,
        ImmutableArray<ImmutableArray<int>> TileEdges,
        ImmutableArray<ImmutableArray<int>> TileNeighbors,
        ImmutableArray<ImmutableArray<int>> VertexTiles,
        ImmutableArray<ImmutableArray<int>> VertexEdges,
        ImmutableArray<ImmutableArray<int>> VertexNeighbors,
        ImmutableArray<ImmutableArray<int>> EdgeTiles)
    BuildAdjacency(
        ImmutableArray<HexCoord> tiles,
        HashSet<HexCoord> tileSet,
        Dictionary<HexCoord, int> tileIndex,
        List<VertexKey> vertices,
        Dictionary<VertexKey, int> vertexIndex,
        List<(int A, int B)> edges)
    {
        var numTiles = tiles.Length;
        var numVertices = vertices.Count;
        var numEdges = edges.Count;

        // Vertex -> on-board tiles (from triplet).
        var vertexTilesBuilder = new List<int>[numVertices];
        for (var vi = 0; vi < numVertices; vi++)
        {
            var key = vertices[vi];
            var onBoard = new List<int>();
            foreach (var c in key.ToArray())
            {
                if (tileIndex.TryGetValue(c, out var ti))
                    onBoard.Add(ti);
            }
            onBoard.Sort();
            vertexTilesBuilder[vi] = onBoard;
        }

        // Tile -> vertices (which vertices reference this tile).
        var tileVertsBuilder = new List<int>[numTiles];
        for (var ti = 0; ti < numTiles; ti++)
            tileVertsBuilder[ti] = [];
        for (var vi = 0; vi < numVertices; vi++)
        {
            foreach (var ti in vertexTilesBuilder[vi])
                tileVertsBuilder[ti].Add(vi);
        }
        for (var ti = 0; ti < numTiles; ti++)
            tileVertsBuilder[ti].Sort();

        // Edge -> on-board tiles (shared coords between the two vertex triplets).
        var edgeTilesBuilder = new List<int>[numEdges];
        for (var ei = 0; ei < numEdges; ei++)
        {
            var (a, b) = edges[ei];
            var setA = vertices[a].ToArray().ToHashSet();
            var shared = vertices[b].ToArray().Where(c => setA.Contains(c));
            var onBoard = new List<int>();
            foreach (var c in shared)
            {
                if (tileIndex.TryGetValue(c, out var ti))
                    onBoard.Add(ti);
            }
            onBoard.Sort();
            edgeTilesBuilder[ei] = onBoard;
        }

        // Tile -> edges.
        var tileEdgesBuilder = new List<int>[numTiles];
        for (var ti = 0; ti < numTiles; ti++)
            tileEdgesBuilder[ti] = [];
        for (var ei = 0; ei < numEdges; ei++)
        {
            foreach (var ti in edgeTilesBuilder[ei])
                tileEdgesBuilder[ti].Add(ei);
        }
        for (var ti = 0; ti < numTiles; ti++)
            tileEdgesBuilder[ti].Sort();

        // Vertex -> edges.
        var vertexEdgesBuilder = new List<int>[numVertices];
        for (var vi = 0; vi < numVertices; vi++)
            vertexEdgesBuilder[vi] = [];
        for (var ei = 0; ei < numEdges; ei++)
        {
            var (a, b) = edges[ei];
            vertexEdgesBuilder[a].Add(ei);
            vertexEdgesBuilder[b].Add(ei);
        }
        for (var vi = 0; vi < numVertices; vi++)
            vertexEdgesBuilder[vi].Sort();

        // Vertex -> adjacent vertices.
        var vertexNeighborsBuilder = new List<int>[numVertices];
        for (var vi = 0; vi < numVertices; vi++)
            vertexNeighborsBuilder[vi] = [];
        for (var ei = 0; ei < numEdges; ei++)
        {
            var (a, b) = edges[ei];
            vertexNeighborsBuilder[a].Add(b);
            vertexNeighborsBuilder[b].Add(a);
        }
        for (var vi = 0; vi < numVertices; vi++)
            vertexNeighborsBuilder[vi].Sort();

        // Tile -> adjacent tiles (tiles sharing an interior edge).
        var tileNeighborsSet = new HashSet<int>[numTiles];
        for (var ti = 0; ti < numTiles; ti++)
            tileNeighborsSet[ti] = [];
        for (var ei = 0; ei < numEdges; ei++)
        {
            var ts = edgeTilesBuilder[ei];
            if (ts.Count == 2)
            {
                tileNeighborsSet[ts[0]].Add(ts[1]);
                tileNeighborsSet[ts[1]].Add(ts[0]);
            }
        }

        return (
            ToImmutable2D(tileVertsBuilder),
            ToImmutable2D(tileEdgesBuilder),
            ToImmutable2DFromSets(tileNeighborsSet),
            ToImmutable2D(vertexTilesBuilder),
            ToImmutable2D(vertexEdgesBuilder),
            ToImmutable2D(vertexNeighborsBuilder),
            ToImmutable2D(edgeTilesBuilder));
    }

    private static ImmutableArray<ImmutableArray<int>> ToImmutable2D(List<int>[] data)
    {
        var builder = ImmutableArray.CreateBuilder<ImmutableArray<int>>(data.Length);
        foreach (var list in data)
            builder.Add([.. list]);
        return builder.MoveToImmutable();
    }

    private static ImmutableArray<ImmutableArray<int>> ToImmutable2DFromSets(HashSet<int>[] data)
    {
        var builder = ImmutableArray.CreateBuilder<ImmutableArray<int>>(data.Length);
        foreach (var set in data)
        {
            var sorted = set.ToList();
            sorted.Sort();
            builder.Add([.. sorted]);
        }
        return builder.MoveToImmutable();
    }

    // ── Coastal edges ───────────────────────────────────────────────

    private static List<int> FindCoastalEdges(
        List<(int A, int B)> edges,
        ImmutableArray<ImmutableArray<int>> edgeTiles)
    {
        var coastal = new List<int>();
        for (var ei = 0; ei < edges.Count; ei++)
        {
            if (edgeTiles[ei].Length == 1)
                coastal.Add(ei);
        }
        return coastal;
    }

    // ── Port generation ─────────────────────────────────────────────

    private static List<(int, int)> GeneratePorts(
        int radius,
        List<VertexKey> vertices,
        Dictionary<VertexKey, int> vertexIndex,
        List<(int A, int B)> edges,
        ImmutableArray<ImmutableArray<int>> edgeTiles)
    {
        var numVertices = vertices.Count;

        // Find coastal edge indices.
        var coastalIndices = new List<int>();
        for (var ei = 0; ei < edges.Count; ei++)
        {
            if (edgeTiles[ei].Length == 1)
                coastalIndices.Add(ei);
        }

        // Build boundary adjacency: vertex -> [(neighbor, edgeIndex), ...].
        var boundaryAdj = new Dictionary<int, List<(int Neighbor, int Edge)>>();
        foreach (var ei in coastalIndices)
        {
            var (a, b) = edges[ei];
            if (!boundaryAdj.ContainsKey(a)) boundaryAdj[a] = [];
            if (!boundaryAdj.ContainsKey(b)) boundaryAdj[b] = [];
            boundaryAdj[a].Add((b, ei));
            boundaryAdj[b].Add((a, ei));
        }

        // Compute vertex pixel positions for sorting.
        // Use a unit-size hex corner pixel to get relative positions.
        var vertexPos = new (double X, double Y)[numVertices];
        for (var vi = 0; vi < numVertices; vi++)
        {
            // Use the first tile in the triplet and find which corner this is.
            // Simpler: compute pixel position from the average of the three tile centers.
            var key = vertices[vi];
            var coords = key.ToArray();
            double sx = 0, sy = 0;
            foreach (var c in coords)
            {
                var (px, py) = AxialToPixel(c);
                sx += px;
                sy += py;
            }
            vertexPos[vi] = (sx / 3.0, sy / 3.0);
        }

        // Start from the topmost (then leftmost) boundary vertex.
        var start = boundaryAdj.Keys
            .OrderBy(v => vertexPos[v].Y)
            .ThenBy(v => vertexPos[v].X)
            .First();

        // Walk clockwise: pick the rightmost (most positive x) neighbor first.
        var visitedEdges = new HashSet<int>();
        var ring = new List<int> { start };

        var startNeighbors = boundaryAdj[start]
            .OrderByDescending(ne => vertexPos[ne.Neighbor].X)
            .ToList();
        var (nextV, nextE) = startNeighbors[0];
        visitedEdges.Add(nextE);
        ring.Add(nextV);
        var current = nextV;

        while (current != start)
        {
            foreach (var (nv, ne) in boundaryAdj[current])
            {
                if (!visitedEdges.Contains(ne))
                {
                    visitedEdges.Add(ne);
                    ring.Add(nv);
                    current = nv;
                    break;
                }
            }
        }

        // Remove duplicate start vertex at the end.
        ring.RemoveAt(ring.Count - 1);

        // Build ordered ring edges.
        var ringEdges = new List<(int A, int B)>(ring.Count);
        for (var i = 0; i < ring.Count; i++)
            ringEdges.Add((ring[i], ring[(i + 1) % ring.Count]));

        // Select evenly-spaced port positions along the ring.
        var total = ringEdges.Count;
        var nports = 3 * (radius + 1);
        var ports = new List<(int, int)>(nports);
        for (var i = 0; i < nports; i++)
        {
            var pos = (1 + i * total / nports) % total;
            ports.Add(ringEdges[pos]);
        }

        return ports;
    }
}
