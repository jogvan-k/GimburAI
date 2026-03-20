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
/// Immutable board topology for a hex grid.
/// Computes tile positions, vertices, edges, ports, and all adjacency lookups.
/// Supports both circular boards (defined by radius) and arbitrary tile sets.
/// </summary>
/// <remarks>
/// All indices follow the ordering defined in docs/topology-reference.md:
/// tiles, vertices sorted by screen position (ascending y, then ascending x);
/// edges sorted by (min vertex index, max vertex index);
/// ports ordered clockwise from the top.
/// </remarks>
public sealed class BoardTopology
{
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

    /// <summary>
    /// Per-vertex flag: <c>true</c> if the vertex is a peak (top of hex),
    /// <c>false</c> if it is a valley (bottom of hex). Peak vertices have
    /// edge directions N/SW/SE; valley vertices have S/NW/NE.
    /// </summary>
    public ImmutableArray<bool> IsPeakVertex { get; }

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
    public static BoardTopology Standard { get; } = FromRadius(2);

    /// <summary>Mini 7-tile board (radius 1).</summary>
    public static BoardTopology Mini { get; } = FromRadius(1);

    /// <summary>
    /// Small 10-tile board: two central hexes (0,0) and (1,0) with one layer
    /// of hexes around them. Non-circular oval shape.
    /// </summary>
    public static BoardTopology Small { get; } = FromTiles(GenerateSmallTileCoords(), portCount: 6);

    // ── Factory methods ─────────────────────────────────────────────

    /// <summary>
    /// Creates a circular board topology of the given radius.
    /// Radius 1 = 7 tiles (mini), radius 2 = 19 tiles (standard).
    /// Port count is computed as 3 * (radius + 1).
    /// </summary>
    public static BoardTopology FromRadius(int radius)
    {
        if (radius < 1)
            throw new ArgumentOutOfRangeException(nameof(radius), radius, "Radius must be >= 1");

        var tileCoords = GenerateCircularTileCoords(radius);
        return new BoardTopology(tileCoords, portCount: 3 * (radius + 1));
    }

    /// <summary>
    /// Creates a board topology from an explicit set of tile coordinates.
    /// </summary>
    public static BoardTopology FromTiles(IEnumerable<HexCoord> tiles, int portCount)
    {
        return new BoardTopology(tiles.ToList(), portCount);
    }

    // ── Constructor ─────────────────────────────────────────────────

    private BoardTopology(List<HexCoord> tileCoords, int portCount)
    {
        if (tileCoords.Count == 0)
            throw new ArgumentException("Must provide at least one tile", nameof(tileCoords));

        // 1) Sort tile coordinates by screen position.
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
        Ports = [.. GeneratePorts(portCount, vertexList, vertexIndex, edgeList, edgeTiles)];

        // 6) Compute peak/valley classification for each vertex.
        IsPeakVertex = ComputePeakValley(vertexList);
    }

    // ── Tile generation ─────────────────────────────────────────────

    /// <summary>
    /// Generates tile coordinates for a circular hex grid of the given radius.
    /// </summary>
    private static List<HexCoord> GenerateCircularTileCoords(int radius)
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

    /// <summary>
    /// Generates tile coordinates for the small oval board: two central hexes
    /// (0,0) and (1,0) with one layer of hexes around them, totalling 10 tiles.
    /// </summary>
    private static List<HexCoord> GenerateSmallTileCoords()
    {
        var centers = new[] { new HexCoord(0, 0), new HexCoord(1, 0) };
        var tileSet = new HashSet<HexCoord>();
        foreach (var center in centers)
        {
            tileSet.Add(center);
            foreach (var dir in HexCoord.Directions)
                tileSet.Add(center + dir);
        }
        return [.. tileSet];
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
        int portCount,
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
        var ports = new List<(int, int)>(portCount);
        for (var i = 0; i < portCount; i++)
        {
            var pos = (1 + i * total / portCount) % total;
            ports.Add(ringEdges[pos]);
        }

        return ports;
    }

    // ── Peak / valley classification ────────────────────────────────

    /// <summary>
    /// Classifies each vertex as peak or valley by comparing the vertex pixel Y
    /// to the average Y of its three surrounding hex centers. In the pointy-top
    /// coordinate system (Y increasing upward), peak vertices sit above the
    /// average center Y and valley vertices sit below.
    /// </summary>
    private static ImmutableArray<bool> ComputePeakValley(List<VertexKey> vertices)
    {
        var builder = ImmutableArray.CreateBuilder<bool>(vertices.Count);
        for (var vi = 0; vi < vertices.Count; vi++)
        {
            var key = vertices[vi];
            var coords = key.ToArray();
            double sumY = 0;
            foreach (var c in coords)
            {
                var (_, py) = AxialToPixel(c);
                sumY += py;
            }
            var avgY = sumY / 3.0;

            // Compute vertex pixel position (average of three hex-center positions).
            // But we need the actual vertex position, not the hex-center average.
            // Use the first hex and find which corner this vertex corresponds to.
            // Simpler: compute from any tile in the triplet.
            double vx = 0, vy = 0;
            foreach (var c in coords)
            {
                var (px, py) = AxialToPixel(c);
                vx += px;
                vy += py;
            }
            vx /= 3.0;
            vy /= 3.0;

            // For pointy-top hexes with Y increasing upward (-1.5*r),
            // peak vertices have pixel Y > average hex center Y.
            // However, the vertex pixel position computed as average of 3 hex centers
            // IS the vertex position (each vertex is equidistant from its 3 hexes).
            // We need a different approach: use the hex corner geometry.
            //
            // Alternative: a peak vertex has its Y coordinate higher than the Y of
            // the hex center it belongs to, looking at the offset from any single
            // adjacent hex center.
            //
            // Simplest approach: check the sign of (vertexY - hexCenterY) for any
            // adjacent on-board tile. For a peak vertex (top of hex), the vertex
            // is above the hex center. For a valley vertex, below.
            //
            // Actually the simplest: in axial coords with pointy-top, the 6 corners
            // alternate peak and valley. Corner 0 (30°) is a valley, corner 1 (90°)
            // is a peak, corner 2 (150°) is a valley, etc. But this depends on
            // which corner index maps to which vertex.
            //
            // Even simpler: for any adjacent hex, compute the corner pixel and compare
            // its Y to the hex center Y. If cornerY > centerY, it's a peak vertex.
            // Let's use the first coordinate in the triplet.
            var (cx, cy) = AxialToPixel(coords[0]);

            // Find which corner of coords[0] this vertex corresponds to.
            // The vertex is at the intersection of coords[0], coords[1], coords[2].
            // Each pair of adjacent coords defines a shared edge, and the vertex is
            // at the corner where all three meet.
            // Instead of complex corner detection, just compute the actual pixel position.
            var dirs = HexCoord.Directions;
            bool found = false;
            bool isPeak = false;
            for (var i = 0; i < 6; i++)
            {
                var n1 = coords[0] + dirs[i];
                var n2 = coords[0] + dirs[(i + 5) % 6];
                var candidate = VertexKey.Create(coords[0], n1, n2);
                if (candidate == key)
                {
                    var (_, cornerY) = HexCornerPixel(cx, cy, i);
                    isPeak = cornerY > cy;
                    found = true;
                    break;
                }
            }

            if (!found)
                throw new InvalidOperationException(
                    $"Could not classify vertex {vi} as peak or valley.");

            builder.Add(isPeak);
        }

        return builder.MoveToImmutable();
    }
}
