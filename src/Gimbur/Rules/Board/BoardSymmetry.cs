using System.Collections.Immutable;
using System.Text;

namespace Gimbur.Rules;

/// <summary>
/// A precomputed permutation table for a single symmetry operation.
/// Each array maps old index → new index: element at old position i moves to position Tiles[i], etc.
/// </summary>
public sealed class SymmetryPermutation
{
    /// <summary>Old tile index → new tile index.</summary>
    public ImmutableArray<int> Tiles { get; }

    /// <summary>Old vertex index → new vertex index.</summary>
    public ImmutableArray<int> Vertices { get; }

    /// <summary>Old edge index → new edge index.</summary>
    public ImmutableArray<int> Edges { get; }

    /// <summary>Old port index → new port index.</summary>
    public ImmutableArray<int> Ports { get; }

    /// <summary>Human-readable label (e.g. "rot120", "reflect_q").</summary>
    public string Label { get; }

    internal SymmetryPermutation(
        ImmutableArray<int> tiles,
        ImmutableArray<int> vertices,
        ImmutableArray<int> edges,
        ImmutableArray<int> ports,
        string label)
    {
        Tiles = tiles;
        Vertices = vertices;
        Edges = edges;
        Ports = ports;
        Label = label;
    }
}

/// <summary>
/// Computes board symmetry permutations for hex-based Catan boards.
/// <para>
/// <b>Mini map (radius 1)</b>: C6 rotational symmetry (order 6) — 5 non-trivial permutations.
/// The hex grid has full D6, but reflections reverse the boundary ring direction, mapping
/// port positions {1,4,7,10,13,16} to non-port positions {0,3,6,9,12,15}. Only rotations
/// preserve the port set.
/// </para>
/// <para>
/// <b>Standard map (radius 2)</b>: C3 rotational symmetry only (order 3) — 2 non-trivial
/// permutations (120° and 240° CW). Port positions break full D6 down to C3 because the
/// 9 ports on 30 coastal edges have gap pattern 3,3,4 repeating 3×, which is only invariant
/// under 120° rotation.
/// </para>
/// <para>
/// <b>Small map (10-tile oval)</b>: C2 rotational symmetry (order 2) — 1 non-trivial
/// permutation (180° rotation). The oval has two central hexes (0,0) and (1,0); the 180°
/// rotation about their midpoint maps (q,r) → (1−q,−r). The 6 ports on 22 coastal edges
/// have gap pattern 3,4,4,3,4,4 which is invariant under the half-ring shift (+11 mod 22),
/// making the port set 180°-symmetric.
/// </para>
/// </summary>
public static class BoardSymmetry
{
    // ── Coordinate transforms ───────────────────────────────────────

    /// <summary>All 6 rotations as (q,r) → (q',r') transforms.</summary>
    private static readonly Func<HexCoord, HexCoord>[] Rotations =
    [
        c => c,                                         // 0° (identity)
        c => new HexCoord(-c.R, c.Q + c.R),             // 60° CW
        c => new HexCoord(-c.Q - c.R, c.Q),             // 120° CW
        c => new HexCoord(-c.Q, -c.R),                  // 180°
        c => new HexCoord(c.R, -c.Q - c.R),             // 240° CW
        c => new HexCoord(c.Q + c.R, -c.Q),             // 300° CW
    ];

    // ── Public API ──────────────────────────────────────────────────

    /// <summary>
    /// Returns the list of non-trivial symmetry permutations for the given topology.
    /// Returns empty for unsupported topologies.
    /// </summary>
    public static ImmutableArray<SymmetryPermutation> GetPermutations(BoardTopology topology)
    {
        if (topology == BoardTopology.Mini)
            return GetC6Permutations(topology);

        if (topology == BoardTopology.Standard)
            return GetC3Permutations(topology);

        if (topology == BoardTopology.Small)
            return GetC2Permutations(topology);

        return [];
    }

    /// <summary>
    /// Returns 2 non-trivial C3 rotational permutations (120° and 240° CW).
    /// </summary>
    private static ImmutableArray<SymmetryPermutation> GetC3Permutations(BoardTopology topology)
    {
        var result = ImmutableArray.CreateBuilder<SymmetryPermutation>(2);

        // 120° CW (index 2) and 240° CW (index 4)
        result.Add(ComputePermutation(topology, Rotations[2], "rot120"));
        result.Add(ComputePermutation(topology, Rotations[4], "rot240"));

        return result.MoveToImmutable();
    }

    /// <summary>
    /// Returns 1 non-trivial C2 rotational permutation (180° rotation about the midpoint
    /// of the two central hexes). The small map's oval shape has no higher-order rotational
    /// symmetry, but the port layout (gap pattern 3,4,4,3,4,4) is invariant under 180°.
    /// </summary>
    private static ImmutableArray<SymmetryPermutation> GetC2Permutations(BoardTopology topology)
    {
        // 180° rotation about the midpoint of (0,0) and (1,0): (q,r) → (1−q, −r)
        return [ComputePermutation(topology, c => new HexCoord(1 - c.Q, -c.R), "rot180")];
    }

    /// <summary>
    /// Returns 5 non-trivial C6 rotational permutations (60° through 300° CW).
    /// Reflections are excluded because they reverse the boundary ring direction,
    /// mapping port positions to non-port coastal edges.
    /// </summary>
    private static ImmutableArray<SymmetryPermutation> GetC6Permutations(BoardTopology topology)
    {
        var result = ImmutableArray.CreateBuilder<SymmetryPermutation>(5);

        // 5 non-trivial rotations (skip identity at index 0)
        for (var i = 1; i < 6; i++)
        {
            var label = $"rot{i * 60}";
            result.Add(ComputePermutation(topology, Rotations[i], label));
        }

        return result.MoveToImmutable();
    }

    // ── Permutation computation ─────────────────────────────────────

    private static SymmetryPermutation ComputePermutation(
        BoardTopology topology,
        Func<HexCoord, HexCoord> transform,
        string label)
    {
        // 1) Build coordinate → tile index lookup for the topology.
        var coordToTile = new Dictionary<HexCoord, int>(topology.TileCount);
        for (var ti = 0; ti < topology.TileCount; ti++)
            coordToTile[topology.Tiles[ti]] = ti;

        // 2) Compute tile permutation: for each old tile index, where does it go?
        var tilePerm = new int[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            var newCoord = transform(topology.Tiles[ti]);
            tilePerm[ti] = coordToTile[newCoord];
        }

        // 3) Build vertex key → vertex index lookup.
        var keyToVertex = new Dictionary<VertexKey, int>(topology.VertexCount);
        for (var vi = 0; vi < topology.VertexCount; vi++)
            keyToVertex[topology.Vertices[vi]] = vi;

        // 4) Compute vertex permutation.
        var vertexPerm = new int[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var oldKey = topology.Vertices[vi];
            var newKey = VertexKey.Create(
                transform(oldKey.A),
                transform(oldKey.B),
                transform(oldKey.C));
            vertexPerm[vi] = keyToVertex[newKey];
        }

        // 5) Build edge lookup: (minVertex, maxVertex) → edge index.
        var endpointsToEdge = new Dictionary<(int, int), int>(topology.EdgeCount);
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            var (a, b) = topology.Edges[ei];
            var key = a <= b ? (a, b) : (b, a);
            endpointsToEdge[key] = ei;
        }

        // 6) Compute edge permutation.
        var edgePerm = new int[topology.EdgeCount];
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            var (oldA, oldB) = topology.Edges[ei];
            var newA = vertexPerm[oldA];
            var newB = vertexPerm[oldB];
            var key = newA <= newB ? (newA, newB) : (newB, newA);
            edgePerm[ei] = endpointsToEdge[key];
        }

        // 7) Build port lookup: (minVertex, maxVertex) → port index.
        var portEndpointsToPort = new Dictionary<(int, int), int>(topology.PortCount);
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            var (a, b) = topology.Ports[pi];
            var key = a <= b ? (a, b) : (b, a);
            portEndpointsToPort[key] = pi;
        }

        // 8) Compute port permutation.
        var portPerm = new int[topology.PortCount];
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            var (oldA, oldB) = topology.Ports[pi];
            var newA = vertexPerm[oldA];
            var newB = vertexPerm[oldB];
            var key = newA <= newB ? (newA, newB) : (newB, newA);
            portPerm[pi] = portEndpointsToPort[key];
        }

        return new SymmetryPermutation(
            [.. tilePerm],
            [.. vertexPerm],
            [.. edgePerm],
            [.. portPerm],
            label);
    }

    // ── Applying permutations to serialized strings ─────────────────

    /// <summary>
    /// Applies a symmetry permutation to a serialized board string.
    /// Board format: "{tile_chars}|{port_chars}" where tiles are 3 chars each
    /// (resource + pips + side) and ports are 1 char each.
    /// </summary>
    public static string PermuteBoard(string boardSerialized, SymmetryPermutation perm)
    {
        var sections = boardSerialized.Split('|');
        if (sections.Length != 2)
            throw new ArgumentException(
                $"Board string has {sections.Length} sections, expected 2.", nameof(boardSerialized));

        var tileSection = sections[0];
        var portSection = sections[1];

        var tileCount = perm.Tiles.Length;
        var portCount = perm.Ports.Length;

        if (tileSection.Length != tileCount * 3)
            throw new ArgumentException(
                $"Tile section has {tileSection.Length} chars, expected {tileCount * 3}.");
        if (portSection.Length != portCount)
            throw new ArgumentException(
                $"Port section has {portSection.Length} chars, expected {portCount}.");

        // Build inverse permutation: newIndex → oldIndex.
        // We need: "what was at old position i is now at new position perm[i]"
        // So for the output at position perm[i], we read from input position i.
        // Equivalently, output[perm[i]] = input[i], or using inverse: output[j] = input[inv[j]].
        var tileInv = InvertPermutation(perm.Tiles);
        var portInv = InvertPermutation(perm.Ports);

        var sb = new StringBuilder(boardSerialized.Length);

        // Permute tiles (3 chars each: resource + pips + side)
        for (var ti = 0; ti < tileCount; ti++)
        {
            var srcTile = tileInv[ti];
            sb.Append(tileSection[srcTile * 3]);
            sb.Append(tileSection[srcTile * 3 + 1]);
            sb.Append(tileSection[srcTile * 3 + 2]);
        }

        sb.Append('|');

        // Permute ports (1 char each)
        for (var pi = 0; pi < portCount; pi++)
        {
            sb.Append(portSection[portInv[pi]]);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Applies a symmetry permutation to a serialized state-only string.
    /// State format: robber|playerStage|longestLargest|vertices|edges|resources|knights|devCards
    /// (8 sections separated by '|').
    /// <para>
    /// Only the robber tile index (section 0), vertices (section 3), and edges (section 4)
    /// are position-dependent and need permutation. All other sections are scalar or
    /// per-player data that is position-independent.
    /// </para>
    /// </summary>
    public static string PermuteState(string stateSerialized, SymmetryPermutation perm)
    {
        var sections = stateSerialized.Split('|');
        if (sections.Length != 8)
            throw new ArgumentException(
                $"State string has {sections.Length} sections, expected 8.", nameof(stateSerialized));

        var vertexCount = perm.Vertices.Length;
        var edgeCount = perm.Edges.Length;

        // Validate vertex section (2 chars per vertex) and edge section lengths.
        if (sections[3].Length != vertexCount * 2)
            throw new ArgumentException(
                $"Vertex section has {sections[3].Length} chars, expected {vertexCount * 2}.");
        if (sections[4].Length != edgeCount)
            throw new ArgumentException(
                $"Edge section has {sections[4].Length} chars, expected {edgeCount}.");

        // Build inverse permutations.
        var tileInv = InvertPermutation(perm.Tiles);
        var vertexInv = InvertPermutation(perm.Vertices);
        var edgeInv = InvertPermutation(perm.Edges);

        var sb = new StringBuilder(stateSerialized.Length);

        // Section 0: Robber tile index (single base-32 char)
        var oldRobber = CrockfordBase32.Decode(sections[0][0]);
        sb.Append(CrockfordBase32.Encode(perm.Tiles[oldRobber]));

        // Section 1: Current Turn (player + stage) — unchanged
        sb.Append('|');
        sb.Append(sections[1]);

        // Section 2: Longest Road / Largest Army — unchanged
        sb.Append('|');
        sb.Append(sections[2]);

        // Section 3: Vertices — permute (2 chars per vertex: building + player)
        sb.Append('|');
        for (var vi = 0; vi < vertexCount; vi++)
        {
            var srcVertex = vertexInv[vi];
            sb.Append(sections[3][srcVertex * 2]);
            sb.Append(sections[3][srcVertex * 2 + 1]);
        }

        // Section 4: Edges — permute (1 char per edge: player ID)
        sb.Append('|');
        for (var ei = 0; ei < edgeCount; ei++)
        {
            sb.Append(sections[4][edgeInv[ei]]);
        }

        // Sections 5-7: Resources, Knights, DevCards — unchanged (per-player, not positional)
        sb.Append('|');
        sb.Append(sections[5]);
        sb.Append('|');
        sb.Append(sections[6]);
        sb.Append('|');
        sb.Append(sections[7]);

        return sb.ToString();
    }


    /// <summary>
    /// Applies a symmetry permutation to a serialized placement-phase state string.
    /// Placement format: tiles|ports|placementVertices|edges (4 sections).
    /// <para>
    /// Tiles (section 0, 3 chars each), ports (section 1, 1 char each), vertices
    /// (section 2, 2 chars each: placementNumber + player), and edges (section 3,
    /// 1 char each) are all position-dependent and need permutation.
    /// </para>
    /// </summary>
    public static string PermutePlacementState(string placementSerialized, SymmetryPermutation perm)
    {
        var sections = placementSerialized.Split('|');
        if (sections.Length != 4)
            throw new ArgumentException(
                $"Placement string has {sections.Length} sections, expected 4.",
                nameof(placementSerialized));

        var tileCount = perm.Tiles.Length;
        var portCount = perm.Ports.Length;
        var vertexCount = perm.Vertices.Length;
        var edgeCount = perm.Edges.Length;

        if (sections[0].Length != tileCount * 3)
            throw new ArgumentException(
                $"Tile section has {sections[0].Length} chars, expected {tileCount * 3}.");
        if (sections[1].Length != portCount)
            throw new ArgumentException(
                $"Port section has {sections[1].Length} chars, expected {portCount}.");
        if (sections[2].Length != vertexCount * 2)
            throw new ArgumentException(
                $"Vertex section has {sections[2].Length} chars, expected {vertexCount * 2}.");
        if (sections[3].Length != edgeCount)
            throw new ArgumentException(
                $"Edge section has {sections[3].Length} chars, expected {edgeCount}.");

        var tileInv = InvertPermutation(perm.Tiles);
        var portInv = InvertPermutation(perm.Ports);
        var vertexInv = InvertPermutation(perm.Vertices);
        var edgeInv = InvertPermutation(perm.Edges);

        var sb = new StringBuilder(placementSerialized.Length);

        // Section 0: Tiles (3 chars each: resource + pips + side)
        for (var ti = 0; ti < tileCount; ti++)
        {
            var src = tileInv[ti];
            sb.Append(sections[0][src * 3]);
            sb.Append(sections[0][src * 3 + 1]);
            sb.Append(sections[0][src * 3 + 2]);
        }

        sb.Append('|');

        // Section 1: Ports (1 char each)
        for (var pi = 0; pi < portCount; pi++)
        {
            sb.Append(sections[1][portInv[pi]]);
        }

        sb.Append('|');

        // Section 2: Placement vertices (2 chars each: placementNumber + player)
        for (var vi = 0; vi < vertexCount; vi++)
        {
            var src = vertexInv[vi];
            sb.Append(sections[2][src * 2]);
            sb.Append(sections[2][src * 2 + 1]);
        }

        sb.Append('|');

        // Section 3: Edges (1 char each: player ID)
        for (var ei = 0; ei < edgeCount; ei++)
        {
            sb.Append(sections[3][edgeInv[ei]]);
        }

        return sb.ToString();
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /// <summary>
    /// Computes the inverse of a permutation array.
    /// If perm[i] = j, then inverse[j] = i.
    /// </summary>
    private static int[] InvertPermutation(ImmutableArray<int> perm)
    {
        var inv = new int[perm.Length];
        for (var i = 0; i < perm.Length; i++)
            inv[perm[i]] = i;
        return inv;
    }
}
