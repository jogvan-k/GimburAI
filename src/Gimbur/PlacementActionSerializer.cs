using System.Collections.Immutable;
using Gimbur.Rules;

namespace Gimbur;

/// <summary>
/// Serializes placement actions as "<vertex_index><road_direction>" strings
/// (e.g., "3S", "12NW", "53NE") and maintains a fixed vocabulary of all valid
/// (vertex, direction) combinations for a given board topology.
/// </summary>
/// <remarks>
/// Each edge can be described from either endpoint, producing 2 action strings
/// per edge. Vocabulary sizes: mini=60, small=82, standard=144.
/// </remarks>
public sealed class PlacementActionSerializer
{
    /// <summary>
    /// Describes a single entry in the action vocabulary: a settlement vertex
    /// plus the compass direction of the road built from that vertex.
    /// </summary>
    public readonly record struct ActionEntry(int Vertex, string Direction, int Edge);

    /// <summary>All valid (vertex, direction) pairs, sorted by vertex then direction.</summary>
    public ImmutableArray<ActionEntry> Vocabulary { get; }

    /// <summary>Maps action string (e.g. "3S") to vocabulary index.</summary>
    private readonly Dictionary<string, int> _actionToIndex;

    /// <summary>Maps (vertex, edge) pair to vocabulary index.</summary>
    private readonly Dictionary<(int Vertex, int Edge), int> _vertexEdgeToIndex;

    public int VocabularySize => Vocabulary.Length;

    /// <summary>
    /// Stable string key identifying this serializer's underlying topology,
    /// suitable for use as a pool/dictionary key. Distinguishes the three
    /// canonical topologies (Mini/Small/Standard) by vocabulary size.
    /// </summary>
    public string TopologyKey => $"vocab{VocabularySize}";

    private PlacementActionSerializer(ImmutableArray<ActionEntry> vocabulary)
    {
        Vocabulary = vocabulary;
        _actionToIndex = new Dictionary<string, int>(vocabulary.Length);
        _vertexEdgeToIndex = new Dictionary<(int, int), int>(vocabulary.Length);
        for (var i = 0; i < vocabulary.Length; i++)
        {
            var entry = vocabulary[i];
            _actionToIndex[$"{entry.Vertex}{entry.Direction}"] = i;
            _vertexEdgeToIndex[(entry.Vertex, entry.Edge)] = i;
        }
    }

    /// <summary>
    /// Serializes a placement action (settlement vertex + road edge) as a string.
    /// </summary>
    public string Serialize(int vertex, int edge)
    {
        var entry = Vocabulary[_vertexEdgeToIndex[(vertex, edge)]];
        return $"{entry.Vertex}{entry.Direction}";
    }

    /// <summary>
    /// Returns the vocabulary index for a given action string.
    /// </summary>
    public int IndexOf(string action) => _actionToIndex[action];

    /// <summary>
    /// Returns the vocabulary index for a given (vertex, edge) pair.
    /// </summary>
    public int IndexOf(int vertex, int edge) => _vertexEdgeToIndex[(vertex, edge)];

    /// <summary>Validates that a dense policy exactly matches this vocabulary.</summary>
    public bool IsValidDensePolicy(IReadOnlyList<double> policy) =>
        policy.Count == VocabularySize
        && policy.All(value => double.IsFinite(value) && value >= 0.0);

    /// <summary>
    /// Masks a dense vocabulary policy to legal composite indices and normalizes it.
    /// Invalid or zero-mass policies produce a uniform legal policy.
    /// </summary>
    public double[] MaskAndNormalize(
        IReadOnlyList<double> policy,
        IReadOnlyList<int> legalVocabularyIndices)
    {
        var result = new double[legalVocabularyIndices.Count];
        if (result.Length == 0)
            return result;

        if (IsValidDensePolicy(policy))
        {
            for (var i = 0; i < result.Length; i++)
                result[i] = policy[legalVocabularyIndices[i]];
        }

        NormalizeOrUniform(result);
        return result;
    }

    /// <summary>
    /// Marginalizes a dense composite policy into settlement-action order.
    /// Each group contains the legal composite vocabulary indices for one settlement.
    /// </summary>
    public double[] SettlementMarginals(
        IReadOnlyList<double> policy,
        IReadOnlyList<int[]> settlementCompositeIndices)
    {
        var result = new double[settlementCompositeIndices.Count];
        if (IsValidDensePolicy(policy))
        {
            for (var i = 0; i < result.Length; i++)
            {
                foreach (var index in settlementCompositeIndices[i])
                    result[i] += policy[index];
            }
        }

        NormalizeOrUniform(result);
        return result;
    }

    /// <summary>
    /// Masks a dense policy to all legal composite groups while retaining dense
    /// vocabulary indexing. Legal mass is normalized globally; illegal entries are zero.
    /// </summary>
    public double[] LegalDensePolicy(
        IReadOnlyList<double> policy,
        IReadOnlyList<int[]> settlementCompositeIndices)
    {
        var result = new double[VocabularySize];
        var legalIndices = settlementCompositeIndices.SelectMany(indices => indices).ToArray();
        if (legalIndices.Length == 0)
            return result;

        if (IsValidDensePolicy(policy))
        {
            foreach (var index in legalIndices)
                result[index] = policy[index];
        }

        var legalValues = legalIndices.Select(index => result[index]).ToArray();
        NormalizeOrUniform(legalValues);
        for (var i = 0; i < legalIndices.Length; i++)
            result[legalIndices[i]] = legalValues[i];
        return result;
    }

    private static void NormalizeOrUniform(double[] values)
    {
        if (values.Length == 0)
            return;

        var total = values.Sum();
        if (double.IsFinite(total) && total > 0.0)
        {
            for (var i = 0; i < values.Length; i++)
                values[i] /= total;
            return;
        }

        Array.Fill(values, 1.0 / values.Length);
    }

    // â”€â”€ Precomputed instances â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    public static PlacementActionSerializer Mini { get; } = Create(BoardTopology.Mini);
    public static PlacementActionSerializer Small { get; } = Create(BoardTopology.Small);
    public static PlacementActionSerializer Standard { get; } = Create(BoardTopology.Standard);

    /// <summary>
    /// Returns the precomputed instance matching the given topology.
    /// </summary>
    public static PlacementActionSerializer ForTopology(BoardTopology topology)
    {
        if (ReferenceEquals(topology, BoardTopology.Mini)) return Mini;
        if (ReferenceEquals(topology, BoardTopology.Small)) return Small;
        if (ReferenceEquals(topology, BoardTopology.Standard)) return Standard;
        return Create(topology);
    }

    /// <summary>
    /// Creates an action serializer for the given topology by computing all
    /// valid (vertex, direction) pairs from the edge list and peak/valley
    /// classification.
    /// </summary>
    public static PlacementActionSerializer Create(BoardTopology topology)
    {
        // Precompute actual pixel positions for each vertex using hex-corner geometry.
        var vertexPositions = ComputeVertexPositions(topology);
        var entries = new List<ActionEntry>();

        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            var (va, vb) = topology.Edges[ei];

            // Each edge produces 2 entries: one from each endpoint.
            entries.Add(new ActionEntry(va, GetDirection(topology, vertexPositions, va, vb), ei));
            entries.Add(new ActionEntry(vb, GetDirection(topology, vertexPositions, vb, va), ei));
        }

        // Sort by vertex index, then by direction string.
        entries.Sort((a, b) =>
        {
            var cmp = a.Vertex.CompareTo(b.Vertex);
            return cmp != 0 ? cmp : string.Compare(a.Direction, b.Direction, StringComparison.Ordinal);
        });

        return new PlacementActionSerializer([.. entries]);
    }

    /// <summary>
    /// Determines the compass direction from <paramref name="from"/> to
    /// <paramref name="to"/> based on peak/valley classification.
    /// </summary>
    private static string GetDirection(
        BoardTopology topology, (double X, double Y)[] positions, int from, int to)
    {
        var isPeak = topology.IsPeakVertex[from];
        var (fx, fy) = positions[from];
        var (tx, ty) = positions[to];

        var dx = tx - fx;
        var dy = ty - fy;

        // The pixel coordinate system uses y = -1.5 * r, so more negative Y
        // corresponds to "up" (north) on the visual board. Compass directions
        // follow the visual convention: N = toward more negative Y.
        if (isPeak)
        {
            // Peak vertex: N (up), SW (down-left), SE (down-right)
            if (dy < 0)
                return "N";     // neighbor has more negative Y (visually above)
            else if (dx < 0)
                return "SW";    // neighbor is visually below-left
            else
                return "SE";    // neighbor is visually below-right
        }
        else
        {
            // Valley vertex: S (down), NW (up-left), NE (up-right)
            if (dy > 0)
                return "S";     // neighbor has more positive Y (visually below)
            else if (dx < 0)
                return "NW";    // neighbor is visually above-left
            else
                return "NE";    // neighbor is visually above-right
        }
    }

    /// <summary>
    /// Computes the actual pixel position of each vertex using the hex-corner
    /// geometry. This gives the true corner position rather than the centroid
    /// of the three surrounding hex centers (which is incorrect for boundary
    /// vertices with off-board hexes).
    /// </summary>
    private static (double X, double Y)[] ComputeVertexPositions(BoardTopology topology)
    {
        var dirs = HexCoord.Directions;
        var positions = new (double X, double Y)[topology.VertexCount];

        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var key = topology.Vertices[vi];
            var coords = key.ToArray();
            var (cx, cy) = BoardTopology.AxialToPixel(coords[0]);

            bool found = false;
            for (var i = 0; i < 6; i++)
            {
                var n1 = coords[0] + dirs[i];
                var n2 = coords[0] + dirs[(i + 5) % 6];
                var candidate = VertexKey.Create(coords[0], n1, n2);
                if (candidate == key)
                {
                    var angle = Math.PI / 180.0 * (60 * i - 30);
                    positions[vi] = (cx + Math.Cos(angle), cy + Math.Sin(angle));
                    found = true;
                    break;
                }
            }

            if (!found)
                throw new InvalidOperationException(
                    $"Could not find corner index for vertex {vi}.");
        }

        return positions;
    }
}
