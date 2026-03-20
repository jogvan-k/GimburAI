using Gimbur;
using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class PlacementSerializationTests
{
    // == StateToken.EncodePlacementNumber / DecodePlacementNumber ====

    [TestCase(0, '.')]
    [TestCase(1, 'a')]
    [TestCase(2, 'b')]
    public void EncodePlacementNumber_RoundTrips(int number, char expected)
    {
        var encoded = StateToken.EncodePlacementNumber(number);
        Assert.That(encoded, Is.EqualTo(expected));
        Assert.That(StateToken.DecodePlacementNumber(encoded), Is.EqualTo(number));
    }

    [Test]
    public void EncodePlacementNumber_InvalidValue_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => StateToken.EncodePlacementNumber(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => StateToken.EncodePlacementNumber(-1));
    }

    [Test]
    public void DecodePlacementNumber_InvalidChar_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => StateToken.DecodePlacementNumber('x'));
        Assert.Throws<ArgumentOutOfRangeException>(() => StateToken.DecodePlacementNumber('c'));
    }

    // == BoardTopology.IsPeakVertex =================================

    [Test]
    public void Mini_IsPeakVertex_Has12Peaks12Valleys()
    {
        var topo = BoardTopology.Mini;
        Assert.That(topo.IsPeakVertex.Length, Is.EqualTo(topo.VertexCount));

        var peakCount = topo.IsPeakVertex.Count(v => v);
        var valleyCount = topo.IsPeakVertex.Count(v => !v);
        Assert.That(peakCount, Is.EqualTo(12));
        Assert.That(valleyCount, Is.EqualTo(12));
    }

    [Test]
    public void Standard_IsPeakVertex_Has27Peaks27Valleys()
    {
        var topo = BoardTopology.Standard;
        var peakCount = topo.IsPeakVertex.Count(v => v);
        var valleyCount = topo.IsPeakVertex.Count(v => !v);
        Assert.That(peakCount, Is.EqualTo(27));
        Assert.That(valleyCount, Is.EqualTo(27));
    }

    [Test]
    public void EveryEdge_ConnectsPeakToValley()
    {
        foreach (var topo in new[] { BoardTopology.Mini, BoardTopology.Small, BoardTopology.Standard })
        {
            for (var ei = 0; ei < topo.EdgeCount; ei++)
            {
                var (va, vb) = topo.Edges[ei];
                Assert.That(
                    topo.IsPeakVertex[va] != topo.IsPeakVertex[vb],
                    Is.True,
                    $"Edge {ei} ({va}-{vb}): both are {(topo.IsPeakVertex[va] ? "peak" : "valley")}");
            }
        }
    }

    // == PlacementActionSerializer ==================================

    [TestCase("Mini", 60)]
    [TestCase("Small", 82)]
    [TestCase("Standard", 144)]
    public void VocabularySize_MatchesExpected(string mapName, int expectedSize)
    {
        var serializer = GetSerializer(mapName);
        Assert.That(serializer.VocabularySize, Is.EqualTo(expectedSize));
    }

    [Test]
    public void AllEntries_AreUnique()
    {
        foreach (var mapName in new[] { "Mini", "Small", "Standard" })
        {
            var serializer = GetSerializer(mapName);
            var strings = serializer.Vocabulary.Select(e => $"{e.Vertex}{e.Direction}").ToHashSet();
            Assert.That(strings.Count, Is.EqualTo(serializer.VocabularySize),
                $"Duplicate action strings in {mapName}");
        }
    }

    [Test]
    public void Vocabulary_IsSortedByVertexThenDirection()
    {
        foreach (var mapName in new[] { "Mini", "Small", "Standard" })
        {
            var serializer = GetSerializer(mapName);
            for (var i = 1; i < serializer.VocabularySize; i++)
            {
                var prev = serializer.Vocabulary[i - 1];
                var curr = serializer.Vocabulary[i];
                var cmp = prev.Vertex.CompareTo(curr.Vertex);
                if (cmp == 0)
                    cmp = string.Compare(prev.Direction, curr.Direction, StringComparison.Ordinal);
                Assert.That(cmp, Is.LessThan(0),
                    $"Entry {i} ({curr.Vertex}{curr.Direction}) is not sorted after {prev.Vertex}{prev.Direction} in {mapName}");
            }
        }
    }

    [Test]
    public void SerializeAndIndexOf_RoundTrip()
    {
        foreach (var mapName in new[] { "Mini", "Small", "Standard" })
        {
            var serializer = GetSerializer(mapName);
            var topo = GetTopology(mapName);

            for (var ei = 0; ei < topo.EdgeCount; ei++)
            {
                var (va, vb) = topo.Edges[ei];

                // From va
                var str = serializer.Serialize(va, ei);
                var idx = serializer.IndexOf(str);
                var idx2 = serializer.IndexOf(va, ei);
                Assert.That(idx, Is.EqualTo(idx2), $"IndexOf mismatch for {str} in {mapName}");

                // From vb
                str = serializer.Serialize(vb, ei);
                idx = serializer.IndexOf(str);
                idx2 = serializer.IndexOf(vb, ei);
                Assert.That(idx, Is.EqualTo(idx2), $"IndexOf mismatch for {str} in {mapName}");
            }
        }
    }

    [Test]
    public void PeakVertices_UseN_SW_SE_Directions()
    {
        var validDirs = new HashSet<string> { "N", "SW", "SE" };
        foreach (var mapName in new[] { "Mini", "Small", "Standard" })
        {
            var serializer = GetSerializer(mapName);
            var topo = GetTopology(mapName);
            foreach (var entry in serializer.Vocabulary)
            {
                if (topo.IsPeakVertex[entry.Vertex])
                {
                    Assert.That(validDirs, Does.Contain(entry.Direction),
                        $"Peak vertex {entry.Vertex} has direction {entry.Direction} in {mapName}");
                }
            }
        }
    }

    [Test]
    public void ValleyVertices_UseS_NW_NE_Directions()
    {
        var validDirs = new HashSet<string> { "S", "NW", "NE" };
        foreach (var mapName in new[] { "Mini", "Small", "Standard" })
        {
            var serializer = GetSerializer(mapName);
            var topo = GetTopology(mapName);
            foreach (var entry in serializer.Vocabulary)
            {
                if (!topo.IsPeakVertex[entry.Vertex])
                {
                    Assert.That(validDirs, Does.Contain(entry.Direction),
                        $"Valley vertex {entry.Vertex} has direction {entry.Direction} in {mapName}");
                }
            }
        }
    }

    [Test]
    public void EachVertex_HasCorrectEntryCount()
    {
        foreach (var mapName in new[] { "Mini", "Small", "Standard" })
        {
            var serializer = GetSerializer(mapName);
            var topo = GetTopology(mapName);

            for (var vi = 0; vi < topo.VertexCount; vi++)
            {
                var expectedEdges = topo.VertexEdges[vi].Length;
                var actualEntries = serializer.Vocabulary.Count(e => e.Vertex == vi);
                Assert.That(actualEntries, Is.EqualTo(expectedEdges),
                    $"Vertex {vi} has {actualEntries} entries, expected {expectedEdges} in {mapName}");
            }
        }
    }

    // == SerializePlacementPhase ====================================

    [Test]
    public void PlacementPhase_HasFourSections()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var serialized = state.SerializePlacementPhase();
        var sections = serialized.Split('|');
        Assert.That(sections.Length, Is.EqualTo(4));
    }

    [Test]
    public void PlacementPhase_TilesAndPortsMatchGameState()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var fullState = state.SerializeHumanReadable();
        var placement = state.SerializePlacementPhase();

        var fullSections = fullState.Split('|');
        var placeSections = placement.Split('|');

        // Tiles (section 0) and ports (section 1) should match
        Assert.That(placeSections[0], Is.EqualTo(fullSections[0]), "Tiles differ");
        Assert.That(placeSections[1], Is.EqualTo(fullSections[1]), "Ports differ");
    }

    [Test]
    public void PlacementPhase_EmptyBoard_AllVerticesEmpty()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var serialized = state.SerializePlacementPhase();
        var sections = serialized.Split('|');
        var vertexSection = sections[2];

        for (var i = 0; i < vertexSection.Length; i += 2)
        {
            Assert.That(vertexSection[i], Is.EqualTo('.'), $"Vertex {i / 2} placement not empty");
            Assert.That(vertexSection[i + 1], Is.EqualTo('_'), $"Vertex {i / 2} owner not none");
        }
    }

    [Test]
    public void PlacementPhase_EmptyBoard_AllEdgesEmpty()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var serialized = state.SerializePlacementPhase();
        var sections = serialized.Split('|');
        var edgeSection = sections[3];

        for (var i = 0; i < edgeSection.Length; i++)
        {
            Assert.That(edgeSection[i], Is.EqualTo('_'), $"Edge {i} not empty");
        }
    }

    [Test]
    public void PlacementPhase_AfterPlacement_VerticesHaveABTokens()
    {
        var state = PlayThroughPlacement(GameConfig.Mini, 2, seed: 42);
        var serialized = state.SerializePlacementPhase();
        var sections = serialized.Split('|');
        var vertexSection = sections[2];

        var aCount = 0;
        var bCount = 0;
        for (var i = 0; i < vertexSection.Length; i += 2)
        {
            if (vertexSection[i] == 'a') aCount++;
            if (vertexSection[i] == 'b') bCount++;
        }

        Assert.That(aCount, Is.EqualTo(2));
        Assert.That(bCount, Is.EqualTo(0));
    }

    [Test]
    public void PlacementPhaseCompact_StripsAllPipes()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var human = state.SerializePlacementPhase();
        var compact = state.SerializePlacementPhaseCompact();

        Assert.That(compact, Does.Not.Contain("|"));
        Assert.That(compact.Length, Is.EqualTo(human.Length - 3));
    }

    [TestCase("Mini", 2, 105)]
    [TestCase("Small", 2, 141)]
    [TestCase("Standard", 3, 246)]
    public void PlacementPhaseCompact_HasExpectedLength(string mapName, int players, int expectedLength)
    {
        var config = GetConfig(mapName);
        var state = new CatanState(config, players, new Random(42));
        var compact = state.SerializePlacementPhaseCompact();
        Assert.That(compact.Length, Is.EqualTo(expectedLength));
    }

    [Test]
    public void PlacementPhase_Standard_FullPlacement_HasCorrectCounts()
    {
        var state = PlayThroughPlacement(GameConfig.Standard, 3, seed: 99);
        var serialized = state.SerializePlacementPhase();
        var sections = serialized.Split('|');
        var vertexSection = sections[2];

        var aCount = 0;
        var bCount = 0;
        for (var i = 0; i < vertexSection.Length; i += 2)
        {
            if (vertexSection[i] == 'a') aCount++;
            if (vertexSection[i] == 'b') bCount++;
        }

        Assert.That(aCount, Is.EqualTo(3));
        Assert.That(bCount, Is.EqualTo(3));
    }

    [Test]
    public void PlacementPhase_EdgeSection_MatchesGameStateEdges()
    {
        var state = PlayThroughPlacement(GameConfig.Mini, 2, seed: 42);
        var fullState = state.SerializeHumanReadable();
        var placement = state.SerializePlacementPhase();

        var fullSections = fullState.Split('|');
        var placeSections = placement.Split('|');

        Assert.That(placeSections[3], Is.EqualTo(fullSections[6]));
    }

    // == Helpers =======================================================

    private static CatanState PlayThroughPlacement(GameConfig config, int players, int seed)
    {
        var state = new CatanState(config, players, new Random(seed));
        while (state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceFirstRoad
                              or TurnStage.PlaceSecondSettlement or TurnStage.PlaceSecondRoad)
        {
            var action = GetCatanActions(state).First();
            state = (CatanState)action.DoCoreAction();
        }
        return state;
    }

    private static IEnumerable<Gimbur.CatanAction> GetCatanActions(Gimbur.CatanState state) =>
        state.Actions().Select(ca => ca.IsDeterministic
            ? (Gimbur.CatanAction)(Gimbur.CatanDeterministicAction)((Kjarni.CoreAction.Deterministic)ca).Item
            : (Gimbur.CatanAction)(Gimbur.CatanStochasticAction)((Kjarni.CoreAction.Stochastic)ca).Item);

    private static PlacementActionSerializer GetSerializer(string mapName) => mapName switch
    {
        "Mini" => PlacementActionSerializer.Mini,
        "Small" => PlacementActionSerializer.Small,
        "Standard" => PlacementActionSerializer.Standard,
        _ => throw new ArgumentException($"Unknown map: {mapName}")
    };

    private static BoardTopology GetTopology(string mapName) => mapName switch
    {
        "Mini" => BoardTopology.Mini,
        "Small" => BoardTopology.Small,
        "Standard" => BoardTopology.Standard,
        _ => throw new ArgumentException($"Unknown map: {mapName}")
    };

    private static GameConfig GetConfig(string mapName) => mapName switch
    {
        "Mini" => GameConfig.Mini,
        "Small" => GameConfig.Small,
        "Standard" => GameConfig.Standard,
        _ => throw new ArgumentException($"Unknown map: {mapName}")
    };
}
