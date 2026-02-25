using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class BoardSymmetryTests
{
    // ── Permutation count ───────────────────────────────────────────

    [Test]
    public void Mini_Returns5Permutations()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        Assert.That(perms.Length, Is.EqualTo(5));
    }

    [Test]
    public void Standard_Returns2Permutations()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Standard);
        Assert.That(perms.Length, Is.EqualTo(2));
    }

    [Test]
    public void Small_ReturnsNoPermutations()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Small);
        Assert.That(perms.Length, Is.EqualTo(0));
    }

    // ── Permutation validity ────────────────────────────────────────

    [Test]
    public void Mini_AllPermutationsAreValidBijections()
    {
        var topo = BoardTopology.Mini;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            AssertValidPermutation(perm.Tiles, topo.TileCount, $"Tiles ({perm.Label})");
            AssertValidPermutation(perm.Vertices, topo.VertexCount, $"Vertices ({perm.Label})");
            AssertValidPermutation(perm.Edges, topo.EdgeCount, $"Edges ({perm.Label})");
            AssertValidPermutation(perm.Ports, topo.PortCount, $"Ports ({perm.Label})");
        }
    }

    [Test]
    public void Standard_AllPermutationsAreValidBijections()
    {
        var topo = BoardTopology.Standard;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            AssertValidPermutation(perm.Tiles, topo.TileCount, $"Tiles ({perm.Label})");
            AssertValidPermutation(perm.Vertices, topo.VertexCount, $"Vertices ({perm.Label})");
            AssertValidPermutation(perm.Edges, topo.EdgeCount, $"Edges ({perm.Label})");
            AssertValidPermutation(perm.Ports, topo.PortCount, $"Ports ({perm.Label})");
        }
    }

    [Test]
    public void Mini_AllPermutationsAreNonTrivial()
    {
        var topo = BoardTopology.Mini;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            // At least one tile must move to a different index.
            var isIdentity = true;
            for (var i = 0; i < topo.TileCount; i++)
            {
                if (perm.Tiles[i] != i)
                {
                    isIdentity = false;
                    break;
                }
            }

            Assert.That(isIdentity, Is.False,
                $"Permutation {perm.Label} is the identity — should not be included.");
        }
    }

    [Test]
    public void Standard_AllPermutationsAreNonTrivial()
    {
        var topo = BoardTopology.Standard;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            var isIdentity = true;
            for (var i = 0; i < topo.TileCount; i++)
            {
                if (perm.Tiles[i] != i)
                {
                    isIdentity = false;
                    break;
                }
            }

            Assert.That(isIdentity, Is.False,
                $"Permutation {perm.Label} is the identity — should not be included.");
        }
    }

    // ── All permutations are distinct ───────────────────────────────

    [Test]
    public void Mini_AllPermutationsAreDistinct()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var tileStrings = new HashSet<string>();

        foreach (var perm in perms)
        {
            var key = string.Join(",", perm.Tiles);
            Assert.That(tileStrings.Add(key), Is.True,
                $"Duplicate permutation found: {perm.Label}");
        }
    }

    [Test]
    public void Standard_AllPermutationsAreDistinct()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Standard);
        var tileStrings = new HashSet<string>();

        foreach (var perm in perms)
        {
            var key = string.Join(",", perm.Tiles);
            Assert.That(tileStrings.Add(key), Is.True,
                $"Duplicate permutation found: {perm.Label}");
        }
    }

    // ── Center tile maps to itself ──────────────────────────────────

    [Test]
    public void Mini_CenterTileMapsToItself()
    {
        // Center tile is (0,0) which is tile index 3 for mini map.
        var topo = BoardTopology.Mini;
        var centerIndex = -1;
        for (var i = 0; i < topo.TileCount; i++)
        {
            if (topo.Tiles[i] == new HexCoord(0, 0))
            {
                centerIndex = i;
                break;
            }
        }

        Assert.That(centerIndex, Is.GreaterThanOrEqualTo(0), "Center tile not found");

        var perms = BoardSymmetry.GetPermutations(topo);
        foreach (var perm in perms)
        {
            Assert.That(perm.Tiles[centerIndex], Is.EqualTo(centerIndex),
                $"Center tile should map to itself under {perm.Label}");
        }
    }

    [Test]
    public void Standard_CenterTileMapsToItself()
    {
        var topo = BoardTopology.Standard;
        var centerIndex = -1;
        for (var i = 0; i < topo.TileCount; i++)
        {
            if (topo.Tiles[i] == new HexCoord(0, 0))
            {
                centerIndex = i;
                break;
            }
        }

        Assert.That(centerIndex, Is.GreaterThanOrEqualTo(0), "Center tile not found");

        var perms = BoardSymmetry.GetPermutations(topo);
        foreach (var perm in perms)
        {
            Assert.That(perm.Tiles[centerIndex], Is.EqualTo(centerIndex),
                $"Center tile should map to itself under {perm.Label}");
        }
    }

    // ── Edge consistency: edges connect permuted vertices ────────────

    [Test]
    public void Mini_EdgePermutationConsistentWithVertexPermutation()
    {
        var topo = BoardTopology.Mini;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            AssertEdgeVertexConsistency(topo, perm);
        }
    }

    [Test]
    public void Standard_EdgePermutationConsistentWithVertexPermutation()
    {
        var topo = BoardTopology.Standard;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            AssertEdgeVertexConsistency(topo, perm);
        }
    }

    // ── C3: applying rot120 three times yields identity ─────────────

    [Test]
    public void Standard_Rot120AppliedThreeTimesIsIdentity()
    {
        var topo = BoardTopology.Standard;
        var perms = BoardSymmetry.GetPermutations(topo);
        var rot120 = perms[0]; // First permutation is rot120

        Assert.That(rot120.Label, Is.EqualTo("rot120"));

        // Apply tile permutation 3 times.
        var current = Enumerable.Range(0, topo.TileCount).ToArray();
        for (var round = 0; round < 3; round++)
        {
            var next = new int[topo.TileCount];
            for (var i = 0; i < topo.TileCount; i++)
            {
                next[i] = rot120.Tiles[current[i]];
            }

            current = next;
        }

        // Should be identity.
        for (var i = 0; i < topo.TileCount; i++)
        {
            Assert.That(current[i], Is.EqualTo(i),
                $"Tile {i} does not return to itself after 3×rot120");
        }
    }

    // ── D6: applying rot60 six times yields identity ────────────────

    [Test]
    public void Mini_Rot60AppliedSixTimesIsIdentity()
    {
        var topo = BoardTopology.Mini;
        var perms = BoardSymmetry.GetPermutations(topo);
        var rot60 = perms[0]; // First permutation is rot60

        Assert.That(rot60.Label, Is.EqualTo("rot60"));

        var current = Enumerable.Range(0, topo.TileCount).ToArray();
        for (var round = 0; round < 6; round++)
        {
            var next = new int[topo.TileCount];
            for (var i = 0; i < topo.TileCount; i++)
            {
                next[i] = rot60.Tiles[current[i]];
            }

            current = next;
        }

        for (var i = 0; i < topo.TileCount; i++)
        {
            Assert.That(current[i], Is.EqualTo(i),
                $"Tile {i} does not return to itself after 6×rot60");
        }
    }

    // ── Serialization permutation: board ─────────────────────────────

    [Test]
    public void Mini_PermuteBoardProducesCorrectLengthString()
    {
        var topo = BoardTopology.Mini;
        var perms = BoardSymmetry.GetPermutations(topo);

        // Create a deterministic board serialization.
        var board = CreateTestBoard(GameConfig.Mini);
        var boardStr = board.SerializeBoard();

        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermuteBoard(boardStr, perm);
            Assert.That(permuted.Length, Is.EqualTo(boardStr.Length),
                $"Permuted board string length mismatch for {perm.Label}");

            // Should still have exactly one '|' separator.
            Assert.That(permuted.Count(c => c == '|'), Is.EqualTo(1),
                $"Permuted board should have 1 separator for {perm.Label}");
        }
    }

    [Test]
    public void Standard_PermuteBoardProducesCorrectLengthString()
    {
        var topo = BoardTopology.Standard;
        var perms = BoardSymmetry.GetPermutations(topo);

        var board = CreateTestBoard(GameConfig.Standard);
        var boardStr = board.SerializeBoard();

        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermuteBoard(boardStr, perm);
            Assert.That(permuted.Length, Is.EqualTo(boardStr.Length),
                $"Permuted board string length mismatch for {perm.Label}");
        }
    }

    [Test]
    public void Mini_PermuteBoardIsDifferentFromOriginal()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var board = CreateTestBoard(GameConfig.Mini);
        var boardStr = board.SerializeBoard();

        // At least some permutations should produce different strings
        // (unless the board happens to be perfectly symmetric, which is extremely unlikely).
        var allSame = true;
        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermuteBoard(boardStr, perm);
            if (permuted != boardStr)
            {
                allSame = false;
                break;
            }
        }

        Assert.That(allSame, Is.False,
            "All permuted board strings are identical to the original — highly unlikely.");
    }

    // ── Serialization permutation: state ─────────────────────────────

    [Test]
    public void Mini_PermuteStateProducesCorrectSectionCount()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var state = CreateTestState(GameConfig.Mini, playerCount: 2);
        var stateStr = state.SerializeStateOnly();

        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermuteState(stateStr, perm);
            var sections = permuted.Split('|');
            Assert.That(sections.Length, Is.EqualTo(8),
                $"Permuted state should have 8 sections for {perm.Label}");
        }
    }

    [Test]
    public void Mini_PermuteStatePreservesPlayerSections()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var state = CreateTestState(GameConfig.Mini, playerCount: 2);
        var stateStr = state.SerializeStateOnly();
        var originalSections = stateStr.Split('|');

        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermuteState(stateStr, perm);
            var permSections = permuted.Split('|');

            // Sections 1 (playerStage), 2 (longestLargest), 5 (resources),
            // 6 (knights), 7 (devCards) should be unchanged.
            Assert.Multiple(() =>
            {
                Assert.That(permSections[1], Is.EqualTo(originalSections[1]),
                    $"Player/stage changed under {perm.Label}");
                Assert.That(permSections[2], Is.EqualTo(originalSections[2]),
                    $"Longest/largest changed under {perm.Label}");
                Assert.That(permSections[5], Is.EqualTo(originalSections[5]),
                    $"Resources changed under {perm.Label}");
                Assert.That(permSections[6], Is.EqualTo(originalSections[6]),
                    $"Knights changed under {perm.Label}");
                Assert.That(permSections[7], Is.EqualTo(originalSections[7]),
                    $"DevCards changed under {perm.Label}");
            });
        }
    }

    [Test]
    public void Standard_PermuteStatePreservesLengths()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Standard);
        var state = CreateTestState(GameConfig.Standard, playerCount: 3);
        var stateStr = state.SerializeStateOnly();

        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermuteState(stateStr, perm);
            Assert.That(permuted.Length, Is.EqualTo(stateStr.Length),
                $"Permuted state string length mismatch for {perm.Label}");
        }
    }

    // ── Round-trip: permuted state can be deserialized ───────────────

    [Test]
    public void Mini_PermutedFullStateIsDeserializable()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var config = GameConfig.Mini;
        var state = CreateTestState(config, playerCount: 2);

        // Serialize the full human-readable state (all 10 sections).
        var fullStr = state.SerializeHumanReadable();
        var fullSections = fullStr.Split('|');

        // Construct a permuted full string: permute tiles+ports for board,
        // and permute the state-only sections.
        foreach (var perm in perms)
        {
            // Permute board part (sections 0 tiles, 6 ports).
            var boardStr = fullSections[0] + "|" + fullSections[6];
            var permBoard = BoardSymmetry.PermuteBoard(boardStr, perm);
            var permBoardSections = permBoard.Split('|');

            // Permute state-only part (sections 1-5, 7-9 → mapped to state-only sections 0-7).
            var stateOnlyStr = string.Join('|',
                fullSections[1], fullSections[2], fullSections[3],
                fullSections[4], fullSections[5],
                fullSections[7], fullSections[8], fullSections[9]);
            var permState = BoardSymmetry.PermuteState(stateOnlyStr, perm);
            var permStateSections = permState.Split('|');

            // Reassemble full permuted string in 10-section format.
            var permFull = string.Join('|',
                permBoardSections[0],   // tiles
                permStateSections[0],   // robber
                permStateSections[1],   // playerStage
                permStateSections[2],   // longestLargest
                permStateSections[3],   // vertices
                permStateSections[4],   // edges
                permBoardSections[1],   // ports
                permStateSections[5],   // resources
                permStateSections[6],   // knights
                permStateSections[7]);  // devCards

            // This should deserialize without error.
            Assert.DoesNotThrow(() =>
            {
                Gimbur.CatanState.DeserializeHumanReadable(config, 2, permFull);
            }, $"Permuted state ({perm.Label}) failed to deserialize");
        }
    }

    // ── Adjacency preservation ──────────────────────────────────────

    [Test]
    public void Mini_PermutationPreservesAdjacency()
    {
        var topo = BoardTopology.Mini;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            // Every pair of adjacent tiles should remain adjacent after permutation.
            for (var ti = 0; ti < topo.TileCount; ti++)
            {
                var permTi = perm.Tiles[ti];
                var originalNeighbors = topo.TileNeighbors[ti];
                var permutedNeighbors = topo.TileNeighbors[permTi];

                foreach (var neighbor in originalNeighbors)
                {
                    var permNeighbor = perm.Tiles[neighbor];
                    Assert.That(permutedNeighbors.Contains(permNeighbor), Is.True,
                        $"Tile adjacency not preserved under {perm.Label}: " +
                        $"tiles {ti} and {neighbor} are adjacent, but their images " +
                        $"{permTi} and {permNeighbor} are not.");
                }
            }
        }
    }

    [Test]
    public void Standard_PermutationPreservesAdjacency()
    {
        var topo = BoardTopology.Standard;
        var perms = BoardSymmetry.GetPermutations(topo);

        foreach (var perm in perms)
        {
            for (var ti = 0; ti < topo.TileCount; ti++)
            {
                var permTi = perm.Tiles[ti];
                var originalNeighbors = topo.TileNeighbors[ti];
                var permutedNeighbors = topo.TileNeighbors[permTi];

                foreach (var neighbor in originalNeighbors)
                {
                    var permNeighbor = perm.Tiles[neighbor];
                    Assert.That(permutedNeighbors.Contains(permNeighbor), Is.True,
                        $"Tile adjacency not preserved under {perm.Label}: " +
                        $"tiles {ti} and {neighbor} are adjacent, but their images " +
                        $"{permTi} and {permNeighbor} are not.");
                }
            }
        }
    }

    // ── Labels ──────────────────────────────────────────────────────

    [Test]
    public void Standard_PermutationLabels()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Standard);
        Assert.That(perms[0].Label, Is.EqualTo("rot120"));
        Assert.That(perms[1].Label, Is.EqualTo("rot240"));
    }

    [Test]
    public void Mini_PermutationLabelsIncludeAllRotations()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var labels = perms.Select(p => p.Label).ToList();

        Assert.That(labels, Does.Contain("rot60"));
        Assert.That(labels, Does.Contain("rot120"));
        Assert.That(labels, Does.Contain("rot180"));
        Assert.That(labels, Does.Contain("rot240"));
        Assert.That(labels, Does.Contain("rot300"));
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private static void AssertValidPermutation(
        System.Collections.Immutable.ImmutableArray<int> perm,
        int expectedCount,
        string description)
    {
        Assert.That(perm.Length, Is.EqualTo(expectedCount),
            $"{description}: wrong length");

        var seen = new HashSet<int>();
        for (var i = 0; i < perm.Length; i++)
        {
            Assert.That(perm[i], Is.InRange(0, expectedCount - 1),
                $"{description}: index {i} maps to out-of-range value {perm[i]}");
            Assert.That(seen.Add(perm[i]), Is.True,
                $"{description}: duplicate mapping to {perm[i]}");
        }
    }

    private static void AssertEdgeVertexConsistency(BoardTopology topo, SymmetryPermutation perm)
    {
        for (var ei = 0; ei < topo.EdgeCount; ei++)
        {
            var (oldA, oldB) = topo.Edges[ei];
            var newA = perm.Vertices[oldA];
            var newB = perm.Vertices[oldB];

            // The permuted edge should connect the permuted vertices.
            var newEdge = perm.Edges[ei];
            var (actualA, actualB) = topo.Edges[newEdge];

            var expected = newA <= newB ? (newA, newB) : (newB, newA);
            var actual = actualA <= actualB ? (actualA, actualB) : (actualB, actualA);

            Assert.That(actual, Is.EqualTo(expected),
                $"Edge {ei} → {newEdge} under {perm.Label}: " +
                $"expected vertices ({expected.Item1},{expected.Item2}) " +
                $"but got ({actual.Item1},{actual.Item2})");
        }
    }

    private static Gimbur.CatanState CreateTestState(GameConfig config, int playerCount)
    {
        var rng = new Random(42);
        return new Gimbur.CatanState(config, playerCount, rng);
    }

    private static Gimbur.CatanState CreateTestBoard(GameConfig config)
    {
        var rng = new Random(42);
        return new Gimbur.CatanState(config, config.MinPlayers, rng);
    }
}
