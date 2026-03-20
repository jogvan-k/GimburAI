using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

public partial class BoardSymmetryTests
{
    // -- Placement state permutation tests --

    [Test]
    public void Mini_PermutePlacementStateProducesCorrectSectionCount()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var state = CreateTestState(GameConfig.Mini, playerCount: 2);
        var placementStr = state.SerializePlacementPhase();
        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermutePlacementState(placementStr, perm);
            var sections = permuted.Split('|');
            Assert.That(sections.Length, Is.EqualTo(4),
                $"Expected 4 sections for {perm.Label}");
        }
    }

    [Test]
    public void Standard_PermutePlacementStateProducesCorrectSectionCount()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Standard);
        var state = CreateTestState(GameConfig.Standard, playerCount: 3);
        var placementStr = state.SerializePlacementPhase();
        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermutePlacementState(placementStr, perm);
            Assert.That(permuted.Split('|').Length, Is.EqualTo(4),
                $"Expected 4 sections for {perm.Label}");
        }
    }

    [Test]
    public void Small_PermutePlacementStateProducesCorrectSectionCount()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Small);
        var state = CreateTestState(GameConfig.Small, playerCount: 2);
        var placementStr = state.SerializePlacementPhase();
        foreach (var perm in perms)
        {
            var permuted = BoardSymmetry.PermutePlacementState(placementStr, perm);
            Assert.That(permuted.Split('|').Length, Is.EqualTo(4),
                $"Expected 4 sections for {perm.Label}");
        }
    }

    [Test]
    public void Mini_PermutePlacementStatePreservesSectionLengths()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var state = CreateTestState(GameConfig.Mini, playerCount: 2);
        var ps = state.SerializePlacementPhase();
        var orig = ps.Split('|');
        foreach (var perm in perms)
        {
            var s = BoardSymmetry.PermutePlacementState(ps, perm).Split('|');
            for (var i = 0; i < 4; i++)
                Assert.That(s[i].Length, Is.EqualTo(orig[i].Length),
                    $"Section {i} length mismatch for {perm.Label}");
        }
    }

    [Test]
    public void Mini_PermutePlacementStateIsDifferentFromOriginal()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var state = CreateTestState(GameConfig.Mini, playerCount: 2);
        var ps = state.SerializePlacementPhase();
        var allSame = true;
        foreach (var perm in perms)
        {
            if (BoardSymmetry.PermutePlacementState(ps, perm) != ps)
            { allSame = false; break; }
        }
        Assert.That(allSame, Is.False,
            "All permuted placement strings identical to original.");
    }

    [Test]
    public void Mini_PlacementTilePortSectionsMatchPermuteBoard()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        var state = CreateTestState(GameConfig.Mini, playerCount: 2);
        var ps = state.SerializePlacementPhase();
        var psSections = ps.Split('|');
        var boardStr = psSections[0] + "|" + psSections[1];
        foreach (var perm in perms)
        {
            var pp = BoardSymmetry.PermutePlacementState(ps, perm).Split('|');
            var pb = BoardSymmetry.PermuteBoard(boardStr, perm).Split('|');
            Assert.That(pp[0], Is.EqualTo(pb[0]),
                $"Tile section mismatch for {perm.Label}");
            Assert.That(pp[1], Is.EqualTo(pb[1]),
                $"Port section mismatch for {perm.Label}");
        }
    }

    [Test]
    public void Mini_PermutePlacementStateWrongSectionCount_Throws()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Mini);
        Assert.That(
            () => BoardSymmetry.PermutePlacementState("a|b|c", perms[0]),
            Throws.TypeOf<ArgumentException>());
    }

    [Test]
    public void Standard_PermutePlacementStatePreservesSectionLengths()
    {
        var perms = BoardSymmetry.GetPermutations(BoardTopology.Standard);
        var state = CreateTestState(GameConfig.Standard, playerCount: 3);
        var ps = state.SerializePlacementPhase();
        var orig = ps.Split('|');
        foreach (var perm in perms)
        {
            var s = BoardSymmetry.PermutePlacementState(ps, perm).Split('|');
            for (var i = 0; i < 4; i++)
                Assert.That(s[i].Length, Is.EqualTo(orig[i].Length),
                    $"Section {i} length mismatch for {perm.Label}");
        }
    }
}
