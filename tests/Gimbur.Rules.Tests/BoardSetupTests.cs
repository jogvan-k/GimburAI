using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class BoardSetupTests
{
    [Test]
    public void Standard_GeneratesValidSetup()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng);

        Assert.Multiple(() =>
        {
            Assert.That(setup.TileResources.Length, Is.EqualTo(19));
            Assert.That(setup.TileNumbers.Length, Is.EqualTo(19));
            Assert.That(setup.PortTypes.Length, Is.EqualTo(9));
        });
    }

    [Test]
    public void Standard_HasCorrectResourceDistribution()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng);

        var counts = new int[6];
        foreach (var r in setup.TileResources)
            counts[(int)r]++;

        Assert.Multiple(() =>
        {
            Assert.That(counts[(int)ResourceType.Desert], Is.EqualTo(1));
            Assert.That(counts[(int)ResourceType.Wood], Is.EqualTo(4));
            Assert.That(counts[(int)ResourceType.Brick], Is.EqualTo(3));
            Assert.That(counts[(int)ResourceType.Sheep], Is.EqualTo(4));
            Assert.That(counts[(int)ResourceType.Wheat], Is.EqualTo(4));
            Assert.That(counts[(int)ResourceType.Ore], Is.EqualTo(3));
        });
    }

    [Test]
    public void Standard_DesertTileHasNumberZero()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng);

        for (var ti = 0; ti < setup.TileResources.Length; ti++)
        {
            if (setup.TileResources[ti] == ResourceType.Desert)
                Assert.That(setup.TileNumbers[ti], Is.EqualTo(0),
                    $"Desert tile {ti} should have number 0");
            else
                Assert.That(setup.TileNumbers[ti], Is.Not.EqualTo(0),
                    $"Non-desert tile {ti} should not have number 0");
        }
    }

    [Test]
    public void Standard_NoAdjacentRedNumbers()
    {
        var rng = new Random(42);
        var topology = BoardTopology.Standard;

        // Generate many boards to increase confidence.
        for (var i = 0; i < 100; i++)
        {
            var setup = BoardSetup.Generate(MapConfig.Standard, new Random(i));

            for (var ti = 0; ti < topology.TileCount; ti++)
            {
                if (setup.TileNumbers[ti] is not (6 or 8))
                    continue;

                foreach (var neighbor in topology.TileNeighbors[ti])
                {
                    Assert.That(setup.TileNumbers[neighbor], Is.Not.EqualTo(6).And.Not.EqualTo(8),
                        $"Seed {i}: tiles {ti} and {neighbor} both have red numbers " +
                        $"({setup.TileNumbers[ti]} and {setup.TileNumbers[neighbor]})");
                }
            }
        }
    }

    [Test]
    public void Standard_HasCorrectNumberTokenDistribution()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng);

        var nonDesertNumbers = setup.TileNumbers
            .Where(n => n != 0)
            .OrderBy(n => n)
            .ToArray();

        var expected = new[] { 2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12 };
        Assert.That(nonDesertNumbers, Is.EqualTo(expected));
    }

    [Test]
    public void Standard_AssignsNumberTokensInSpiralOrder_SkippingDesert()
    {
        var setup = BoardSetup.Generate(MapConfig.Standard, new Random(42));

        // The canonical CCW spiral tile rings (before random rotation per ring).
        var outerRing = new[] { 0, 3, 7, 12, 16, 17, 18, 15, 11, 6, 2, 1 };
        var innerRing = new[] { 4, 8, 13, 14, 10, 5 };
        var center = new[] { 9 };
        var tokens = MapConfig.Standard.NumberTokens.ToArray();

        AssertRingsAreRotatedSubsequencesOfTokenPool(setup, [outerRing, innerRing, center], tokens);
    }

    [Test]
    public void Standard_HasCorrectPortTypeDistribution()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng);

        var portCounts = new int[7];
        foreach (var pt in setup.PortTypes)
            portCounts[(int)pt]++;

        Assert.Multiple(() =>
        {
            Assert.That(portCounts[(int)PortType.Generic], Is.EqualTo(4));
            Assert.That(portCounts[(int)PortType.Wood], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Brick], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Sheep], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Wheat], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Ore], Is.EqualTo(1));
        });
    }

    [Test]
    public void Standard_RobberStartsOnDesert()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng);

        Assert.That(setup.TileResources[setup.InitialRobberTile],
            Is.EqualTo(ResourceType.Desert));
    }

    [Test]
    public void Standard_DifferentSeedsProduceDifferentLayouts()
    {
        var setup1 = BoardSetup.Generate(MapConfig.Standard, new Random(1));
        var setup2 = BoardSetup.Generate(MapConfig.Standard, new Random(2));

        // At least one tile resource should differ (extremely unlikely to be identical).
        var anyDifference = false;
        for (var ti = 0; ti < 19; ti++)
        {
            if (setup1.TileResources[ti] != setup2.TileResources[ti])
            {
                anyDifference = true;
                break;
            }
        }
        Assert.That(anyDifference, Is.True,
            "Two different seeds should produce different board layouts");
    }

    [Test]
    public void Standard_AllowAdjacentRedNumbers_WhenConstraintDisabled()
    {
        // With the constraint disabled, we should be able to generate boards
        // without retries (just verify it doesn't throw).
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Standard, rng, noAdjacentRedNumbers: false);
        Assert.That(setup.TileResources.Length, Is.EqualTo(19));
    }

    // ── Mini map ────────────────────────────────────────────────────

    [Test]
    public void Mini_GeneratesValidSetup()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Mini, rng);

        Assert.Multiple(() =>
        {
            Assert.That(setup.TileResources.Length, Is.EqualTo(7));
            Assert.That(setup.TileNumbers.Length, Is.EqualTo(7));
            Assert.That(setup.PortTypes.Length, Is.EqualTo(6));
        });
    }

    [Test]
    public void Mini_DesertTileHasNumberZero()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Mini, rng);

        for (var ti = 0; ti < setup.TileResources.Length; ti++)
        {
            if (setup.TileResources[ti] == ResourceType.Desert)
                Assert.That(setup.TileNumbers[ti], Is.EqualTo(0));
        }
    }

    [Test]
    public void Mini_AssignsNumberTokensInSpiralOrder_SkippingDesert()
    {
        var setup = BoardSetup.Generate(MapConfig.Mini, new Random(42));

        var outerRing = new[] { 0, 2, 5, 6, 4, 1 };
        var center = new[] { 3 };
        var tokens = MapConfig.Mini.NumberTokens.ToArray();

        AssertRingsAreRotatedSubsequencesOfTokenPool(setup, [outerRing, center], tokens);
    }

    // ── Small map ───────────────────────────────────────────────────

    [Test]
    public void Small_GeneratesValidSetup()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Small, rng);

        Assert.Multiple(() =>
        {
            Assert.That(setup.TileResources.Length, Is.EqualTo(10));
            Assert.That(setup.TileNumbers.Length, Is.EqualTo(10));
            Assert.That(setup.PortTypes.Length, Is.EqualTo(6));
        });
    }

    [Test]
    public void Small_HasCorrectResourceDistribution()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Small, rng);

        var counts = new int[6];
        foreach (var r in setup.TileResources)
            counts[(int)r]++;

        Assert.Multiple(() =>
        {
            Assert.That(counts[(int)ResourceType.Desert], Is.EqualTo(1));
            Assert.That(counts[(int)ResourceType.Wood], Is.EqualTo(2));
            Assert.That(counts[(int)ResourceType.Brick], Is.EqualTo(2));
            Assert.That(counts[(int)ResourceType.Sheep], Is.EqualTo(2));
            Assert.That(counts[(int)ResourceType.Wheat], Is.EqualTo(2));
            Assert.That(counts[(int)ResourceType.Ore], Is.EqualTo(1));
        });
    }

    [Test]
    public void Small_DesertTileHasNumberZero()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Small, rng);

        for (var ti = 0; ti < setup.TileResources.Length; ti++)
        {
            if (setup.TileResources[ti] == ResourceType.Desert)
                Assert.That(setup.TileNumbers[ti], Is.EqualTo(0),
                    $"Desert tile {ti} should have number 0");
            else
                Assert.That(setup.TileNumbers[ti], Is.Not.EqualTo(0),
                    $"Non-desert tile {ti} should not have number 0");
        }
    }

    [Test]
    public void Small_NoAdjacentRedNumbers()
    {
        var topology = BoardTopology.Small;

        for (var i = 0; i < 100; i++)
        {
            var setup = BoardSetup.Generate(MapConfig.Small, new Random(i));

            for (var ti = 0; ti < topology.TileCount; ti++)
            {
                if (setup.TileNumbers[ti] is not (6 or 8))
                    continue;

                foreach (var neighbor in topology.TileNeighbors[ti])
                {
                    Assert.That(setup.TileNumbers[neighbor], Is.Not.EqualTo(6).And.Not.EqualTo(8),
                        $"Seed {i}: tiles {ti} and {neighbor} both have red numbers " +
                        $"({setup.TileNumbers[ti]} and {setup.TileNumbers[neighbor]})");
                }
            }
        }
    }

    [Test]
    public void Small_HasCorrectNumberTokenDistribution()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Small, rng);

        var nonDesertNumbers = setup.TileNumbers
            .Where(n => n != 0)
            .OrderBy(n => n)
            .ToArray();

        var expected = new[] { 3, 4, 5, 6, 8, 9, 10, 11, 12 };
        Assert.That(nonDesertNumbers, Is.EqualTo(expected));
    }

    [Test]
    public void Small_AssignsNumberTokensInSpiralOrder_SkippingDesert()
    {
        var setup = BoardSetup.Generate(MapConfig.Small, new Random(42));

        var outerRing = new[] { 0, 3, 7, 8, 9, 6, 2, 1 };
        var innerRing = new[] { 4, 5 };
        var tokens = MapConfig.Small.NumberTokens.ToArray();

        AssertRingsAreRotatedSubsequencesOfTokenPool(setup, [outerRing, innerRing], tokens);
    }

    [Test]
    public void Small_HasCorrectPortTypeDistribution()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Small, rng);

        var portCounts = new int[7];
        foreach (var pt in setup.PortTypes)
            portCounts[(int)pt]++;

        Assert.Multiple(() =>
        {
            Assert.That(portCounts[(int)PortType.Generic], Is.EqualTo(2));
            Assert.That(portCounts[(int)PortType.Wood], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Brick], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Sheep], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Wheat], Is.EqualTo(1));
            Assert.That(portCounts[(int)PortType.Ore], Is.EqualTo(0));
        });
    }

    [Test]
    public void Small_RobberStartsOnDesert()
    {
        var rng = new Random(42);
        var setup = BoardSetup.Generate(MapConfig.Small, rng);

        Assert.That(setup.TileResources[setup.InitialRobberTile],
            Is.EqualTo(ResourceType.Desert));
    }

    [Test]
    public void Small_DifferentSeedsProduceDifferentLayouts()
    {
        var setup1 = BoardSetup.Generate(MapConfig.Small, new Random(1));
        var setup2 = BoardSetup.Generate(MapConfig.Small, new Random(2));

        var anyDifference = false;
        for (var ti = 0; ti < 10; ti++)
        {
            if (setup1.TileResources[ti] != setup2.TileResources[ti])
            {
                anyDifference = true;
                break;
            }
        }
        Assert.That(anyDifference, Is.True,
            "Two different seeds should produce different board layouts");
    }

    [Test]
    public void Standard_DifferentSeedsCanProduceDifferentSpiralStartCorners()
    {
        // Generate many boards and collect the number assigned to the first tile
        // in the canonical spiral order. With a randomized starting corner, the
        // first token in canonical order should vary across seeds.
        var canonicalFirst = new HashSet<int>();
        for (var seed = 0; seed < 100; seed++)
        {
            var setup = BoardSetup.Generate(MapConfig.Standard, new Random(seed),
                noAdjacentRedNumbers: false);

            // Find the number on the canonical first non-desert spiral tile.
            var canonicalSpiral = new[] { 0, 3, 7, 12, 16, 17, 18, 15, 11, 6, 2, 1 };
            foreach (var ti in canonicalSpiral)
            {
                if (setup.TileResources[ti] != ResourceType.Desert)
                {
                    canonicalFirst.Add(setup.TileNumbers[ti]);
                    break;
                }
            }
        }

        Assert.That(canonicalFirst.Count, Is.GreaterThan(1),
            "With randomized starting corners, the first canonical spiral tile " +
            "should receive different numbers across seeds");
    }

    /// <summary>
    /// Verifies that the numbers assigned to tiles follow the spiral pattern: each ring's
    /// non-desert numbers, read in CCW order from a random starting position, form a
    /// contiguous subsequence of the token pool in the correct relative order.
    /// </summary>
    private static void AssertRingsAreRotatedSubsequencesOfTokenPool(
        BoardSetup setup, int[][] rings, int[] tokens)
    {
        var poolIndex = 0;

        foreach (var ring in rings)
        {
            // Extract non-desert numbers from this ring in canonical CCW order.
            var ringNumbers = ring
                .Where(ti => setup.TileResources[ti] != ResourceType.Desert)
                .Select(ti => setup.TileNumbers[ti])
                .ToArray();

            if (ringNumbers.Length == 0)
                continue;

            // The expected tokens for this ring segment.
            var expected = tokens[poolIndex..(poolIndex + ringNumbers.Length)];
            poolIndex += ringNumbers.Length;

            // ringNumbers should be a rotation of expected.
            var doubled = ringNumbers.Concat(ringNumbers).ToArray();
            var found = false;
            for (var offset = 0; offset < ringNumbers.Length; offset++)
            {
                var match = true;
                for (var i = 0; i < expected.Length; i++)
                {
                    if (doubled[offset + i] != expected[i])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    found = true;
                    break;
                }
            }

            Assert.That(found, Is.True,
                $"Ring numbers should be a rotation of the expected token subsequence.\n" +
                $"  Expected tokens: [{string.Join(", ", expected)}]\n" +
                $"  Ring numbers: [{string.Join(", ", ringNumbers)}]");
        }

        Assert.That(poolIndex, Is.EqualTo(tokens.Length),
            "All tokens should be consumed by the rings");
    }
}
