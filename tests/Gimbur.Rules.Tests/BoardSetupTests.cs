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

        var spiralTileOrder = new[] { 0, 1, 2, 6, 11, 15, 18, 17, 16, 12, 7, 3, 4, 5, 10, 14, 13, 8, 9 };
        var assigned = spiralTileOrder
            .Where(ti => setup.TileResources[ti] != ResourceType.Desert)
            .Select(ti => setup.TileNumbers[ti])
            .ToArray();

        Assert.That(assigned, Is.EqualTo(MapConfig.Standard.NumberTokens.ToArray()));
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

        var spiralTileOrder = new[] { 0, 1, 4, 6, 5, 2, 3 };
        var assigned = spiralTileOrder
            .Where(ti => setup.TileResources[ti] != ResourceType.Desert)
            .Select(ti => setup.TileNumbers[ti])
            .ToArray();

        Assert.That(assigned, Is.EqualTo(MapConfig.Mini.NumberTokens.ToArray()));
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
            Assert.That(setup.PortTypes.Length, Is.EqualTo(7));
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

        var spiralTileOrder = new[] { 0, 1, 2, 6, 9, 8, 7, 3, 4, 5 };
        var assigned = spiralTileOrder
            .Where(ti => setup.TileResources[ti] != ResourceType.Desert)
            .Select(ti => setup.TileNumbers[ti])
            .ToArray();

        Assert.That(assigned, Is.EqualTo(MapConfig.Small.NumberTokens.ToArray()));
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
            Assert.That(portCounts[(int)PortType.Generic], Is.EqualTo(3));
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
}
