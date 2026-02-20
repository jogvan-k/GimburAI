using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class BoardTests
{
    private Board CreateStandardBoard(int seed = 42)
    {
        var setup = BoardSetup.Generate(MapConfig.Standard, new Random(seed));
        return new Board(setup);
    }

    [Test]
    public void NewBoard_AllVerticesEmpty()
    {
        var board = CreateStandardBoard();
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            Assert.That(board.VertexOccupancy[vi].IsEmpty, Is.True,
                $"Vertex {vi} should be empty");
        }
    }

    [Test]
    public void NewBoard_AllEdgesEmpty()
    {
        var board = CreateStandardBoard();
        for (var ei = 0; ei < board.Topology.EdgeCount; ei++)
        {
            Assert.That(board.EdgeOccupancy[ei].IsEmpty, Is.True,
                $"Edge {ei} should be empty");
        }
    }

    [Test]
    public void NewBoard_RobberOnDesert()
    {
        var board = CreateStandardBoard();
        Assert.That(board.TileResource(board.RobberTile),
            Is.EqualTo(ResourceType.Desert));
    }

    [Test]
    public void Clone_CreatesIndependentCopy()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);
        board.EdgeOccupancy[0] = new EdgeOccupancy(1);

        var clone = board.Clone();

        // Clone should have the same state.
        Assert.That(clone.VertexOccupancy[0].Player, Is.EqualTo(1));
        Assert.That(clone.EdgeOccupancy[0].Player, Is.EqualTo(1));

        // Modifying the clone should not affect the original.
        clone.VertexOccupancy[0] = VertexOccupancy.Empty;
        Assert.That(board.VertexOccupancy[0].Player, Is.EqualTo(1));
    }

    // ── CanPlaceSettlement ───────────────────────────────────────────

    [Test]
    public void CanPlaceSettlement_EmptyBoard_AllVerticesAvailable()
    {
        var board = CreateStandardBoard();
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            Assert.That(board.CanPlaceSettlement(vi), Is.True,
                $"Vertex {vi} should be available on empty board");
        }
    }

    [Test]
    public void CanPlaceSettlement_OccupiedVertex_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        Assert.That(board.CanPlaceSettlement(10), Is.False);
    }

    [Test]
    public void CanPlaceSettlement_AdjacentToBuilding_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        // All neighbors of vertex 10 should be blocked by the distance rule.
        foreach (var neighbor in board.Topology.VertexNeighbors[10])
        {
            Assert.That(board.CanPlaceSettlement(neighbor), Is.False,
                $"Vertex {neighbor} (neighbor of 10) should be blocked");
        }
    }

    [Test]
    public void CanPlaceSettlement_TwoAway_ReturnsTrue()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        // Find a vertex that is 2 edges away (neighbor of a neighbor, not a direct neighbor).
        var neighbors = board.Topology.VertexNeighbors[10].ToHashSet();
        neighbors.Add(10);

        var found = false;
        foreach (var n in board.Topology.VertexNeighbors[10])
        {
            foreach (var nn in board.Topology.VertexNeighbors[n])
            {
                if (!neighbors.Contains(nn))
                {
                    Assert.That(board.CanPlaceSettlement(nn), Is.True,
                        $"Vertex {nn} (2 away from 10) should be available");
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        Assert.That(found, Is.True, "Should find at least one vertex 2 away");
    }

    // ── CanUpgradeToCity ────────────────────────────────────────────

    [Test]
    public void CanUpgradeToCity_OwnSettlement_ReturnsTrue()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        Assert.That(board.CanUpgradeToCity(10, 1), Is.True);
    }

    [Test]
    public void CanUpgradeToCity_OpponentSettlement_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 2);

        Assert.That(board.CanUpgradeToCity(10, 1), Is.False);
    }

    [Test]
    public void CanUpgradeToCity_EmptyVertex_ReturnsFalse()
    {
        var board = CreateStandardBoard();

        Assert.That(board.CanUpgradeToCity(10, 1), Is.False);
    }

    [Test]
    public void CanUpgradeToCity_AlreadyCity_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.City, 1);

        Assert.That(board.CanUpgradeToCity(10, 1), Is.False);
    }

    // ── CanPlaceRoad ────────────────────────────────────────────────

    [Test]
    public void CanPlaceRoad_ConnectedToBuilding_ReturnsTrue()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        // Any edge connected to vertex 10 should be valid.
        foreach (var ei in board.Topology.VertexEdges[10])
        {
            Assert.That(board.CanPlaceRoad(ei, 1), Is.True,
                $"Edge {ei} should be placeable (connected to building at 10)");
        }
    }

    [Test]
    public void CanPlaceRoad_ConnectedToExistingRoad_ReturnsTrue()
    {
        var board = CreateStandardBoard();
        // Place a road on the first edge of vertex 10.
        var firstEdge = board.Topology.VertexEdges[10][0];
        board.EdgeOccupancy[firstEdge] = new EdgeOccupancy(1);

        // Other edges at vertex 10 should also be valid for player 1.
        foreach (var ei in board.Topology.VertexEdges[10])
        {
            if (ei == firstEdge) continue;
            Assert.That(board.CanPlaceRoad(ei, 1), Is.True,
                $"Edge {ei} should be placeable (connected via road at edge {firstEdge})");
        }
    }

    [Test]
    public void CanPlaceRoad_NoConnection_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        // Edge 35 connects vertices 24 and 31. With nothing nearby, player 1 can't build here.
        Assert.That(board.CanPlaceRoad(35, 1), Is.False);
    }

    [Test]
    public void CanPlaceRoad_AlreadyOccupied_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);
        var ei = board.Topology.VertexEdges[10][0];
        board.EdgeOccupancy[ei] = new EdgeOccupancy(1);

        Assert.That(board.CanPlaceRoad(ei, 1), Is.False, "Already has a road");
        Assert.That(board.CanPlaceRoad(ei, 2), Is.False, "Already has a road (other player)");
    }

    [Test]
    public void CanPlaceRoad_BlockedByOpponentBuilding()
    {
        var board = CreateStandardBoard();
        // Player 1 has a road, player 2 has a building at the junction.
        var t = board.Topology;
        var edge = t.VertexEdges[10][0];
        board.EdgeOccupancy[edge] = new EdgeOccupancy(1);

        // Put opponent building at vertex 10.
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 2);

        // Player 1 should not be able to extend through vertex 10 (blocked by opponent).
        foreach (var ei in t.VertexEdges[10])
        {
            if (ei == edge) continue;
            Assert.That(board.CanPlaceRoad(ei, 1), Is.False,
                $"Edge {ei} should be blocked by opponent building at vertex 10");
        }
    }

    // ── TradeRatio ──────────────────────────────────────────────────

    [Test]
    public void TradeRatio_NoPort_Returns4()
    {
        var board = CreateStandardBoard();
        Assert.That(board.TradeRatio(1, ResourceType.Wood), Is.EqualTo(4));
    }

    [Test]
    public void TradeRatio_GenericPort_Returns3()
    {
        var board = CreateStandardBoard();
        // Find a generic port and place a settlement on one of its vertices.
        for (var pi = 0; pi < board.Topology.PortCount; pi++)
        {
            if (board.Setup.PortTypes[pi] != PortType.Generic) continue;

            var (va, _) = board.Topology.Ports[pi];
            board.VertexOccupancy[va] = new VertexOccupancy(BuildingType.Settlement, 1);

            Assert.That(board.TradeRatio(1, ResourceType.Wood), Is.EqualTo(3));
            Assert.That(board.TradeRatio(1, ResourceType.Ore), Is.EqualTo(3));
            return;
        }
        Assert.Fail("No generic port found");
    }

    [Test]
    public void TradeRatio_ResourcePort_Returns2ForMatchingResource()
    {
        var board = CreateStandardBoard();
        // Find a wood port and place a settlement on one of its vertices.
        for (var pi = 0; pi < board.Topology.PortCount; pi++)
        {
            if (board.Setup.PortTypes[pi] != PortType.Wood) continue;

            var (va, _) = board.Topology.Ports[pi];
            board.VertexOccupancy[va] = new VertexOccupancy(BuildingType.Settlement, 1);

            Assert.That(board.TradeRatio(1, ResourceType.Wood), Is.EqualTo(2));
            // Non-matching resource should still be 4.
            Assert.That(board.TradeRatio(1, ResourceType.Ore), Is.EqualTo(4));
            return;
        }
        Assert.Fail("No wood port found");
    }

    // ── TilesForRoll ────────────────────────────────────────────────

    [Test]
    public void TilesForRoll_ExcludesRobberTile()
    {
        var board = CreateStandardBoard();
        var robberNumber = board.TileNumber(board.RobberTile);

        // If robber is on desert, it has number 0, which is never rolled.
        // Move robber to a non-desert tile.
        var nonDesert = Enumerable.Range(0, board.Topology.TileCount)
            .First(ti => board.TileResource(ti) != ResourceType.Desert);
        board.RobberTile = nonDesert;
        var number = board.TileNumber(nonDesert);

        var tiles = board.TilesForRoll(number).ToList();
        Assert.That(tiles, Does.Not.Contain(nonDesert),
            "Robber tile should be excluded from production");
    }
}
