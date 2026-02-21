using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class BoardTests
{
    private Board CreateStandardBoard(int seed = 42)
    {
        var setup = BoardSetup.Generate(MapConfig.Standard, new Random(seed));
        return new Board(setup, GameConfig.Standard);
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
            Assert.That(board.CanPlaceSettlement(vi, 1), Is.True,
                $"Vertex {vi} should be available on empty board");
        }
    }

    [Test]
    public void CanPlaceSettlement_OccupiedVertex_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        Assert.That(board.CanPlaceSettlement(10, 1), Is.False);
    }

    [Test]
    public void CanPlaceSettlement_AdjacentToBuilding_ReturnsFalse()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 1);

        // All neighbors of vertex 10 should be blocked by the distance rule.
        foreach (var neighbor in board.Topology.VertexNeighbors[10])
        {
            Assert.That(board.CanPlaceSettlement(neighbor, 2), Is.False,
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
                    Assert.That(board.CanPlaceSettlement(nn, 1), Is.True,
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

    // ── Piece counting ────────────────────────────────────────────

    [Test]
    public void SettlementCount_ReturnsCorrectCountPerPlayer()
    {
        var board = CreateStandardBoard();
        Assert.That(board.SettlementCount(1), Is.EqualTo(0));

        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);
        board.VertexOccupancy[5] = new VertexOccupancy(BuildingType.Settlement, 1);
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 2);

        Assert.That(board.SettlementCount(1), Is.EqualTo(2));
        Assert.That(board.SettlementCount(2), Is.EqualTo(1));
        Assert.That(board.SettlementCount(3), Is.EqualTo(0));
    }

    [Test]
    public void CityCount_ReturnsCorrectCountPerPlayer()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.City, 1);
        board.VertexOccupancy[5] = new VertexOccupancy(BuildingType.City, 2);
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.City, 2);

        Assert.That(board.CityCount(1), Is.EqualTo(1));
        Assert.That(board.CityCount(2), Is.EqualTo(2));
    }

    [Test]
    public void RoadCount_ReturnsCorrectCountPerPlayer()
    {
        var board = CreateStandardBoard();
        board.EdgeOccupancy[0] = new EdgeOccupancy(1);
        board.EdgeOccupancy[1] = new EdgeOccupancy(1);
        board.EdgeOccupancy[2] = new EdgeOccupancy(1);
        board.EdgeOccupancy[5] = new EdgeOccupancy(2);

        Assert.That(board.RoadCount(1), Is.EqualTo(3));
        Assert.That(board.RoadCount(2), Is.EqualTo(1));
        Assert.That(board.RoadCount(3), Is.EqualTo(0));
    }

    [Test]
    public void SettlementCount_DoesNotCountCities()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);
        board.VertexOccupancy[5] = new VertexOccupancy(BuildingType.City, 1);

        Assert.That(board.SettlementCount(1), Is.EqualTo(1));
    }

    [Test]
    public void CityCount_DoesNotCountSettlements()
    {
        var board = CreateStandardBoard();
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);
        board.VertexOccupancy[5] = new VertexOccupancy(BuildingType.City, 1);

        Assert.That(board.CityCount(1), Is.EqualTo(1));
    }

    // ── Supply limit tests ──────────────────────────────────────────

    private Board CreateMiniBoard(int seed = 42)
    {
        var setup = BoardSetup.Generate(MapConfig.Mini, new Random(seed));
        return new Board(setup, GameConfig.Mini);
    }

    /// <summary>
    /// Finds vertex indices that satisfy the distance rule (no two adjacent)
    /// using a greedy scan. Returns the requested count or fewer if not enough exist.
    /// </summary>
    private static List<int> FindNonAdjacentVertices(BoardTopology topology, int count)
    {
        var result = new List<int>();
        var blocked = new HashSet<int>();

        for (var vi = 0; vi < topology.VertexCount && result.Count < count; vi++)
        {
            if (blocked.Contains(vi)) continue;
            result.Add(vi);
            blocked.Add(vi);
            foreach (var neighbor in topology.VertexNeighbors[vi])
                blocked.Add(neighbor);
        }
        return result;
    }

    [Test]
    public void CanPlaceSettlement_SupplyLimitReached_ReturnsFalse()
    {
        // Mini config: MaxSettlements = 4, 24 vertices
        var board = CreateMiniBoard();
        var vertices = FindNonAdjacentVertices(board.Topology, GameConfig.Mini.MaxSettlements + 1);
        Assert.That(vertices.Count, Is.GreaterThanOrEqualTo(GameConfig.Mini.MaxSettlements + 1),
            "Need at least MaxSettlements+1 non-adjacent vertices for this test");

        // Place settlements up to the limit.
        for (var i = 0; i < GameConfig.Mini.MaxSettlements; i++)
        {
            Assert.That(board.CanPlaceSettlement(vertices[i], 1), Is.True,
                $"Should be able to place settlement {i + 1} of {GameConfig.Mini.MaxSettlements}");
            board.VertexOccupancy[vertices[i]] = new VertexOccupancy(BuildingType.Settlement, 1);
        }

        // The next placement should fail due to supply limit.
        Assert.That(board.CanPlaceSettlement(vertices[GameConfig.Mini.MaxSettlements], 1), Is.False,
            "Should not be able to exceed settlement supply limit");
    }

    [Test]
    public void CanPlaceSettlement_SupplyLimitIsPerPlayer()
    {
        var board = CreateMiniBoard();
        var vertices = FindNonAdjacentVertices(board.Topology, GameConfig.Mini.MaxSettlements + 1);

        // Fill player 1's supply.
        for (var i = 0; i < GameConfig.Mini.MaxSettlements; i++)
            board.VertexOccupancy[vertices[i]] = new VertexOccupancy(BuildingType.Settlement, 1);

        // Player 2 should still be able to place (on a vertex not adjacent to player 1's).
        // Find a vertex not blocked by adjacency.
        var available = -1;
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            if (!board.VertexOccupancy[vi].IsEmpty) continue;
            var blocked = false;
            foreach (var n in board.Topology.VertexNeighbors[vi])
            {
                if (!board.VertexOccupancy[n].IsEmpty)
                {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) { available = vi; break; }
        }

        Assert.That(available, Is.GreaterThanOrEqualTo(0), "Should find available vertex for player 2");
        Assert.That(board.CanPlaceSettlement(available, 2), Is.True,
            "Player 2 should not be affected by player 1's supply limit");
    }

    [Test]
    public void CanUpgradeToCity_SupplyLimitReached_ReturnsFalse()
    {
        var board = CreateMiniBoard();
        var vertices = FindNonAdjacentVertices(board.Topology, GameConfig.Mini.MaxCities + 1);
        Assert.That(vertices.Count, Is.GreaterThanOrEqualTo(GameConfig.Mini.MaxCities + 1),
            "Need at least MaxCities+1 non-adjacent vertices for this test");

        // Place settlements and upgrade to cities up to the limit.
        for (var i = 0; i < GameConfig.Mini.MaxCities; i++)
        {
            board.VertexOccupancy[vertices[i]] = new VertexOccupancy(BuildingType.City, 1);
        }

        // Place one more settlement to try to upgrade.
        board.VertexOccupancy[vertices[GameConfig.Mini.MaxCities]] =
            new VertexOccupancy(BuildingType.Settlement, 1);

        Assert.That(board.CanUpgradeToCity(vertices[GameConfig.Mini.MaxCities], 1), Is.False,
            "Should not be able to exceed city supply limit");
    }

    [Test]
    public void CanPlaceRoad_SupplyLimitReached_ReturnsFalse()
    {
        var board = CreateMiniBoard();
        // Place a building so roads have a valid connection point.
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);

        // Build a chain of roads from vertex 0 up to the limit.
        var placedRoads = 0;
        var visited = new HashSet<int>();
        var frontier = new Queue<int>();
        frontier.Enqueue(0);
        visited.Add(0);

        // BFS-style: place roads emanating from connected vertices.
        while (placedRoads < GameConfig.Mini.MaxRoads && frontier.Count > 0)
        {
            var vertex = frontier.Dequeue();
            foreach (var ei in board.Topology.VertexEdges[vertex])
            {
                if (placedRoads >= GameConfig.Mini.MaxRoads) break;
                if (!board.EdgeOccupancy[ei].IsEmpty) continue;
                board.EdgeOccupancy[ei] = new EdgeOccupancy(1);
                placedRoads++;

                // Add the other endpoint of this edge to the frontier.
                var (va, vb) = board.Topology.Edges[ei];
                var other = (va == vertex) ? vb : va;
                if (visited.Add(other))
                    frontier.Enqueue(other);
            }
        }

        Assert.That(placedRoads, Is.EqualTo(GameConfig.Mini.MaxRoads),
            "Should have placed exactly MaxRoads roads");
        Assert.That(board.RoadCount(1), Is.EqualTo(GameConfig.Mini.MaxRoads));

        // Find an unoccupied edge connected to our network.
        var blockedEdge = -1;
        foreach (var v in visited)
        {
            foreach (var ei in board.Topology.VertexEdges[v])
            {
                if (board.EdgeOccupancy[ei].IsEmpty)
                {
                    blockedEdge = ei;
                    break;
                }
            }
            if (blockedEdge >= 0) break;
        }

        Assert.That(blockedEdge, Is.GreaterThanOrEqualTo(0),
            "Should find an unoccupied connected edge");
        Assert.That(board.CanPlaceRoad(blockedEdge, 1), Is.False,
            "Should not be able to exceed road supply limit");
    }

    // ── GameConfig values ───────────────────────────────────────────

    [Test]
    public void StandardConfig_HasExpectedValues()
    {
        var cfg = GameConfig.Standard;
        Assert.That(cfg.MinPlayers, Is.EqualTo(3));
        Assert.That(cfg.MaxPlayers, Is.EqualTo(4));
        Assert.That(cfg.MaxSettlements, Is.EqualTo(5));
        Assert.That(cfg.MaxCities, Is.EqualTo(4));
        Assert.That(cfg.MaxRoads, Is.EqualTo(15));
        Assert.That(cfg.VictoryPointsToWin, Is.EqualTo(10));
        Assert.That(cfg.LongestRoadMinimum, Is.EqualTo(5));
        Assert.That(cfg.LargestArmyMinimum, Is.EqualTo(3));
        Assert.That(cfg.DiscardThreshold, Is.EqualTo(7));
        Assert.That(cfg.ResourceCardsPerType, Is.EqualTo(19));
        Assert.That(cfg.InitialPlacementRounds, Is.EqualTo(2));
        Assert.That(cfg.TotalDevCards, Is.EqualTo(25));
        Assert.That(cfg.Map, Is.SameAs(MapConfig.Standard));
    }

    [Test]
    public void MiniConfig_HasExpectedValues()
    {
        var cfg = GameConfig.Mini;
        Assert.That(cfg.MinPlayers, Is.EqualTo(2));
        Assert.That(cfg.MaxPlayers, Is.EqualTo(2));
        Assert.That(cfg.MaxSettlements, Is.EqualTo(4));
        Assert.That(cfg.MaxCities, Is.EqualTo(3));
        Assert.That(cfg.MaxRoads, Is.EqualTo(10));
        Assert.That(cfg.VictoryPointsToWin, Is.EqualTo(5));
        Assert.That(cfg.LongestRoadMinimum, Is.EqualTo(4));
        Assert.That(cfg.LargestArmyMinimum, Is.EqualTo(2));
        Assert.That(cfg.DiscardThreshold, Is.EqualTo(5));
        Assert.That(cfg.ResourceCardsPerType, Is.EqualTo(10));
        Assert.That(cfg.InitialPlacementRounds, Is.EqualTo(1));
        Assert.That(cfg.TotalDevCards, Is.EqualTo(12));
        Assert.That(cfg.Map, Is.SameAs(MapConfig.Mini));
    }

    [Test]
    public void StandardConfig_BuildingCosts_AreCorrect()
    {
        var cfg = GameConfig.Standard;

        // Road: 1 wood + 1 brick
        Assert.That(cfg.RoadCost[ResourceType.Wood], Is.EqualTo(1));
        Assert.That(cfg.RoadCost[ResourceType.Brick], Is.EqualTo(1));
        Assert.That(cfg.RoadCost.Count, Is.EqualTo(2));

        // Settlement: 1 wood + 1 brick + 1 sheep + 1 wheat
        Assert.That(cfg.SettlementCost[ResourceType.Wood], Is.EqualTo(1));
        Assert.That(cfg.SettlementCost[ResourceType.Brick], Is.EqualTo(1));
        Assert.That(cfg.SettlementCost[ResourceType.Sheep], Is.EqualTo(1));
        Assert.That(cfg.SettlementCost[ResourceType.Wheat], Is.EqualTo(1));
        Assert.That(cfg.SettlementCost.Count, Is.EqualTo(4));

        // City: 2 wheat + 3 ore
        Assert.That(cfg.CityCost[ResourceType.Wheat], Is.EqualTo(2));
        Assert.That(cfg.CityCost[ResourceType.Ore], Is.EqualTo(3));
        Assert.That(cfg.CityCost.Count, Is.EqualTo(2));

        // Dev card: 1 sheep + 1 wheat + 1 ore
        Assert.That(cfg.DevCardCost[ResourceType.Sheep], Is.EqualTo(1));
        Assert.That(cfg.DevCardCost[ResourceType.Wheat], Is.EqualTo(1));
        Assert.That(cfg.DevCardCost[ResourceType.Ore], Is.EqualTo(1));
        Assert.That(cfg.DevCardCost.Count, Is.EqualTo(3));
    }

    [Test]
    public void StandardConfig_DevCardCounts_AreCorrect()
    {
        var cfg = GameConfig.Standard;
        Assert.That(cfg.DevCardCounts[DevCardType.Knight], Is.EqualTo(14));
        Assert.That(cfg.DevCardCounts[DevCardType.VictoryPoint], Is.EqualTo(5));
        Assert.That(cfg.DevCardCounts[DevCardType.RoadBuilding], Is.EqualTo(2));
        Assert.That(cfg.DevCardCounts[DevCardType.Monopoly], Is.EqualTo(2));
        Assert.That(cfg.DevCardCounts[DevCardType.YearOfPlenty], Is.EqualTo(2));
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
