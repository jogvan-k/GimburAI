using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class GameConfigTests
{
    // ── Standard config values ──────────────────────────────────────

    [Test]
    public void Standard_PlayerRange_3To4()
    {
        Assert.That(GameConfig.Standard.MinPlayers, Is.EqualTo(3));
        Assert.That(GameConfig.Standard.MaxPlayers, Is.EqualTo(4));
    }

    [Test]
    public void Standard_BuildingSupply()
    {
        Assert.That(GameConfig.Standard.MaxSettlements, Is.EqualTo(5));
        Assert.That(GameConfig.Standard.MaxCities, Is.EqualTo(4));
        Assert.That(GameConfig.Standard.MaxRoads, Is.EqualTo(15));
    }

    [Test]
    public void Standard_VictoryConditions()
    {
        Assert.That(GameConfig.Standard.VictoryPointsToWin, Is.EqualTo(10));
        Assert.That(GameConfig.Standard.LongestRoadMinimum, Is.EqualTo(5));
        Assert.That(GameConfig.Standard.LargestArmyMinimum, Is.EqualTo(3));
    }

    [Test]
    public void Standard_DiscardThreshold_Is7()
    {
        Assert.That(GameConfig.Standard.DiscardThreshold, Is.EqualTo(7));
    }

    [Test]
    public void Standard_ResourceCardsPerType_Is19()
    {
        Assert.That(GameConfig.Standard.ResourceCardsPerType, Is.EqualTo(19));
    }

    [Test]
    public void Standard_InitialPlacementRounds_Is2()
    {
        Assert.That(GameConfig.Standard.InitialPlacementRounds, Is.EqualTo(2));
    }

    [Test]
    public void Standard_Map_IsStandard()
    {
        Assert.That(GameConfig.Standard.Map, Is.SameAs(MapConfig.Standard));
    }

    // ── Mini config values ──────────────────────────────────────────

    [Test]
    public void Mini_PlayerRange_2To2()
    {
        Assert.That(GameConfig.Mini.MinPlayers, Is.EqualTo(2));
        Assert.That(GameConfig.Mini.MaxPlayers, Is.EqualTo(2));
    }

    [Test]
    public void Mini_BuildingSupply()
    {
        Assert.That(GameConfig.Mini.MaxSettlements, Is.EqualTo(4));
        Assert.That(GameConfig.Mini.MaxCities, Is.EqualTo(3));
        Assert.That(GameConfig.Mini.MaxRoads, Is.EqualTo(10));
    }

    [Test]
    public void Mini_VictoryConditions()
    {
        Assert.That(GameConfig.Mini.VictoryPointsToWin, Is.EqualTo(5));
        Assert.That(GameConfig.Mini.LongestRoadMinimum, Is.EqualTo(4));
        Assert.That(GameConfig.Mini.LargestArmyMinimum, Is.EqualTo(2));
    }

    [Test]
    public void Mini_DiscardThreshold_Is5()
    {
        Assert.That(GameConfig.Mini.DiscardThreshold, Is.EqualTo(5));
    }

    [Test]
    public void Mini_ResourceCardsPerType_Is10()
    {
        Assert.That(GameConfig.Mini.ResourceCardsPerType, Is.EqualTo(10));
    }

    [Test]
    public void Mini_InitialPlacementRounds_Is1()
    {
        Assert.That(GameConfig.Mini.InitialPlacementRounds, Is.EqualTo(1));
    }

    [Test]
    public void Mini_Map_IsMini()
    {
        Assert.That(GameConfig.Mini.Map, Is.SameAs(MapConfig.Mini));
    }

    // ── Development card pool ───────────────────────────────────────

    [Test]
    public void Standard_DevCardCounts()
    {
        var counts = GameConfig.Standard.DevCardCounts;
        Assert.That(counts[DevCardType.Knight], Is.EqualTo(14));
        Assert.That(counts[DevCardType.VictoryPoint], Is.EqualTo(5));
        Assert.That(counts[DevCardType.RoadBuilding], Is.EqualTo(2));
        Assert.That(counts[DevCardType.Monopoly], Is.EqualTo(2));
        Assert.That(counts[DevCardType.YearOfPlenty], Is.EqualTo(2));
    }

    [Test]
    public void Standard_TotalDevCards_Is25()
    {
        Assert.That(GameConfig.Standard.TotalDevCards, Is.EqualTo(25));
    }

    [Test]
    public void Mini_DevCardCounts()
    {
        var counts = GameConfig.Mini.DevCardCounts;
        Assert.That(counts[DevCardType.Knight], Is.EqualTo(7));
        Assert.That(counts[DevCardType.VictoryPoint], Is.EqualTo(2));
        Assert.That(counts[DevCardType.RoadBuilding], Is.EqualTo(1));
        Assert.That(counts[DevCardType.Monopoly], Is.EqualTo(1));
        Assert.That(counts[DevCardType.YearOfPlenty], Is.EqualTo(1));
    }

    [Test]
    public void Mini_TotalDevCards_Is12()
    {
        Assert.That(GameConfig.Mini.TotalDevCards, Is.EqualTo(12));
    }

    // ── Building costs ──────────────────────────────────────────────

    [Test]
    public void RoadCost_WoodAndBrick()
    {
        var cost = GameConfig.Standard.RoadCost;
        Assert.That(cost[ResourceType.Wood], Is.EqualTo(1));
        Assert.That(cost[ResourceType.Brick], Is.EqualTo(1));
        Assert.That(cost.Count, Is.EqualTo(2));
    }

    [Test]
    public void SettlementCost_WoodBrickSheepWheat()
    {
        var cost = GameConfig.Standard.SettlementCost;
        Assert.That(cost[ResourceType.Wood], Is.EqualTo(1));
        Assert.That(cost[ResourceType.Brick], Is.EqualTo(1));
        Assert.That(cost[ResourceType.Sheep], Is.EqualTo(1));
        Assert.That(cost[ResourceType.Wheat], Is.EqualTo(1));
        Assert.That(cost.Count, Is.EqualTo(4));
    }

    [Test]
    public void CityCost_WheatAndOre()
    {
        var cost = GameConfig.Standard.CityCost;
        Assert.That(cost[ResourceType.Wheat], Is.EqualTo(2));
        Assert.That(cost[ResourceType.Ore], Is.EqualTo(3));
        Assert.That(cost.Count, Is.EqualTo(2));
    }

    [Test]
    public void DevCardCost_SheepWheatOre()
    {
        var cost = GameConfig.Standard.DevCardCost;
        Assert.That(cost[ResourceType.Sheep], Is.EqualTo(1));
        Assert.That(cost[ResourceType.Wheat], Is.EqualTo(1));
        Assert.That(cost[ResourceType.Ore], Is.EqualTo(1));
        Assert.That(cost.Count, Is.EqualTo(3));
    }

    [Test]
    public void Mini_SharesSameBuildingCosts()
    {
        // Mini variant uses the same building costs as Standard.
        Assert.That(GameConfig.Mini.RoadCost, Is.EqualTo(GameConfig.Standard.RoadCost));
        Assert.That(GameConfig.Mini.SettlementCost, Is.EqualTo(GameConfig.Standard.SettlementCost));
        Assert.That(GameConfig.Mini.CityCost, Is.EqualTo(GameConfig.Standard.CityCost));
        Assert.That(GameConfig.Mini.DevCardCost, Is.EqualTo(GameConfig.Standard.DevCardCost));
    }

    // ── Small config values ─────────────────────────────────────────

    [Test]
    public void Small_PlayerRange_2To3()
    {
        Assert.That(GameConfig.Small.MinPlayers, Is.EqualTo(2));
        Assert.That(GameConfig.Small.MaxPlayers, Is.EqualTo(3));
    }

    [Test]
    public void Small_BuildingSupply()
    {
        Assert.That(GameConfig.Small.MaxSettlements, Is.EqualTo(5));
        Assert.That(GameConfig.Small.MaxCities, Is.EqualTo(3));
        Assert.That(GameConfig.Small.MaxRoads, Is.EqualTo(12));
    }

    [Test]
    public void Small_VictoryConditions()
    {
        Assert.That(GameConfig.Small.VictoryPointsToWin, Is.EqualTo(7));
        Assert.That(GameConfig.Small.LongestRoadMinimum, Is.EqualTo(5));
        Assert.That(GameConfig.Small.LargestArmyMinimum, Is.EqualTo(3));
    }

    [Test]
    public void Small_DiscardThreshold_Is6()
    {
        Assert.That(GameConfig.Small.DiscardThreshold, Is.EqualTo(6));
    }

    [Test]
    public void Small_ResourceCardsPerType_Is14()
    {
        Assert.That(GameConfig.Small.ResourceCardsPerType, Is.EqualTo(14));
    }

    [Test]
    public void Small_InitialPlacementRounds_Is2()
    {
        Assert.That(GameConfig.Small.InitialPlacementRounds, Is.EqualTo(2));
    }

    [Test]
    public void Small_Map_IsSmall()
    {
        Assert.That(GameConfig.Small.Map, Is.SameAs(MapConfig.Small));
    }

    [Test]
    public void Small_DevCardCounts()
    {
        var counts = GameConfig.Small.DevCardCounts;
        Assert.That(counts[DevCardType.Knight], Is.EqualTo(10));
        Assert.That(counts[DevCardType.VictoryPoint], Is.EqualTo(3));
        Assert.That(counts[DevCardType.RoadBuilding], Is.EqualTo(1));
        Assert.That(counts[DevCardType.Monopoly], Is.EqualTo(1));
        Assert.That(counts[DevCardType.YearOfPlenty], Is.EqualTo(1));
    }

    [Test]
    public void Small_TotalDevCards_Is16()
    {
        Assert.That(GameConfig.Small.TotalDevCards, Is.EqualTo(16));
    }

    [Test]
    public void Small_SharesSameBuildingCosts()
    {
        Assert.That(GameConfig.Small.RoadCost, Is.EqualTo(GameConfig.Standard.RoadCost));
        Assert.That(GameConfig.Small.SettlementCost, Is.EqualTo(GameConfig.Standard.SettlementCost));
        Assert.That(GameConfig.Small.CityCost, Is.EqualTo(GameConfig.Standard.CityCost));
        Assert.That(GameConfig.Small.DevCardCost, Is.EqualTo(GameConfig.Standard.DevCardCost));
    }

    // ── Supply limit enforcement via Board ──────────────────────────

    private Board CreateMiniBoard(int seed = 42)
    {
        var setup = BoardSetup.Generate(MapConfig.Mini, new Random(seed));
        return new Board(setup, GameConfig.Mini);
    }

    [Test]
    public void CanPlaceSettlement_AtLimit_ReturnsFalse()
    {
        var board = CreateMiniBoard(); // MaxSettlements = 4
        var topology = board.Topology;

        // Place 4 settlements for player 1 on well-spaced vertices.
        var placed = PlaceSettlements(board, player: 1, count: 4);
        Assert.That(placed, Is.EqualTo(4), "Should be able to place 4 settlements");
        Assert.That(board.SettlementCount(1), Is.EqualTo(4));

        // Find an empty vertex with no neighbors occupied — should still be blocked by supply limit.
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            if (!board.VertexOccupancy[vi].IsEmpty) continue;
            if (HasOccupiedNeighbor(board, vi)) continue;
            Assert.That(board.CanPlaceSettlement(vi, 1), Is.False,
                $"Player 1 at settlement limit should not be able to place at vertex {vi}");
            return;
        }
        Assert.Fail("Could not find an eligible vertex to test supply limit");
    }

    [Test]
    public void CanPlaceSettlement_BelowLimit_ReturnsTrue()
    {
        var board = CreateMiniBoard(); // MaxSettlements = 4

        // Place 3 settlements (below limit).
        var placed = PlaceSettlements(board, player: 1, count: 3);
        Assert.That(placed, Is.EqualTo(3));
        Assert.That(board.SettlementCount(1), Is.EqualTo(3));

        // Should still be able to place one more.
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            if (!board.VertexOccupancy[vi].IsEmpty) continue;
            if (HasOccupiedNeighbor(board, vi)) continue;
            Assert.That(board.CanPlaceSettlement(vi, 1), Is.True,
                $"Player 1 below limit should be able to place at vertex {vi}");
            return;
        }
        Assert.Fail("Could not find an eligible vertex");
    }

    [Test]
    public void CanPlaceSettlement_SupplyLimitIsPerPlayer()
    {
        var board = CreateMiniBoard(); // MaxSettlements = 4

        // Fill player 1's supply.
        var placed = PlaceSettlements(board, player: 1, count: 4);
        Assert.That(placed, Is.EqualTo(4));

        // Player 2 should still be able to place (their supply is independent).
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            if (!board.VertexOccupancy[vi].IsEmpty) continue;
            if (HasOccupiedNeighbor(board, vi)) continue;
            Assert.That(board.CanPlaceSettlement(vi, 2), Is.True,
                $"Player 2 should not be affected by player 1's limit");
            return;
        }
        Assert.Fail("Could not find an eligible vertex for player 2");
    }

    [Test]
    public void CanUpgradeToCity_AtLimit_ReturnsFalse()
    {
        var board = CreateMiniBoard(); // MaxCities = 3

        // Place settlements and upgrade 3 to cities.
        var placed = PlaceSettlements(board, player: 1, count: 4);
        Assert.That(placed, Is.EqualTo(4));

        var upgraded = 0;
        for (var vi = 0; vi < board.Topology.VertexCount && upgraded < 3; vi++)
        {
            var occ = board.VertexOccupancy[vi];
            if (occ.Building == BuildingType.Settlement && occ.Player == 1)
            {
                board.VertexOccupancy[vi] = new VertexOccupancy(BuildingType.City, 1);
                upgraded++;
            }
        }
        Assert.That(board.CityCount(1), Is.EqualTo(3));

        // Find the remaining settlement — upgrading should be blocked.
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            var occ = board.VertexOccupancy[vi];
            if (occ.Building == BuildingType.Settlement && occ.Player == 1)
            {
                Assert.That(board.CanUpgradeToCity(vi, 1), Is.False,
                    "Player 1 at city limit should not be able to upgrade");
                return;
            }
        }
        Assert.Fail("Should have one settlement remaining");
    }

    [Test]
    public void CanPlaceRoad_AtLimit_ReturnsFalse()
    {
        var board = CreateMiniBoard(); // MaxRoads = 10

        // Place a building so roads have something to connect to.
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);

        // Fill road supply by placing 10 roads along connected edges.
        var roadCount = PlaceRoadsFromVertex(board, startVertex: 0, player: 1, count: 10);
        Assert.That(roadCount, Is.EqualTo(10), "Should place 10 roads");
        Assert.That(board.RoadCount(1), Is.EqualTo(10));

        // Any further road placement should be blocked.
        for (var ei = 0; ei < board.Topology.EdgeCount; ei++)
        {
            if (board.EdgeOccupancy[ei].IsEmpty)
            {
                Assert.That(board.CanPlaceRoad(ei, 1), Is.False,
                    $"Player 1 at road limit should not be able to place at edge {ei}");
                return;
            }
        }
        Assert.Fail("No empty edges found to test");
    }

    // ── Counting helpers ────────────────────────────────────────────

    [Test]
    public void SettlementCount_NewBoard_IsZero()
    {
        var board = CreateMiniBoard();
        Assert.That(board.SettlementCount(1), Is.EqualTo(0));
        Assert.That(board.SettlementCount(2), Is.EqualTo(0));
    }

    [Test]
    public void SettlementCount_CountsOnlySettlements()
    {
        var board = CreateMiniBoard();
        board.VertexOccupancy[0] = new VertexOccupancy(BuildingType.Settlement, 1);
        board.VertexOccupancy[5] = new VertexOccupancy(BuildingType.City, 1);
        board.VertexOccupancy[10] = new VertexOccupancy(BuildingType.Settlement, 2);

        Assert.That(board.SettlementCount(1), Is.EqualTo(1));
        Assert.That(board.CityCount(1), Is.EqualTo(1));
        Assert.That(board.SettlementCount(2), Is.EqualTo(1));
    }

    [Test]
    public void RoadCount_CountsOnlyPlayerRoads()
    {
        var board = CreateMiniBoard();
        board.EdgeOccupancy[0] = new EdgeOccupancy(1);
        board.EdgeOccupancy[1] = new EdgeOccupancy(1);
        board.EdgeOccupancy[2] = new EdgeOccupancy(2);

        Assert.That(board.RoadCount(1), Is.EqualTo(2));
        Assert.That(board.RoadCount(2), Is.EqualTo(1));
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// <summary>
    /// Place settlements for a player on well-spaced vertices (respecting distance rule).
    /// Returns the number of settlements actually placed.
    /// </summary>
    private static int PlaceSettlements(Board board, int player, int count)
    {
        var placed = 0;
        for (var vi = 0; vi < board.Topology.VertexCount && placed < count; vi++)
        {
            if (!board.VertexOccupancy[vi].IsEmpty) continue;
            if (HasOccupiedNeighbor(board, vi)) continue;

            board.VertexOccupancy[vi] = new VertexOccupancy(BuildingType.Settlement, player);
            placed++;
        }
        return placed;
    }

    private static bool HasOccupiedNeighbor(Board board, int vertexIndex)
    {
        foreach (var neighbor in board.Topology.VertexNeighbors[vertexIndex])
        {
            if (!board.VertexOccupancy[neighbor].IsEmpty)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Places roads for a player starting from a vertex, doing a BFS along connected edges.
    /// Returns the number of roads actually placed.
    /// </summary>
    private static int PlaceRoadsFromVertex(Board board, int startVertex, int player, int count)
    {
        var placed = 0;
        var visited = new HashSet<int>();
        var queue = new Queue<int>();
        queue.Enqueue(startVertex);
        visited.Add(startVertex);

        while (queue.Count > 0 && placed < count)
        {
            var vertex = queue.Dequeue();
            foreach (var ei in board.Topology.VertexEdges[vertex])
            {
                if (placed >= count) break;
                if (!board.EdgeOccupancy[ei].IsEmpty) continue;

                board.EdgeOccupancy[ei] = new EdgeOccupancy(player);
                placed++;

                // Find the other end of this edge and enqueue it.
                var (va, vb) = board.Topology.Edges[ei];
                var otherEnd = va == vertex ? vb : va;
                if (visited.Add(otherEnd))
                    queue.Enqueue(otherEnd);
            }
        }
        return placed;
    }
}
