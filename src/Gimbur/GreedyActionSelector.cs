namespace Gimbur;

public sealed class GreedyActionSelector
{
    private static readonly Dictionary<int, int> NumberPips = new()
    {
        [2] = 1,
        [3] = 2,
        [4] = 3,
        [5] = 4,
        [6] = 5,
        [8] = 5,
        [9] = 4,
        [10] = 3,
        [11] = 2,
        [12] = 1,
    };

    public CatanAction? ChooseAction(CatanState state, Random rng)
    {
        var actions = state.Actions().OfType<CatanAction>().ToArray();
        if (actions.Length == 0)
        {
            return null;
        }

        if (actions.All(a => a.ActionType == CatanActionType.RollDice))
        {
            var roll = rng.Next(1, 7) + rng.Next(1, 7);
            return actions.FirstOrDefault(a => a.Arg1 == roll) ?? actions[0];
        }

        if (actions.All(a => a.ActionType == CatanActionType.BuyDevCard))
        {
            var weighted = actions
                .Select(a => new
                {
                    Action = a,
                    Weight = state.DevCardsRemaining((Rules.DevCardType)a.Arg1),
                })
                .Where(x => x.Weight > 0)
                .ToArray();

            if (weighted.Length == 0)
            {
                return actions[0];
            }

            var totalWeight = weighted.Sum(x => x.Weight);
            var pick = rng.Next(totalWeight);
            var cumulative = 0;
            foreach (var option in weighted)
            {
                cumulative += option.Weight;
                if (pick < cumulative)
                {
                    return option.Action;
                }
            }

            return weighted[^1].Action;
        }

        var player = state.CurrentPlayer;
        var bestActions = new List<CatanAction>();
        var bestScore = double.NegativeInfinity;
        foreach (var action in actions)
        {
            var next = (CatanState)action.DoCoreAction();
            var score = Evaluate(next, player) + ActionHeuristic(state, action, next, player);
            if (score > bestScore)
            {
                bestScore = score;
                bestActions.Clear();
                bestActions.Add(action);
            }
            else if (Math.Abs(score - bestScore) < 0.0001)
            {
                bestActions.Add(action);
            }
        }

        return bestActions[rng.Next(bestActions.Count)];
    }

    private static double Evaluate(CatanState state, int player)
    {
        if (state.WinnerPlayer == player)
        {
            return 1_000_000_000.0;
        }

        if (state.WinnerPlayer != 0)
        {
            return -1_000_000_000.0;
        }

        var vp = state.VictoryPointsFor(player);
        var resources = state.TotalResourceCards(player);
        var roads = state.Board.RoadCount(player);
        var settlements = state.Board.SettlementCount(player);
        var cities = state.Board.CityCount(player);
        var knights = state.KnightsPlayedFor(player);
        var longestRoadBonus = state.LongestRoadOwner == player ? 1 : 0;
        var largestArmyBonus = state.LargestArmyOwner == player ? 1 : 0;
        var pendingRoadBuilding = state.PendingRoadBuildingPlacementsFor(player);

        var maxOpponentVp = 0;
        for (var p = 1; p <= state.PlayerCount; p++)
        {
            if (p == player)
            {
                continue;
            }

            maxOpponentVp = Math.Max(maxOpponentVp, state.VictoryPointsFor(p));
        }

        return (vp * 500.0)
            + (cities * 120.0)
            + (settlements * 70.0)
            + (roads * 8.0)
            + (resources * 8.0)
            + (knights * 20.0)
            + (longestRoadBonus * 100.0)
            + (largestArmyBonus * 100.0)
            + (pendingRoadBuilding * 50.0)
            - (maxOpponentVp * 220.0);
    }

    private static double ActionHeuristic(CatanState state, CatanAction action, CatanState next, int player)
    {
        return action.ActionType switch
        {
            CatanActionType.PlaceSettlement => ScoreSettlementPlacement(state, action.Arg1),
            CatanActionType.PlaceRoad => ScoreRoadPlacement(state, next, action.Arg1, player),
            CatanActionType.ChooseRobberTile => ScoreRobberPlacement(state, next, action.Arg1, player),
            _ => 0.0,
        };
    }

    private static double ScoreSettlementPlacement(CatanState state, int vertex)
    {
        var board = state.Board;
        var expectedYield = 0;
        var uniqueResources = new HashSet<Rules.ResourceType>();
        foreach (var tile in board.Topology.VertexTiles[vertex])
        {
            var resource = board.TileResource(tile);
            if (resource == Rules.ResourceType.Desert)
            {
                continue;
            }

            uniqueResources.Add(resource);
            var number = board.TileNumber(tile);
            expectedYield += NumberPips.GetValueOrDefault(number, 0);
        }

        var touchesPort = false;
        for (var port = 0; port < board.Topology.PortCount; port++)
        {
            var (a, b) = board.Topology.Ports[port];
            if (a == vertex || b == vertex)
            {
                touchesPort = true;
                break;
            }
        }

        var adjacencyCoverage = board.Topology.VertexTiles[vertex].Length;

        return (expectedYield * 120.0)
            + (uniqueResources.Count * 170.0)
            + (adjacencyCoverage * 35.0)
            + (touchesPort ? 25.0 : 0.0);
    }

    private static double ScoreRoadPlacement(CatanState before, CatanState after, int edgeIndex, int player)
    {
        if (after.LongestRoadOwner == player && before.LongestRoadOwner != player)
        {
            return 1000.0;
        }

        var (a, b) = after.Board.Topology.Edges[edgeIndex];
        var candidates = new HashSet<int> { a, b };
        foreach (var n in after.Board.Topology.VertexNeighbors[a])
        {
            candidates.Add(n);
        }

        foreach (var n in after.Board.Topology.VertexNeighbors[b])
        {
            candidates.Add(n);
        }

        var bestProspect = 0.0;
        foreach (var vertex in candidates)
        {
            if (!IsFutureSettlementCandidate(after.Board, vertex))
            {
                continue;
            }

            if (!TouchesPlayerRoad(after.Board, vertex, player))
            {
                continue;
            }

            var prospect = ScoreSettlementPlacement(after, vertex);
            if (prospect > bestProspect)
            {
                bestProspect = prospect;
            }
        }

        return bestProspect * 0.55;
    }

    private static bool IsFutureSettlementCandidate(Rules.Board board, int vertex)
    {
        if (!board.VertexOccupancy[vertex].IsEmpty)
        {
            return false;
        }

        foreach (var neighbor in board.Topology.VertexNeighbors[vertex])
        {
            if (!board.VertexOccupancy[neighbor].IsEmpty)
            {
                return false;
            }
        }

        return true;
    }

    private static bool TouchesPlayerRoad(Rules.Board board, int vertex, int player)
    {
        foreach (var edge in board.Topology.VertexEdges[vertex])
        {
            if (board.EdgeOccupancy[edge].Player == player)
            {
                return true;
            }
        }

        return false;
    }

    private static double ScoreRobberPlacement(CatanState before, CatanState after, int tileIndex, int player)
    {
        var board = before.Board;
        var number = board.TileNumber(tileIndex);
        var pips = NumberPips.GetValueOrDefault(number, 0);

        double opponentBlocked = 0;
        double selfBlocked = 0;
        var victimsWithCards = new HashSet<int>();

        foreach (var vertex in board.Topology.TileVertices[tileIndex])
        {
            var occ = board.VertexOccupancy[vertex];
            if (occ.IsEmpty)
            {
                continue;
            }

            var buildingWeight = occ.Building == Rules.BuildingType.City ? 2.0 : 1.0;
            var blocked = pips * buildingWeight;
            if (occ.Player == player)
            {
                selfBlocked += blocked;
            }
            else
            {
                opponentBlocked += blocked;
                if (before.TotalResourceCards(occ.Player) > 0)
                {
                    victimsWithCards.Add(occ.Player);
                }
            }
        }

        var stolenAmount =
            (after.TotalResourceCards(player) - before.TotalResourceCards(player)) > 0
                ? 1.0
                : 0.0;

        return (opponentBlocked * 140.0)
            - (selfBlocked * 170.0)
            + (victimsWithCards.Count * 40.0)
            + (stolenAmount * 120.0);
    }
}
