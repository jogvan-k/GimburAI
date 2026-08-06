using Gimbur.Rules;
using Kjarni;

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

    /// <summary>
    /// Build goals the greedy AI considers, ordered by priority.
    /// Each goal is a cost dictionary and a base desirability weight.
    /// </summary>
    private static readonly (string Name, Func<GameConfig, IReadOnlyDictionary<ResourceType, int>> Cost, double BaseWeight)[] BuildGoals =
    [
        ("City", c => c.CityCost, 280.0),
        ("Settlement", c => c.SettlementCost, 200.0),
        ("DevCard", c => c.DevCardCost, 100.0),
        ("Road", c => c.RoadCost, 60.0),
    ];

    public CatanAction? ChooseAction(CatanState state, Random rng)
    {
        var coreActions = state.Actions();
        var actions = new List<CatanAction>(coreActions.Length);
        foreach (var ca in coreActions)
        {
            if (ca.IsDeterministic)
                actions.Add((CatanDeterministicAction)((CoreAction.Deterministic)ca).Item);
            else if (ca.IsStochastic)
                actions.Add((CatanStochasticAction)((CoreAction.Stochastic)ca).Item);
        }

        if (actions.Count == 0)
        {
            return null;
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

        // Score how close the player is to being able to build something useful.
        // This gives the AI a sense of "momentum" -- having the right resources
        // is worth more than having random cards.
        var buildReadiness = ScoreBuildReadiness(state, player);

        return (vp * 500.0)
            + (cities * 120.0)
            + (settlements * 70.0)
            + (roads * 8.0)
            + (resources * 5.0)
            + (knights * 20.0)
            + (longestRoadBonus * 100.0)
            + (largestArmyBonus * 100.0)
            + (pendingRoadBuilding * 50.0)
            + buildReadiness
            - (maxOpponentVp * 220.0);
    }

    /// <summary>
    /// Scores how close the player is to affording each build goal.
    /// Resources that contribute toward a build goal are worth more than idle cards.
    /// </summary>
    private static double ScoreBuildReadiness(CatanState state, int player)
    {
        var score = 0.0;
        var config = state.Config;

        foreach (var (name, costFunc, baseWeight) in BuildGoals)
        {
            var cost = costFunc(config);

            // Check if the build is even possible (supply limits).
            if (name == "City" && state.Board.CityCount(player) >= config.MaxCities)
                continue;
            if (name == "City" && state.Board.SettlementCount(player) == 0)
                continue;
            if (name == "Settlement" && state.Board.SettlementCount(player) >= config.MaxSettlements)
                continue;
            if (name == "Road" && state.Board.RoadCount(player) >= config.MaxRoads)
                continue;

            var totalCostItems = 0;
            var satisfied = 0;
            foreach (var pair in cost)
            {
                totalCostItems += pair.Value;
                satisfied += Math.Min(state.ResourceCountFor(player, pair.Key), pair.Value);
            }

            if (totalCostItems == 0)
                continue;

            var progress = (double)satisfied / totalCostItems;
            // Quadratic scaling: being 80% of the way there is worth much more
            // than being 20% of the way there.
            score += baseWeight * progress * progress;
        }

        return score;
    }

    private static double ActionHeuristic(CatanState state, CatanAction action, CatanState next, int player)
    {
        return action switch
        {
            PlaceSettlementAction => ScoreSettlementPlacement(state, action.Arg1),
            PlaceRoadAction => ScoreRoadPlacement(state, next, action.Arg1, player),
            ChooseRobberTileAction => ScoreRobberPlacement(state, next, action.Arg1, player),
            PlaceCityAction => ScoreCityUpgrade(state, action.Arg1),
            BuyRoadAction => 100.0,
            BuySettlementAction => 300.0,
            UpgradeCityAction => 450.0,
            ChooseBankTradeReceiveAction trade when state._pendingBankTradeGive.HasValue =>
                ScoreBankTrade(state, next, state._pendingBankTradeGive.Value, trade.Resource, player),
            BuyDevCardAction => ScoreDevCardPurchase(state, player),
            _ => 0.0,
        };
    }

    private static double ScoreSettlementPlacement(CatanState state, int vertex)
    {
        var board = state.Board;
        var expectedYield = 0;
        var uniqueResources = new HashSet<ResourceType>();
        foreach (var tile in board.Topology.VertexTiles[vertex])
        {
            var resource = board.TileResource(tile);
            if (resource == ResourceType.Desert)
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

    private static bool IsFutureSettlementCandidate(Board board, int vertex)
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

    private static bool TouchesPlayerRoad(Board board, int vertex, int player)
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

            var buildingWeight = occ.Building == BuildingType.City ? 2.0 : 1.0;
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

    /// <summary>
    /// Scores a city upgrade by the production value of the vertex being upgraded.
    /// Cities double resource production, so upgrading high-pip vertices is very valuable.
    /// </summary>
    private static double ScoreCityUpgrade(CatanState state, int vertex)
    {
        var board = state.Board;
        var productionValue = 0.0;
        foreach (var tile in board.Topology.VertexTiles[vertex])
        {
            var resource = board.TileResource(tile);
            if (resource == ResourceType.Desert)
            {
                continue;
            }

            var number = board.TileNumber(tile);
            var pips = NumberPips.GetValueOrDefault(number, 0);
            // The extra production from city (settlement already produces 1x,
            // city produces 2x, so the upgrade adds 1x production).
            productionValue += pips * 30.0;

            // Bonus for wheat and ore production (helps build more cities).
            if (resource == ResourceType.Wheat || resource == ResourceType.Ore)
            {
                productionValue += pips * 10.0;
            }
        }

        return productionValue;
    }

    /// <summary>
    /// Scores a bank trade by measuring whether it brings the player closer to
    /// affording a useful build. Penalizes trades that don't progress toward any goal.
    /// </summary>
    private static double ScoreBankTrade(CatanState before, CatanState after, ResourceType give, ResourceType receive, int player)
    {
        var config = before.Config;
        var ratio = before.Board.TradeRatio(player, give);

        // Measure improvement in build readiness before vs after trade.
        var readinessBefore = ScoreBuildReadiness(before, player);
        var readinessAfter = ScoreBuildReadiness(after, player);
        var readinessGain = readinessAfter - readinessBefore;

        // Check if the trade enables an immediate build that wasn't possible before.
        var enablesNewBuild = EnablesNewBuild(before, after, player);

        // Penalize trades at worse ratios more heavily.
        var ratioPenalty = ratio >= 4 ? -15.0 : (ratio >= 3 ? -8.0 : -3.0);

        // Large penalty for trades that don't improve build readiness.
        // This is the key anti-cycling mechanism: if a trade doesn't bring
        // the player closer to building something, it's heavily penalized,
        // making EndTurn (which scores 0) the preferred choice.
        if (readinessGain <= 0 && !enablesNewBuild)
        {
            return -200.0 + ratioPenalty;
        }

        // Bonus for trades that enable an immediate build.
        var enableBonus = enablesNewBuild ? 150.0 : 0.0;

        return readinessGain + enableBonus + ratioPenalty;
    }

    /// <summary>
    /// Returns true if the 'after' state can afford a build that the 'before' state could not.
    /// </summary>
    private static bool EnablesNewBuild(CatanState before, CatanState after, int player)
    {
        var config = before.Config;

        // Check city (most VP-valuable build).
        if (before.Board.SettlementCount(player) > 0
            && before.Board.CityCount(player) < config.MaxCities
            && !CanAffordCost(before, player, config.CityCost)
            && CanAffordCost(after, player, config.CityCost))
        {
            return true;
        }

        // Check settlement.
        if (before.Board.SettlementCount(player) < config.MaxSettlements
            && !CanAffordCost(before, player, config.SettlementCost)
            && CanAffordCost(after, player, config.SettlementCost))
        {
            return true;
        }

        // Check dev card.
        if (!CanAffordCost(before, player, config.DevCardCost)
            && CanAffordCost(after, player, config.DevCardCost))
        {
            return true;
        }

        // Check road.
        if (before.Board.RoadCount(player) < config.MaxRoads
            && !CanAffordCost(before, player, config.RoadCost)
            && CanAffordCost(after, player, config.RoadCost))
        {
            return true;
        }

        return false;
    }

    private static bool CanAffordCost(CatanState state, int player, IReadOnlyDictionary<ResourceType, int> cost)
    {
        foreach (var pair in cost)
        {
            if (state.ResourceCountFor(player, pair.Key) < pair.Value)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Scores dev card purchase. Considers proximity to largest army and
    /// the chance of drawing a VP card.
    /// </summary>
    private static double ScoreDevCardPurchase(CatanState state, int player)
    {
        var score = 0.0;
        var config = state.Config;
        var knights = state.KnightsPlayedFor(player);

        // Bonus when close to or competing for largest army.
        var knightsNeeded = config.LargestArmyMinimum - knights;
        if (state.LargestArmyOwner == 0 && knightsNeeded <= 2)
        {
            // Close to claiming largest army (2 VP).
            score += (3 - knightsNeeded) * 60.0;
        }
        else if (state.LargestArmyOwner != 0 && state.LargestArmyOwner != player)
        {
            // Opponent has largest army; buying knights can steal it.
            var opponentKnights = state.KnightsPlayedFor(state.LargestArmyOwner);
            if (knights >= opponentKnights - 1)
            {
                score += 80.0;
            }
        }

        // Consider the chance of drawing a VP card.
        var vpRemaining = state.DevCardsRemaining(DevCardType.VictoryPoint);
        var totalRemaining = 0;
        foreach (DevCardType dt in Enum.GetValues(typeof(DevCardType)))
        {
            totalRemaining += state.DevCardsRemaining(dt);
        }

        if (totalRemaining > 0)
        {
            var vpChance = (double)vpRemaining / totalRemaining;
            // VP cards are very valuable when close to winning.
            var vpToWin = config.VictoryPointsToWin - state.VictoryPointsFor(player);
            if (vpToWin <= 2)
            {
                score += vpChance * 300.0;
            }
            else
            {
                score += vpChance * 100.0;
            }
        }

        return score;
    }
}
