using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

public class CatanStateTests
{
    [Test]
    public void StandardInitialPlacement_UsesSnakeOrder()
    {
        var state = new Gimbur.CatanState(GameConfig.Standard, 3, new Random(123));
        var playersAtSettlementTurns = new List<int>();

        while (state.Stage is not TurnStage.PreRoll)
        {
            if (state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
            {
                playersAtSettlementTurns.Add(state.CurrentPlayer);
            }

            var action = state.Actions().Cast<Gimbur.CatanAction>().First();
            state = (Gimbur.CatanState)action.DoCoreAction();
        }

        Assert.That(playersAtSettlementTurns, Is.EqualTo(new[] { 1, 2, 3, 3, 2, 1 }));
    }

    [Test]
    public void RoadActions_AreAdjacentToPendingSettlement()
    {
        var state = new Gimbur.CatanState(GameConfig.Standard, 3, new Random(123));
        var settlement = state.Actions().Cast<Gimbur.CatanAction>().First();
        state = (Gimbur.CatanState)settlement.DoCoreAction();

        Assert.That(state.Stage, Is.EqualTo(TurnStage.PlaceFirstRoad));
        Assert.That(state.PendingSettlementVertex, Is.Not.Null);

        var pending = state.PendingSettlementVertex!.Value;
        var expected = state.Board.Topology.VertexEdges[pending].ToHashSet();
        var actual = state.Actions().Cast<Gimbur.CatanAction>().Select(a => a.TargetIndex).ToHashSet();

        Assert.That(actual, Is.SubsetOf(expected));
        Assert.That(actual.Count, Is.GreaterThan(0));
    }

    [Test]
    public void MiniInitialPlacement_CompletesToPreRollTurnOne()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(55));

        while (state.Stage is not TurnStage.PreRoll)
        {
            var action = state.Actions().Cast<Gimbur.CatanAction>().First();
            state = (Gimbur.CatanState)action.DoCoreAction();
        }

        Assert.That(state.TurnNumber, Is.EqualTo(1));
        Assert.That(state.CurrentPlayer, Is.EqualTo(1));
        Assert.That(state.Board.SettlementCount(1), Is.EqualTo(1));
        Assert.That(state.Board.SettlementCount(2), Is.EqualTo(1));
        Assert.That(state.Board.RoadCount(1), Is.EqualTo(1));
        Assert.That(state.Board.RoadCount(2), Is.EqualTo(1));
    }

    [Test]
    public void Serialization_RoundTrips()
    {
        var state = new Gimbur.CatanState(GameConfig.Standard, 3, new Random(777));

        for (var i = 0; i < 4; i++)
        {
            var action = state.Actions().Cast<Gimbur.CatanAction>().First();
            state = (Gimbur.CatanState)action.DoCoreAction();
        }

        var serialized = state.SerializeHumanReadable();
        var parsed = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Standard, 3, serialized);

        Assert.That(parsed.SerializeHumanReadable(), Is.EqualTo(serialized));
        Assert.That(parsed.GetHashCode(), Is.EqualTo(state.GetHashCode()));
    }

    [Test]
    public void SecondSettlement_GrantsAdjacentNonDesertResources()
    {
        var state = new Gimbur.CatanState(GameConfig.Standard, 3, new Random(123));

        while (!(state.Stage == TurnStage.PlaceSecondSettlement && state.CurrentPlayer == 3))
        {
            var step = state.Actions().Cast<Gimbur.CatanAction>().First();
            state = (Gimbur.CatanState)step.DoCoreAction();
        }

        var player = state.CurrentPlayer;
        var settlementAction = state.Actions().Cast<Gimbur.CatanAction>().First();
        var targetVertex = settlementAction.TargetIndex;

        var expected = new Dictionary<ResourceType, int>
        {
            [ResourceType.Wood] = 0,
            [ResourceType.Brick] = 0,
            [ResourceType.Sheep] = 0,
            [ResourceType.Wheat] = 0,
            [ResourceType.Ore] = 0,
        };

        foreach (var tileIndex in state.Board.Topology.VertexTiles[targetVertex])
        {
            var resource = state.Board.TileResource(tileIndex);
            if (resource != ResourceType.Desert)
            {
                expected[resource]++;
            }
        }

        var next = (Gimbur.CatanState)settlementAction.DoCoreAction();
        Assert.That(next.Stage, Is.EqualTo(TurnStage.PlaceSecondRoad));

        Assert.That(next.ResourceCountFor(player, ResourceType.Wood), Is.EqualTo(expected[ResourceType.Wood]));
        Assert.That(next.ResourceCountFor(player, ResourceType.Brick), Is.EqualTo(expected[ResourceType.Brick]));
        Assert.That(next.ResourceCountFor(player, ResourceType.Sheep), Is.EqualTo(expected[ResourceType.Sheep]));
        Assert.That(next.ResourceCountFor(player, ResourceType.Wheat), Is.EqualTo(expected[ResourceType.Wheat]));
        Assert.That(next.ResourceCountFor(player, ResourceType.Ore), Is.EqualTo(expected[ResourceType.Ore]));
    }

    [Test]
    public void RollDice_ProducesResources_AndSevenTriggersRobberStage()
    {
        var state = ReachPreRoll(new Gimbur.CatanState(GameConfig.Standard, 3, new Random(42)));
        var beforeTotal = TotalResources(state);

        var rollAction = state.Actions()
            .Cast<Gimbur.CatanAction>()
            .Where(a => a.ActionType == Gimbur.CatanActionType.RollDice)
            .OrderByDescending(a => ExpectedProductionGain(state, a.Arg1))
            .ThenBy(a => a.Arg1 == 7 ? 1 : 0)
            .First();

        var expectedGain = ExpectedProductionGain(state, rollAction.Arg1);
        var next = (Gimbur.CatanState)rollAction.DoCoreAction();

        if (rollAction.Arg1 == 7)
        {
            Assert.That(next.Stage, Is.EqualTo(TurnStage.ChooseRobberLocation));
        }
        else
        {
            Assert.That(next.Stage, Is.EqualTo(TurnStage.BuildTrade));
            Assert.That(TotalResources(next), Is.EqualTo(beforeTotal + expectedGain));
        }
    }

    [Test]
    public void SevenRoll_DiscardsHalf_WhenAboveThreshold()
    {
        var state = ReachPreRoll(new Gimbur.CatanState(GameConfig.Standard, 3, new Random(42)));
        var serialized = state.SerializeHumanReadable();
        for (var player = 1; player <= 3; player++)
        {
            serialized = SetResource(serialized, state, player, ResourceType.Wood, 8);
            serialized = SetResource(serialized, state, player, ResourceType.Brick, 0);
            serialized = SetResource(serialized, state, player, ResourceType.Sheep, 0);
            serialized = SetResource(serialized, state, player, ResourceType.Wheat, 0);
            serialized = SetResource(serialized, state, player, ResourceType.Ore, 0);
        }

        var loaded = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Standard, 3, serialized);
        var roll7 = loaded.Actions()
            .Cast<Gimbur.CatanAction>()
            .Single(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 == 7);
        var next = (Gimbur.CatanState)roll7.DoCoreAction();

        Assert.That(next.Stage, Is.EqualTo(TurnStage.ChooseRobberLocation));
        for (var player = 1; player <= 3; player++)
        {
            Assert.That(next.TotalResourceCards(player), Is.EqualTo(4));
        }
    }

    [Test]
    public void RobberPlacement_StealsOneCardFromAdjacentOpponent()
    {
        var baseState = ReachPreRoll(new Gimbur.CatanState(GameConfig.Standard, 3, new Random(42)));
        var roll7 = baseState.Actions()
            .Cast<Gimbur.CatanAction>()
            .Single(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 == 7);
        var robberState = (Gimbur.CatanState)roll7.DoCoreAction();

        var current = robberState.CurrentPlayer;
        var targetTile = FindRobberTargetWithVictim(robberState, out var victim);
        Assert.That(victim, Is.GreaterThan(0));

        var serialized = robberState.SerializeHumanReadable();
        serialized = SetResource(serialized, robberState, current, ResourceType.Brick, 0);
        serialized = SetResource(serialized, robberState, victim, ResourceType.Brick, 3);
        serialized = SetResource(serialized, robberState, victim, ResourceType.Wood, 0);
        serialized = SetResource(serialized, robberState, victim, ResourceType.Sheep, 0);
        serialized = SetResource(serialized, robberState, victim, ResourceType.Wheat, 0);
        serialized = SetResource(serialized, robberState, victim, ResourceType.Ore, 0);

        var loaded = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Standard, 3, serialized);
        var beforeCurrent = loaded.ResourceCountFor(current, ResourceType.Brick);
        var beforeVictim = loaded.ResourceCountFor(victim, ResourceType.Brick);
        var place = new Gimbur.CatanAction(loaded, Gimbur.CatanActionType.ChooseRobberTile, targetTile);
        var next = (Gimbur.CatanState)place.DoCoreAction();

        Assert.That(next.Stage, Is.EqualTo(TurnStage.BuildTrade));
        Assert.That(next.ResourceCountFor(current, ResourceType.Brick), Is.EqualTo(beforeCurrent + 1));
        Assert.That(next.ResourceCountFor(victim, ResourceType.Brick), Is.EqualTo(beforeVictim - 1));
    }

    [Test]
    public void BuildTrade_GeneratesTradeBuildAndDevActions()
    {
        var preRoll = ReachPreRoll(new Gimbur.CatanState(GameConfig.Standard, 3, new Random(42)));
        var roll = preRoll.Actions()
            .Cast<Gimbur.CatanAction>()
            .Where(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 != 7)
            .First();
        var buildTrade = (Gimbur.CatanState)roll.DoCoreAction();

        var current = buildTrade.CurrentPlayer;
        var serialized = buildTrade.SerializeHumanReadable();
        foreach (var resource in new[] { ResourceType.Wood, ResourceType.Brick, ResourceType.Sheep, ResourceType.Wheat, ResourceType.Ore })
        {
            serialized = SetResource(serialized, buildTrade, current, resource, 6);
        }

        serialized = SetDevCard(serialized, buildTrade, current, DevCardType.Monopoly, 1);
        serialized = SetDevCard(serialized, buildTrade, current, DevCardType.YearOfPlenty, 1);
        serialized = SetDevCard(serialized, buildTrade, current, DevCardType.RoadBuilding, 1);
        serialized = SetDevCard(serialized, buildTrade, current, DevCardType.Knight, 1);
        var loaded = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Standard, 3, serialized);

        var actions = loaded.Actions().Cast<Gimbur.CatanAction>().ToArray();
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.EndTurn), Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.BankTrade), Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.PlaceRoad), Is.True);
        Assert.That(
            actions.Any(a => a.ActionType is Gimbur.CatanActionType.PlaceSettlement or Gimbur.CatanActionType.BuildCity),
            Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.BuyDevCard), Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.PlayMonopoly), Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.PlayYearOfPlenty), Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.PlayRoadBuilding), Is.True);
        Assert.That(actions.Any(a => a.ActionType == Gimbur.CatanActionType.PlayKnight), Is.True);
    }

    [Test]
    public void DevCardBoughtThisTurn_CannotBePlayedUntilNextTurn()
    {
        var preRoll = ReachPreRoll(new Gimbur.CatanState(GameConfig.Mini, 2, new Random(42)));
        var toBuildTrade = preRoll.Actions()
            .Cast<Gimbur.CatanAction>()
            .First(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 != 7);
        var buildTrade = (Gimbur.CatanState)toBuildTrade.DoCoreAction();

        var current = buildTrade.CurrentPlayer;
        var serialized = buildTrade.SerializeHumanReadable();
        foreach (var resource in new[] { ResourceType.Wood, ResourceType.Brick, ResourceType.Sheep, ResourceType.Wheat, ResourceType.Ore })
        {
            serialized = SetResource(serialized, buildTrade, current, resource, 6);
        }

        var loaded = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Mini, 2, serialized);
        var buyKnight = loaded.Actions()
            .Cast<Gimbur.CatanAction>()
            .First(a => a.ActionType == Gimbur.CatanActionType.BuyDevCard && a.Arg1 == (int)DevCardType.Knight);
        var afterBuy = (Gimbur.CatanState)buyKnight.DoCoreAction();

        Assert.That(afterBuy.Actions().Cast<Gimbur.CatanAction>().Any(a => a.ActionType == Gimbur.CatanActionType.PlayKnight), Is.False);

        var endTurnP1 = afterBuy.Actions().Cast<Gimbur.CatanAction>().First(a => a.ActionType == Gimbur.CatanActionType.EndTurn);
        var p2PreRoll = (Gimbur.CatanState)endTurnP1.DoCoreAction();
        var p2Roll = p2PreRoll.Actions().Cast<Gimbur.CatanAction>().First(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 != 7);
        var p2BuildTrade = (Gimbur.CatanState)p2Roll.DoCoreAction();
        var endTurnP2 = p2BuildTrade.Actions().Cast<Gimbur.CatanAction>().First(a => a.ActionType == Gimbur.CatanActionType.EndTurn);
        var p1PreRollAgain = (Gimbur.CatanState)endTurnP2.DoCoreAction();
        var p1RollAgain = p1PreRollAgain.Actions().Cast<Gimbur.CatanAction>().First(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 != 7);
        var p1BuildTradeAgain = (Gimbur.CatanState)p1RollAgain.DoCoreAction();

        Assert.That(p1BuildTradeAgain.Actions().Cast<Gimbur.CatanAction>().Any(a => a.ActionType == Gimbur.CatanActionType.PlayKnight), Is.True);
    }

    [Test]
    public void PlayKnight_IsSingleAction_AndTransitionsToRobberPlacementStage()
    {
        var preRoll = ReachPreRoll(new Gimbur.CatanState(GameConfig.Mini, 2, new Random(13)));
        var toBuildTrade = preRoll.Actions()
            .Cast<Gimbur.CatanAction>()
            .First(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 != 7);
        var buildTrade = (Gimbur.CatanState)toBuildTrade.DoCoreAction();

        var current = buildTrade.CurrentPlayer;
        var serialized = buildTrade.SerializeHumanReadable();
        serialized = SetDevCard(serialized, buildTrade, current, DevCardType.Knight, 1);
        var loaded = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Mini, 2, serialized);

        var knightActions = loaded.Actions()
            .Cast<Gimbur.CatanAction>()
            .Where(a => a.ActionType == Gimbur.CatanActionType.PlayKnight)
            .ToArray();
        Assert.That(knightActions.Length, Is.EqualTo(1));
        Assert.That(knightActions[0].Arg1, Is.EqualTo(0));

        var afterKnight = (Gimbur.CatanState)knightActions[0].DoCoreAction();
        Assert.That(afterKnight.Stage, Is.EqualTo(TurnStage.ChooseRobberLocation));
        Assert.That(afterKnight.Actions().Cast<Gimbur.CatanAction>().All(a => a.ActionType == Gimbur.CatanActionType.ChooseRobberTile), Is.True);
    }

    [Test]
    public void PlayRoadBuilding_StartsTwoConsecutiveFreeRoadPlacements()
    {
        Gimbur.CatanState? buildTrade = null;
        for (var seed = 1; seed <= 100; seed++)
        {
            var preRollCandidate = ReachPreRoll(new Gimbur.CatanState(GameConfig.Mini, 2, new Random(seed)));
            var toBuildTradeCandidate = preRollCandidate.Actions()
                .Cast<Gimbur.CatanAction>()
                .First(a => a.ActionType == Gimbur.CatanActionType.RollDice && a.Arg1 != 7);
            var buildTradeCandidate = (Gimbur.CatanState)toBuildTradeCandidate.DoCoreAction();

            var currentCandidate = buildTradeCandidate.CurrentPlayer;
            var serializedCandidate = buildTradeCandidate.SerializeHumanReadable();
            serializedCandidate = SetDevCard(serializedCandidate, buildTradeCandidate, currentCandidate, DevCardType.RoadBuilding, 1);
            var loadedCandidate = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Mini, 2, serializedCandidate);

            if (loadedCandidate.Actions().Cast<Gimbur.CatanAction>().Any(a => a.ActionType == Gimbur.CatanActionType.PlayRoadBuilding))
            {
                buildTrade = buildTradeCandidate;
                break;
            }
        }

        Assert.That(buildTrade, Is.Not.Null);

        var current = buildTrade!.CurrentPlayer;
        var serialized = buildTrade.SerializeHumanReadable();
        serialized = SetDevCard(serialized, buildTrade, current, DevCardType.RoadBuilding, 1);
        var loaded = Gimbur.CatanState.DeserializeHumanReadable(GameConfig.Mini, 2, serialized);

        var playRoadBuilding = loaded.Actions()
            .Cast<Gimbur.CatanAction>()
            .Single(a => a.ActionType == Gimbur.CatanActionType.PlayRoadBuilding);
        var pending2 = (Gimbur.CatanState)playRoadBuilding.DoCoreAction();
        Assert.That(pending2.PendingRoadBuildingPlacementsFor(current), Is.EqualTo(2));
        Assert.That(
            pending2.Actions().Cast<Gimbur.CatanAction>().All(a => a.ActionType == Gimbur.CatanActionType.PlaceRoad),
            Is.True);

        var firstRoad = pending2.Actions().Cast<Gimbur.CatanAction>().First();
        var pending1 = (Gimbur.CatanState)firstRoad.DoCoreAction();
        Assert.That(pending1.PendingRoadBuildingPlacementsFor(current), Is.EqualTo(1));
        Assert.That(
            pending1.Actions().Cast<Gimbur.CatanAction>().All(a => a.ActionType == Gimbur.CatanActionType.PlaceRoad),
            Is.True);

        var secondRoad = pending1.Actions().Cast<Gimbur.CatanAction>().First();
        var backToBuildTrade = (Gimbur.CatanState)secondRoad.DoCoreAction();
        Assert.That(backToBuildTrade.PendingRoadBuildingPlacementsFor(current), Is.EqualTo(0));
        Assert.That(backToBuildTrade.Actions().Cast<Gimbur.CatanAction>().Any(a => a.ActionType == Gimbur.CatanActionType.EndTurn), Is.True);
    }

    private static Gimbur.CatanState ReachPreRoll(Gimbur.CatanState state)
    {
        while (state.Stage != TurnStage.PreRoll)
        {
            var action = state.Actions().Cast<Gimbur.CatanAction>().First();
            state = (Gimbur.CatanState)action.DoCoreAction();
        }

        return state;
    }

    private static int TotalResources(Gimbur.CatanState state)
    {
        var total = 0;
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            total += state.TotalResourceCards(player);
        }

        return total;
    }

    private static int ExpectedProductionGain(Gimbur.CatanState state, int roll)
    {
        if (roll == 7)
        {
            return 0;
        }

        var gain = 0;
        foreach (var tile in state.Board.TilesForRoll(roll))
        {
            foreach (var vertex in state.Board.Topology.TileVertices[tile])
            {
                var occ = state.Board.VertexOccupancy[vertex];
                if (occ.IsEmpty)
                {
                    continue;
                }

                gain += occ.Building == BuildingType.City ? 2 : 1;
            }
        }

        return gain;
    }

    private static int FindRobberTargetWithVictim(Gimbur.CatanState state, out int victim)
    {
        for (var tile = 0; tile < state.Board.Topology.TileCount; tile++)
        {
            if (tile == state.Board.RobberTile)
            {
                continue;
            }

            var candidates = state.Board.Topology.TileVertices[tile]
                .Select(v => state.Board.VertexOccupancy[v].Player)
                .Where(p => p != 0 && p != state.CurrentPlayer && state.TotalResourceCards(p) > 0)
                .Distinct()
                .OrderBy(p => p)
                .ToArray();

            if (candidates.Length > 0)
            {
                victim = candidates[0];
                return tile;
            }
        }

        victim = 0;
        return 0;
    }

    private static string SetResource(
        string serialized,
        Gimbur.CatanState state,
        int player,
        ResourceType resource,
        int value)
    {
        var topology = state.Board.Topology;
        var resourceBase =
            (topology.TileCount * 2)
            + 1
            + 2
            + 2
            + topology.VertexCount
            + topology.EdgeCount
            + topology.PortCount;
        var index = resourceBase + ((player - 1) * 5) + ResourceIndex(resource);
        return ReplaceToken(serialized, index, value);
    }

    private static string SetDevCard(
        string serialized,
        Gimbur.CatanState state,
        int player,
        DevCardType card,
        int value)
    {
        var topology = state.Board.Topology;
        var devBase =
            (topology.TileCount * 2)
            + 1
            + 2
            + 2
            + topology.VertexCount
            + topology.EdgeCount
            + topology.PortCount
            + (state.PlayerCount * 5)
            + state.PlayerCount;
        var index = devBase + ((player - 1) * 5) + (int)card;
        return ReplaceToken(serialized, index, value);
    }

    private static int ResourceIndex(ResourceType resource) => resource switch
    {
        ResourceType.Wood => 0,
        ResourceType.Brick => 1,
        ResourceType.Sheep => 2,
        ResourceType.Wheat => 3,
        ResourceType.Ore => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(resource), resource, null),
    };

    private static string ReplaceToken(string serialized, int index, int value)
    {
        var tokens = serialized.Split('|');
        tokens[index] = value.ToString("D2");
        return string.Join('|', tokens);
    }
}
