using System.Collections.Immutable;
using System.Text.Json;
using Gimbur.Cli;
using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
internal sealed class CatanPolicySerializerTests
{
    [TestCaseSource(nameof(Configs))]
    public void PolicySize_MatchesDocumentedFormula(
        GameConfig config,
        int players,
        int expected)
    {
        var serializer = new CatanPolicySerializer(config.Map.Topology, players);

        Assert.That(serializer.PolicySize, Is.EqualTo(expected));
    }

    [Test]
    public void InitialPlacementActions_HaveUniqueInRangeIndices()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
        var actions = state.Actions().Select(Unwrap).ToArray();
        var indices = actions.Select(action => serializer.IndexOf(state, action)).ToArray();

        Assert.That(indices, Is.Unique);
        Assert.That(indices, Is.All.InRange(0, serializer.PolicySize - 1));
    }

    [Test]
    public void EveryActionClass_UsesItsDocumentedSegment()
    {
        var state = new CatanState(GameConfig.Standard, 3, new Random(42));
        var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
        var actions = new (CatanAction Action, int Expected)[]
        {
            (new ChooseRobberTileAction(state, 2), serializer.TilesOffset + 2),
            (new PlaceSettlementAction(state, 3), serializer.VerticesOffset + 3),
            (new PlaceCityAction(state, 4), serializer.VerticesOffset + 4),
            (new PlaceRoadAction(state, 5), serializer.EdgesOffset + 5),
            (new ChooseBankTradeGiveAction(state, ResourceType.Wood), serializer.ResourcesOffset),
            (new ChooseBankTradeReceiveAction(state, ResourceType.Brick), serializer.ResourcesOffset + 1),
            (new ChooseMonopolyResourceAction(state, ResourceType.Sheep), serializer.ResourcesOffset + 2),
            (new ChooseYearOfPlentyResourceAction(state, ResourceType.Wheat), serializer.ResourcesOffset + 3),
            (new BuyRoadAction(state), serializer.BuyTradeOffset),
            (new BuySettlementAction(state), serializer.BuyTradeOffset + 1),
            (new UpgradeCityAction(state), serializer.BuyTradeOffset + 2),
            (new BuyDevCardAction(state), serializer.BuyTradeOffset + 3),
            (new TradeWithBankAction(state), serializer.BuyTradeOffset + 4),
            (new PlayKnightAction(state), serializer.PlayDevCardOffset),
            (new PlayRoadBuildingAction(state), serializer.PlayDevCardOffset + 1),
            (new PlayMonopolyAction(state), serializer.PlayDevCardOffset + 2),
            (new PlayYearOfPlentyAction(state), serializer.PlayDevCardOffset + 3),
            (new ChooseRobberVictimAction(state, 1), serializer.VictimsOffset),
            (new ChooseRobberVictimAction(state, 2), serializer.VictimsOffset + 1),
            (new ChooseRobberVictimAction(state, 3), serializer.VictimsOffset + 2),
            (new RollDiceAction(state), serializer.ControlsOffset),
            (new EndTurnAction(state), serializer.ControlsOffset + 1),
        };

        Assert.Multiple(() =>
        {
            foreach (var (action, expected) in actions)
                Assert.That(serializer.IndexOf(state, action), Is.EqualTo(expected), action.GetType().Name);
        });
    }

    [Test]
    public void ConstructedStages_HaveUniqueLegalPolicyIndices_AndCoverActionClasses()
    {
        var settlement = new CatanState(GameConfig.Standard, 3, new Random(123));
        var road = Apply(settlement, Actions(settlement).First());
        var preRoll = ReachPreRoll(settlement);
        var buildTrade = ReachBuildTradeWithResourcesAndDevCards(preRoll);
        var tradeGive = Apply(buildTrade, Actions(buildTrade).OfType<TradeWithBankAction>().Single());
        var tradeReceive = Apply(tradeGive,
            Actions(tradeGive).OfType<ChooseBankTradeGiveAction>().First());
        var monopoly = Apply(buildTrade, Actions(buildTrade).OfType<PlayMonopolyAction>().Single());
        var plentyFirst = Apply(buildTrade, Actions(buildTrade).OfType<PlayYearOfPlentyAction>().Single());
        var plentySecond = Apply(plentyFirst,
            Actions(plentyFirst).OfType<ChooseYearOfPlentyResourceAction>().First());
        var robber = Apply(buildTrade, Actions(buildTrade).OfType<PlayKnightAction>().Single());
        var buyRoad = Apply(buildTrade, Actions(buildTrade).OfType<BuyRoadAction>().Single());
        var buySettlement = ReachBuySettlement(buildTrade);
        var upgradeCity = Apply(buildTrade, Actions(buildTrade).OfType<UpgradeCityAction>().Single());
        var roadBuilding = Apply(buildTrade,
            Actions(buildTrade).OfType<PlayRoadBuildingAction>().Single());
        var victim = ReachRobberVictimChoice(robber);
        var states = new[]
        {
            settlement, road, preRoll, buildTrade, tradeGive, tradeReceive, monopoly,
            plentyFirst, plentySecond, robber, buyRoad, buySettlement, upgradeCity, roadBuilding, victim,
        };

        foreach (var state in states)
        {
            var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
            var indices = Actions(state).Select(action => serializer.IndexOf(state, action)).ToArray();
            Assert.That(indices, Is.Unique, $"stage {state.Stage}");
            Assert.That(indices, Is.All.InRange(0, serializer.PolicySize - 1), $"stage {state.Stage}");
        }

        var covered = states.SelectMany(Actions).Select(action => action.GetType()).ToHashSet();
        var expected = typeof(CatanAction).Assembly.GetTypes()
            .Where(type => type.IsSealed && type.IsSubclassOf(typeof(CatanAction)))
            .ToArray();
        Assert.That(covered, Is.SupersetOf(expected));
    }

    [Test]
    public void Symmetry_TransformsOnlySpatialSegments()
    {
        var topology = BoardTopology.Mini;
        var serializer = new CatanPolicySerializer(topology, 2);
        var permutation = BoardSymmetry.GetPermutations(topology)[0];

        Assert.Multiple(() =>
        {
            Assert.That(serializer.TransformIndex(serializer.TilesOffset + 2, permutation),
                Is.EqualTo(serializer.TilesOffset + permutation.Tiles[2]));
            Assert.That(serializer.TransformIndex(serializer.VerticesOffset + 3, permutation),
                Is.EqualTo(serializer.VerticesOffset + permutation.Vertices[3]));
            Assert.That(serializer.TransformIndex(serializer.EdgesOffset + 4, permutation),
                Is.EqualTo(serializer.EdgesOffset + permutation.Edges[4]));
            Assert.That(serializer.TransformIndex(serializer.ResourcesOffset + 1, permutation),
                Is.EqualTo(serializer.ResourcesOffset + 1));
        });
    }

    [TestCaseSource(nameof(Topologies))]
    public void Symmetry_TransformsEveryPolicyIndexBySegment(BoardTopology topology)
    {
        var serializer = new CatanPolicySerializer(topology, 3);
        foreach (var permutation in BoardSymmetry.GetPermutations(topology))
        {
            for (var index = 0; index < serializer.PolicySize; index++)
            {
                var expected = index < serializer.VerticesOffset
                    ? permutation.Tiles[index]
                    : index < serializer.EdgesOffset
                        ? serializer.VerticesOffset + permutation.Vertices[index - serializer.VerticesOffset]
                        : index < serializer.ResourcesOffset
                            ? serializer.EdgesOffset + permutation.Edges[index - serializer.EdgesOffset]
                            : index;
                Assert.That(serializer.TransformIndex(index, permutation), Is.EqualTo(expected),
                    $"{permutation.Label}, index {index}");
            }
        }
    }

    [Test]
    public void FullStateExport_UsesDescriptiveActionAndSymmetryPermutation()
    {
        var topology = BoardTopology.Mini;
        var permutation = BoardSymmetry.GetPermutations(topology)[0];
        var game = new GameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 1,
            Turns = 1,
            SearchTimeMs = 1,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = new CatanState(GameConfig.Mini, 2, new Random(42)).SerializeBoard(),
            States =
            [
                new StateRecord
                {
                    PlayerTurn = 1,
                    TurnNumber = 1,
                    Stage = "t",
                    SerializedState = new CatanState(GameConfig.Mini, 2, new Random(42)).SerializeStateOnly(),
                    Scores = [0.0, 0.0],
                    Wins = [],
                    Actions =
                    [
                        new StateActionRecord
                        {
                            Action = "PlaceRoad:4",
                            Wins = [],
                        },
                    ],
                },
            ],
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildGameJsonObject(game, [permutation]),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);
        var exported = document.RootElement.GetProperty("states")[0].GetProperty("actions")[0];

        Assert.Multiple(() =>
        {
            Assert.That(exported.GetProperty("action").GetString(), Is.EqualTo("PlaceRoad:4"));
            Assert.That(exported.TryGetProperty("policyIndex", out _), Is.False);
            Assert.That(exported.GetProperty("permutations")[0].GetString(),
                Is.EqualTo($"PlaceRoad:{permutation.Edges[4]}"));
        });
    }

    [Test]
    public void PriorFullStateMapping_MasksNormalizesAndRepeatsStochasticOutcomes()
    {
        var serializer = new CatanPolicySerializer(BoardTopology.Mini, 2);
        var policy = new double[serializer.PolicySize];
        policy[serializer.ControlsOffset] = 3;
        policy[serializer.ControlsOffset + 1] = 1;

        var mapped = PriorClient.MapFullStatePolicy(
            serializer,
            policy,
            [serializer.ControlsOffset, serializer.ControlsOffset + 1],
            [3, 1]);

        Assert.That(mapped, Is.EqualTo(new[] { 0.75, 0.75, 0.75, 0.25 }));
    }

    [Test]
    public void PriorFullStateMapping_InvalidWidthFallsBackToUniformActions()
    {
        var serializer = new CatanPolicySerializer(BoardTopology.Mini, 2);

        var mapped = PriorClient.MapFullStatePolicy(
            serializer,
            new double[serializer.PolicySize - 1],
            [serializer.ControlsOffset, serializer.ControlsOffset + 1],
            [2, 1]);

        Assert.That(mapped, Is.EqualTo(new[] { 0.5, 0.5, 0.5 }));
    }

    private static IEnumerable<TestCaseData> Configs()
    {
        yield return new TestCaseData(GameConfig.Mini, 2, 79);
        yield return new TestCaseData(GameConfig.Small, 2, 101);
        yield return new TestCaseData(GameConfig.Small, 3, 102);
        yield return new TestCaseData(GameConfig.Standard, 3, 164);
        yield return new TestCaseData(GameConfig.Standard, 4, 165);
    }

    private static IEnumerable<BoardTopology> Topologies()
    {
        yield return BoardTopology.Mini;
        yield return BoardTopology.Small;
        yield return BoardTopology.Standard;
    }

    private static CatanState ReachPreRoll(CatanState state)
    {
        while (state.Stage != TurnStage.PreRoll)
            state = Apply(state, Actions(state).First());
        return state;
    }

    private static CatanState ReachBuildTradeWithResourcesAndDevCards(CatanState state)
    {
        var roll = Actions(state).OfType<RollDiceAction>().Single();
        state = roll.Outcomes().Select(outcome => (CatanState)outcome.Item2)
            .First(outcome => outcome.Stage == TurnStage.BuildTrade);
        var sections = state.SerializeHumanReadable().Split('|');
        var resources = sections[7].Split('/');
        var abundant = CrockfordBase32.Encode(20);
        resources[state.CurrentPlayer - 1] = new string(abundant, 5);
        sections[7] = string.Join('/', resources);
        var cards = sections[9].Split('/');
        cards[state.CurrentPlayer - 1] = "10111";
        sections[9] = string.Join('/', cards);
        return CatanState.DeserializeHumanReadable(
            GameConfig.Standard, state.PlayerCount, string.Join('|', sections));
    }

    private static CatanState ReachRobberVictimChoice(CatanState robber)
    {
        var tile = Enumerable.Range(0, robber.Board.Topology.TileCount).First(tileIndex =>
            tileIndex != robber.Board.RobberTile
            && robber.Board.Topology.TileVertices[tileIndex]
                .Select(vertex => robber.Board.VertexOccupancy[vertex].Player)
                .Where(player => player != 0 && player != robber.CurrentPlayer)
                .Distinct().Count() >= 2);
        var sections = robber.SerializeHumanReadable().Split('|');
        var resources = sections[7].Split('/');
        for (var player = 1; player <= robber.PlayerCount; player++)
        {
            if (player != robber.CurrentPlayer)
                resources[player - 1] = "10000";
        }
        sections[7] = string.Join('/', resources);
        var loaded = CatanState.DeserializeHumanReadable(
            GameConfig.Standard, robber.PlayerCount, string.Join('|', sections));
        return Apply(loaded, new ChooseRobberTileAction(loaded, tile));
    }

    private static CatanState ReachBuySettlement(CatanState state)
    {
        var frontier = new Queue<(CatanState State, int Roads)>();
        frontier.Enqueue((state, 0));
        while (frontier.TryDequeue(out var candidate))
        {
            if (Actions(candidate.State).Any(action => action is BuySettlementAction))
                return candidate.State;
            if (candidate.Roads == 8)
                continue;
            var buyRoad = Actions(candidate.State).OfType<BuyRoadAction>().SingleOrDefault();
            if (buyRoad is null)
                continue;
            var committed = Apply(candidate.State, buyRoad);
            foreach (var road in Actions(committed).OfType<PlaceRoadAction>())
                frontier.Enqueue((Apply(committed, road), candidate.Roads + 1));
        }
        throw new InvalidOperationException("Could not construct a legal settlement purchase.");
    }

    private static CatanState Apply(CatanState state, CatanAction action) =>
        (CatanState)action.DoCoreAction();

    private static CatanAction[] Actions(CatanState state) =>
        state.Actions().Select(Unwrap).ToArray();

    private static CatanAction Unwrap(Kjarni.CoreAction action) => action.IsDeterministic
        ? (CatanDeterministicAction)((Kjarni.CoreAction.Deterministic)action).Item
        : (CatanStochasticAction)((Kjarni.CoreAction.Stochastic)action).Item;
}
