using System.Collections.Immutable;
using System.Text.Json;
using Gimbur.Cli;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class SimulationExportTests
{
    private static SimulationOptions Options => new() { NumberOfGames = 1 };

    [Test]
    public void ForcedStateHasNoPolicyChoice()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        while (state.Actions().Length != 1)
        {
            var action = state.Actions()[0];
            state = (CatanState)(action.IsDeterministic
                ? ((CatanDeterministicAction)((Kjarni.CoreAction.Deterministic)action).Item)
                    .DoCoreAction()
                : ((CatanStochasticAction)((Kjarni.CoreAction.Stochastic)action).Item)
                    .DoCoreAction());
        }

        Assert.That(state.Actions(), Has.Length.EqualTo(1));
    }

    [Test]
    public void GameStateExport_IncludesTurnStageAndScores()
    {
        var game = new GameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 1,
            Turns = 3,
            SearchTimeMs = 1,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = "board",
            States =
            [
                new StateRecord
                {
                    PlayerTurn = 1,
                    TurnNumber = 1,
                    Stage = "r",
                     SerializedState = "state",
                     Scores = [2.0, 3.0],
                     Wins = [1.0, 0.0],
                     Actions = [],
                },
            ],
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildGameJsonObject(game, ImmutableArray<SymmetryPermutation>.Empty),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);
        var state = document.RootElement.GetProperty("states")[0];

        Assert.Multiple(() =>
        {
            Assert.That(state.GetProperty("turnNumber").GetInt32(), Is.EqualTo(1));
            Assert.That(state.GetProperty("stage").GetString(), Is.EqualTo("r"));
            Assert.That(
                state.GetProperty("scores").EnumerateArray().Select(value => value.GetDouble()),
                Is.EqualTo(new[] { 2.0, 3.0 }));
        });
    }

    [Test]
    public void GameStateExport_IncludesActionDiagnosticsAndSelection()
    {
        var game = new GameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 1,
            Turns = 3,
            SearchTimeMs = 1,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = "board",
            States =
            [
                new StateRecord
                {
                    PlayerTurn = 1,
                    TurnNumber = 1,
                    Stage = "r",
                    SerializedState = "state",
                    Scores = [2.0, 3.0],
                    Wins = [1.0, 0.0],
                    Actions =
                    [
                        new StateActionRecord
                        {
                            Action = "Roll",
                            PolicyIndex = 77,
                            Wins = [3.0, 1.0],
                            Visits = 4,
                            WinRate = 0.75,
                            ModelPrior = 0.6,
                            Selected = true,
                        },
                    ],
                },
            ],
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildGameJsonObject(game, ImmutableArray<SymmetryPermutation>.Empty),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);
        var action = document.RootElement.GetProperty("states")[0].GetProperty("actions")[0];

        Assert.Multiple(() =>
        {
            Assert.That(action.GetProperty("action").GetString(), Is.EqualTo("Roll"));
            Assert.That(action.GetProperty("wins")[0].GetDouble(), Is.EqualTo(3.0));
            Assert.That(action.GetProperty("visits").GetInt32(), Is.EqualTo(4));
            Assert.That(action.GetProperty("winRate").GetDouble(), Is.EqualTo(0.75));
            Assert.That(action.GetProperty("modelPrior").GetDouble(), Is.EqualTo(0.6));
            Assert.That(action.GetProperty("selected").GetBoolean(), Is.True);
        });
    }

    [Test]
    public void GameStateActions_UseDescriptiveDomainNames()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));

        Assert.Multiple(() =>
        {
            Assert.That(
                SimulationRunner.DescribeAction(new PlaceSettlementAction(state, 9)),
                Is.EqualTo("PlaceSettlement:9"));
            Assert.That(
                SimulationRunner.DescribeAction(new PlaceRoadAction(state, 14)),
                Is.EqualTo("PlaceRoad:14"));
            Assert.That(
                SimulationRunner.DescribeAction(new RollDiceAction(state)),
                Is.EqualTo("Roll"));
            Assert.That(
                SimulationRunner.DescribeAction(new ChooseBankTradeGiveAction(state, ResourceType.Wood)),
                Is.EqualTo("ChooseBankTradeGive:Wood"));
        });
    }

    [Test]
    public void GameStateExport_IncludesExactValueTarget()
    {
        var game = new GameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 2,
            Turns = 3,
            SearchTimeMs = 1,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = "board",
            States =
            [
                new StateRecord
                {
                    PlayerTurn = 1,
                    TurnNumber = 3,
                    Stage = "t",
                    SerializedState = "state",
                    Scores = [3.0, 5.0],
                    Wins = [4.0, 6.0],
                    ValueTarget = [0.25, 0.75],
                    Actions = [],
                    ReachedTerminal = true,
                },
            ],
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildGameJsonObject(game, ImmutableArray<SymmetryPermutation>.Empty),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);

        Assert.That(
            document.RootElement.GetProperty("states")[0].GetProperty("valueTarget")
                .EnumerateArray().Select(value => value.GetDouble()),
            Is.EqualTo(new[] { 0.25, 0.75 }));
    }

    [Test]
    public void PlacementExport_IncludesWinner()
    {
        var game = new PlacementGameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 2,
            SearchTimeMs = 1,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = "board",
            States = [],
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildPlacementGameJsonObject(
                game, ImmutableArray<SymmetryPermutation>.Empty),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);

        Assert.That(document.RootElement.GetProperty("winner").GetInt32(), Is.EqualTo(2));
    }

    [Test]
    public void CombinedExport_UsesSharedWinnerAndBothStateArrays()
    {
        var game = new GameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 2,
            Turns = 3,
            SearchTimeMs = 8,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = "board",
            States =
            [
                new StateRecord
                {
                    PlayerTurn = 1,
                    TurnNumber = 0,
                    Stage = "a",
                     SerializedState = "state",
                     Scores = [1.0, 1.0],
                     Wins = [0.0, 1.0],
                     Actions = [],
                },
            ],
        };
        var combined = new CombinedGameResult
        {
            Game = game,
            PlacementStates =
            [
                new PlacementStateRecord
                {
                    PlayerTurn = 1,
                    Stage = "a",
                    SerializedState = "placement",
                    Actions = [],
                },
            ],
            PlacementSearchTimeMs = 16,
            MainGameSearchTimeMs = 8,
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildCombinedGameJsonObject(
                combined, ImmutableArray<SymmetryPermutation>.Empty),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        Assert.Multiple(() =>
        {
            Assert.That(root.GetProperty("winner").GetInt32(), Is.EqualTo(2));
            Assert.That(root.GetProperty("states").GetArrayLength(), Is.EqualTo(1));
            Assert.That(root.GetProperty("states")[0].GetProperty("scores").GetArrayLength(), Is.EqualTo(2));
            Assert.That(root.GetProperty("placementStates").GetArrayLength(), Is.EqualTo(1));
            Assert.That(root.GetProperty("constraints").GetProperty("placementSearchTimeMs").GetInt32(), Is.EqualTo(16));
            Assert.That(root.GetProperty("constraints").GetProperty("mainGameSearchTimeMs").GetInt32(), Is.EqualTo(8));
        });
    }

    [Test]
    public void CombinedExport_RoutesPlacementAndMainGamePriorsSeparately()
    {
        Assert.Multiple(() =>
        {
            Assert.That(SimulationRouting.PriorModeFor(
                ExportType.PlacementAndState, placementPhase: true), Is.EqualTo(PriorMode.State));
            Assert.That(SimulationRouting.PriorModeFor(
                ExportType.PlacementAndState, placementPhase: false), Is.EqualTo(PriorMode.State));
            Assert.That(SimulationRouting.PriorModeFor(
                ExportType.GameState, placementPhase: true), Is.EqualTo(PriorMode.State));
        });
    }

    [Test]
    public void ActionWinData_UsesParentEdgeStatistics()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var root = new Kjarni.MCTS.Types.MCTSState((Kjarni.ICoreState)state);
        root.ActionStats[0].CompletedVisits = 4;
        root.ActionStats[0].ValueSums[0] = 3;
        root.ActionStats[0].ValueSums[1] = 1;

        var (wins, rate, rollouts) = SimulationRunner.GetActionWinData(root, 0, 0);

        Assert.Multiple(() =>
        {
            Assert.That(rollouts, Is.EqualTo(4));
            Assert.That(rate, Is.EqualTo(0.75));
            Assert.That(wins, Is.EqualTo(new[] { 3.0, 1.0 }));
        });
    }

    [Test]
    public void ActionWinData_UnvisitedEdgeHasNoTrainingTarget()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var root = new Kjarni.MCTS.Types.MCTSState((Kjarni.ICoreState)state);

        var (wins, rate, rollouts) = SimulationRunner.GetActionWinData(root, 0, 0);

        Assert.Multiple(() =>
        {
            Assert.That(rollouts, Is.Zero);
            Assert.That(rate, Is.Zero);
            Assert.That(wins, Is.Empty);
        });
    }

    [Test]
    public void PlacementExport_SerializesSettlementAndRoadStagePolicyIndices()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var settlementState = state.SerializePlacementPhase();
        var settlementAction = Unwrap(state.Actions()[0]);
        state = (CatanState)settlementAction.DoCoreAction();
        var road = (PlaceRoadAction)Unwrap(state.Actions()[0]);
        var serializer = PlacementActionSerializer.Mini;
        var roadDirection = serializer.DirectionIndexOf(state.PendingSettlementVertex!.Value, road.EdgeIndex);
        var permutation = BoardSymmetry.GetPermutations(BoardTopology.Mini)[0];
        var game = new PlacementGameResult
        {
            Seed = 42,
            Map = "mini",
            Players = 2,
            Winner = 1,
            SearchTimeMs = 1,
            MaxSimulations = 1,
            MaxRolloutDepth = 1,
            ActionRolloutLimit = 1,
            BoardSerialized = state.SerializeBoard(),
            States =
            [
                new PlacementStateRecord
                {
                    PlayerTurn = 1,
                    Stage = "a",
                    SerializedState = settlementState,
                    Actions =
                    [
                        new PlacementActionRecord
                        {
                            PolicyIndex = ((PlaceSettlementAction)settlementAction).VertexIndex,
                            Visits = 4,
                            Wins = [3.0, 1.0],
                            WinRate = 0.75,
                            ModelPrior = 1.0,
                        },
                    ],
                },
                new PlacementStateRecord
                {
                    PlayerTurn = 1,
                    PendingVertex = state.PendingSettlementVertex,
                    Stage = "e",
                    SerializedState = state.SerializePlacementPhase(),
                    Actions =
                    [
                        new PlacementActionRecord
                        {
                            PolicyIndex = roadDirection,
                            RoadEdge = road.EdgeIndex,
                            Visits = 2,
                            Wins = [1.0, 1.0],
                            WinRate = 0.5,
                        },
                    ],
                },
            ],
        };

        var json = JsonSerializer.Serialize(
            SimulationRunner.BuildPlacementGameJsonObject(game, [permutation]),
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);
        var settlement = document.RootElement.GetProperty("states")[0];
        var exportedRoad = document.RootElement.GetProperty("states")[1];

        Assert.Multiple(() =>
        {
            Assert.That(settlement.GetProperty("stage").GetString(), Is.EqualTo("a"));
            Assert.That(settlement.GetProperty("actions")[0].GetProperty("visits").GetInt32(), Is.EqualTo(4));
            Assert.That(settlement.GetProperty("actions")[0].TryGetProperty("action", out _), Is.False);
            Assert.That(settlement.GetProperty("actions")[0].GetProperty("modelPrior").GetDouble(), Is.EqualTo(1.0));
            Assert.That(settlement.GetProperty("actions")[0].TryGetProperty("policyTarget", out _), Is.False);
            Assert.That(
                settlement.GetProperty("actions")[0].GetProperty("permutations")[0].GetInt32(),
                Is.EqualTo(permutation.Vertices[((PlaceSettlementAction)settlementAction).VertexIndex]));
            Assert.That(exportedRoad.GetProperty("stage").GetString(), Is.EqualTo("e"));
            Assert.That(
                exportedRoad.GetProperty("actions")[0].GetProperty("permutations")[0].GetInt32(),
                Is.EqualTo(serializer.TransformDirectionIndex(
                    state.PendingSettlementVertex.Value, road.EdgeIndex, permutation)));
        });
    }

    private static CatanAction Unwrap(Kjarni.CoreAction action) => action.IsDeterministic
        ? (CatanDeterministicAction)((Kjarni.CoreAction.Deterministic)action).Item
        : (CatanStochasticAction)((Kjarni.CoreAction.Stochastic)action).Item;

    [Test]
    public void EvaluationDiagnostics_AggregatesLogInfoAndSerializes()
    {
        var diagnostics = new EvaluationDiagnostics();
        var log = new Kjarni.MCTS.Types.LogInfo
        {
            leafEvaluationsSubmitted = 10,
            leafEvaluationsApplied = 8,
            leafEvaluationTimeouts = 1,
            leafEvaluationsInvalid = 2,
            leafEvaluationsCancelled = 3,
            leafEvaluationFallbacks = 4,
            leafEvaluationOrphans = 5,
            leafEvaluationBatches = 6,
            leafEvaluationStates = 20,
            leafEvaluationLatencyMs = 70,
            priorResponsesOrphaned = 7,
        };

        diagnostics.Add(log);
        diagnostics.Add(log);
        var json = JsonSerializer.Serialize(diagnostics,
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        using var document = JsonDocument.Parse(json);

        Assert.Multiple(() =>
        {
            Assert.That(diagnostics.Submitted, Is.EqualTo(20));
            Assert.That(diagnostics.HardErrors, Is.EqualTo(16));
            Assert.That(document.RootElement.GetProperty("invalidResponses").GetInt32(), Is.EqualTo(4));
            Assert.That(document.RootElement.GetProperty("latencyMs").GetInt64(), Is.EqualTo(140));
        });
    }

    [Test]
    public void ErrorPolicy_DiscardsForAbsoluteRateAndFallbackRules()
    {
        Assert.Multiple(() =>
        {
            Assert.That(SimulationErrorPolicy.GetGameDiscardReason(
                new EvaluationDiagnostics { Timeouts = 6 }, Options), Does.StartWith("hard errors"));
            Assert.That(SimulationErrorPolicy.GetGameDiscardReason(
                new EvaluationDiagnostics { Submitted = 50, InvalidResponses = 2 }, Options),
                Does.StartWith("hard error rate"));
            Assert.That(SimulationErrorPolicy.GetGameDiscardReason(
                new EvaluationDiagnostics { Fallbacks = 1 }, Options with { DiscardGamesWithFallbacks = true }),
                Does.StartWith("fallbacks prohibited"));
            Assert.That(SimulationErrorPolicy.GetGameDiscardReason(new EvaluationDiagnostics(), Options), Is.Null);
        });
    }

    [Test]
    public void ErrorPolicy_StopsGenerationAtConfiguredThresholds()
    {
        Assert.Multiple(() =>
        {
            Assert.That(SimulationErrorPolicy.GetGenerationStopReason(22, 21, 1, Options),
                Does.StartWith("discarded games"));
            Assert.That(SimulationErrorPolicy.GetGenerationStopReason(50, 3, 1, Options),
                Does.StartWith("discard rate"));
            Assert.That(SimulationErrorPolicy.GetGenerationStopReason(6, 6, 6, Options),
                Does.StartWith("consecutive discards"));
        });
    }
}
