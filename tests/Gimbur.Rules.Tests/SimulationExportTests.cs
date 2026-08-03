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
                ExportType.PlacementAndState, placementPhase: true), Is.EqualTo(PriorMode.Placement));
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
