using System.Collections.Immutable;
using System.Text.Json;
using Gimbur.Cli;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class SimulationExportTests
{
    private static SimulationOptions Options => new() { NumberOfGames = 1 };

    [Test]
    public void GameStateExport_IncludesTurnNumberAndStage()
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
