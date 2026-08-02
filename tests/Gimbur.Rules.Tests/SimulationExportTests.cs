using System.Collections.Immutable;
using System.Text.Json;
using Gimbur.Cli;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class SimulationExportTests
{
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
}
