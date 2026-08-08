using Gimbur.Cli;
using Gimbur.Commands;
namespace Gimbur.Rules.Tests;

[TestFixture]
internal sealed class BenchmarkRunnerTests
{
    [TestCase("random", AiKind.Random)]
    [TestCase("greedy", AiKind.Greedy)]
    [TestCase("mcts", AiKind.Mcts)]
    [TestCase("nn", AiKind.Nn)]
    [TestCase("nn-placement", AiKind.NnPlacement)]
    [TestCase("nn-main-game", AiKind.NnMainGame)]
    [TestCase("nn-value", AiKind.NnValue)]
    [TestCase("nn-value-placement", AiKind.NnValuePlacement)]
    [TestCase("nn-value-main-game", AiKind.NnValueMainGame)]
    [TestCase("server-mcts", AiKind.ServerMcts)]
    [TestCase("server-mcts-nn", AiKind.ServerMctsNn)]
    public void ParsesStableAiNames(string name, AiKind expected)
    {
        Assert.That(RootCommandFactory.TryParseAiKind(name, out var actual), Is.True);
        Assert.That(actual, Is.EqualTo(expected));
        Assert.That(AiKindNames.Format(actual), Is.EqualTo(name));
    }

    [TestCase("nn-state")]
    [TestCase("nn-placement-state")]
    [TestCase("nn-mcts-placement")]
    [TestCase("nn-mcts-state")]
    [TestCase("nn-mcts-placement-state")]
    [TestCase("mcts-placement")]
    public void RejectsRemovedAiNames(string name)
    {
        Assert.That(RootCommandFactory.TryParseAiKind(name, out _), Is.False);
    }

    [Test]
    public void TenThousandGamesHasSubOnePercentWorstCaseMargin()
    {
        Assert.That(BenchmarkConfidence.WorstCaseWald95Margin(10_000), Is.EqualTo(0.0098).Within(1e-12));
        Assert.That(BenchmarkConfidence.RequiredGamesForWorstCase95Margin(0.0098), Is.EqualTo(10_000));
    }

    [Test]
    public void BenchmarkOptionsAcceptExplicitParallelism()
    {
        var options = new BenchmarkOptions
        {
            NumberOfGames = 10,
            Players = [AiKind.Random, AiKind.Greedy],
            Parallelism = 8,
        };

        Assert.That(options.Parallelism, Is.EqualTo(8));
    }

    [Test]
    public void BenchmarkCompetitorMetadataResolvesPerPlayer()
    {
        var options = new BenchmarkOptions
        {
            NumberOfGames = 10,
            Players = [AiKind.Nn, AiKind.Nn],
            NnUrls = ["http://localhost:8000", "http://localhost:8001"],
            PlayerLabels = ["challenger", "champion"],
        };

        Assert.That(BenchmarkRunner.ResolveNnUrls(options), Is.EqualTo(options.NnUrls));
        Assert.That(BenchmarkRunner.ResolvePlayerLabels(options), Is.EqualTo(options.PlayerLabels));
    }

    [Test]
    public void BenchmarkCompetitorMetadataRejectsMismatchedLengths()
    {
        var options = new BenchmarkOptions
        {
            NumberOfGames = 10,
            Players = [AiKind.Nn, AiKind.Greedy],
            NnUrls = ["http://localhost:8000"],
        };

        Assert.That(() => BenchmarkRunner.ResolveNnUrls(options), Throws.ArgumentException);
    }

    [Test]
    public void EmptyCompetitorMetadataUsesLegacyDefaults()
    {
        var options = new BenchmarkOptions
        {
            NumberOfGames = 10,
            Players = [AiKind.Nn, AiKind.Greedy],
            NnUrl = "http://localhost:8123",
            NnUrls = [],
            PlayerLabels = [],
        };

        Assert.That(BenchmarkRunner.ResolveNnUrls(options),
            Is.EqualTo(new[] { "http://localhost:8123", "http://localhost:8123" }));
        Assert.That(BenchmarkRunner.ResolvePlayerLabels(options),
            Is.EqualTo(new[] { "nn", "greedy" }));
    }

    [Test]
    public void PhaseSwitching_OnlyTreatsInitialSetupAsPlacement()
    {
        Assert.Multiple(() =>
        {
            Assert.That(PhaseSwitchingPlayer.IsInitialPlacement(TurnStage.PlaceFirstSettlement), Is.True);
            Assert.That(PhaseSwitchingPlayer.IsInitialPlacement(TurnStage.PlaceSecondRoad), Is.True);
            Assert.That(PhaseSwitchingPlayer.IsInitialPlacement(TurnStage.PlaceRoadCommitted), Is.False);
            Assert.That(PhaseSwitchingPlayer.IsInitialPlacement(TurnStage.PlaceCityCommitted), Is.False);
            Assert.That(PhaseSwitchingPlayer.IsInitialPlacement(TurnStage.BuildTrade), Is.False);
        });
    }

    [Test]
    public void ValuePlayer_MapsParentPlayerIntoSuccessorCanonicalSlots()
    {
        Assert.Multiple(() =>
        {
            Assert.That(NnValuePlayer.CanonicalPlayerSlot(1, 1, 2), Is.Zero);
            Assert.That(NnValuePlayer.CanonicalPlayerSlot(1, 2, 2), Is.EqualTo(1));
            Assert.That(NnValuePlayer.CanonicalPlayerSlot(3, 2, 3), Is.EqualTo(1));
        });
    }

}
