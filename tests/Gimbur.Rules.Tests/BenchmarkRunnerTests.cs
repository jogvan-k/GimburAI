using Gimbur.Cli;
using Gimbur.Commands;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Rules.Tests;

[TestFixture]
internal sealed class BenchmarkRunnerTests
{
    [TestCase("nn-placement-state", AiKind.NnPlacementState)]
    [TestCase("nn-mcts-placement-state", AiKind.NnMctsPlacementState)]
    [TestCase("nn-state", AiKind.NnState)]
    [TestCase("nn-mcts-state", AiKind.NnMctsState)]
    public void ParsesStableAiNames(string name, AiKind expected)
    {
        Assert.That(RootCommandFactory.TryParseAiKind(name, out var actual), Is.True);
        Assert.That(actual, Is.EqualTo(expected));
        Assert.That(AiKindNames.Format(actual), Is.EqualTo(name));
    }

    [Test]
    public void PhaseSwitchingPlayerRoutesSetupAndMainGameAndAggregatesStats()
    {
        var placement = new RecordingPlayer(1, 2, 3, 4, 5);
        var mainGame = new RecordingPlayer(10, 20, 30, 40, 50);
        using var player = new PhaseSwitchingPlayer(placement, mainGame);
        var rng = new Random(1);
        var state = new CatanState(GameConfig.Mini, 2, rng);

        player.Act(state, rng);
        Assert.That(placement.ActCount, Is.EqualTo(1));
        Assert.That(mainGame.ActCount, Is.Zero);

        while (PhaseSwitchingPlayer.IsPlacement(state.Stage))
        {
            var action = Unwrap(state.Actions()[0]);
            state = (CatanState)action.DoCoreAction();
        }

        player.Act(state, rng);
        Assert.That(mainGame.ActCount, Is.EqualTo(1));
        Assert.Multiple(() =>
        {
            Assert.That(player.TotalNnRequests, Is.EqualTo(11));
            Assert.That(player.TotalNnStatesEvaluated, Is.EqualTo(22));
            Assert.That(player.TotalPriorActionsApplied, Is.EqualTo(33));
            Assert.That(player.TotalPriorActionsRequested, Is.EqualTo(44));
            Assert.That(player.TotalPriorInferencesRequested, Is.EqualTo(55));
        });
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
            Players = [AiKind.NnPlacementState, AiKind.NnPlacementState],
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

    private static CatanAction Unwrap(CoreAction action) => action.IsDeterministic
        ? (CatanDeterministicAction)((CoreAction.Deterministic)action).Item
        : (CatanStochasticAction)((CoreAction.Stochastic)action).Item;

    private sealed class RecordingPlayer(
        int requests,
        int states,
        int applied,
        int requested,
        int inferences) : IBenchmarkPlayer, IPriorStatsProvider, IDisposable
    {
        public int ActCount { get; private set; }
        public int TotalNnRequests => requests;
        public int TotalNnStatesEvaluated => states;
        public int TotalPriorNodesRequested => requests;
        public int TotalPriorActionsApplied => applied;
        public int TotalPriorActionsRequested => requested;
        public int TotalPriorInferencesRequested => inferences;

        public CatanState Act(CatanState state, Random rng)
        {
            ActCount++;
            return state;
        }

        public void Dispose()
        {
        }
    }
}
