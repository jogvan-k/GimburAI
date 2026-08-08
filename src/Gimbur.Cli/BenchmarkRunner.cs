using System.Collections.Concurrent;
using System.Net.Http;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Encodings.Web;
using Gimbur;
using Gimbur.Rules;
using Kjarni;
using static Kjarni.MCTS.Algorithm;

namespace Gimbur.Cli;

/// <summary>
/// Identifies an AI strategy that can be used in benchmark games.
/// New strategies should be added here and registered in
/// <see cref="BenchmarkRunner.CreatePlayer"/>.
/// </summary>
internal enum AiKind
{
    Random,
    Greedy,
    Mcts,
    Nn,
    NnPlacement,
    NnMainGame,
    NnValue,
    NnValuePlacement,
    NnValueMainGame,
    ServerMcts,
    ServerMctsNn,
}

/// <summary>
/// Configuration options for running AI benchmark games.
/// </summary>
internal record BenchmarkOptions
{
    public required uint NumberOfGames { get; init; }
    public int Seed { get; init; }
    public string? MapConfig { get; init; }
    public string Verbosity { get; init; } = "normal";

    /// <summary>
    /// AI assignments per player seat (index 0 = player 1).
    /// The length also determines the player count.
    /// </summary>
    public required AiKind[] Players { get; init; }

    /// <summary>
    /// Optional path to write a JSON results file.
    /// </summary>
    public FileInfo? OutputPath { get; init; }

    /// <summary>
    /// MCTS search time limit in milliseconds per decision. Defaults to 1000ms.
    /// </summary>
    public int SearchTimeMs { get; init; } = 1000;

    /// <summary>
    /// Maximum number of MCTS simulations per decision. Defaults to int.MaxValue (time-limited).
    /// </summary>
    public int MaxSimulations { get; init; } = int.MaxValue;

    /// <summary>
    /// Maximum rollout depth for MCTS simulations. Defaults to 500.
    /// </summary>
    public int MaxRolloutDepth { get; init; } = 500;

    /// <summary>
    /// Base URL for the neural network inference server.
    /// Defaults to <c>http://localhost:8000</c>.
    /// </summary>
    public string NnUrl { get; init; } = "http://localhost:8000";
    public string[]? NnUrls { get; init; }
    public string[]? PlayerLabels { get; init; }

    /// <summary>
    /// Base URL for the Gimbur.Server (used by server-mcts and server-mcts-nn AI kinds).
    /// Defaults to <c>http://localhost:5123</c>.
    /// </summary>
    public string ServerUrl { get; init; } = "http://localhost:5123";

    /// <summary>
    /// Maximum tree depth for NN prior requests (server-mcts-nn only).
    /// Defaults to unlimited.
    /// </summary>
    public int? ServerMaxPriorDepth { get; init; }
    public int Parallelism { get; init; }
}

/// <summary>
/// Abstraction for a benchmark AI player. Implementations may be stateless
/// (random, greedy) or carry per-game state (MCTS tree reuse).
/// A new instance is created for each game.
/// </summary>
internal interface IBenchmarkPlayer
{
    /// <summary>
    /// Choose and apply the next action, returning the resulting state.
    /// Returns null if no action can be taken (game should stop).
    /// </summary>
    CatanState? Act(CatanState state, Random rng);
}

/// <summary>Routes initial setup and normal play to separate benchmark players.</summary>
internal sealed class PhaseSwitchingPlayer : IBenchmarkPlayer, INnStatsProvider, IDisposable
{
    private readonly IBenchmarkPlayer _placement;
    private readonly IBenchmarkPlayer _mainGame;

    public PhaseSwitchingPlayer(IBenchmarkPlayer placement, IBenchmarkPlayer mainGame)
    {
        _placement = placement;
        _mainGame = mainGame;
    }

    public int TotalNnRequests => Stats(_placement).Requests + Stats(_mainGame).Requests;
    public int TotalNnStatesEvaluated => Stats(_placement).States + Stats(_mainGame).States;

    public CatanState? Act(CatanState state, Random rng) => IsInitialPlacement(state.Stage)
        ? _placement.Act(state, rng)
        : _mainGame.Act(state, rng);

    internal static bool IsInitialPlacement(TurnStage stage) => stage is
        TurnStage.PlaceFirstSettlement or TurnStage.PlaceFirstRoad or
        TurnStage.PlaceSecondSettlement or TurnStage.PlaceSecondRoad;

    public void Dispose()
    {
        if (_placement is IDisposable placement) placement.Dispose();
        if (_mainGame is IDisposable mainGame) mainGame.Dispose();
    }

    private static (int Requests, int States) Stats(IBenchmarkPlayer player) =>
        player is INnStatsProvider stats
            ? (stats.TotalNnRequests, stats.TotalNnStatesEvaluated)
            : (0, 0);
}


/// <summary>
/// Optional interface for benchmark players that use a neural network
/// inference server. Provides aggregate NN call statistics for reporting.
/// </summary>
internal interface INnStatsProvider
{
    /// <summary>Total NN prediction requests (batch calls) made during this game.</summary>
    int TotalNnRequests { get; }

    /// <summary>Total individual states evaluated by the NN server across all requests.</summary>
    int TotalNnStatesEvaluated { get; }
}

internal interface IPriorStatsProvider : INnStatsProvider
{
    int TotalPriorNodesRequested { get; }
    int TotalPriorActionsApplied { get; }
    int TotalPriorActionsRequested { get; }
    int TotalPriorInferencesRequested { get; }
}

internal static class BenchmarkConfidence
{
    private const double Confidence95Z = 1.96;

    public static double Wald95Margin(double rate, int games) =>
        games > 0 ? Confidence95Z * Math.Sqrt(rate * (1.0 - rate) / games) : 0.0;

    public static double WorstCaseWald95Margin(int games) => Wald95Margin(0.5, games);

    public static int RequiredGamesForWorstCase95Margin(double margin) =>
        margin is > 0 and < 1
            ? (int)Math.Ceiling(Confidence95Z * Confidence95Z * 0.25 / (margin * margin))
            : throw new ArgumentOutOfRangeException(nameof(margin));
}

/// <summary>
/// Picks a random action from the available actions.
/// </summary>
internal sealed class RandomPlayer : IBenchmarkPlayer
{
    public CatanState? Act(CatanState state, Random rng)
    {
        var coreActions = state.Actions();
        if (coreActions.Length == 0) return null;

        var roll = rng.Next(coreActions.Length);
        var chosen = coreActions[roll];

        return (CatanState)UnwrapCoreAction(chosen).DoCoreAction();
    }

    private static CatanAction UnwrapCoreAction(CoreAction coreAction)
    {
        if (coreAction.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
        if (coreAction.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
    }
}

/// <summary>
/// Greedy one-step lookahead player.
/// </summary>
internal sealed class GreedyPlayer : IBenchmarkPlayer
{
    private readonly GreedyActionSelector _selector = new();

    public CatanState? Act(CatanState state, Random rng)
    {
        var action = _selector.ChooseAction(state, rng);
        if (action is null) return null;
        return (CatanState)action.DoCoreAction();
    }
}

/// <summary>
/// MCTS-based player that mirrors SimulationRunner behavior:
/// skips forced moves, follows the best path, and reuses the search tree.
/// Accumulates prior stats across all MCTS decisions for reporting.
/// </summary>
internal sealed class MctsPlayer : IBenchmarkPlayer, INnStatsProvider
{
    private readonly Kjarni.MCTS.AI.MonteCarloTreeSearch _mcts;
    private Kjarni.MCTS.Types.MCTSState? _mctsRoot;

    /// <summary>Total nodes for which a prior was requested across all MCTS decisions in this game.</summary>
    public int TotalPriorNodesRequested { get; private set; }

    /// <summary>Total individual action states whose priors were successfully applied across all MCTS decisions.</summary>
    public int TotalPriorActionsApplied { get; private set; }

    /// <summary>Total MCTS-level action states sent to the prior client across all decisions.</summary>
    public int TotalPriorActionsRequested { get; private set; }

    /// <summary>Total (state, action) inference pairs actually sent to the NN model across all decisions.</summary>
    public int TotalPriorInferencesRequested { get; private set; }

    /// <summary>Per-depth count of MCTS-level action states sent to the client across all MCTS decisions.</summary>
    public Dictionary<int, int>? TotalPriorActionsPerDepth { get; private set; }

    /// <summary>Per-depth count of model inference pairs across all MCTS decisions.</summary>
    public Dictionary<int, int>? TotalPriorInferencesPerDepth { get; private set; }

    // INnStatsProvider — TotalNnStatesEvaluated reflects actual NN inference work.
    int INnStatsProvider.TotalNnRequests => TotalPriorNodesRequested;
    int INnStatsProvider.TotalNnStatesEvaluated => TotalPriorInferencesRequested;

    public MctsPlayer(MCTSConfig config)
    {
        _mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(config);
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var actions = state.Actions();
        if (actions.Length == 0) return null;

        if (actions.Length == 1)
        {
            // Forced action — apply without running MCTS.
            var next = (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();
            _mctsRoot = AdvanceMctsRoot(_mctsRoot, 0, (ICoreState)next);
            return next;
        }

        // Multiple actions — run MCTS to decide.
        _mctsRoot ??= new Kjarni.MCTS.Types.MCTSState((ICoreState)state);
        _mcts.RunSimulation(_mctsRoot);
        var bestPath = extractBestPath(_mctsRoot);

        // Accumulate prior stats from this decision.
        var logInfo = _mcts.LatestLogInfo();
        TotalPriorNodesRequested += logInfo.priorNodesRequested;
        TotalPriorActionsApplied += logInfo.priorActionsApplied;
        TotalPriorActionsRequested += logInfo.priorActionsRequested;
        TotalPriorInferencesRequested += logInfo.priorInferencesRequested;
        if (logInfo.priorActionsPerDepth is { Count: > 0 })
        {
            TotalPriorActionsPerDepth ??= new Dictionary<int, int>();
            foreach (var kv in logInfo.priorActionsPerDepth)
            {
                TotalPriorActionsPerDepth.TryGetValue(kv.Key, out var existing);
                TotalPriorActionsPerDepth[kv.Key] = existing + kv.Value;
            }
        }
        if (logInfo.priorInferencesPerDepth is { Count: > 0 })
        {
            TotalPriorInferencesPerDepth ??= new Dictionary<int, int>();
            foreach (var kv in logInfo.priorInferencesPerDepth)
            {
                TotalPriorInferencesPerDepth.TryGetValue(kv.Key, out var existing);
                TotalPriorInferencesPerDepth[kv.Key] = existing + kv.Value;
            }
        }

        if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
        {
            var next = (CatanState)UnwrapCoreAction(actions[bestPath.Head]).DoCoreAction();
            _mctsRoot = AdvanceMctsRoot(_mctsRoot, bestPath.Head, (ICoreState)next);
            return next;
        }

        return null;
    }

    private static CatanAction UnwrapCoreAction(CoreAction coreAction)
    {
        if (coreAction.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
        if (coreAction.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
    }

    /// <summary>
    /// Follows the MCTS tree to the child at the given action index, reusing
    /// prior search results. For deterministic actions, returns the child node
    /// directly. For stochastic actions, matches the actual game result against
    /// the outcome states. Returns null if the action was not expanded or no
    /// matching outcome is found.
    /// </summary>
    private static Kjarni.MCTS.Types.MCTSState? AdvanceMctsRoot(
        Kjarni.MCTS.Types.MCTSState? current, int actionIndex, ICoreState actualResult)
    {
        if (current is null) return null;
        if (actionIndex < 0 || actionIndex >= current.Actions.Length) return null;

        var action = current.Actions[actionIndex];
        if (action.IsDeterministicAction)
            return ((Kjarni.MCTS.Types.Action.DeterministicAction)action).Item;

        if (action.IsHorizonAction)
            return ((Kjarni.MCTS.Types.Action.HorizonAction)action).Item;

        if (action.IsStochasticAction)
        {
            var outcomes = ((Kjarni.MCTS.Types.Action.StochasticAction)action).Item;
            foreach (var outcome in outcomes)
            {
                if (outcome.State.State.Equals(actualResult))
                    return outcome.State;
            }
        }

        return null;
    }
}

/// <summary>
/// Per-game result captured by the benchmark runner.
/// </summary>
internal record BenchmarkGameResult
{
    public required int GameNumber { get; init; }
    public required int Seed { get; init; }

    /// <summary>
    /// The winning AI kind, or null for a draw (no winner before safety limits).
    /// </summary>
    public required AiKind? WinnerAi { get; init; }
    public required string? WinnerLabel { get; init; }

    /// <summary>
    /// The seat assignment used for this game (index 0 = seat 1).
    /// Rotates across games to eliminate positional bias.
    /// </summary>
    public required AiKind[] SeatAssignment { get; init; }
    public required string[] SeatLabels { get; init; }

    /// <summary>
    /// The winning seat number (1-based), or 0 for a draw.
    /// </summary>
    public required int WinnerSeat { get; init; }

    public required int Turns { get; init; }
    public required TimeSpan Elapsed { get; init; }

    public int NnRequests { get; init; }
    public int NnStatesEvaluated { get; init; }

    /// <summary>
    /// Total nodes for which a prior was requested across all MCTS decisions in this game.
    /// Zero when no MCTS player uses priors.
    /// </summary>
    public int PriorNodesRequested { get; init; }

    /// <summary>
    /// Total individual action states whose priors were successfully applied across all MCTS decisions.
    /// </summary>
    public int PriorActionsApplied { get; init; }

    /// <summary>
    /// Total individual action states sent to the NN server across all decisions.
    /// </summary>
    public int PriorActionsRequested { get; init; }

    /// <summary>
    /// Total (state, action) inference pairs actually sent to the NN model across all decisions.
    /// </summary>
    public int PriorInferencesRequested { get; init; }

    /// <summary>
    /// Per-depth count of action states sent to the NN across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorActionsPerDepth { get; init; }

    /// <summary>
    /// Per-depth count of model inference pairs across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorInferencesPerDepth { get; init; }
}

/// <summary>
/// Aggregate benchmark results with per-AI win rates.
/// </summary>
internal record BenchmarkStats
{
    public required List<BenchmarkGameResult> Games { get; init; }
    public required int TotalGames { get; init; }
    public required TimeSpan TotalElapsed { get; init; }
    public required AiKind[] PlayerAis { get; init; }
    public required string[] PlayerLabels { get; init; }
}

/// <summary>
/// Runs benchmark games between configurable AI strategies and reports win rates.
/// Games are executed in parallel across available CPU cores.
/// </summary>
internal class BenchmarkRunner
{
    private readonly BenchmarkOptions _options;
    private readonly bool _quiet;
    private readonly Dictionary<string, NnClient> _nnClients = new(StringComparer.OrdinalIgnoreCase);

    public BenchmarkRunner(BenchmarkOptions options)
    {
        _options = options;
        _quiet = options.Verbosity is "quiet" or "q";
    }

    public void Run()
    {
        var config = ResolveGameConfig();
        var playerCount = _options.Players.Length;
        var nnUrls = ResolveNnUrls(_options);
        var playerLabels = ResolvePlayerLabels(_options);

        if (playerCount < config.MinPlayers || playerCount > config.MaxPlayers)
        {
            Console.Error.WriteLine(
                $"Map '{_options.MapConfig ?? "standard"}' requires {config.MinPlayers}-{config.MaxPlayers} players, " +
                $"but {playerCount} AI(s) were specified.");
            return;
        }

        if (UsesNn(_options.Players))
        {
            foreach (var url in nnUrls.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                var client = new NnClient(url);
                _nnClients[url] = client;
                if (!client.IsHealthyAsync().GetAwaiter().GetResult())
                {
                    Console.Error.WriteLine($"NN inference server at {url} is not reachable.");
                    DisposeNnClients();
                    return;
                }
            }
        }

        if (UsesServer(_options.Players))
        {
            using var serverClient = new HttpClient();
            try
            {
                var response = serverClient.GetAsync($"{_options.ServerUrl.TrimEnd('/')}/health")
                    .GetAwaiter().GetResult();
                if (!response.IsSuccessStatusCode)
                {
                    Console.Error.WriteLine($"Game server at {_options.ServerUrl} returned {(int)response.StatusCode}.");
                    DisposeNnClients();
                    return;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Game server at {_options.ServerUrl} is not reachable: {ex.Message}");
                DisposeNnClients();
                return;
            }
        }

        // When NN is in use, limit parallelism to avoid overwhelming the
        // inference server with concurrent requests.
        var maxParallelism = _options.Parallelism > 0
            ? Math.Min(_options.Parallelism, Environment.ProcessorCount)
            : (UsesNn(_options.Players) || UsesServer(_options.Players))
                ? Math.Min(4, Environment.ProcessorCount)
                : Environment.ProcessorCount;

        if (!_quiet)
        {
            Console.WriteLine($"Starting benchmark: {_options.NumberOfGames} game(s)");
            Console.WriteLine($"  Map: {_options.MapConfig ?? "standard"}");
            Console.WriteLine($"  Players: {string.Join(" vs ", _options.Players.Select((ai, i) => $"P{i + 1}={ai}"))}");
            Console.WriteLine($"  Seed: {_options.Seed}");
            Console.WriteLine($"  Parallelism: {maxParallelism} cores");
            if (_options.Players.Any(ai => ai is AiKind.Mcts))
            {
                Console.WriteLine($"  MCTS search time: {_options.SearchTimeMs}ms");
                Console.WriteLine($"  MCTS max simulations: {(_options.MaxSimulations == int.MaxValue ? "unlimited" : _options.MaxSimulations.ToString())}");
                Console.WriteLine($"  MCTS max rollout depth: {_options.MaxRolloutDepth}");
            }
            if (UsesNn(_options.Players))
            {
                Console.WriteLine($"  NN servers: {string.Join(", ", nnUrls.Distinct())}");
            }
            if (UsesServer(_options.Players))
            {
                Console.WriteLine($"  Game server: {_options.ServerUrl}");
            }
            Console.WriteLine();
        }

        var gameResults = new ConcurrentBag<BenchmarkGameResult>();
        var failedGames = new ConcurrentBag<(int GameIndex, Exception Error)>();
        var totalStopwatch = Stopwatch.StartNew();

        Parallel.For(0, (int)_options.NumberOfGames, new ParallelOptions
        {
            MaxDegreeOfParallelism = maxParallelism,
        }, gameIndex =>
        {
            var gameSeed = unchecked(_options.Seed + gameIndex);
            var rng = new Random(gameSeed);

            // Rotate seat assignments to eliminate first-player bias.
            var rotation = gameIndex % playerCount;
            var seatAssignment = new AiKind[playerCount];
            var seatLabels = new string[playerCount];
            var seatNnUrls = new string[playerCount];
            for (var i = 0; i < playerCount; i++)
            {
                var competitor = (i + rotation) % playerCount;
                seatAssignment[i] = _options.Players[competitor];
                seatLabels[i] = playerLabels[competitor];
                seatNnUrls[i] = nnUrls[competitor];
            }

            try
            {
                var gameStopwatch = Stopwatch.StartNew();
                var (winnerSeat, turns, nnRequests, nnStatesEvaluated, priorNodesRequested, priorActionsApplied, priorActionsRequested, priorInferencesRequested, priorActionsPerDepth, priorInferencesPerDepth) = RunSingleGame(config, rng, seatAssignment, seatNnUrls);
                gameStopwatch.Stop();

                AiKind? winnerAi = winnerSeat > 0 ? seatAssignment[winnerSeat - 1] : null;

                var result = new BenchmarkGameResult
                {
                    GameNumber = gameIndex + 1,
                    Seed = gameSeed,
                    WinnerAi = winnerAi,
                    WinnerLabel = winnerSeat > 0 ? seatLabels[winnerSeat - 1] : null,
                    SeatAssignment = seatAssignment,
                    SeatLabels = seatLabels,
                    WinnerSeat = winnerSeat,
                    Turns = turns,
                    Elapsed = gameStopwatch.Elapsed,
                    NnRequests = nnRequests,
                    NnStatesEvaluated = nnStatesEvaluated,
                    PriorNodesRequested = priorNodesRequested,
                    PriorActionsApplied = priorActionsApplied,
                    PriorActionsRequested = priorActionsRequested,
                    PriorInferencesRequested = priorInferencesRequested,
                    PriorActionsPerDepth = priorActionsPerDepth,
                    PriorInferencesPerDepth = priorInferencesPerDepth,
                };

                gameResults.Add(result);

                if (!_quiet)
                {
                    var winnerLabel = winnerSeat == 0
                        ? "draw"
                        : $"P{winnerSeat}({winnerAi})";
                    Console.WriteLine(
                        $"Game {gameIndex + 1}: winner={winnerLabel}, turns={turns}, " +
                        $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
                }
            }
            catch (Exception ex)
            {
                failedGames.Add((gameIndex + 1, ex));
                if (!_quiet)
                {
                    Console.Error.WriteLine(
                        $"Game {gameIndex + 1}: FAILED — {ex.GetType().Name}: {ex.Message}");
                }
            }
        });

        totalStopwatch.Stop();

        if (!failedGames.IsEmpty && !_quiet)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine($"WARNING: {failedGames.Count} game(s) failed and were excluded from results.");
        }

        var completedGames = gameResults.OrderBy(g => g.GameNumber).ToList();

        var stats = new BenchmarkStats
        {
            Games = completedGames,
            TotalGames = completedGames.Count,
            TotalElapsed = totalStopwatch.Elapsed,
            PlayerAis = _options.Players,
            PlayerLabels = playerLabels,
        };

        if (!_quiet)
        {
            PrintSummary(stats);
        }

        if (_options.OutputPath is not null)
        {
            ExportResults(stats, _options.OutputPath);
        }

        DisposeNnClients();

        // Only fail the benchmark if more than half the games failed.
        // A few sporadic failures (e.g. from edge-case states the NN server
        // cannot tokenize) are tolerable — the partial results are still valid.
        if (failedGames.Count > (int)_options.NumberOfGames / 2)
        {
            throw new InvalidOperationException(
                $"{failedGames.Count}/{_options.NumberOfGames} game(s) failed. " +
                $"First error: {failedGames.First().Error.Message}");
        }
    }

    /// <summary>
    /// Creates a new player instance for the given AI kind.
    /// A new instance is created per game to allow stateful players (e.g. MCTS tree reuse).
    /// </summary>
    private IBenchmarkPlayer CreatePlayer(AiKind kind, string nnUrl)
    {
        var nnClient = _nnClients.GetValueOrDefault(nnUrl);

        return kind switch
        {
            AiKind.Random => new RandomPlayer(),
            AiKind.Greedy => new GreedyPlayer(),
            AiKind.Mcts => new MctsPlayer(new MCTSConfig(
                searchTime.NewMilliSeconds(_options.SearchTimeMs),
                _options.MaxSimulations,
                _options.MaxRolloutDepth,
                System.Math.Sqrt(2.0),
                int.MaxValue,
                null,
                null,
                null,
                int.MaxValue,
                32,
                500,
                1000)),
            AiKind.Nn => new NnPlayer(nnClient!),
            AiKind.NnPlacement => new PhaseSwitchingPlayer(
                new NnPlayer(nnClient!), new GreedyPlayer()),
            AiKind.NnMainGame => new PhaseSwitchingPlayer(
                new GreedyPlayer(), new NnPlayer(nnClient!)),
            AiKind.NnValue => new NnValuePlayer(nnClient!),
            AiKind.NnValuePlacement => new PhaseSwitchingPlayer(
                new NnValuePlayer(nnClient!), new GreedyPlayer()),
            AiKind.NnValueMainGame => new PhaseSwitchingPlayer(
                new GreedyPlayer(), new NnValuePlayer(nnClient!)),
            AiKind.ServerMcts => new ServerPlayer(
                _options.ServerUrl, "mcts-ai", _options.MapConfig ?? "standard",
                _options.Players.Length, _options.SearchTimeMs, _options.MaxRolloutDepth),
            AiKind.ServerMctsNn => new ServerPlayer(
                _options.ServerUrl, "mcts-nn-ai", _options.MapConfig ?? "standard",
                _options.Players.Length, _options.SearchTimeMs, _options.MaxRolloutDepth,
                nnUrl, _options.ServerMaxPriorDepth),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, $"Unknown AI kind: {kind}"),
        };
    }

    /// <summary>
    /// Returns true if any player in the array requires an NN inference server.
    /// </summary>
    /// <summary>
    /// Returns true if any player in the array requires a Gimbur.Server instance.
    /// </summary>
    private static bool UsesServer(AiKind[] players) =>
        players.Any(ai => ai is AiKind.ServerMcts or AiKind.ServerMctsNn);

    private static bool UsesNn(AiKind[] players) =>
        players.Any(ai => ai is AiKind.Nn or AiKind.NnPlacement or AiKind.NnMainGame
            or AiKind.NnValue or AiKind.NnValuePlacement or AiKind.NnValueMainGame
            or AiKind.ServerMctsNn);

    internal static string[] ResolveNnUrls(BenchmarkOptions options)
    {
        var urls = options.NnUrls is { Length: > 0 }
            ? options.NnUrls
            : Enumerable.Repeat(options.NnUrl, options.Players.Length).ToArray();
        if (urls.Length != options.Players.Length)
            throw new ArgumentException("nnUrls must contain one URL per configured AI.");
        if (urls.Any(url => !Uri.TryCreate(url, UriKind.Absolute, out var uri)
            || uri.Scheme is not ("http" or "https")))
            throw new ArgumentException("Each nnUrls entry must be an absolute HTTP or HTTPS URL.");
        return urls;
    }

    internal static string[] ResolvePlayerLabels(BenchmarkOptions options)
    {
        var labels = options.PlayerLabels is { Length: > 0 } ? options.PlayerLabels : null;
        if (labels is null)
        {
            var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            labels = options.Players.Select(ai =>
            {
                var label = AiKindNames.Format(ai);
                counts.TryGetValue(label, out var count);
                counts[label] = ++count;
                return count == 1 ? label : $"{label}-{count}";
            }).ToArray();
        }
        if (labels.Length != options.Players.Length || labels.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("playerLabels must contain one non-empty label per configured AI.");
        if (labels.Distinct(StringComparer.OrdinalIgnoreCase).Count() != labels.Length)
            throw new ArgumentException("playerLabels must be unique.");
        return labels;
    }

    private void DisposeNnClients()
    {
        foreach (var client in _nnClients.Values)
            client.Dispose();
        _nnClients.Clear();
    }

    private (int WinnerSeat, int Turns, int NnRequests, int NnStatesEvaluated, int PriorNodesRequested, int PriorActionsApplied, int PriorActionsRequested, int PriorInferencesRequested, Dictionary<int, int>? PriorActionsPerDepth, Dictionary<int, int>? PriorInferencesPerDepth) RunSingleGame(GameConfig config, Random rng, AiKind[] seatAssignment, string[] seatNnUrls)
    {
        var playerCount = seatAssignment.Length;
        var state = new CatanState(config, playerCount, rng);

        // Build per-player AI instances (1-indexed; index 0 unused).
        var players = new IBenchmarkPlayer[playerCount + 1];
        for (var i = 0; i < playerCount; i++)
        {
            players[i + 1] = CreatePlayer(seatAssignment[i], seatNnUrls[i]);
        }

        const int maxTotalActions = 10_000;
        var totalActions = 0;

        while (state.WinnerPlayer == 0)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;

            var player = players[state.CurrentPlayer];
            var next = player.Act(state, rng);
            if (next is null) break;

            state = next;

            totalActions++;
            if (totalActions >= maxTotalActions || state.TurnNumber > 500)
            {
                break;
            }
        }

        // Aggregate NN stats from all players that use neural network inference.
        var nnRequests = 0;
        var nnStatesEvaluated = 0;
        var priorNodesRequested = 0;
        var priorActionsApplied = 0;
        var priorActionsRequested = 0;
        var priorInferencesRequested = 0;
        Dictionary<int, int>? priorActionsPerDepth = null;
        Dictionary<int, int>? priorInferencesPerDepth = null;
        foreach (var player in players)
        {
            if (player is INnStatsProvider nnStats)
            {
                nnRequests += nnStats.TotalNnRequests;
                nnStatesEvaluated += nnStats.TotalNnStatesEvaluated;
            }

            if (player is MctsPlayer mctsPlayer)
            {
                priorNodesRequested += mctsPlayer.TotalPriorNodesRequested;
                priorActionsApplied += mctsPlayer.TotalPriorActionsApplied;
                priorActionsRequested += mctsPlayer.TotalPriorActionsRequested;
                priorInferencesRequested += mctsPlayer.TotalPriorInferencesRequested;
                if (mctsPlayer.TotalPriorActionsPerDepth is { Count: > 0 })
                {
                    priorActionsPerDepth ??= new Dictionary<int, int>();
                    foreach (var kv in mctsPlayer.TotalPriorActionsPerDepth)
                    {
                        priorActionsPerDepth.TryGetValue(kv.Key, out var existing);
                        priorActionsPerDepth[kv.Key] = existing + kv.Value;
                    }
                }
                if (mctsPlayer.TotalPriorInferencesPerDepth is { Count: > 0 })
                {
                    priorInferencesPerDepth ??= new Dictionary<int, int>();
                    foreach (var kv in mctsPlayer.TotalPriorInferencesPerDepth)
                    {
                        priorInferencesPerDepth.TryGetValue(kv.Key, out var existing);
                        priorInferencesPerDepth[kv.Key] = existing + kv.Value;
                    }
                }
            }
            else if (player is IPriorStatsProvider priorStats)
            {
                priorNodesRequested += priorStats.TotalPriorNodesRequested;
                priorActionsApplied += priorStats.TotalPriorActionsApplied;
                priorActionsRequested += priorStats.TotalPriorActionsRequested;
                priorInferencesRequested += priorStats.TotalPriorInferencesRequested;
            }
        }


        foreach (var player in players)
        {
            if (player is IDisposable disposable) disposable.Dispose();
        }

        return (state.WinnerPlayer, state.TurnNumber, nnRequests, nnStatesEvaluated,
            priorNodesRequested, priorActionsApplied, priorActionsRequested,
            priorInferencesRequested, priorActionsPerDepth, priorInferencesPerDepth);
    }

    private GameConfig ResolveGameConfig()
    {
        return _options.MapConfig?.ToLowerInvariant() switch
        {
            "mini" or "m" => GameConfig.Mini,
            "small" or "sm" => GameConfig.Small,
            "standard" or "s" or null or "" => GameConfig.Standard,
            _ => GameConfig.Standard,
        };
    }

    private static void PrintSummary(BenchmarkStats stats)
    {
        Console.WriteLine();
        Console.WriteLine("=== Benchmark Results ===");
        Console.WriteLine($"Total games: {stats.TotalGames}");
        Console.WriteLine($"Total time: {stats.TotalElapsed.TotalSeconds:F2}s");

        if (stats.TotalGames > 0)
        {
            var avgTime = TimeSpan.FromTicks(stats.TotalElapsed.Ticks / stats.TotalGames);
            Console.WriteLine($"Avg time/game: {avgTime.TotalSeconds:F3}s");
        }

        var avgTurns = stats.Games.Count > 0
            ? stats.Games.Average(g => g.Turns)
            : 0;
        Console.WriteLine($"Avg turns/game: {avgTurns:F1}");

        // Aggregate win rates by AI kind (the primary metric).
        Console.WriteLine();
        Console.WriteLine("Win rates by AI:");

        var draws = stats.Games.Count(g => g.WinnerLabel is null);

        foreach (var label in stats.PlayerLabels)
        {
            var wins = stats.Games.Count(g => g.WinnerLabel == label);
            var rate = stats.TotalGames > 0 ? (double)wins / stats.TotalGames * 100 : 0;
            Console.WriteLine($"  {label}: {wins}/{stats.TotalGames} ({rate:F1}%)");
        }

        if (draws > 0)
        {
            var drawRate = (double)draws / stats.TotalGames * 100;
            Console.WriteLine($"  Draws: {draws}/{stats.TotalGames} ({drawRate:F1}%)");
        }

        // Per-seat win rates to verify rotation eliminates positional bias.
        Console.WriteLine();
        Console.WriteLine("Win rates by seat (should be roughly equal with rotation):");
        var playerCount = stats.PlayerAis.Length;
        for (var seat = 1; seat <= playerCount; seat++)
        {
            var seatWins = stats.Games.Count(g => g.WinnerSeat == seat);
            var rate = stats.TotalGames > 0 ? (double)seatWins / stats.TotalGames * 100 : 0;
            Console.WriteLine($"  Seat {seat}: {seatWins}/{stats.TotalGames} ({rate:F1}%)");
        }

        // Prior stats (only shown when priors were used).
        var totalPriorNodesRequested = stats.Games.Sum(g => g.PriorNodesRequested);
        var totalNnRequests = stats.Games.Sum(g => g.NnRequests);
        if (totalNnRequests > 0)
        {
            Console.WriteLine();
            Console.WriteLine("NN stats:");
            Console.WriteLine($"  Requests: {totalNnRequests}");
            Console.WriteLine($"  States evaluated: {stats.Games.Sum(g => g.NnStatesEvaluated)}");
        }

        if (totalPriorNodesRequested > 0)
        {
            var totalPriorActionsApplied = stats.Games.Sum(g => g.PriorActionsApplied);
            var totalPriorActionsRequested = stats.Games.Sum(g => g.PriorActionsRequested);
            var totalPriorInferencesRequested = stats.Games.Sum(g => g.PriorInferencesRequested);
            Console.WriteLine();
            Console.WriteLine("Prior stats:");
            Console.WriteLine($"  Prior nodes requested: {totalPriorNodesRequested}");
            Console.WriteLine($"  Prior actions applied: {totalPriorActionsApplied}");
            Console.WriteLine($"  Prior actions requested: {totalPriorActionsRequested}");
            Console.WriteLine($"  Prior inferences requested: {totalPriorInferencesRequested}");
            if (stats.TotalGames > 0)
            {
                Console.WriteLine($"  Avg nodes requested/game: {(double)totalPriorNodesRequested / stats.TotalGames:F1}");
                Console.WriteLine($"  Avg actions applied/game: {(double)totalPriorActionsApplied / stats.TotalGames:F1}");
                Console.WriteLine($"  Avg inferences requested/game: {(double)totalPriorInferencesRequested / stats.TotalGames:F1}");
            }

            // Per-depth breakdown of action states sent to the NN.
            var aggregatedDepths = new SortedDictionary<int, int>();
            foreach (var game in stats.Games)
            {
                if (game.PriorActionsPerDepth is not { Count: > 0 }) continue;
                foreach (var kv in game.PriorActionsPerDepth)
                {
                    aggregatedDepths.TryGetValue(kv.Key, out var existing);
                    aggregatedDepths[kv.Key] = existing + kv.Value;
                }
            }

            if (aggregatedDepths.Count > 0)
            {
                var depthParts = aggregatedDepths.Select(kv => $"{kv.Key}:{kv.Value}");
                Console.WriteLine($"  Prior actions by depth: {string.Join(", ", depthParts)}");
            }

            var aggregatedInferenceDepths = new SortedDictionary<int, int>();
            foreach (var game in stats.Games)
            {
                if (game.PriorInferencesPerDepth is not { Count: > 0 }) continue;
                foreach (var kv in game.PriorInferencesPerDepth)
                {
                    aggregatedInferenceDepths.TryGetValue(kv.Key, out var existing);
                    aggregatedInferenceDepths[kv.Key] = existing + kv.Value;
                }
            }

            if (aggregatedInferenceDepths.Count > 0)
            {
                var depthParts = aggregatedInferenceDepths.Select(kv => $"{kv.Key}:{kv.Value}");
                Console.WriteLine($"  Prior inferences by depth: {string.Join(", ", depthParts)}");
            }
        }
    }

    private static void ExportResults(BenchmarkStats stats, FileInfo outputPath)
    {
        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        var output = new
        {
            aiKinds = stats.PlayerAis.Select(AiKindNames.Format).ToArray(),
            playerLabels = stats.PlayerLabels,
            totalGames = stats.TotalGames,
            totalElapsedSeconds = Math.Round(stats.TotalElapsed.TotalSeconds, 2),
            winRates = stats.PlayerLabels.Select((label, index) =>
            {
                var wins = stats.Games.Count(g => g.WinnerLabel == label);
                var rate = stats.TotalGames > 0 ? (double)wins / stats.TotalGames : 0.0;
                return new
                {
                    ai = AiKindNames.Format(stats.PlayerAis[index]),
                    label,
                    wins,
                    rate = Math.Round(rate, 4),
                    confidence95Margin = Math.Round(
                        BenchmarkConfidence.Wald95Margin(rate, stats.TotalGames), 6),
                    worstCaseConfidence95Margin = Math.Round(
                        BenchmarkConfidence.WorstCaseWald95Margin(stats.TotalGames), 6),
                };
            }).ToArray(),
            draws = stats.Games.Count(g => g.WinnerAi is null),
            games = stats.Games.Select(g => new
            {
                game = g.GameNumber,
                seed = g.Seed,
                seatAssignment = g.SeatAssignment.Select(AiKindNames.Format).ToArray(),
                seatLabels = g.SeatLabels,
                winnerAi = g.WinnerAi is { } winner ? AiKindNames.Format(winner) : null,
                winnerLabel = g.WinnerLabel,
                winnerSeat = g.WinnerSeat,
                turns = g.Turns,
                elapsedSeconds = Math.Round(g.Elapsed.TotalSeconds, 3),
                nnRequests = g.NnRequests,
                nnStatesEvaluated = g.NnStatesEvaluated,
                priorNodesRequested = g.PriorNodesRequested,
                priorActionsApplied = g.PriorActionsApplied,
                priorActionsRequested = g.PriorActionsRequested,
                priorInferencesRequested = g.PriorInferencesRequested,
                priorActionsPerDepth = g.PriorActionsPerDepth != null
                    ? g.PriorActionsPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
                priorInferencesPerDepth = g.PriorInferencesPerDepth != null
                    ? g.PriorInferencesPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
            }).ToArray(),
        };

        Directory.CreateDirectory(outputPath.DirectoryName ?? ".");
        var json = JsonSerializer.Serialize(output, jsonOptions);
        var temporaryPath = outputPath.FullName + ".tmp";
        File.WriteAllText(temporaryPath, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        File.Move(temporaryPath, outputPath.FullName, overwrite: true);

        Console.WriteLine($"Results exported to {outputPath.FullName}");
    }
}
