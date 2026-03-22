using System.Collections.Concurrent;
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
    NnState,
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
internal sealed class MctsPlayer : IBenchmarkPlayer
{
    private readonly Kjarni.MCTS.AI.MonteCarloTreeSearch _mcts;
    private Kjarni.MCTS.Types.MCTSState? _mctsRoot;

    /// <summary>Total prior requests across all MCTS decisions in this game.</summary>
    public int TotalPriorsRequested { get; private set; }

    /// <summary>Total prior responses applied across all MCTS decisions in this game.</summary>
    public int TotalPriorsApplied { get; private set; }

    /// <summary>Total individual states evaluated by the NN server across all decisions.</summary>
    public int TotalPriorStatesEvaluated { get; private set; }

    /// <summary>Per-depth count of prior states evaluated across all MCTS decisions.</summary>
    public Dictionary<int, int>? TotalPriorStatesPerDepth { get; private set; }

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
        TotalPriorsRequested += logInfo.priorsRequested;
        TotalPriorsApplied += logInfo.priorsApplied;
        TotalPriorStatesEvaluated += logInfo.priorStatesEvaluated;
        if (logInfo.priorStatesPerDepth is { Count: > 0 })
        {
            TotalPriorStatesPerDepth ??= new Dictionary<int, int>();
            foreach (var kv in logInfo.priorStatesPerDepth)
            {
                TotalPriorStatesPerDepth.TryGetValue(kv.Key, out var existing);
                TotalPriorStatesPerDepth[kv.Key] = existing + kv.Value;
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

    /// <summary>
    /// The seat assignment used for this game (index 0 = seat 1).
    /// Rotates across games to eliminate positional bias.
    /// </summary>
    public required AiKind[] SeatAssignment { get; init; }

    /// <summary>
    /// The winning seat number (1-based), or 0 for a draw.
    /// </summary>
    public required int WinnerSeat { get; init; }

    public required int Turns { get; init; }
    public required TimeSpan Elapsed { get; init; }

    /// <summary>
    /// Total prior requests across all MCTS decisions in this game.
    /// Zero when no MCTS player uses priors.
    /// </summary>
    public int PriorsRequested { get; init; }

    /// <summary>
    /// Total prior responses applied across all MCTS decisions in this game.
    /// </summary>
    public int PriorsApplied { get; init; }

    /// <summary>
    /// Total individual states evaluated by the NN server across all decisions.
    /// </summary>
    public int PriorStatesEvaluated { get; init; }

    /// <summary>
    /// Per-depth count of prior states evaluated across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorsCalculated { get; init; }
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
}

/// <summary>
/// Runs benchmark games between configurable AI strategies and reports win rates.
/// Games are executed in parallel across available CPU cores.
/// </summary>
internal class BenchmarkRunner
{
    private readonly BenchmarkOptions _options;
    private readonly bool _quiet;
    private NnClient? _nnClient;

    public BenchmarkRunner(BenchmarkOptions options)
    {
        _options = options;
        _quiet = options.Verbosity is "quiet" or "q";
    }

    public void Run()
    {
        var config = ResolveGameConfig();
        var playerCount = _options.Players.Length;

        if (playerCount < config.MinPlayers || playerCount > config.MaxPlayers)
        {
            Console.Error.WriteLine(
                $"Map '{_options.MapConfig ?? "standard"}' requires {config.MinPlayers}-{config.MaxPlayers} players, " +
                $"but {playerCount} AI(s) were specified.");
            return;
        }

        if (UsesNn(_options.Players))
        {
            _nnClient = new NnClient(_options.NnUrl);
            if (!_nnClient.IsHealthyAsync().GetAwaiter().GetResult())
            {
                Console.Error.WriteLine($"NN inference server at {_options.NnUrl} is not reachable.");
                _nnClient.Dispose();
                return;
            }
        }

        // When NN is in use, limit parallelism to avoid overwhelming the
        // inference server with concurrent requests.
        var maxParallelism = UsesNn(_options.Players)
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
                Console.WriteLine($"  NN server: {_options.NnUrl}");
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
            for (var i = 0; i < playerCount; i++)
            {
                seatAssignment[i] = _options.Players[(i + rotation) % playerCount];
            }

            try
            {
                var gameStopwatch = Stopwatch.StartNew();
                var (winnerSeat, turns, priorsRequested, priorsApplied, priorStatesEvaluated, priorsCalculated) = RunSingleGame(config, rng, seatAssignment);
                gameStopwatch.Stop();

                AiKind? winnerAi = winnerSeat > 0 ? seatAssignment[winnerSeat - 1] : null;

                var result = new BenchmarkGameResult
                {
                    GameNumber = gameIndex + 1,
                    Seed = gameSeed,
                    WinnerAi = winnerAi,
                    SeatAssignment = seatAssignment,
                    WinnerSeat = winnerSeat,
                    Turns = turns,
                    Elapsed = gameStopwatch.Elapsed,
                    PriorsRequested = priorsRequested,
                    PriorsApplied = priorsApplied,
                    PriorStatesEvaluated = priorStatesEvaluated,
                    PriorsCalculated = priorsCalculated,
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
        };

        if (!_quiet)
        {
            PrintSummary(stats);
        }

        if (_options.OutputPath is not null)
        {
            ExportResults(stats, _options.OutputPath);
        }

        _nnClient?.Dispose();

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
    private IBenchmarkPlayer CreatePlayer(AiKind kind, GameConfig config)
    {
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
                null)),
            AiKind.Nn => new NnPlayer(_nnClient!),
            AiKind.NnPlacement => new NnPlacementPlayer(
                _nnClient!, PlacementActionSerializer.ForTopology(config.Map.Topology)),
            AiKind.NnState => new NnStatePlayer(_nnClient!),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, $"Unknown AI kind: {kind}"),
        };
    }

    /// <summary>
    /// Returns true if any player in the array requires an NN inference server.
    /// </summary>
    private static bool UsesNn(AiKind[] players) =>
        players.Any(ai => ai is AiKind.Nn or AiKind.NnPlacement or AiKind.NnState);

    private (int WinnerSeat, int Turns, int PriorsRequested, int PriorsApplied, int PriorStatesEvaluated, Dictionary<int, int>? PriorsCalculated) RunSingleGame(GameConfig config, Random rng, AiKind[] seatAssignment)
    {
        var playerCount = seatAssignment.Length;
        var state = new CatanState(config, playerCount, rng);

        // Build per-player AI instances (1-indexed; index 0 unused).
        var players = new IBenchmarkPlayer[playerCount + 1];
        for (var i = 0; i < playerCount; i++)
        {
            players[i + 1] = CreatePlayer(seatAssignment[i], config);
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

        // Aggregate prior stats from all MCTS players in this game.
        var priorsRequested = 0;
        var priorsApplied = 0;
        var priorStatesEvaluated = 0;
        Dictionary<int, int>? priorsCalculated = null;
        foreach (var player in players)
        {
            if (player is MctsPlayer mctsPlayer)
            {
                priorsRequested += mctsPlayer.TotalPriorsRequested;
                priorsApplied += mctsPlayer.TotalPriorsApplied;
                priorStatesEvaluated += mctsPlayer.TotalPriorStatesEvaluated;
                if (mctsPlayer.TotalPriorStatesPerDepth is { Count: > 0 })
                {
                    priorsCalculated ??= new Dictionary<int, int>();
                    foreach (var kv in mctsPlayer.TotalPriorStatesPerDepth)
                    {
                        priorsCalculated.TryGetValue(kv.Key, out var existing);
                        priorsCalculated[kv.Key] = existing + kv.Value;
                    }
                }
            }
        }

        return (state.WinnerPlayer, state.TurnNumber, priorsRequested, priorsApplied, priorStatesEvaluated, priorsCalculated);
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

        var distinctAis = stats.PlayerAis.Distinct().ToList();
        var draws = stats.Games.Count(g => g.WinnerAi is null);

        foreach (var ai in distinctAis)
        {
            var wins = stats.Games.Count(g => g.WinnerAi == ai);
            var rate = stats.TotalGames > 0 ? (double)wins / stats.TotalGames * 100 : 0;
            Console.WriteLine($"  {ai}: {wins}/{stats.TotalGames} ({rate:F1}%)");
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
        var totalPriorsRequested = stats.Games.Sum(g => g.PriorsRequested);
        if (totalPriorsRequested > 0)
        {
            var totalPriorsApplied = stats.Games.Sum(g => g.PriorsApplied);
            var totalPriorStatesEvaluated = stats.Games.Sum(g => g.PriorStatesEvaluated);
            Console.WriteLine();
            Console.WriteLine("Prior stats:");
            Console.WriteLine($"  Priors requested: {totalPriorsRequested}");
            Console.WriteLine($"  Priors applied: {totalPriorsApplied}");
            Console.WriteLine($"  Prior states evaluated: {totalPriorStatesEvaluated}");
            if (stats.TotalGames > 0)
            {
                Console.WriteLine($"  Avg requested/game: {(double)totalPriorsRequested / stats.TotalGames:F1}");
                Console.WriteLine($"  Avg applied/game: {(double)totalPriorsApplied / stats.TotalGames:F1}");
            }

            // Per-depth breakdown of prior states evaluated.
            var aggregatedDepths = new SortedDictionary<int, int>();
            foreach (var game in stats.Games)
            {
                if (game.PriorsCalculated is not { Count: > 0 }) continue;
                foreach (var kv in game.PriorsCalculated)
                {
                    aggregatedDepths.TryGetValue(kv.Key, out var existing);
                    aggregatedDepths[kv.Key] = existing + kv.Value;
                }
            }

            if (aggregatedDepths.Count > 0)
            {
                var depthParts = aggregatedDepths.Select(kv => $"{kv.Key}:{kv.Value}");
                Console.WriteLine($"  Prior states by depth: {string.Join(", ", depthParts)}");
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
            aiKinds = stats.PlayerAis.Distinct().Select(ai => ai.ToString().ToLowerInvariant()).ToArray(),
            totalGames = stats.TotalGames,
            totalElapsedSeconds = Math.Round(stats.TotalElapsed.TotalSeconds, 2),
            winRates = stats.PlayerAis.Distinct().Select(ai =>
            {
                var wins = stats.Games.Count(g => g.WinnerAi == ai);
                return new
                {
                    ai = ai.ToString().ToLowerInvariant(),
                    wins,
                    rate = stats.TotalGames > 0
                        ? Math.Round((double)wins / stats.TotalGames, 4)
                        : 0.0,
                };
            }).ToArray(),
            draws = stats.Games.Count(g => g.WinnerAi is null),
            games = stats.Games.Select(g => new
            {
                game = g.GameNumber,
                seed = g.Seed,
                seatAssignment = g.SeatAssignment.Select(ai => ai.ToString().ToLowerInvariant()).ToArray(),
                winnerAi = g.WinnerAi?.ToString().ToLowerInvariant(),
                winnerSeat = g.WinnerSeat,
                turns = g.Turns,
                elapsedSeconds = Math.Round(g.Elapsed.TotalSeconds, 3),
                priorsRequested = g.PriorsRequested,
                priorsApplied = g.PriorsApplied,
                priorStatesEvaluated = g.PriorStatesEvaluated,
                priorsCalculated = g.PriorsCalculated != null
                    ? g.PriorsCalculated.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
            }).ToArray(),
        };

        Directory.CreateDirectory(outputPath.DirectoryName ?? ".");
        var json = JsonSerializer.Serialize(output, jsonOptions);
        File.WriteAllText(outputPath.FullName, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        Console.WriteLine($"Results exported to {outputPath.FullName}");
    }
}
