using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
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
/// </summary>
internal sealed class MctsPlayer : IBenchmarkPlayer
{
    private readonly Kjarni.MCTS.AI.MonteCarloTreeSearch _mcts;
    private Kjarni.MCTS.Types.MCTSState? _mctsRoot;

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

        if (_options.Players.Contains(AiKind.Nn))
        {
            _nnClient = new NnClient(_options.NnUrl);
            if (!_nnClient.IsHealthyAsync().GetAwaiter().GetResult())
            {
                Console.Error.WriteLine($"NN inference server at {_options.NnUrl} is not reachable.");
                _nnClient.Dispose();
                return;
            }
        }

        if (!_quiet)
        {
            Console.WriteLine($"Starting benchmark: {_options.NumberOfGames} game(s)");
            Console.WriteLine($"  Map: {_options.MapConfig ?? "standard"}");
            Console.WriteLine($"  Players: {string.Join(" vs ", _options.Players.Select((ai, i) => $"P{i + 1}={ai}"))}");
            Console.WriteLine($"  Seed: {_options.Seed}");
            Console.WriteLine($"  Parallelism: {Environment.ProcessorCount} cores");
            if (_options.Players.Contains(AiKind.Mcts))
            {
                Console.WriteLine($"  MCTS search time: {_options.SearchTimeMs}ms");
                Console.WriteLine($"  MCTS max simulations: {(_options.MaxSimulations == int.MaxValue ? "unlimited" : _options.MaxSimulations.ToString())}");
                Console.WriteLine($"  MCTS max rollout depth: {_options.MaxRolloutDepth}");
            }
            if (_options.Players.Contains(AiKind.Nn))
            {
                Console.WriteLine($"  NN server: {_options.NnUrl}");
            }
            Console.WriteLine();
        }

        var gameResults = new ConcurrentBag<BenchmarkGameResult>();
        var totalStopwatch = Stopwatch.StartNew();

        Parallel.For(0, (int)_options.NumberOfGames, new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount,
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

            var gameStopwatch = Stopwatch.StartNew();
            var (winnerSeat, turns) = RunSingleGame(config, rng, seatAssignment);
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
        });

        totalStopwatch.Stop();

        var stats = new BenchmarkStats
        {
            Games = gameResults.OrderBy(g => g.GameNumber).ToList(),
            TotalGames = (int)_options.NumberOfGames,
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
    }

    /// <summary>
    /// Creates a new player instance for the given AI kind.
    /// A new instance is created per game to allow stateful players (e.g. MCTS tree reuse).
    /// </summary>
    private IBenchmarkPlayer CreatePlayer(AiKind kind)
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
                int.MaxValue)),
            AiKind.Nn => new NnPlayer(_nnClient!),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, $"Unknown AI kind: {kind}"),
        };
    }

    private (int WinnerSeat, int Turns) RunSingleGame(GameConfig config, Random rng, AiKind[] seatAssignment)
    {
        var playerCount = seatAssignment.Length;
        var state = new CatanState(config, playerCount, rng);

        // Build per-player AI instances (1-indexed; index 0 unused).
        var players = new IBenchmarkPlayer[playerCount + 1];
        for (var i = 0; i < playerCount; i++)
        {
            players[i + 1] = CreatePlayer(seatAssignment[i]);
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

        return (state.WinnerPlayer, state.TurnNumber);
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
    }

    private static void ExportResults(BenchmarkStats stats, FileInfo outputPath)
    {
        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
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
            }).ToArray(),
        };

        Directory.CreateDirectory(outputPath.DirectoryName ?? ".");
        var json = JsonSerializer.Serialize(output, jsonOptions);
        File.WriteAllText(outputPath.FullName, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        Console.WriteLine($"Results exported to {outputPath.FullName}");
    }
}
