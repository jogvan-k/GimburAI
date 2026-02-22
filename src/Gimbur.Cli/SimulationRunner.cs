using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using Gimbur;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Configuration options for running game simulations.
/// </summary>
internal record SimulationOptions
{
    public required uint NumberOfGames { get; init; }
    public int Seed { get; init; }
    public int NumberOfPlayers { get; init; }
    public string? MapConfig { get; init; }
    public FileInfo? ExportPath { get; init; }
    public string Verbosity { get; init; } = "normal";

    /// <summary>
    /// MCTS search time limit in milliseconds. Defaults to 1000ms.
    /// </summary>
    public int SearchTimeMs { get; init; } = 1000;

    /// <summary>
    /// Maximum number of MCTS simulations per decision. Defaults to int.MaxValue (time-limited).
    /// </summary>
    public int MaxSimulations { get; init; } = int.MaxValue;

    /// <summary>
    /// Kjarni MCTS configuration flags (TranspositionTable, AsyncExecution, etc.).
    /// </summary>
    public Kjarni.MCTS.AI.configuration MctsConfig { get; init; } = Kjarni.MCTS.AI.configuration.None;

    /// <summary>
    /// Maximum rollout depth for MCTS simulations. Defaults to 500.
    /// When exceeded, rollout terminates with score-based outcome.
    /// </summary>
    public int MaxRolloutDepth { get; init; } = 500;
}

/// <summary>
/// Per-state MCTS result: the serialized state before the action is taken,
/// the win counts from the MCTS root node after search (raw counts, not
/// normalized), and the number of rollouts (simulations) performed.
/// Win rates can be inferred as winCounts[i] / rollouts.
/// </summary>
internal record SimulationResult
{
    public required string SerializedState { get; init; }
    public required float[] WinCounts { get; init; }
    public required int Rollouts { get; init; }
}

/// <summary>
/// Aggregate container for all simulation results plus metadata.
/// </summary>
internal record SimulationStats
{
    public required List<SimulationResult> Results { get; init; }
    public required int PlayerCount { get; init; }
    public required string MapConfig { get; init; }
    public required int SearchTimeMs { get; init; }
    public required int MaxSimulations { get; init; }
    public required int TotalGames { get; init; }
    public required TimeSpan TotalElapsed { get; init; }
    public required TimeSpan AverageTimePerGame { get; init; }
}

/// <summary>
/// Runs Settlers of Catan game simulations using MCTS-based AI.
/// Supports parallel batch execution and optional training data export.
/// </summary>
internal class SimulationRunner
{
    private readonly SimulationOptions _options;
    private readonly bool _verbose;
    private readonly bool _quiet;

    public SimulationRunner(SimulationOptions options)
    {
        _options = options;
        _verbose = options.Verbosity is "verbose" or "detailed" or "diagnostic" or "d" or "diag";
        _quiet = options.Verbosity is "quiet" or "q";
    }

    public void Run()
    {
        var config = ResolveGameConfig();
        var playerCount = ResolvePlayerCount(config);

        if (!_quiet)
        {
            Console.WriteLine($"Starting {_options.NumberOfGames} game simulation(s)...");
            Console.WriteLine($"  Map: {(_options.MapConfig ?? "standard")}");
            Console.WriteLine($"  Players: {playerCount}");
            Console.WriteLine($"  Seed: {_options.Seed}");
            Console.WriteLine($"  MCTS search time: {_options.SearchTimeMs}ms");
            Console.WriteLine($"  MCTS max simulations: {(_options.MaxSimulations == int.MaxValue ? "unlimited" : _options.MaxSimulations.ToString())}");
            Console.WriteLine($"  Parallelism: {Environment.ProcessorCount} cores");
            if (_options.ExportPath is not null)
            {
                Console.WriteLine($"  Export: {_options.ExportPath.FullName}");
            }

            Console.WriteLine();
        }

        var gameResults = new ConcurrentBag<(int GameNumber, List<SimulationResult> Results, TimeSpan Elapsed)>();

        var totalStopwatch = Stopwatch.StartNew();

        Parallel.For(0, (int)_options.NumberOfGames, new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount,
        }, gameIndex =>
        {
            // Each game gets a deterministic seed derived from the base seed + game index.
            var gameSeed = unchecked(_options.Seed + gameIndex);
            var rng = new Random(gameSeed);

            var gameStopwatch = Stopwatch.StartNew();
            var results = RunSingleGame(config, playerCount, rng, gameIndex + 1);
            gameStopwatch.Stop();

            gameResults.Add((gameIndex + 1, results, gameStopwatch.Elapsed));

            if (!_quiet)
            {
                Console.WriteLine(
                    $"Game {gameIndex + 1}: {results.Count} decision points, " +
                    $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
            }
        });

        totalStopwatch.Stop();

        var allResults = gameResults
            .OrderBy(g => g.GameNumber)
            .SelectMany(g => g.Results)
            .ToList();

        var stats = new SimulationStats
        {
            Results = allResults,
            PlayerCount = playerCount,
            MapConfig = _options.MapConfig ?? "standard",
            SearchTimeMs = _options.SearchTimeMs,
            MaxSimulations = _options.MaxSimulations,
            TotalGames = (int)_options.NumberOfGames,
            TotalElapsed = totalStopwatch.Elapsed,
            AverageTimePerGame = TimeSpan.FromTicks(totalStopwatch.Elapsed.Ticks / Math.Max(1, (int)_options.NumberOfGames)),
        };

        if (!_quiet)
        {
            PrintSummary(stats);
        }

        if (_options.ExportPath is not null)
        {
            ExportTrainingData(stats, _options.ExportPath);
        }
    }

    private List<SimulationResult> RunSingleGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameNumber)
    {
        var state = new CatanState(config, playerCount, rng);
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(
            searchTime.NewMilliSeconds(_options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MctsConfig,
            _options.MaxRolloutDepth);
        var ai = (IGameAI)mcts;
        var results = new List<SimulationResult>();

        // Total action counter guards against infinite loops from non-advancing
        // actions (e.g., cyclic bank trades) where TurnNumber never increments.
        const int maxTotalActions = 10_000;
        var totalActions = 0;
        var lastReportedTurn = -1;

        while (state.WinnerPlayer == 0)
        {
            // Only record MCTS decision points during the main BuildTrade stage.
            if (state.Stage == TurnStage.BuildTrade)
            {
                // Progress reporting: log on turn 1 and then every 10 turns.
                if (!_quiet && (lastReportedTurn < 0 || state.TurnNumber / 10 > lastReportedTurn / 10))
                {
                    lastReportedTurn = state.TurnNumber;
                    Console.WriteLine(
                        $"  Game {gameNumber}: turn {state.TurnNumber}, " +
                        $"{results.Count} decisions so far...");
                }

                var serialized = state.SerializeHumanReadable();

                // Run MCTS from the current state.
                var bestPath = ai.DetermineAction(state);
                var logInfo = mcts.LatestLogInfo();

                // Convert double[] win counts to float[].
                var winCounts = new float[logInfo.winCounts.Length];
                for (var i = 0; i < logInfo.winCounts.Length; i++)
                {
                    winCounts[i] = (float)logInfo.winCounts[i];
                }

                results.Add(new SimulationResult
                {
                    SerializedState = serialized,
                    WinCounts = winCounts,
                    Rollouts = logInfo.simulations,
                });

                // Apply the best action from MCTS.
                var actions = state.Actions();
                if (bestPath.Length > 0 && bestPath[0] < actions.Length)
                {
                    state = (CatanState)actions[bestPath[0]].DoCoreAction();
                }
                else
                {
                    break;
                }
            }
            else
            {
                // For non-BuildTrade stages (dice rolls, setup, etc.), use greedy/comparable sort.
                var actions = state.Actions();
                if (actions.Length == 0) break;

                // Sort by IComparable and pick first (same as MCTS rollout policy).
                Array.Sort(actions);
                state = (CatanState)actions[0].DoCoreAction();
            }

            totalActions++;

            // Safety: prevent infinite loops from non-advancing actions.
            if (totalActions >= maxTotalActions || state.TurnNumber > 500)
            {
                break;
            }
        }

        return results;
    }

    private GameConfig ResolveGameConfig()
    {
        return _options.MapConfig?.ToLowerInvariant() switch
        {
            "mini" or "m" => GameConfig.Mini,
            "standard" or "s" or null or "" => GameConfig.Standard,
            _ => GameConfig.Standard,
        };
    }

    private int ResolvePlayerCount(GameConfig config)
    {
        if (_options.NumberOfPlayers > 0)
        {
            var requested = Math.Clamp(_options.NumberOfPlayers, config.MinPlayers, config.MaxPlayers);
            return requested;
        }

        // Default: use min players (fastest for simulation).
        return config.MinPlayers;
    }

    private static void PrintSummary(SimulationStats stats)
    {
        Console.WriteLine();
        Console.WriteLine("=== Simulation Summary ===");
        Console.WriteLine($"Total games: {stats.TotalGames}");
        Console.WriteLine($"Total decision points: {stats.Results.Count}");
        Console.WriteLine($"Total time: {stats.TotalElapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Avg time/game: {stats.AverageTimePerGame.TotalSeconds:F3}s");
        Console.WriteLine($"MCTS search time: {stats.SearchTimeMs}ms");
        Console.WriteLine($"MCTS max simulations: {(stats.MaxSimulations == int.MaxValue ? "unlimited" : stats.MaxSimulations.ToString())}");

        if (stats.Results.Count > 0)
        {
            var avgRollouts = stats.Results.Average(r => r.Rollouts);
            Console.WriteLine($"Avg rollouts/decision: {avgRollouts:F0}");
        }
    }

    private static void ExportTrainingData(SimulationStats stats, FileInfo exportPath)
    {
        if (stats.Results.Count == 0)
        {
            Console.WriteLine("No simulation results to export.");
            return;
        }

        // Format: one result per line
        // "rollouts|p1_wins,p2_wins,...,pN_wins|serialized_state"
        // Win counts are raw MCTS simulation wins (not normalized).
        // Win rate = winCount / rollouts.
        var sb = new StringBuilder(stats.Results.Count * 600);
        foreach (var result in stats.Results)
        {
            sb.Append(result.Rollouts);
            sb.Append('|');
            // Win counts for players 1..N (skip index 0 which is unused).
            var counts = result.WinCounts
                .Skip(1)
                .Take(stats.PlayerCount)
                .Select(c => c.ToString("F2"));
            sb.Append(string.Join(',', counts));
            sb.Append('|');
            sb.AppendLine(result.SerializedState);
        }

        Directory.CreateDirectory(exportPath.DirectoryName ?? ".");
        File.WriteAllText(exportPath.FullName, sb.ToString());
        Console.WriteLine($"Exported {stats.Results.Count} simulation results to {exportPath.FullName}");
    }
}
