using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
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

    /// <summary>
    /// Whether to include board symmetry permutations in the export.
    /// Defaults to true (all valid symmetries for the map).
    /// </summary>
    public bool Symmetries { get; init; } = true;
}

/// <summary>
/// Per-state record capturing the serialized state and any MCTS search results.
/// States without MCTS search have Simulations=0 and empty Wins.
/// </summary>
internal record StateRecord
{
    public required int PlayerTurn { get; init; }
    public required string SerializedState { get; init; }
    public int Simulations { get; init; }
    public int ElapsedMs { get; init; }
    public double WinRate { get; init; }

    /// <summary>
    /// Raw MCTS win counts, 0-indexed (index 0 = player 1). Empty if no search.
    /// </summary>
    public required double[] Wins { get; init; }
}

/// <summary>
/// Complete result for a single game, including metadata and all state records.
/// </summary>
internal record GameResult
{
    public required int Seed { get; init; }
    public required string Map { get; init; }
    public required int Players { get; init; }
    public required int Winner { get; init; }
    public required int Turns { get; init; }
    public required int SearchTimeMs { get; init; }
    public required int MaxSimulations { get; init; }
    public required int MaxRolloutDepth { get; init; }
    public required string BoardSerialized { get; init; }
    public required List<StateRecord> States { get; init; }
}

/// <summary>
/// Aggregate container for all game results plus timing metadata.
/// </summary>
internal record SimulationStats
{
    public required List<GameResult> Games { get; init; }
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

        // Precompute symmetry permutations (same for all games with the same topology).
        var symmetryPerms = _options.Symmetries
            ? BoardSymmetry.GetPermutations(config.Map.Topology)
            : [];

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
                if (_options.Symmetries)
                {
                    Console.WriteLine($"  Symmetries: {symmetryPerms.Length} permutation(s)");
                    if (symmetryPerms.Length == 0 && _options.Symmetries)
                    {
                        Console.WriteLine("  WARNING: No symmetries available for this map. Permutation arrays will be empty.");
                    }
                }
                else
                {
                    Console.WriteLine("  Symmetries: disabled");
                }
            }

            Console.WriteLine();
        }

        var gameResults = new ConcurrentBag<(int GameNumber, GameResult Result, TimeSpan Elapsed)>();

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
            var result = RunSingleGame(config, playerCount, rng, gameSeed, gameIndex + 1);
            gameStopwatch.Stop();

            gameResults.Add((gameIndex + 1, result, gameStopwatch.Elapsed));

            if (!_quiet)
            {
                Console.WriteLine(
                    $"Game {gameIndex + 1}: {result.States.Count} states, " +
                    $"winner=P{result.Winner}, turns={result.Turns}, " +
                    $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
            }
        });

        totalStopwatch.Stop();

        var allGames = gameResults
            .OrderBy(g => g.GameNumber)
            .Select(g => g.Result)
            .ToList();

        var stats = new SimulationStats
        {
            Games = allGames,
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
            ExportJsonl(stats, _options.ExportPath, symmetryPerms);
        }
    }

    private GameResult RunSingleGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber)
    {
        var state = new CatanState(config, playerCount, rng);
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(
            searchTime.NewMilliSeconds(_options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MctsConfig,
            _options.MaxRolloutDepth);
        var ai = (IGameAI)mcts;
        var states = new List<StateRecord>();

        // Capture the board serialization once (invariant across turns).
        var boardSerialized = state.SerializeBoard();

        // Total action counter guards against infinite loops from non-advancing
        // actions (e.g., cyclic bank trades) where TurnNumber never increments.
        const int maxTotalActions = 10_000;
        var totalActions = 0;
        var lastReportedTurn = -1;

        while (state.WinnerPlayer == 0)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;

            if (actions.Length == 1)
            {
                // Forced action (e.g., dice roll) — no decision to make.
                // Apply it without recording or running MCTS.
                state = (CatanState)actions[0].DoCoreAction();
            }
            else
            {
                // Multiple actions available — run MCTS to decide.
                // Progress reporting: log on turn 1 and then every 10 turns.
                if (!_quiet && state.Stage == TurnStage.BuildTrade
                    && (lastReportedTurn < 0 || state.TurnNumber / 10 > lastReportedTurn / 10))
                {
                    lastReportedTurn = state.TurnNumber;
                    Console.WriteLine(
                        $"  Game {gameNumber}: turn {state.TurnNumber}, " +
                        $"{states.Count} states so far...");
                }

                var serialized = state.SerializeStateOnly();

                // Run MCTS from the current state.
                var bestPath = ai.DetermineAction(state);
                var logInfo = mcts.LatestLogInfo();

                // Convert 1-indexed winCounts to 0-indexed doubles.
                var wins = new double[playerCount];
                for (var i = 0; i < playerCount; i++)
                {
                    if (i + 1 < logInfo.winCounts.Length)
                    {
                        wins[i] = logInfo.winCounts[i + 1];
                    }
                }

                var winRate = logInfo.simulations > 0
                    ? logInfo.estimatedAiWinChance
                    : 0.0;

                states.Add(new StateRecord
                {
                    PlayerTurn = state.CurrentPlayer,
                    SerializedState = serialized,
                    Simulations = logInfo.simulations,
                    ElapsedMs = (int)logInfo.elapsedTime.TotalMilliseconds,
                    WinRate = winRate,
                    Wins = wins,
                });

                // Apply the best action from MCTS.
                if (bestPath.Length > 0 && bestPath[0] < actions.Length)
                {
                    state = (CatanState)actions[bestPath[0]].DoCoreAction();
                }
                else
                {
                    break;
                }
            }

            totalActions++;

            // Safety: prevent infinite loops from non-advancing actions.
            if (totalActions >= maxTotalActions || state.TurnNumber > 500)
            {
                break;
            }
        }

        return new GameResult
        {
            Seed = gameSeed,
            Map = _options.MapConfig ?? "standard",
            Players = playerCount,
            Winner = state.WinnerPlayer,
            Turns = state.TurnNumber,
            SearchTimeMs = _options.SearchTimeMs,
            MaxSimulations = _options.MaxSimulations,
            MaxRolloutDepth = _options.MaxRolloutDepth,
            BoardSerialized = boardSerialized,
            States = states,
        };
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

        var totalStates = stats.Games.Sum(g => g.States.Count);
        var mctsStates = stats.Games.Sum(g => g.States.Count(s => s.Simulations > 0));
        Console.WriteLine($"Total states: {totalStates} ({mctsStates} with MCTS)");
        Console.WriteLine($"Total time: {stats.TotalElapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Avg time/game: {stats.AverageTimePerGame.TotalSeconds:F3}s");

        if (mctsStates > 0)
        {
            var avgRollouts = stats.Games
                .SelectMany(g => g.States)
                .Where(s => s.Simulations > 0)
                .Average(s => s.Simulations);
            Console.WriteLine($"Avg rollouts/MCTS decision: {avgRollouts:F0}");
        }

        var wins = new int[5]; // index 0 = no winner
        foreach (var game in stats.Games)
        {
            if (game.Winner >= 0 && game.Winner < wins.Length)
            {
                wins[game.Winner]++;
            }
        }

        if (stats.TotalGames > 1)
        {
            var winParts = new List<string>();
            for (var p = 1; p < wins.Length; p++)
            {
                if (wins[p] > 0)
                {
                    winParts.Add($"P{p}={wins[p]}");
                }
            }

            if (wins[0] > 0)
            {
                winParts.Add($"none={wins[0]}");
            }

            Console.WriteLine($"Wins: {string.Join(", ", winParts)}");
        }
    }

    private static void ExportJsonl(
        SimulationStats stats,
        FileInfo exportPath,
        ImmutableArray<SymmetryPermutation> symmetryPerms)
    {
        if (stats.Games.Count == 0)
        {
            Console.WriteLine("No simulation results to export.");
            return;
        }

        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };

        Directory.CreateDirectory(exportPath.DirectoryName ?? ".");
        using var writer = new StreamWriter(exportPath.FullName, append: false, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        foreach (var game in stats.Games)
        {
            // Compute board permutations once per game.
            var boardPermutations = symmetryPerms.Length > 0
                ? symmetryPerms.Select(p => BoardSymmetry.PermuteBoard(game.BoardSerialized, p)).ToArray()
                : Array.Empty<string>();

            var jsonObj = new
            {
                version = 1,
                game.Seed,
                game.Map,
                game.Players,
                game.Winner,
                game.Turns,
                constraints = new
                {
                    game.SearchTimeMs,
                    game.MaxSimulations,
                    game.MaxRolloutDepth,
                },
                board = new
                {
                    serialized = game.BoardSerialized,
                    permutations = boardPermutations,
                },
                states = game.States.Select(s => new
                {
                    s.PlayerTurn,
                    s.SerializedState,
                    s.Simulations,
                    s.ElapsedMs,
                    s.WinRate,
                    s.Wins,
                    permutations = symmetryPerms.Length > 0
                        ? symmetryPerms.Select(p => BoardSymmetry.PermuteState(s.SerializedState, p)).ToArray()
                        : Array.Empty<string>(),
                }).ToArray(),
            };

            writer.WriteLine(JsonSerializer.Serialize(jsonObj, jsonOptions));
        }

        var totalStates = stats.Games.Sum(g => g.States.Count);
        Console.WriteLine($"Exported {stats.Games.Count} game(s) ({totalStates} states) to {exportPath.FullName}");
    }
}
