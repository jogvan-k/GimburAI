using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Gimbur;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Identifies an AI strategy that can be used in benchmark games.
/// New strategies should be added here and registered in
/// <see cref="BenchmarkRunner.CreateActionSelector"/>.
/// </summary>
internal enum AiKind
{
    Random,
    Greedy,
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
}

/// <summary>
/// Per-game result captured by the benchmark runner.
/// </summary>
internal record BenchmarkGameResult
{
    public required int GameNumber { get; init; }
    public required int Seed { get; init; }
    public required int Winner { get; init; }
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

        if (!_quiet)
        {
            Console.WriteLine($"Starting benchmark: {_options.NumberOfGames} game(s)");
            Console.WriteLine($"  Map: {_options.MapConfig ?? "standard"}");
            Console.WriteLine($"  Players: {string.Join(" vs ", _options.Players.Select((ai, i) => $"P{i + 1}={ai}"))}");
            Console.WriteLine($"  Seed: {_options.Seed}");
            Console.WriteLine($"  Parallelism: {Environment.ProcessorCount} cores");
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

            var gameStopwatch = Stopwatch.StartNew();
            var (winner, turns) = RunSingleGame(config, rng);
            gameStopwatch.Stop();

            var result = new BenchmarkGameResult
            {
                GameNumber = gameIndex + 1,
                Seed = gameSeed,
                Winner = winner,
                Turns = turns,
                Elapsed = gameStopwatch.Elapsed,
            };

            gameResults.Add(result);

            if (!_quiet)
            {
                var winnerLabel = winner == 0
                    ? "draw"
                    : $"P{winner}({_options.Players[winner - 1]})";
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
    }

    /// <summary>
    /// Creates an action selector function for the given AI kind.
    /// Returns a function that takes a CatanState and a Random, and returns the chosen CatanAction.
    /// </summary>
    private static Func<CatanState, Random, CatanAction?> CreateActionSelector(AiKind kind)
    {
        return kind switch
        {
            AiKind.Greedy => new GreedyActionSelector().ChooseAction,
            AiKind.Random => ChooseRandomAction,
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, $"Unknown AI kind: {kind}"),
        };
    }

    private static CatanAction? ChooseRandomAction(CatanState state, Random rng)
    {
        var coreActions = state.Actions();
        if (coreActions.Length == 0) return null;

        var roll = rng.Next(coreActions.Length);
        var chosen = coreActions[roll];

        if (chosen.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)chosen).Item;
        if (chosen.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)chosen).Item;

        return null;
    }

    private (int Winner, int Turns) RunSingleGame(GameConfig config, Random rng)
    {
        var playerCount = _options.Players.Length;
        var state = new CatanState(config, playerCount, rng);

        // Build per-player action selectors (1-indexed; index 0 unused).
        var selectors = new Func<CatanState, Random, CatanAction?>[playerCount + 1];
        for (var i = 0; i < playerCount; i++)
        {
            selectors[i + 1] = CreateActionSelector(_options.Players[i]);
        }

        const int maxTotalActions = 10_000;
        var totalActions = 0;

        while (state.WinnerPlayer == 0)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;

            var selector = selectors[state.CurrentPlayer];
            var chosen = selector(state, rng);
            if (chosen is null) break;

            state = (CatanState)chosen.DoCoreAction();

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

        Console.WriteLine();
        Console.WriteLine("Win rates:");

        var draws = stats.Games.Count(g => g.Winner == 0);
        for (var p = 0; p < stats.PlayerAis.Length; p++)
        {
            var playerNumber = p + 1;
            var wins = stats.Games.Count(g => g.Winner == playerNumber);
            var rate = stats.TotalGames > 0 ? (double)wins / stats.TotalGames * 100 : 0;
            Console.WriteLine($"  P{playerNumber} ({stats.PlayerAis[p]}): {wins}/{stats.TotalGames} ({rate:F1}%)");
        }

        if (draws > 0)
        {
            var drawRate = (double)draws / stats.TotalGames * 100;
            Console.WriteLine($"  Draws: {draws}/{stats.TotalGames} ({drawRate:F1}%)");
        }

        // Per-AI aggregate when multiple players share the same AI.
        var aiGroups = stats.PlayerAis
            .Select((ai, i) => (Ai: ai, PlayerIndex: i))
            .GroupBy(x => x.Ai)
            .Where(g => g.Count() < stats.PlayerAis.Length) // Only show if AIs differ.
            .ToList();

        if (aiGroups.Count > 1)
        {
            Console.WriteLine();
            Console.WriteLine("Aggregate by AI:");
            foreach (var group in aiGroups)
            {
                var playerNumbers = group.Select(x => x.PlayerIndex + 1).ToHashSet();
                var wins = stats.Games.Count(g => playerNumbers.Contains(g.Winner));
                var rate = stats.TotalGames > 0 ? (double)wins / stats.TotalGames * 100 : 0;
                var seats = string.Join(",", playerNumbers.Select(n => $"P{n}"));
                Console.WriteLine($"  {group.Key} [{seats}]: {wins}/{stats.TotalGames} ({rate:F1}%)");
            }
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
            players = stats.PlayerAis.Select((ai, i) => new
            {
                seat = i + 1,
                ai = ai.ToString().ToLowerInvariant(),
            }).ToArray(),
            totalGames = stats.TotalGames,
            totalElapsedSeconds = Math.Round(stats.TotalElapsed.TotalSeconds, 2),
            winRates = stats.PlayerAis.Select((ai, i) =>
            {
                var playerNumber = i + 1;
                var wins = stats.Games.Count(g => g.Winner == playerNumber);
                return new
                {
                    seat = playerNumber,
                    ai = ai.ToString().ToLowerInvariant(),
                    wins,
                    rate = stats.TotalGames > 0
                        ? Math.Round((double)wins / stats.TotalGames, 4)
                        : 0.0,
                };
            }).ToArray(),
            draws = stats.Games.Count(g => g.Winner == 0),
            games = stats.Games.Select(g => new
            {
                game = g.GameNumber,
                seed = g.Seed,
                winner = g.Winner,
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
