using Gimbur;
using Kjarni;
using Kjarni.MCTS;
using static Kjarni.MCTS.AI;

namespace Gimbur.Cli;

/// <summary>
/// Configuration options for running game simulations.
/// </summary>
internal record SimulationOptions
{
    public required uint NumberOfGames { get; init; }
    public required TimeSpan SearchTime { get; init; }
    public uint? MaxSimulations { get; init; }
    public int Seed { get; init; }
    public int NumberOfPlayers { get; init; }
    public string? MapConfig { get; init; }
    public FileInfo? ExportPath { get; init; }
    public string Verbosity { get; init; } = "normal";
}

/// <summary>
/// Results from a simulation run.
/// </summary>
internal record SimulationResult
{
    public required int GameNumber { get; init; }
    public required Player Winner { get; init; }
    public required int TotalTurns { get; init; }
    public required int TotalSimulations { get; init; }
    public required TimeSpan ElapsedTime { get; init; }
}

/// <summary>
/// Runs Settlers of Catan game simulations using the Kjarni MCTS engine.
/// </summary>
internal class SimulationRunner
{
    private readonly SimulationOptions _options;
    private readonly bool _verbose;

    public SimulationRunner(SimulationOptions options)
    {
        _options = options;
        _verbose = options.Verbosity is "verbose" or "detailed" or "diagnostic" or "d" or "diag";
    }

    public void Run()
    {
        Console.WriteLine($"Starting {_options.NumberOfGames} game simulation(s)...");
        Console.WriteLine($"  Search time: {_options.SearchTime}");
        Console.WriteLine($"  Max simulations: {_options.MaxSimulations?.ToString() ?? "unlimited"}");
        Console.WriteLine($"  Seed: {_options.Seed}");
        Console.WriteLine();

        var results = new List<SimulationResult>();

        for (uint game = 1; game <= _options.NumberOfGames; game++)
        {
            var result = RunSingleGame(game);
            results.Add(result);

            if (_verbose)
            {
                Console.WriteLine($"Game {game}: Winner = {result.Winner}, Turns = {result.TotalTurns}, " +
                                  $"Simulations = {result.TotalSimulations}, Time = {result.ElapsedTime.TotalSeconds:F2}s");
            }
        }

        PrintSummary(results);

        if (_options.ExportPath is not null)
        {
            Console.WriteLine($"Export to {_options.ExportPath.FullName} not yet implemented.");
        }
    }

    private SimulationResult RunSingleGame(uint gameNumber)
    {
        var searchTime = ConvertToSearchTime(_options.SearchTime);
        var maxSimulations = _options.MaxSimulations.HasValue
            ? (int)_options.MaxSimulations.Value
            : int.MaxValue;

        var ai = new MonteCarloTreeSearch(searchTime, maxSimulations, configuration.AsyncExecution);

        ICoreState state = new CatanState();
        var totalSimulations = 0;
        var startTime = DateTime.UtcNow;

        // Run game loop until terminal state (no actions available)
        while (true)
        {
            var actions = state.Actions();
            if (actions.Length == 0)
            {
                break;
            }

            // Use MCTS to determine best action
            var bestPath = ((IGameAI)ai).DetermineAction(state);
            var logInfo = ai.LatestLogInfo();
            totalSimulations += logInfo.simulations;

            if (bestPath.Length > 0 && bestPath[0] < actions.Length)
            {
                state = actions[bestPath[0]].DoCoreAction();
            }
            else
            {
                // Fallback: take first action if MCTS returns invalid path
                state = actions[0].DoCoreAction();
            }

            // Safety: prevent infinite loops in placeholder implementation
            if (state.TurnNumber > 1000)
            {
                Console.WriteLine("  Warning: Game exceeded 1000 turns, terminating.");
                break;
            }
        }

        var elapsedTime = DateTime.UtcNow - startTime;

        return new SimulationResult
        {
            GameNumber = (int)gameNumber,
            Winner = state.PlayerTurn,
            TotalTurns = state.TurnNumber,
            TotalSimulations = totalSimulations,
            ElapsedTime = elapsedTime
        };
    }

    private static searchTime ConvertToSearchTime(TimeSpan timeSpan)
    {
        var totalMs = (int)timeSpan.TotalMilliseconds;
        return searchTime.NewMilliSeconds(totalMs);
    }

    private void PrintSummary(List<SimulationResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("=== Simulation Summary ===");
        Console.WriteLine($"Total games: {results.Count}");
        Console.WriteLine($"Total simulations: {results.Sum(r => r.TotalSimulations)}");
        Console.WriteLine($"Total time: {results.Sum(r => r.ElapsedTime.TotalSeconds):F2}s");

        var winCounts = results
            .GroupBy(r => r.Winner)
            .OrderBy(g => g.Key)
            .Select(g => $"{g.Key}: {g.Count()}")
            .ToList();

        Console.WriteLine($"Wins by player: {string.Join(", ", winCounts)}");
    }
}
