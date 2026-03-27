using System.Collections.Concurrent;
using System.Collections.Immutable;
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
/// Specifies the format for exported training data.
/// </summary>
internal enum ExportFormat
{
    /// <summary>No export.</summary>
    None,
    /// <summary>
    /// All games appended to a single JSONL file (one JSON object per line).
    /// The <see cref="SimulationOptions.ExportPath"/> is the file path.
    /// </summary>
    Jsonl,
    /// <summary>
    /// Each game written to its own JSON file with a random GUID filename
    /// inside the directory specified by <see cref="SimulationOptions.ExportPath"/>.
    /// </summary>
    Json,
}

/// <summary>
/// Specifies the type of training data to export.
/// </summary>
internal enum ExportType
{
    /// <summary>
    /// Export game state data for the GimburStateEvaluator model.
    /// Records state-only serialization and best-action stats.
    /// </summary>
    GameState,
    /// <summary>
    /// Export initial placement data for the GimburPlacementActionEvaluator model.
    /// Records placement phase states and all candidate composite actions with per-action stats.
    /// Implies placement-only mode.
    /// </summary>
    InitialPlacement,
}

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
    public ExportFormat ExportFormat { get; init; } = ExportFormat.Jsonl;
    public ExportType ExportType { get; init; } = ExportType.GameState;
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
    /// Maximum rollout depth for MCTS simulations. Defaults to 500.
    /// When exceeded, rollout terminates with score-based outcome.
    /// </summary>
    public int MaxRolloutDepth { get; init; } = 500;

    /// <summary>
    /// Maximum rollouts for any single action before MCTS search stops.
    /// Search stops when any action reaches this count, or when
    /// time/simulation limits are hit first.
    /// Defaults to int.MaxValue (disabled, only time/simulation limits apply).
    /// </summary>
    public int ActionRolloutLimit { get; init; } = int.MaxValue;

    /// <summary>
    /// Whether to include board symmetry permutations in the export.
    /// Defaults to true (all valid symmetries for the map).
    /// </summary>
    public bool Symmetries { get; init; } = true;

    /// <summary>
    /// Whether to enable async NN prior evaluation during MCTS search.
    /// Requires a running inference server at <see cref="NnUrl"/>.
    /// </summary>
    public bool Prior { get; init; }

    /// <summary>
    /// Base URL of the NN inference server. Used when <see cref="Prior"/> is true.
    /// Defaults to http://localhost:8000.
    /// </summary>
    public string NnUrl { get; init; } = "http://localhost:8000";

    /// <summary>
    /// When true, stops the MCTS tree expansion at the placement/main-game boundary.
    /// The expansion guard blocks any action whose underlying CoreAction is a
    /// Stochastic action wrapping a RollDiceAction. The game simulation loop also
    /// stops when the best MCTS action is a HorizonAction.
    /// </summary>
    public bool PlacementOnly { get; init; }

    /// <summary>
    /// Maximum tree depth at which prior (NN) requests are sent.
    /// Nodes deeper than this are left with uniform priors.
    /// Defaults to int.MaxValue (no limit).
    /// </summary>
    public int MaxPriorDepth { get; init; } = int.MaxValue;

    /// <summary>
    /// When set to a positive value, placement simulation enumerates every composite
    /// (settlement + road) action and runs a fresh MCTS search with this many simulations
    /// from each resulting post-placement state. This ensures uniform evaluation coverage
    /// across all actions, including poor ones, producing more balanced training data.
    /// Defaults to 0 (disabled — uses standard MCTS with UCB allocation).
    /// Only applies when <see cref="ExportType"/> is <see cref="ExportType.InitialPlacement"/>.
    /// </summary>
    public int SimulationsPerAction { get; init; }
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
    /// Raw MCTS win counts at the root, 0-indexed (index 0 = player 1). Empty if no search.
    /// </summary>
    public required double[] Wins { get; init; }

    /// <summary>
    /// Win rate of the acting player at the child state reached by the best action.
    /// </summary>
    public double BestActionWinRate { get; init; }

    /// <summary>
    /// Raw MCTS win counts at the child state reached by the best action,
    /// 0-indexed (index 0 = player 1). Empty if no search.
    /// </summary>
    public required double[] BestActionWins { get; init; }

    /// <summary>
    /// Total rollouts at the child state reached by the best action.
    /// For stochastic actions, this is the sum of rollouts across all outcomes.
    /// </summary>
    public int BestActionRollouts { get; init; }

    /// <summary>
    /// Whether the MCTS search fully resolved the tree via terminal propagation.
    /// When true, all root actions are Terminal and the win estimates are exact.
    /// </summary>
    public bool ReachedTerminal { get; init; }

    /// <summary>
    /// Number of prior requests sent to the NN inference server during this MCTS search.
    /// </summary>
    public int PriorsRequested { get; init; }

    /// <summary>
    /// Number of prior responses successfully applied to tree nodes during this MCTS search.
    /// </summary>
    public int PriorsApplied { get; init; }

    /// <summary>
    /// Number of individual states evaluated by the NN server during this MCTS search.
    /// A single request may contain multiple states (one per action outcome).
    /// </summary>
    public int PriorStatesEvaluated { get; init; }

    /// <summary>
    /// Per-depth count of prior states evaluated during this MCTS search.
    /// Key is depth (0 = root), value is number of states evaluated at that depth.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorStatesPerDepth { get; init; }

    /// <summary>
    /// Number of nodes skipped by the ShouldRequestPrior pre-check during this MCTS search.
    /// </summary>
    public int PriorsSkipped { get; init; }

    /// <summary>
    /// Number of prior states that could not be matched with their original state.
    /// </summary>
    public int PriorStatesNotFound { get; set; }
}

/// <summary>
/// Per-action record for a composite placement action (settlement + road).
/// </summary>
internal record PlacementActionRecord
{
    /// <summary>
    /// Composite action string: vertex index + road direction (e.g. "6N", "12NW").
    /// </summary>
    public required string Action { get; init; }

    /// <summary>
    /// Settlement vertex index.
    /// </summary>
    public required int Vertex { get; init; }

    /// <summary>
    /// Road edge index.
    /// </summary>
    public required int Edge { get; init; }

    /// <summary>
    /// MCTS win counts at the road grandchild node, 0-indexed. Empty if unexplored.
    /// </summary>
    public required double[] Wins { get; init; }

    /// <summary>
    /// Total rollouts at the road grandchild node.
    /// </summary>
    public int Rollouts { get; init; }

    /// <summary>
    /// Acting player's win rate at the road grandchild.
    /// </summary>
    public double WinRate { get; init; }
}

/// <summary>
/// Per-state record for a placement decision, capturing all candidate composite actions.
/// </summary>
internal record PlacementStateRecord
{
    public required int PlayerTurn { get; init; }

    /// <summary>
    /// Turn stage character: 'a' (1st settlement) or 'f' (2nd settlement).
    /// </summary>
    public required string Stage { get; init; }

    /// <summary>
    /// 4-section placement phase state: tiles|ports|placementVertices|edges.
    /// </summary>
    public required string SerializedState { get; init; }

    public int Simulations { get; init; }
    public int ElapsedMs { get; init; }

    /// <summary>
    /// Number of prior requests sent to the NN inference server during this MCTS search.
    /// </summary>
    public int PriorsRequested { get; init; }

    /// <summary>
    /// Number of prior responses successfully applied to tree nodes during this MCTS search.
    /// </summary>
    public int PriorsApplied { get; init; }

    /// <summary>
    /// Number of individual states evaluated by the NN server during this MCTS search.
    /// </summary>
    public int PriorStatesEvaluated { get; init; }

    /// <summary>
    /// Per-depth count of prior states evaluated during this MCTS search.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorStatesPerDepth { get; init; }

    /// <summary>
    /// Number of nodes skipped by the ShouldRequestPrior pre-check during this MCTS search.
    /// </summary>
    public int PriorsSkipped { get; init; }

    /// <summary>
    /// All candidate composite actions with per-action MCTS statistics.
    /// </summary>
    public required List<PlacementActionRecord> Actions { get; init; }

    /// <summary>
    /// Number of prior states that could not be matched with their original state.
    /// </summary>
    public int PriorStatesNotFound { get; set; }
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
    public required int ActionRolloutLimit { get; init; }
    public required string BoardSerialized { get; init; }
    public required List<StateRecord> States { get; init; }

    /// <summary>
    /// Per-depth count of prior states evaluated across all MCTS decisions in this game.
    /// Key is depth (0 = root), value is total number of states evaluated at that depth.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorsCalculated { get; init; }
}

/// <summary>
/// Complete result for a single placement-only game.
/// </summary>
internal record PlacementGameResult
{
    public required int Seed { get; init; }
    public required string Map { get; init; }
    public required int Players { get; init; }
    public required int SearchTimeMs { get; init; }
    public required int MaxSimulations { get; init; }
    public required int MaxRolloutDepth { get; init; }
    public required int ActionRolloutLimit { get; init; }
    public int SimulationsPerAction { get; init; }
    public required string BoardSerialized { get; init; }
    public required List<PlacementStateRecord> States { get; init; }

    /// <summary>
    /// Per-depth count of prior states evaluated across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorsCalculated { get; init; }
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
            Console.WriteLine($"  MCTS action rollout limit: {(_options.ActionRolloutLimit == int.MaxValue ? "unlimited" : _options.ActionRolloutLimit.ToString())}");
            Console.WriteLine($"  Prior: {(_options.Prior ? "enabled" : "disabled")}");
            if (_options.Prior)
                Console.WriteLine($"  NN server: {_options.NnUrl}");
            if (_options.PlacementOnly)
                Console.WriteLine("  Mode: placement-only (expansion guard active)");
            if (_options.ExportType == ExportType.InitialPlacement)
                Console.WriteLine("  Export type: InitialPlacement");
            Console.WriteLine($"  Parallelism: {Environment.ProcessorCount} cores");
            if (_options.ExportPath is not null && _options.ExportFormat != ExportFormat.None)
            {
                Console.WriteLine($"  Export: {_options.ExportPath.FullName} ({_options.ExportFormat.ToString().ToLowerInvariant()})");
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
        var placementResults = new ConcurrentBag<(int GameNumber, PlacementGameResult Result, TimeSpan Elapsed)>();
        var isPlacementExport = _options.ExportType == ExportType.InitialPlacement;

        // Prepare export: for JSONL, open a StreamWriter; for JSON, ensure the directory exists.
        var exportFormat = _options.ExportPath is not null ? _options.ExportFormat : ExportFormat.None;
        StreamWriter? exportWriter = null;
        string? exportDir = null;
        var exportLock = new object();
        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        if (exportFormat == ExportFormat.Jsonl && _options.ExportPath is not null)
        {
            exportWriter = CreateExportWriter(_options.ExportPath);
        }
        else if (exportFormat == ExportFormat.Json && _options.ExportPath is not null)
        {
            exportDir = _options.ExportPath.FullName;
            Directory.CreateDirectory(exportDir);
        }

        var totalStopwatch = Stopwatch.StartNew();

        // Create shared PriorClient when prior evaluation is enabled.
        PriorClient? priorClient = null;
        if (_options.Prior)
        {
            var priorMode = _options.ExportType == ExportType.InitialPlacement
                ? PriorMode.Placement
                : PriorMode.State;
            PlacementActionSerializer? priorActionSerializer = null;
            if (priorMode == PriorMode.Placement)
            {
                var priorConfig = ResolveGameConfig();
                priorActionSerializer = PlacementActionSerializer.ForTopology(priorConfig.Map.Topology);
            }
            priorClient = new PriorClient(_options.NnUrl, priorMode, priorActionSerializer);
        }

        try
        {
            Parallel.For(0, (int)_options.NumberOfGames, new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount,
            }, gameIndex =>
            {
                // Each game gets a deterministic seed derived from the base seed + game index.
                var gameSeed = unchecked(_options.Seed + gameIndex);
                var rng = new Random(gameSeed);

                var gameStopwatch = Stopwatch.StartNew();

                if (isPlacementExport)
                {
                    var result = RunSinglePlacementGame(config, playerCount, rng, gameSeed, gameIndex + 1, priorClient);
                    gameStopwatch.Stop();
                    placementResults.Add((gameIndex + 1, result, gameStopwatch.Elapsed));

                    if (exportFormat == ExportFormat.Jsonl && exportWriter is not null)
                    {
                        var jsonObj = BuildPlacementGameJsonObject(result, symmetryPerms);
                        var line = JsonSerializer.Serialize(jsonObj, jsonOptions);
                        lock (exportLock)
                        {
                            exportWriter.WriteLine(line);
                            exportWriter.Flush();
                        }
                    }
                    else if (exportFormat == ExportFormat.Json && exportDir is not null)
                    {
                        var prettyOptions = new JsonSerializerOptions(jsonOptions) { WriteIndented = true };
                        var jsonObj = BuildPlacementGameJsonObject(result, symmetryPerms);
                        var json = JsonSerializer.Serialize(jsonObj, prettyOptions);
                        var path = Path.Combine(exportDir, $"{Guid.NewGuid()}.json");
                        File.WriteAllText(path, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
                    }

                    if (!_quiet)
                    {
                        Console.WriteLine(
                            $"Game {gameIndex + 1}: {result.States.Count} placement states, " +
                            $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
                    }
                }
                else
                {
                    var result = RunSingleGame(config, playerCount, rng, gameSeed, gameIndex + 1, priorClient);
                    gameStopwatch.Stop();
                    gameResults.Add((gameIndex + 1, result, gameStopwatch.Elapsed));

                    if (exportFormat == ExportFormat.Jsonl && exportWriter is not null)
                    {
                        var line = SerializeGameJsonl(result, symmetryPerms, jsonOptions);
                        lock (exportLock)
                        {
                            exportWriter.WriteLine(line);
                            exportWriter.Flush();
                        }
                    }
                    else if (exportFormat == ExportFormat.Json && exportDir is not null)
                    {
                        var json = SerializeGameJson(result, symmetryPerms, jsonOptions);
                        var path = Path.Combine(exportDir, $"{Guid.NewGuid()}.json");
                        File.WriteAllText(path, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
                    }

                    if (!_quiet)
                    {
                        Console.WriteLine(
                            $"Game {gameIndex + 1}: {result.States.Count} states, " +
                            $"winner=P{result.Winner}, turns={result.Turns}, " +
                            $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
                    }
                }
            });
        }
        finally
        {
            exportWriter?.Dispose();
            priorClient?.Dispose();
        }

        totalStopwatch.Stop();

        if (isPlacementExport)
        {
            var allPlacement = placementResults
                .OrderBy(g => g.GameNumber)
                .Select(g => g.Result)
                .ToList();

            if (!_quiet)
            {
                Console.WriteLine();
                Console.WriteLine("=== Placement Simulation Summary ===");
                Console.WriteLine($"Total games: {allPlacement.Count}");
                var totalStates = allPlacement.Sum(g => g.States.Count);
                Console.WriteLine($"Total placement states: {totalStates}");
                Console.WriteLine($"Total time: {totalStopwatch.Elapsed.TotalSeconds:F2}s");
                Console.WriteLine($"Avg time/game: {(totalStopwatch.Elapsed.TotalSeconds / Math.Max(1, allPlacement.Count)):F3}s");
            }

            if (_options.ExportPath is not null && exportFormat != ExportFormat.None)
            {
                var totalStates = allPlacement.Sum(g => g.States.Count);
                Console.WriteLine($"Exported {allPlacement.Count} game(s) ({totalStates} placement states) to {_options.ExportPath.FullName}");
            }
        }
        else
        {
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

            if (_options.ExportPath is not null && exportFormat != ExportFormat.None)
            {
                var totalStates = allGames.Sum(g => g.States.Count);
                Console.WriteLine($"Exported {allGames.Count} game(s) ({totalStates} states) to {_options.ExportPath.FullName}");
            }
        }
    }

    /// <summary>
    /// Extracts the underlying CatanAction from an F# CoreAction discriminated union.
    /// </summary>
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

        if (action.IsHorizonAction)
            return ((Kjarni.MCTS.Types.Action.HorizonAction)action).Item;
        return null;
    }

    /// <summary>
    /// Extracts win counts and win rate for a given player from a child action
    /// in the MCTS tree. For deterministic actions, reads the child state directly.
    /// For stochastic actions, computes probability-weighted averages across outcomes.
    /// For terminal actions, returns the resolved outcome directly.
    /// Returns empty data for unexplored actions.
    /// </summary>
    private static (double[] Wins, double WinRate, int Rollouts) GetChildWinData(
        Kjarni.MCTS.Types.Action childAction, int playerIndex)
    {
        if (childAction.IsTerminal)
        {
            var outcome = ((Kjarni.MCTS.Types.Action.Terminal)childAction).Item;
            var wins = (double[])outcome.Clone();
            var wr = playerIndex < wins.Length ? wins[playerIndex] : 0.0;
            // Rollouts = 0 signals that this is a resolved outcome, not a rollout estimate.
            return (wins, wr, 0);
        }

        if (childAction.IsDeterministicAction)
        {
            var child = ((Kjarni.MCTS.Types.Action.DeterministicAction)childAction).Item;
            var wins = child.WinCounts is { Length: > 0 }
                ? (double[])child.WinCounts.Clone()
                : Array.Empty<double>();
            var wr = child.Rollouts > 0 && playerIndex < wins.Length
                ? wins[playerIndex] / child.Rollouts
                : 0.0;
            return (wins, wr, child.Rollouts);
        }

        if (childAction.IsStochasticAction)
        {
            var outcomes = ((Kjarni.MCTS.Types.Action.StochasticAction)childAction).Item;
            var totalRollouts = 0;
            var playerCount = 0;
            foreach (var o in outcomes)
            {
                totalRollouts += o.State.Rollouts;
                if (playerCount == 0 && o.State.WinCounts is { Length: > 0 })
                    playerCount = o.State.WinCounts.Length;
            }

            if (totalRollouts == 0 || playerCount == 0)
                return (Array.Empty<double>(), 0.0, 0);

            // Probability-weighted average of win counts across outcomes.
            var aggregated = new double[playerCount];
            var totalWeight = 0;
            foreach (var o in outcomes)
            {
                if (o.State.Rollouts == 0) continue;
                totalWeight += o.ProbabilityWeight;
                for (var i = 0; i < Math.Min(playerCount, o.State.WinCounts.Length); i++)
                    aggregated[i] += (double)o.ProbabilityWeight * o.State.WinCounts[i] / o.State.Rollouts;
            }

            if (totalWeight > 0)
            {
                for (var i = 0; i < aggregated.Length; i++)
                    aggregated[i] /= totalWeight;
            }

            var rate = playerIndex < aggregated.Length ? aggregated[playerIndex] : 0.0;
            return (aggregated, rate, totalRollouts);
        }

        if (childAction.IsHorizonAction)
        {
            var child = ((Kjarni.MCTS.Types.Action.HorizonAction)childAction).Item;
            var wins = child.WinCounts is { Length: > 0 }
                ? (double[])child.WinCounts.Clone()
                : Array.Empty<double>();
            var wr = child.Rollouts > 0 && playerIndex < wins.Length
                ? wins[playerIndex] / child.Rollouts
                : 0.0;
            return (wins, wr, child.Rollouts);
        }

        return (Array.Empty<double>(), 0.0, 0);
    }

    private GameResult RunSingleGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        PriorClient? priorClient)
    {
        var state = new CatanState(config, playerCount, rng);
        // Build the expansion guard when placement-only mode is active.
        var expansionGuard = _options.PlacementOnly
            ? Microsoft.FSharp.Core.FSharpOption<Microsoft.FSharp.Core.FSharpFunc<Kjarni.ICoreState, Microsoft.FSharp.Core.FSharpFunc<Kjarni.CoreAction, bool>>>.Some(
                Microsoft.FSharp.Core.FuncConvert.FromFunc<Kjarni.ICoreState, Kjarni.CoreAction, bool>(
                    (state, action) => action.IsStochastic && ((Kjarni.CoreAction.Stochastic)action).Item is RollDiceAction))
            : null;

        var mctsConfig = new Kjarni.MCTSConfig(
            searchTime.NewMilliSeconds(_options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MaxRolloutDepth,
            System.Math.Sqrt(2.0),
            _options.ActionRolloutLimit,
            priorClient,
            expansionGuard,
            _options.MaxPriorDepth);
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(mctsConfig);
        var states = new List<StateRecord>();

        // Capture the board serialization once (invariant across turns).
        var boardSerialized = state.SerializeBoard();

        // Total action counter guards against infinite loops from non-advancing
        // actions (e.g., cyclic bank trades) where TurnNumber never increments.
        const int maxTotalActions = 10_000;
        var totalActions = 0;
        var lastReportedTurn = -1;

        // Persistent MCTS root for tree reuse across actions.
        Kjarni.MCTS.Types.MCTSState? mctsRoot = null;

        while (state.WinnerPlayer == 0)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;

            // TODO: we still need to estimate win chance here
            if (actions.Length == 1)
            {
                // Forced action (e.g., dice roll) — no decision to make.
                // Apply it without recording or running MCTS.
                state = (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();

                // Try to follow the tree so prior calculations are reused.
                mctsRoot = AdvanceMctsRoot(mctsRoot, 0, (ICoreState)state);
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

                // Reuse the existing tree if available; otherwise create a fresh root.
                mctsRoot ??= new Kjarni.MCTS.Types.MCTSState((ICoreState)state);
                mcts.RunSimulation(mctsRoot);
                var bestPath = extractBestPath(mctsRoot);
                var logInfo = mcts.LatestLogInfo();

                // Read win data from the MCTS root node directly (LogInfo fields
                // estimatedAiWinChance and winCounts are not currently populated).
                // Clone the array since WinCounts is mutable and may change with tree reuse.
                var winCounts = mctsRoot.WinCounts is { Length: > 0 }
                    ? (double[])mctsRoot.WinCounts.Clone()
                    : Array.Empty<double>();
                var playerIndex = (int)state.PlayerTurn;
                var winRate = mctsRoot.Rollouts > 0 && playerIndex < winCounts.Length
                    ? winCounts[playerIndex] / mctsRoot.Rollouts
                    : 0.0;

                // Read win data from the child state reached by the best action.
                var bestActionWins = Array.Empty<double>();
                var bestActionWinRate = 0.0;
                var bestActionRollouts = 0;
                if (!bestPath.IsEmpty && bestPath.Head < mctsRoot.Actions.Length)
                {
                    var bestChild = mctsRoot.Actions[bestPath.Head];
                    (bestActionWins, bestActionWinRate, bestActionRollouts) = GetChildWinData(bestChild, playerIndex);
                }

                states.Add(new StateRecord
                {
                    PlayerTurn = state.CurrentPlayer,
                    SerializedState = serialized,
                    Simulations = logInfo.simulations,
                    ElapsedMs = (int)logInfo.elapsedTime.TotalMilliseconds,
                    WinRate = winRate,
                    Wins = winCounts,
                    BestActionWinRate = bestActionWinRate,
                    BestActionWins = bestActionWins,
                    BestActionRollouts = bestActionRollouts,
                    ReachedTerminal = logInfo.reachedTerminal,
                    PriorsRequested = logInfo.priorStatesRequested,
                    PriorStatesNotFound = logInfo.stateNotFound,
                    PriorsApplied = logInfo.priorsApplied,
                    PriorStatesEvaluated = logInfo.priorStatesEvaluated,
                    PriorStatesPerDepth = logInfo.priorStatesPerDepth is { Count: > 0 }
                        ? new Dictionary<int, int>(logInfo.priorStatesPerDepth)
                        : null,
                    PriorsSkipped = logInfo.priorsSkipped,
                });

                // Apply the best action from MCTS and advance the tree.
                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
                    // When the best action is a HorizonAction, the search has reached
                    // the expansion boundary — stop the game loop.
                    if (mctsRoot.Actions[bestPath.Head].IsHorizonAction)
                        break;

                    state = (CatanState)UnwrapCoreAction(actions[bestPath.Head]).DoCoreAction();
                    mctsRoot = AdvanceMctsRoot(mctsRoot, bestPath.Head, (ICoreState)state);
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

        // Aggregate per-depth prior stats across all MCTS decisions.
        Dictionary<int, int>? priorsCalculated = null;
        foreach (var s in states)
        {
            if (s.PriorStatesPerDepth is not { Count: > 0 }) continue;
            priorsCalculated ??= new Dictionary<int, int>();
            foreach (var (depth, count) in s.PriorStatesPerDepth)
            {
                priorsCalculated.TryGetValue(depth, out var existing);
                priorsCalculated[depth] = existing + count;
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
            ActionRolloutLimit = _options.ActionRolloutLimit,
            BoardSerialized = boardSerialized,
            States = states,
            PriorsCalculated = priorsCalculated,
        };
    }

    private PlacementGameResult RunSinglePlacementGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        PriorClient? priorClient)
    {
        var state = new CatanState(config, playerCount, rng);
        var expansionGuard = Microsoft.FSharp.Core.FSharpOption<Microsoft.FSharp.Core.FSharpFunc<Kjarni.ICoreState, Microsoft.FSharp.Core.FSharpFunc<Kjarni.CoreAction, bool>>>.Some(
            Microsoft.FSharp.Core.FuncConvert.FromFunc<Kjarni.ICoreState, Kjarni.CoreAction, bool>(
                (s, action) => action.IsStochastic && ((Kjarni.CoreAction.Stochastic)action).Item is RollDiceAction));

        var mctsConfig = new Kjarni.MCTSConfig(
            searchTime.NewMilliSeconds(_options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MaxRolloutDepth,
            System.Math.Sqrt(2.0),
            _options.ActionRolloutLimit,
            priorClient,
            expansionGuard,
            _options.MaxPriorDepth);
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(mctsConfig);
        var placementStates = new List<PlacementStateRecord>();

        var boardSerialized = state.SerializeBoard();
        var actionSerializer = PlacementActionSerializer.ForTopology(config.Map.Topology);

        const int maxTotalActions = 1000;
        var totalActions = 0;

        Kjarni.MCTS.Types.MCTSState? mctsRoot = null;

        while (state.WinnerPlayer == 0)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;

            var isSettlementStage = state.Stage is TurnStage.PlaceFirstSettlement
                                                 or TurnStage.PlaceSecondSettlement;

            if (actions.Length == 1)
            {
                state = (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();
                mctsRoot = AdvanceMctsRoot(mctsRoot, 0, (ICoreState)state);
            }
            else if (isSettlementStage && _options.SimulationsPerAction > 0)
            {
                // Per-action MCTS mode: enumerate every composite (settlement + road)
                // action and run a fresh MCTS search from each post-placement state.
                // This ensures uniform evaluation coverage across all actions.
                var timer = Stopwatch.StartNew();
                var playerIndex = (int)state.PlayerTurn;
                var stageChar = StateToken.EncodeTurnStage(state.Stage).ToString();
                var serializedState = state.SerializePlacementPhase();

                var compositeActions = new List<PlacementActionRecord>();
                var totalSimulations = 0;

                // Create a per-action MCTS config: simulation-limited, no time limit.
                var perActionMctsConfig = new Kjarni.MCTSConfig(
                    searchTime.Unlimited,
                    _options.SimulationsPerAction,
                    _options.MaxRolloutDepth,
                    System.Math.Sqrt(2.0),
                    _options.ActionRolloutLimit,
                    priorClient,
                    expansionGuard,
                    _options.MaxPriorDepth);
                var perActionMcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(perActionMctsConfig);

                int bestSettlementIdx = 0;
                double bestWinRate = double.NegativeInfinity;

                for (var si = 0; si < actions.Length; si++)
                {
                    var coreSettlement = UnwrapCoreAction(actions[si]);
                    if (coreSettlement is not PlaceSettlementAction psa) continue;
                    var vertex = psa.VertexIndex;

                    // Apply settlement to get road-stage state.
                    var roadState = (CatanState)coreSettlement.DoCoreAction();
                    var roadActions = roadState.Actions();

                    for (var ri = 0; ri < roadActions.Length; ri++)
                    {
                        var coreRoad = UnwrapCoreAction(roadActions[ri]);
                        if (coreRoad is not PlaceRoadAction pra) continue;
                        var edge = pra.EdgeIndex;

                        // Apply road to get post-placement state.
                        var postPlacementState = (ICoreState)coreRoad.DoCoreAction();

                        // Run fresh MCTS from the post-placement state.
                        var actionRoot = new Kjarni.MCTS.Types.MCTSState(postPlacementState);
                        perActionMcts.RunSimulation(actionRoot);

                        var wins = actionRoot.WinCounts is { Length: > 0 }
                            ? (double[])actionRoot.WinCounts.Clone()
                            : Array.Empty<double>();
                        var rollouts = actionRoot.Rollouts;
                        var winRate = rollouts > 0 && playerIndex < wins.Length
                            ? wins[playerIndex] / rollouts
                            : 0.0;
                        totalSimulations += rollouts;

                        var actionString = actionSerializer.Serialize(vertex, edge);
                        compositeActions.Add(new PlacementActionRecord
                        {
                            Action = actionString,
                            Vertex = vertex,
                            Edge = edge,
                            Wins = wins,
                            Rollouts = rollouts,
                            WinRate = winRate,
                        });

                        if (winRate > bestWinRate)
                        {
                            bestWinRate = winRate;
                            bestSettlementIdx = si;
                        }
                    }
                }

                placementStates.Add(new PlacementStateRecord
                {
                    PlayerTurn = state.CurrentPlayer,
                    Stage = stageChar,
                    SerializedState = serializedState,
                    Simulations = totalSimulations,
                    ElapsedMs = (int)timer.Elapsed.TotalMilliseconds,
                    Actions = compositeActions,
                });

                // Apply the best settlement action and advance.
                state = (CatanState)UnwrapCoreAction(actions[bestSettlementIdx]).DoCoreAction();
                mctsRoot = null; // Tree not reusable in per-action mode.
            }
            else if (isSettlementStage)
            {
                // Standard MCTS mode: run a single search from the settlement root.
                mctsRoot ??= new Kjarni.MCTS.Types.MCTSState((ICoreState)state);
                mcts.RunSimulation(mctsRoot);
                var bestPath = extractBestPath(mctsRoot);
                var logInfo = mcts.LatestLogInfo();

                var playerIndex = (int)state.PlayerTurn;
                var stageChar = StateToken.EncodeTurnStage(state.Stage).ToString();
                var serializedState = state.SerializePlacementPhase();

                // Build composite actions by iterating settlement children and their road grandchildren.
                var compositeActions = new List<PlacementActionRecord>();
                for (var si = 0; si < mctsRoot.Actions.Length; si++)
                {
                    var settlementAction = mctsRoot.Actions[si];
                    // Extract the vertex index from the underlying PlaceSettlementAction.
                    var coreSettlement = UnwrapCoreAction(actions[si]);
                    if (coreSettlement is not PlaceSettlementAction psa) continue;
                    var vertex = psa.VertexIndex;

                    // Get the child MCTSState for this settlement action.
                    Kjarni.MCTS.Types.MCTSState? childMctsState = null;
                    if (settlementAction.IsDeterministicAction)
                        childMctsState = ((Kjarni.MCTS.Types.Action.DeterministicAction)settlementAction).Item;
                    else if (settlementAction.IsHorizonAction)
                        childMctsState = ((Kjarni.MCTS.Types.Action.HorizonAction)settlementAction).Item;

                    if (childMctsState is null)
                    {
                        // Unexplored settlement — we don't know the road actions.
                        // Skip (no composite actions to record for this vertex).
                        continue;
                    }

                    // The child state's actions are PlaceRoadActions.
                    var childCoreState = childMctsState.State;
                    var roadCoreActions = ((CatanState)childCoreState).Actions();

                    for (var ri = 0; ri < childMctsState.Actions.Length; ri++)
                    {
                        var roadAction = childMctsState.Actions[ri];
                        var coreRoad = UnwrapCoreAction(roadCoreActions[ri]);
                        if (coreRoad is not PlaceRoadAction pra) continue;
                        var edge = pra.EdgeIndex;

                        var (wins, winRate, rollouts) = GetChildWinData(roadAction, playerIndex);
                        var actionString = actionSerializer.Serialize(vertex, edge);

                        compositeActions.Add(new PlacementActionRecord
                        {
                            Action = actionString,
                            Vertex = vertex,
                            Edge = edge,
                            Wins = wins,
                            Rollouts = rollouts,
                            WinRate = winRate,
                        });
                    }
                }

                placementStates.Add(new PlacementStateRecord
                {
                    PlayerTurn = state.CurrentPlayer,
                    Stage = stageChar,
                    SerializedState = serializedState,
                    Simulations = logInfo.simulations,
                    ElapsedMs = (int)logInfo.elapsedTime.TotalMilliseconds,
                    PriorsRequested = logInfo.priorStatesRequested,
                    PriorsApplied = logInfo.priorsApplied,
                    PriorStatesEvaluated = logInfo.priorStatesEvaluated,
                    PriorStatesNotFound = logInfo.stateNotFound,
                    PriorStatesPerDepth = logInfo.priorStatesPerDepth is { Count: > 0 }
                        ? new Dictionary<int, int>(logInfo.priorStatesPerDepth)
                        : null,
                    PriorsSkipped = logInfo.priorsSkipped,
                    Actions = compositeActions,
                });

                // Apply the best action and advance.
                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
                    if (mctsRoot.Actions[bestPath.Head].IsHorizonAction)
                        break;
                    state = (CatanState)UnwrapCoreAction(actions[bestPath.Head]).DoCoreAction();
                    mctsRoot = AdvanceMctsRoot(mctsRoot, bestPath.Head, (ICoreState)state);
                }
                else
                {
                    break;
                }
            }
            else
            {
                // Road stage or any other non-settlement stage: run MCTS and apply best action.
                mctsRoot ??= new Kjarni.MCTS.Types.MCTSState((ICoreState)state);
                mcts.RunSimulation(mctsRoot);
                var bestPath = extractBestPath(mctsRoot);

                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
                    if (mctsRoot.Actions[bestPath.Head].IsHorizonAction)
                        break;
                    state = (CatanState)UnwrapCoreAction(actions[bestPath.Head]).DoCoreAction();
                    mctsRoot = AdvanceMctsRoot(mctsRoot, bestPath.Head, (ICoreState)state);
                }
                else
                {
                    break;
                }
            }

            totalActions++;
            if (totalActions >= maxTotalActions) break;
        }

        // Aggregate per-depth prior stats across all MCTS decisions.
        Dictionary<int, int>? priorsCalculated = null;
        foreach (var s in placementStates)
        {
            if (s.PriorStatesPerDepth is not { Count: > 0 }) continue;
            priorsCalculated ??= new Dictionary<int, int>();
            foreach (var (depth, count) in s.PriorStatesPerDepth)
            {
                priorsCalculated.TryGetValue(depth, out var existing);
                priorsCalculated[depth] = existing + count;
            }
        }

        return new PlacementGameResult
        {
            Seed = gameSeed,
            Map = _options.MapConfig ?? "standard",
            Players = playerCount,
            SearchTimeMs = _options.SearchTimeMs,
            MaxSimulations = _options.MaxSimulations,
            MaxRolloutDepth = _options.MaxRolloutDepth,
            ActionRolloutLimit = _options.ActionRolloutLimit,
            SimulationsPerAction = _options.SimulationsPerAction,
            BoardSerialized = boardSerialized,
            States = placementStates,
            PriorsCalculated = priorsCalculated,
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

            var terminalCount = stats.Games
                .SelectMany(g => g.States)
                .Count(s => s.ReachedTerminal);
            if (terminalCount > 0)
            {
                Console.WriteLine($"Reached terminal: {terminalCount}/{mctsStates} ({100.0 * terminalCount / mctsStates:F1}%)");
            }

            // Prior stats (only shown when prior evaluation was used).
            var totalPriorsRequested = stats.Games
                .SelectMany(g => g.States)
                .Sum(s => s.PriorsRequested);
            if (totalPriorsRequested > 0)
            {
                var totalPriorsApplied = stats.Games
                    .SelectMany(g => g.States)
                    .Sum(s => s.PriorsApplied);
                var totalPriorStatesEvaluated = stats.Games
                    .SelectMany(g => g.States)
                    .Sum(s => s.PriorStatesEvaluated);
                var avgRequested = (double)totalPriorsRequested / mctsStates;
                var avgApplied = (double)totalPriorsApplied / mctsStates;
                var avgStatesEvaluated = (double)totalPriorStatesEvaluated / mctsStates;
                Console.WriteLine($"Priors requested: {totalPriorsRequested} (avg {avgRequested:F1}/decision)");
                Console.WriteLine($"Priors applied: {totalPriorsApplied} (avg {avgApplied:F1}/decision)");
                Console.WriteLine($"Prior states evaluated: {totalPriorStatesEvaluated} (avg {avgStatesEvaluated:F1}/decision)");

                // Per-depth breakdown of prior states evaluated.
                var aggregatedDepths = new SortedDictionary<int, int>();
                foreach (var game in stats.Games)
                {
                    if (game.PriorsCalculated is not { Count: > 0 }) continue;
                    foreach (var (depth, count) in game.PriorsCalculated)
                    {
                        aggregatedDepths.TryGetValue(depth, out var existing);
                        aggregatedDepths[depth] = existing + count;
                    }
                }

                if (aggregatedDepths.Count > 0)
                {
                    var depthParts = aggregatedDepths.Select(kv => $"{kv.Key}:{kv.Value}");
                    Console.WriteLine($"Prior states by depth: {string.Join(", ", depthParts)}");
                }
            }
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

    /// <summary>
    /// Serializes a single game result to a JSON string for JSONL output.
    /// Thread-safe (no shared mutable state).
    /// </summary>
    /// <summary>
    /// Creates a StreamWriter for JSONL export, ensuring the directory exists.
    /// </summary>
    private static StreamWriter CreateExportWriter(FileInfo exportPath)
    {
        Directory.CreateDirectory(exportPath.DirectoryName ?? ".");
        return new StreamWriter(exportPath.FullName, append: false, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    /// <summary>
    /// Serializes a single game result to a compact JSON string for JSONL output.
    /// Thread-safe (no shared mutable state).
    /// </summary>
    private static string SerializeGameJsonl(
        GameResult game,
        ImmutableArray<SymmetryPermutation> symmetryPerms,
        JsonSerializerOptions jsonOptions)
    {
        var jsonObj = BuildGameJsonObject(game, symmetryPerms);
        return JsonSerializer.Serialize(jsonObj, jsonOptions);
    }

    /// <summary>
    /// Serializes a single game result to a pretty-printed JSON string for
    /// per-game file export. Same structure as <see cref="SerializeGameJsonl"/>
    /// but with indentation for readability.
    /// </summary>
    private static string SerializeGameJson(
        GameResult game,
        ImmutableArray<SymmetryPermutation> symmetryPerms,
        JsonSerializerOptions baseOptions)
    {
        var prettyOptions = new JsonSerializerOptions(baseOptions) { WriteIndented = true };
        // Reuse the same serialization logic — SerializeGameJsonl produces the
        // object, we just need to re-serialize with indentation.
        var jsonObj = BuildGameJsonObject(game, symmetryPerms);
        return JsonSerializer.Serialize(jsonObj, prettyOptions);
    }

    /// <summary>
    /// Builds the anonymous object representing a game result for JSON serialization.
    /// Shared between JSONL (compact) and JSON (pretty) export formats.
    /// </summary>
    private static object BuildGameJsonObject(
        GameResult game,
        ImmutableArray<SymmetryPermutation> symmetryPerms)
    {
        var boardPermutations = symmetryPerms.Length > 0
            ? symmetryPerms.Select(p => BoardSymmetry.PermuteBoard(game.BoardSerialized, p)).ToArray()
            : Array.Empty<string>();

        return new
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
                game.ActionRolloutLimit,
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
                s.BestActionWinRate,
                s.BestActionWins,
                s.BestActionRollouts,
                s.ReachedTerminal,
                s.PriorsRequested,
                s.PriorsApplied,
                s.PriorStatesEvaluated,
                s.PriorsSkipped,
                permutations = symmetryPerms.Length > 0
                    ? symmetryPerms.Select(p => BoardSymmetry.PermuteState(s.SerializedState, p)).ToArray()
                    : Array.Empty<string>(),
            }).ToArray(),
            priorsCalculated = game.PriorsCalculated != null
                ? game.PriorsCalculated.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
        };
    }

    /// <summary>
    /// Builds the anonymous object representing a placement game result for JSON serialization.
    /// </summary>
    private static object BuildPlacementGameJsonObject(
        PlacementGameResult game,
        ImmutableArray<SymmetryPermutation> symmetryPerms)
    {
        var boardPermutations = symmetryPerms.Length > 0
            ? symmetryPerms.Select(p => BoardSymmetry.PermuteBoard(game.BoardSerialized, p)).ToArray()
            : Array.Empty<string>();

        var actionSerializer = PlacementActionSerializer.ForTopology(
            (game.Map?.ToLowerInvariant()) switch
            {
                "mini" or "m" => BoardTopology.Mini,
                "small" or "sm" => BoardTopology.Small,
                _ => BoardTopology.Standard,
            });

        return new
        {
            game.Seed,
            game.Map,
            game.Players,
            exportType = "initialPlacement",
            constraints = new
            {
                game.SearchTimeMs,
                game.MaxSimulations,
                game.MaxRolloutDepth,
                game.ActionRolloutLimit,
                game.SimulationsPerAction,
            },
            board = new
            {
                serialized = game.BoardSerialized,
                permutations = boardPermutations,
            },
            states = game.States.Select(s => new
            {
                s.PlayerTurn,
                s.Stage,
                s.SerializedState,
                s.Simulations,
                s.ElapsedMs,
                s.PriorsRequested,
                s.PriorsApplied,
                s.PriorStatesEvaluated,
                s.PriorsSkipped,
                actions = s.Actions.Select(a => new
                {
                    a.Action,
                    a.Wins,
                    a.Rollouts,
                    a.WinRate,
                    permutations = symmetryPerms.Length > 0
                        ? symmetryPerms.Select(p =>
                            actionSerializer.Serialize(p.Vertices[a.Vertex], p.Edges[a.Edge]))
                            .ToArray()
                        : Array.Empty<string>(),
                }).ToArray(),
                permutations = symmetryPerms.Length > 0
                    ? symmetryPerms.Select(p => BoardSymmetry.PermutePlacementState(s.SerializedState, p)).ToArray()
                    : Array.Empty<string>(),
            }).ToArray(),
            priorsCalculated = game.PriorsCalculated != null
                ? game.PriorsCalculated.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
        };
    }
}
