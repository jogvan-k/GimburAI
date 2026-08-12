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
    /// Export initial placement diagnostics using the compact placement representation.
    /// Records every non-forced settlement and road root with per-action edge statistics.
    /// Implies placement-only mode.
    /// </summary>
    InitialPlacement,
    /// <summary>Export placement policy roots and full-game states from one game.</summary>
    PlacementAndState,
}

internal static class SimulationRouting
{
    public static PriorMode PriorModeFor(ExportType exportType, bool placementPhase) =>
        PriorMode.State;
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
    public bool GreedyPrior { get; init; }
    public double GreedyPriorUniformMix { get; init; } = 0.25;
    public string Verbosity { get; init; } = "normal";

    /// <summary>
    /// MCTS search time limit in milliseconds. Defaults to 1000ms.
    /// </summary>
    public int SearchTimeMs { get; init; } = 1000;
    public int PlacementSearchTimeMs { get; init; } = 16000;
    public int MainGameSearchTimeMs { get; init; } = 8000;

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
    /// The leaf boundary matches the post-placement PreRoll state. The game simulation
    /// loop also stops when the best MCTS action is a HorizonAction.
    /// </summary>
    public bool PlacementOnly { get; init; }

    /// <summary>
    /// Maximum tree depth at which prior (NN) requests are sent.
    /// Nodes deeper than this are left with uniform priors.
    /// Defaults to int.MaxValue (no limit).
    /// </summary>

    /// <summary>
    /// Maximum games simulated concurrently. Zero selects the existing default:
    /// four with NN priors, otherwise all logical processors.
    /// </summary>
    public int Parallelism { get; init; }
    public int MaxPendingEvaluations { get; init; } = 32;
    public int LeafEvaluationTimeoutMs { get; init; } = 500;
    public int DrainTimeoutMs { get; init; } = 1000;
    public int MaxErrorsPerGame { get; init; } = 5;
    public double MaxErrorRatePerGame { get; init; } = 0.02;
    public int MinimumRequestsForRate { get; init; } = 50;
    public bool DiscardGamesWithFallbacks { get; init; }
    public int MaxDiscardedGames { get; init; } = 20;
    public double MaxDiscardRate { get; init; } = 0.05;
    public int MinimumAttemptsForDiscardRate { get; init; } = 50;
    public int MaxConsecutiveDiscards { get; init; } = 5;
}

internal record EvaluationDiagnostics
{
    public int Submitted { get; set; }
    public int Applied { get; set; }
    public int Timeouts { get; set; }
    public int InvalidResponses { get; set; }
    public int Cancelled { get; set; }
    public int Fallbacks { get; set; }
    public int Orphans { get; set; }
    public int Batches { get; set; }
    public int States { get; set; }
    public long LatencyMs { get; set; }
    public int PriorResponsesOrphaned { get; set; }

    public int HardErrors => Timeouts + InvalidResponses + Orphans;

    public void Add(Kjarni.MCTS.Types.LogInfo info)
    {
        Submitted += info.leafEvaluationsSubmitted;
        Applied += info.leafEvaluationsApplied;
        Timeouts += info.leafEvaluationTimeouts;
        InvalidResponses += info.leafEvaluationsInvalid;
        Cancelled += info.leafEvaluationsCancelled;
        Fallbacks += info.leafEvaluationFallbacks;
        Orphans += info.leafEvaluationOrphans;
        Batches += info.leafEvaluationBatches;
        States += info.leafEvaluationStates;
        LatencyMs += info.leafEvaluationLatencyMs;
        PriorResponsesOrphaned += info.priorResponsesOrphaned;
    }
}

internal static class SimulationErrorPolicy
{
    public static string? GetGameDiscardReason(EvaluationDiagnostics diagnostics, SimulationOptions options)
    {
        if (diagnostics.HardErrors > options.MaxErrorsPerGame)
            return $"hard errors {diagnostics.HardErrors} exceeded {options.MaxErrorsPerGame}";
        if (diagnostics.Submitted >= options.MinimumRequestsForRate
            && diagnostics.HardErrors / (double)Math.Max(1, diagnostics.Submitted) > options.MaxErrorRatePerGame)
            return $"hard error rate {diagnostics.HardErrors / (double)diagnostics.Submitted:P2} exceeded {options.MaxErrorRatePerGame:P2}";
        if (options.DiscardGamesWithFallbacks && diagnostics.Fallbacks > 0)
            return $"fallbacks prohibited ({diagnostics.Fallbacks})";
        return null;
    }

    public static string? GetGenerationStopReason(
        int attempted, int discarded, int consecutiveDiscards, SimulationOptions options)
    {
        if (discarded > options.MaxDiscardedGames)
            return $"discarded games {discarded} exceeded {options.MaxDiscardedGames}";
        if (attempted >= options.MinimumAttemptsForDiscardRate
            && discarded / (double)Math.Max(1, attempted) > options.MaxDiscardRate)
            return $"discard rate {discarded / (double)attempted:P2} exceeded {options.MaxDiscardRate:P2}";
        if (consecutiveDiscards > options.MaxConsecutiveDiscards)
            return $"consecutive discards {consecutiveDiscards} exceeded {options.MaxConsecutiveDiscards}";
        return null;
    }
}

/// <summary>
/// Per-state record capturing the serialized state and any MCTS search results.
/// States without MCTS search have Simulations=0 and empty Wins.
/// </summary>
internal record StateRecord
{
    public required int PlayerTurn { get; init; }
    public required int TurnNumber { get; init; }
    public required string Stage { get; init; }
    public required string SerializedState { get; init; }
    public required double[] Scores { get; init; }
    public int Simulations { get; init; }
    public int ElapsedMs { get; init; }
    public double WinRate { get; init; }

    /// <summary>
    /// Raw MCTS win counts at the root, 0-indexed (index 0 = player 1). Empty if no search.
    /// </summary>
    public required double[] Wins { get; init; }

    /// <summary>Exact resolved value distribution when available.</summary>
    public double[]? ValueTarget { get; init; }

    /// <summary>Per-action MCTS diagnostics at this root.</summary>
    public required List<StateActionRecord> Actions { get; init; }

    /// <summary>
    /// Whether the MCTS search fully resolved the tree via terminal propagation.
    /// When true, all root actions are Terminal and the win estimates are exact.
    /// </summary>
    public bool ReachedTerminal { get; init; }

    /// <summary>
    /// Number of nodes for which a prior was requested during this MCTS search.
    /// </summary>
    public int PriorNodesRequested { get; init; }

    /// <summary>
    /// Number of individual action states whose priors were successfully applied.
    /// </summary>
    public int PriorActionsApplied { get; init; }

    /// <summary>
    /// Number of MCTS-level action states sent to the prior client during this MCTS search.
    /// </summary>
    public int PriorActionsRequested { get; init; }

    /// <summary>
    /// Number of (state, action) inference pairs actually sent to the NN model during this MCTS search.
    /// In placement mode this counts post-fan-out composite (settlement, road) pairs.
    /// </summary>
    public int PriorInferencesRequested { get; init; }

    /// <summary>
    /// Per-depth count of MCTS-level action states sent to the client during this MCTS search.
    /// Key is depth (0 = root), value is number of action states at that depth.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorActionsPerDepth { get; init; }

    /// <summary>
    /// Per-depth count of model inference pairs during this MCTS search.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorInferencesPerDepth { get; init; }

    /// <summary>
    /// Number of nodes refused by the client's ShouldRequestPrior pre-check during this MCTS search.
    /// </summary>
    public int PriorNodesSkipped { get; init; }

    /// <summary>
    /// Number of prior responses returned for nodes the search no longer tracks.
    /// </summary>
    public int PriorResponsesOrphaned { get; set; }
}

/// <summary>Per-action diagnostics at a full-game MCTS root.</summary>
internal record StateActionRecord
{
    public required string Action { get; init; }
    public required double[] Wins { get; init; }
    public int Visits { get; init; }
    public double WinRate { get; init; }
    public double? ModelPrior { get; init; }
    public bool Selected { get; init; }
    public List<StateActionOutcomeRecord> Outcomes { get; init; } = [];
}

internal record StateActionOutcomeRecord
{
    public required string Outcome { get; init; }
    public required double[] Wins { get; init; }
    public int Visits { get; init; }
    public double WinRate { get; init; }
    public double? ModelPrior { get; init; }
    public bool Selected { get; init; }
}

/// <summary>
/// Per-action statistics at a placement-stage MCTS root.
/// </summary>
internal record PlacementActionRecord
{
    public required int PolicyIndex { get; init; }

    /// <summary>Road edge used only to transform direction indices under symmetry.</summary>
    public int? RoadEdge { get; init; }

    /// <summary>
    /// Per-player value sums on the root action edge. Empty if unexplored.
    /// </summary>
    public required double[] Wins { get; init; }

    /// <summary>
    /// Completed visits on the root action edge.
    /// </summary>
    public int Visits { get; init; }

    /// <summary>
    /// Acting player's root-edge value average.
    /// </summary>
    public double WinRate { get; init; }

    /// <summary>
    /// NN probability aligned with this root action.
    /// </summary>
    public double? ModelPrior { get; init; }

    /// <summary>Whether this action was selected for the played game.</summary>
    public bool Selected { get; init; }

}

/// <summary>
/// Per-state record for a settlement or road placement decision.
/// </summary>
internal record PlacementStateRecord
{
    public required int PlayerTurn { get; init; }

    /// <summary>Pending settlement vertex for road-stage symmetry transforms.</summary>
    public int? PendingVertex { get; init; }

    /// <summary>
    /// Turn stage character: settlement 'a'/'f' or road 'e'/'i'.
    /// </summary>
    public required string Stage { get; init; }

    /// <summary>
    /// 5-section placement phase state: tiles|ports|stage|placementVertices|edges.
    /// </summary>
    public required string SerializedState { get; init; }

    public int Simulations { get; init; }
    public int ElapsedMs { get; init; }

    /// <summary>
    /// Number of nodes for which a prior was requested during this MCTS search.
    /// </summary>
    public int PriorNodesRequested { get; init; }

    /// <summary>
    /// Number of individual action states whose priors were successfully applied.
    /// </summary>
    public int PriorActionsApplied { get; init; }

    /// <summary>
    /// Number of MCTS-level action states sent to the prior client during this MCTS search.
    /// </summary>
    public int PriorActionsRequested { get; init; }

    /// <summary>
    /// Number of (state, action) inference pairs actually sent to the NN model during this MCTS search.
    /// </summary>
    public int PriorInferencesRequested { get; init; }

    /// <summary>
    /// Per-depth count of MCTS-level action states sent to the client during this MCTS search.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorActionsPerDepth { get; init; }

    /// <summary>
    /// Per-depth count of model inference pairs during this MCTS search.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorInferencesPerDepth { get; init; }

    /// <summary>
    /// Number of nodes refused by the client's ShouldRequestPrior pre-check during this MCTS search.
    /// </summary>
    public int PriorNodesSkipped { get; init; }

    /// <summary>
    /// All legal root actions with per-edge MCTS statistics.
    /// </summary>
    public required List<PlacementActionRecord> Actions { get; init; }

    /// <summary>
    /// Number of prior responses returned for nodes the search no longer tracks.
    /// </summary>
    public int PriorResponsesOrphaned { get; set; }

    /// <summary>
    /// NN value distribution for this state, indexed by player.
    /// Null when no value estimate is available (non-combined model or no NN prior).
    /// </summary>
    public double[]? ModelValue { get; init; }

    /// <summary>Rollout-weighted per-player win distribution across legal actions.</summary>
    public double[]? ValueTarget { get; init; }
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
    /// Per-depth count of MCTS-level action states sent to the client across all MCTS decisions in this game.
    /// Key is depth (0 = root), value is total number of action states at that depth.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorActionsPerDepth { get; init; }

    /// <summary>
    /// Per-depth count of model inference pairs across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorInferencesPerDepth { get; init; }
    public EvaluationDiagnostics EvaluationDiagnostics { get; init; } = new();
}

/// <summary>
/// Complete result for a single placement-only game.
/// </summary>
internal record PlacementGameResult
{
    public required int Seed { get; init; }
    public required string Map { get; init; }
    public required int Players { get; init; }
    public required int Winner { get; init; }
    public required int SearchTimeMs { get; init; }
    public required int MaxSimulations { get; init; }
    public required int MaxRolloutDepth { get; init; }
    public required int ActionRolloutLimit { get; init; }
    public required string BoardSerialized { get; init; }
    public required List<PlacementStateRecord> States { get; init; }

    /// <summary>
    /// Per-depth count of MCTS-level action states sent to the client across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorActionsPerDepth { get; init; }

    /// <summary>
    /// Per-depth count of model inference pairs across all MCTS decisions in this game.
    /// Null when no priors were used.
    /// </summary>
    public Dictionary<int, int>? PriorInferencesPerDepth { get; init; }
    public EvaluationDiagnostics EvaluationDiagnostics { get; init; } = new();
}

internal record CombinedGameResult
{
    public required GameResult Game { get; init; }
    public required List<PlacementStateRecord> PlacementStates { get; init; }
    public required int PlacementSearchTimeMs { get; init; }
    public required int MainGameSearchTimeMs { get; init; }
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

    public int Run()
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
                Console.WriteLine("  Mode: placement-only (leaf boundary active)");
            if (_options.ExportType == ExportType.InitialPlacement)
                Console.WriteLine("  Export type: InitialPlacement");
            var parallelism = _options.Parallelism > 0
                ? Math.Min(_options.Parallelism, Environment.ProcessorCount)
                : _options.Prior
                    ? Math.Min(4, Environment.ProcessorCount)
                    : Environment.ProcessorCount;
            Console.WriteLine($"  Parallelism: {parallelism} cores");
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
        var combinedResults = new ConcurrentBag<(int GameNumber, CombinedGameResult Result, TimeSpan Elapsed)>();
        var isPlacementExport = _options.ExportType == ExportType.InitialPlacement;
        var isCombinedExport = _options.ExportType == ExportType.PlacementAndState;

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
        var discardedDir = exportDir is not null
            ? Path.Combine(exportDir, "discarded")
            : _options.ExportPath?.Directory is { } parent
                ? Path.Combine(parent.FullName, "discarded")
                : Path.Combine(Environment.CurrentDirectory, "discarded");
        var attempted = 0;
        var nextAttempt = 0;
        var accepted = 0;
        var discarded = 0;
        var consecutiveDiscards = 0;
        string? stopReason = null;
        using var stop = new CancellationTokenSource();
        var targetGames = checked((int)_options.NumberOfGames);

        // Create shared PriorClient when prior evaluation is enabled.
        IPriorClient? priorClient = null;
        IPriorClient? placementPriorClient = null;
        if (_options.Prior)
        {
            priorClient = new PriorClient(_options.NnUrl);
            if (isCombinedExport)
            {
                placementPriorClient = priorClient;
            }
        }
        else if (_options.GreedyPrior)
        {
            priorClient = new GreedyPriorClient(_options.GreedyPriorUniformMix);
            placementPriorClient = isCombinedExport ? priorClient : null;
        }

        try
        {
            var parallelism = _options.Parallelism > 0
                ? Math.Min(_options.Parallelism, Environment.ProcessorCount)
                : priorClient is not null
                    ? Math.Min(4, Environment.ProcessorCount)
                    : Environment.ProcessorCount;
            Parallel.For(0, parallelism, new ParallelOptions
            {
                // When NN priors are active the inference server is the bottleneck;
                // limit parallelism to avoid OOM from too many concurrent MCTS
                // trees blocking on root-prior waits.
                MaxDegreeOfParallelism = parallelism,
                CancellationToken = stop.Token,
            }, _ =>
            {
                while (!stop.IsCancellationRequested)
                {
                    int attemptNumber;
                    lock (exportLock)
                    {
                        if (accepted >= targetGames || stopReason is not null)
                            break;
                        attemptNumber = ++nextAttempt;
                    }

                    var gameSeed = unchecked(_options.Seed + attemptNumber - 1);
                    var rng = new Random(gameSeed);

                    var gameStopwatch = Stopwatch.StartNew();

                    if (isPlacementExport)
                    {
                    var result = RunSinglePlacementGame(config, playerCount, rng, gameSeed, attemptNumber, priorClient);
                    gameStopwatch.Stop();
                        lock (exportLock)
                        {
                            if (stopReason is not null || accepted >= targetGames)
                                continue;
                            attempted++;
                            var reason = SimulationErrorPolicy.GetGameDiscardReason(result.EvaluationDiagnostics, _options);
                            if (reason is not null)
                            {
                                discarded++;
                                consecutiveDiscards++;
                                if (exportFormat != ExportFormat.None)
                                    WriteDiscardDiagnostic(discardedDir, result.Seed, result.Map, result.Players,
                                        "initialPlacement", result.Winner, null, reason, result.EvaluationDiagnostics, jsonOptions);
                            }
                            else if (accepted < targetGames)
                            {
                                accepted++;
                                consecutiveDiscards = 0;
                                placementResults.Add((accepted, result, gameStopwatch.Elapsed));
                                WriteAcceptedPlacement(result, symmetryPerms, exportFormat, exportWriter, exportDir, jsonOptions);
                            }
                            stopReason = SimulationErrorPolicy.GetGenerationStopReason(
                                attempted, discarded, consecutiveDiscards, _options);
                            if (accepted >= targetGames || stopReason is not null)
                                stop.Cancel();
                        }

                    if (!_quiet)
                    {
                        Console.WriteLine(
                            $"Attempt {attemptNumber}: {result.States.Count} placement states, " +
                            $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
                    }
                    }
                    else
                    {
                    var combined = isCombinedExport
                        ? RunSingleCombinedGame(config, playerCount, rng, gameSeed, attemptNumber,
                            placementPriorClient, priorClient)
                        : null;
                    var result = combined?.Game
                        ?? RunSingleGame(config, playerCount, rng, gameSeed, attemptNumber, priorClient);
                    gameStopwatch.Stop();
                        lock (exportLock)
                        {
                            if (stopReason is not null || accepted >= targetGames)
                                continue;
                            attempted++;
                            var reason = SimulationErrorPolicy.GetGameDiscardReason(result.EvaluationDiagnostics, _options);
                            if (reason is not null)
                            {
                                discarded++;
                                consecutiveDiscards++;
                                if (exportFormat != ExportFormat.None)
                                    WriteDiscardDiagnostic(discardedDir, result.Seed, result.Map, result.Players,
                                        "gameState", result.Winner, result.Turns, reason, result.EvaluationDiagnostics, jsonOptions);
                            }
                            else if (accepted < targetGames)
                            {
                                accepted++;
                                consecutiveDiscards = 0;
                                if (combined is not null)
                                {
                                    combinedResults.Add((accepted, combined, gameStopwatch.Elapsed));
                                    WriteAcceptedCombined(combined, symmetryPerms, exportFormat, exportWriter, exportDir, jsonOptions);
                                }
                                else
                                {
                                    gameResults.Add((accepted, result, gameStopwatch.Elapsed));
                                    WriteAcceptedGame(result, symmetryPerms, exportFormat, exportWriter, exportDir, jsonOptions);
                                }
                            }
                            stopReason = SimulationErrorPolicy.GetGenerationStopReason(
                                attempted, discarded, consecutiveDiscards, _options);
                            if (accepted >= targetGames || stopReason is not null)
                                stop.Cancel();
                        }

                    if (!_quiet)
                    {
                        Console.WriteLine(
                            $"Attempt {attemptNumber}: {result.States.Count} states, " +
                            $"winner=P{result.Winner}, turns={result.Turns}, " +
                            $"{gameStopwatch.Elapsed.TotalSeconds:F1}s");
                    }
                    }
                }
            });
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested) { }
        finally
        {
            exportWriter?.Dispose();
            if (priorClient is IDisposable priorDisposable)
                priorDisposable.Dispose();
            if (!ReferenceEquals(placementPriorClient, priorClient)
                && placementPriorClient is IDisposable placementDisposable)
                placementDisposable.Dispose();
        }

        totalStopwatch.Stop();
        Console.WriteLine($"Simulation attempts: {attempted}, accepted: {accepted}, discarded: {discarded}");
        if (stopReason is not null)
            Console.Error.WriteLine($"Simulation stopped: {stopReason}");

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
            var allGames = (isCombinedExport
                    ? combinedResults.Select(g => (g.GameNumber, Result: g.Result.Game, g.Elapsed))
                    : gameResults)
                .OrderBy(g => g.GameNumber)
                .Select(g => g.Result)
                .ToList();

            var stats = new SimulationStats
            {
                Games = allGames,
                TotalGames = allGames.Count,
                TotalElapsed = totalStopwatch.Elapsed,
                AverageTimePerGame = TimeSpan.FromTicks(totalStopwatch.Elapsed.Ticks / Math.Max(1, allGames.Count)),
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
        return stopReason is null && accepted >= targetGames ? 0 : 2;
    }

    private static void WriteAcceptedGame(
        GameResult result, ImmutableArray<SymmetryPermutation> symmetryPerms, ExportFormat format,
        StreamWriter? writer, string? exportDir, JsonSerializerOptions options)
    {
        if (format == ExportFormat.Jsonl && writer is not null)
        {
            writer.WriteLine(SerializeGameJsonl(result, symmetryPerms, options));
            writer.Flush();
        }
        else if (format == ExportFormat.Json && exportDir is not null)
        {
            File.WriteAllText(Path.Combine(exportDir, $"{Guid.NewGuid()}.json"),
                SerializeGameJson(result, symmetryPerms, options), new UTF8Encoding(false));
        }
    }

    private static void WriteAcceptedPlacement(
        PlacementGameResult result, ImmutableArray<SymmetryPermutation> symmetryPerms, ExportFormat format,
        StreamWriter? writer, string? exportDir, JsonSerializerOptions options)
    {
        var obj = BuildPlacementGameJsonObject(result, symmetryPerms);
        if (format == ExportFormat.Jsonl && writer is not null)
        {
            writer.WriteLine(JsonSerializer.Serialize(obj, options));
            writer.Flush();
        }
        else if (format == ExportFormat.Json && exportDir is not null)
        {
            var pretty = new JsonSerializerOptions(options) { WriteIndented = true };
            File.WriteAllText(Path.Combine(exportDir, $"{Guid.NewGuid()}.json"),
                JsonSerializer.Serialize(obj, pretty), new UTF8Encoding(false));
        }
    }

    private static void WriteAcceptedCombined(
        CombinedGameResult result, ImmutableArray<SymmetryPermutation> symmetryPerms, ExportFormat format,
        StreamWriter? writer, string? exportDir, JsonSerializerOptions options)
    {
        var obj = BuildCombinedGameJsonObject(result, symmetryPerms);
        if (format == ExportFormat.Jsonl && writer is not null)
        {
            writer.WriteLine(JsonSerializer.Serialize(obj, options));
            writer.Flush();
        }
        else if (format == ExportFormat.Json && exportDir is not null)
        {
            var pretty = new JsonSerializerOptions(options) { WriteIndented = true };
            File.WriteAllText(Path.Combine(exportDir, $"{Guid.NewGuid()}.json"),
                JsonSerializer.Serialize(obj, pretty), new UTF8Encoding(false));
        }
    }

    private static void WriteDiscardDiagnostic(
        string directory, int seed, string map, int players, string exportType, int winner, int? turns,
        string reason, EvaluationDiagnostics diagnostics, JsonSerializerOptions options)
    {
        Directory.CreateDirectory(directory);
        var payload = new { seed, map, players, exportType, winner, turns, reason, evaluationDiagnostics = diagnostics };
        var pretty = new JsonSerializerOptions(options) { WriteIndented = true };
        File.WriteAllText(Path.Combine(directory, $"discarded-{seed}.json"),
            JsonSerializer.Serialize(payload, pretty), new UTF8Encoding(false));
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

    internal static (double[] Wins, double WinRate, int Rollouts) GetActionWinData(
        Kjarni.MCTS.Types.MCTSState parent, int actionIndex, int playerIndex)
    {
        var stats = parent.ActionStats[actionIndex];
        if (stats.CompletedVisits <= 0)
            return (Array.Empty<double>(), 0.0, 0);

        var wins = (double[])stats.ValueSums.Clone();
        var rate = playerIndex < wins.Length ? wins[playerIndex] / stats.CompletedVisits : 0.0;
        return (wins, rate, stats.CompletedVisits);
    }

    private static StateActionRecord CreateStateActionRecord(
        Kjarni.MCTS.Types.MCTSState mctsRoot,
        CoreAction action,
        int actionIndex,
        int playerIndex,
        bool selected,
        string? selectedOutcome,
        int flattenedPriorOffset)
    {
        var (wins, winRate, visits) = GetActionWinData(mctsRoot, actionIndex, playerIndex);
        var coreAction = UnwrapCoreAction(action);
        var outcomes = new List<StateActionOutcomeRecord>();
        if (coreAction is CatanStochasticAction stochastic)
        {
            var ruleOutcomes = stochastic.Outcomes();
            var treeOutcomes = mctsRoot.Actions[actionIndex].IsStochasticAction
                ? ((Kjarni.MCTS.Types.Action.StochasticAction)mctsRoot.Actions[actionIndex]).Item
                : [];
            for (var outcomeIndex = 0; outcomeIndex < ruleOutcomes.Length; outcomeIndex++)
            {
                var outcomeState = (CatanState)ruleOutcomes[outcomeIndex].Item2;
                var treeState = outcomeIndex < treeOutcomes.Length
                    ? treeOutcomes[outcomeIndex].State
                    : null;
                var outcomeVisits = treeState?.Rollouts ?? 0;
                var outcomeWins = outcomeVisits > 0 && treeState is not null
                    ? (double[])treeState.WinCounts.Clone()
                    : [];
                outcomes.Add(new StateActionOutcomeRecord
                {
                    Outcome = DescribeOutcome(stochastic, outcomeState, outcomeIndex),
                    Wins = outcomeWins,
                    Visits = outcomeVisits,
                    WinRate = outcomeVisits > 0 && playerIndex < outcomeWins.Length
                        ? outcomeWins[playerIndex] / outcomeVisits
                        : 0.0,
                    ModelPrior = mctsRoot.FlattenedPriors is { } flattened
                        && flattenedPriorOffset + outcomeIndex < flattened.Value.Length
                            ? flattened.Value[flattenedPriorOffset + outcomeIndex]
                            : null,
                    Selected = selected && selectedOutcome == DescribeOutcome(stochastic, outcomeState, outcomeIndex),
                });
            }
        }
        return new StateActionRecord
        {
            Action = DescribeAction(coreAction),
            Wins = wins,
            Visits = visits,
            WinRate = winRate,
            ModelPrior = mctsRoot.Priors is { } priors && actionIndex < priors.Value.Length
                    ? priors.Value[actionIndex]
                    : null,
            Selected = selected,
            Outcomes = outcomes,
        };
    }

    internal static string DescribeAction(CatanAction action) => action switch
    {
        PlaceSettlementAction settlement => $"PlaceSettlement:{settlement.VertexIndex}",
        PlaceRoadAction road => $"PlaceRoad:{road.EdgeIndex}",
        RollDiceAction => "Roll",
        ChooseRobberTileAction robber => $"PlaceRobber:{robber.TileIndex}",
        ChooseRobberVictimAction victim => $"ChooseRobberVictim:Player{victim.VictimPlayer}",
        PlaceCityAction city => $"PlaceCity:{city.VertexIndex}",
        BuyRoadAction => "BuyRoad",
        BuySettlementAction => "BuySettlement",
        UpgradeCityAction => "UpgradeCity",
        TradeWithBankAction => "TradeWithBank",
        ChooseBankTradeGiveAction give => $"ChooseBankTradeGive:{give.Resource}",
        ChooseBankTradeReceiveAction receive => $"ChooseBankTradeReceive:{receive.Resource}",
        BuyDevCardAction => "BuyDevCard",
        PlayKnightAction => "PlayKnight",
        PlayRoadBuildingAction => "PlayRoadBuilding",
        PlayMonopolyAction => "PlayMonopoly",
        ChooseMonopolyResourceAction monopoly => $"ChooseMonopolyResource:{monopoly.Resource}",
        PlayYearOfPlentyAction => "PlayYearOfPlenty",
        ChooseYearOfPlentyResourceAction plenty => $"ChooseYearOfPlentyResource:{plenty.Resource}",
        EndTurnAction => "EndTurn",
        _ => action.GetType().Name,
    };

    internal static string TransformActionDescription(
        string action,
        SymmetryPermutation permutation)
    {
        var separator = action.LastIndexOf(':');
        if (separator < 0 || !int.TryParse(action[(separator + 1)..], out var index))
            return action;
        var name = action[..separator];
        return name switch
        {
            "PlaceRobber" => $"{name}:{permutation.Tiles[index]}",
            "PlaceSettlement" or "PlaceCity" => $"{name}:{permutation.Vertices[index]}",
            "PlaceRoad" => $"{name}:{permutation.Edges[index]}",
            _ => action,
        };
    }

    private static string DescribeOutcome(
        CatanStochasticAction action,
        CatanState outcome,
        int outcomeIndex)
    {
        if (action is RollDiceAction)
            return outcome.LastDiceRoll.ToString();
        if (action is BuyDevCardAction)
        {
            foreach (var type in Enum.GetValues<DevCardType>())
            {
                if (outcome.DevCardsInHand(action.OriginState.CurrentPlayer, type)
                    > action.OriginState.DevCardsInHand(action.OriginState.CurrentPlayer, type))
                    return type.ToString();
            }
        }
        if (action is ChooseRobberTileAction or ChooseRobberVictimAction)
        {
            for (var resource = ResourceType.Wood; resource <= ResourceType.Ore; resource++)
            {
                if (outcome.ResourceCountFor(action.OriginState.CurrentPlayer, resource)
                    > action.OriginState.ResourceCountFor(action.OriginState.CurrentPlayer, resource))
                    return resource.ToString();
            }
            return "NoSteal";
        }
        return RuleOutcomeLabel(action, outcomeIndex);
    }

    private static string RuleOutcomeLabel(CatanStochasticAction action, int outcomeIndex) =>
        $"Outcome{outcomeIndex}";

    private static double[]? ComputePlacementValueTarget(IReadOnlyList<PlacementActionRecord> actions)
    {
        var totalVisits = actions.Sum(action => action.Visits);
        if (totalVisits == 0)
            return null;

        var playerCount = actions.Max(action => action.Wins.Length);
        if (playerCount == 0)
            return null;

        var target = new double[playerCount];
        foreach (var action in actions)
        {
            for (var player = 0; player < action.Wins.Length; player++)
                target[player] += action.Wins[player];
        }

        for (var player = 0; player < target.Length; player++)
            target[player] /= totalVisits;
        return target;
    }

    private static PlacementStateRecord CreatePlacementStateRecord(
        CatanState state,
        CoreAction[] actions,
        Kjarni.MCTS.Types.MCTSState mctsRoot,
        Kjarni.MCTS.Types.LogInfo logInfo,
        PlacementActionSerializer actionSerializer,
        int selectedActionIndex = -1)
    {
        var playerIndex = (int)state.PlayerTurn;
        var stageActions = new List<PlacementActionRecord>(actions.Length);
        var settlementStage = state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement;
        var roadStage = state.Stage is TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad;
        if (!settlementStage && !roadStage)
            throw new InvalidOperationException($"Stage {state.Stage} is not an initial placement decision.");

        for (var actionIndex = 0; actionIndex < actions.Length; actionIndex++)
        {
            var coreAction = UnwrapCoreAction(actions[actionIndex]);
            var (policyIndex, roadEdge) = coreAction switch
            {
                PlaceSettlementAction settlement when settlementStage => (settlement.VertexIndex, (int?)null),
                PlaceRoadAction road when roadStage && state.PendingSettlementVertex is { } pending =>
                    (actionSerializer.DirectionIndexOf(pending, road.EdgeIndex), road.EdgeIndex),
                _ => throw new InvalidOperationException(
                    $"Action {coreAction.GetType().Name} does not match placement stage {state.Stage}."),
            };
            var (wins, winRate, visits) = GetActionWinData(mctsRoot, actionIndex, playerIndex);
            stageActions.Add(new PlacementActionRecord
            {
                PolicyIndex = policyIndex,
                RoadEdge = roadEdge,
                Wins = wins,
                Visits = visits,
                 WinRate = winRate,
                 ModelPrior = mctsRoot.DensePriors is { } dense && actionIndex < dense.Value.Length
                    ? dense.Value[actionIndex]
                    : mctsRoot.Priors is { } priors && actionIndex < priors.Value.Length
                         ? priors.Value[actionIndex]
                         : null,
                Selected = actionIndex == selectedActionIndex,
            });
        }

        return new PlacementStateRecord
        {
            PlayerTurn = state.CurrentPlayer,
            PendingVertex = state.PendingSettlementVertex,
            Stage = StateToken.EncodeTurnStage(state.Stage).ToString(),
            SerializedState = state.SerializePlacementPhase(),
            Simulations = logInfo.simulations,
            ElapsedMs = (int)logInfo.elapsedTime.TotalMilliseconds,
            PriorNodesRequested = logInfo.priorNodesRequested,
            PriorActionsApplied = logInfo.priorActionsApplied,
            PriorActionsRequested = logInfo.priorActionsRequested,
            PriorInferencesRequested = logInfo.priorInferencesRequested,
            PriorResponsesOrphaned = logInfo.priorResponsesOrphaned,
            PriorActionsPerDepth = logInfo.priorActionsPerDepth is { Count: > 0 }
                ? new Dictionary<int, int>(logInfo.priorActionsPerDepth) : null,
            PriorInferencesPerDepth = logInfo.priorInferencesPerDepth is { Count: > 0 }
                ? new Dictionary<int, int>(logInfo.priorInferencesPerDepth) : null,
            PriorNodesSkipped = logInfo.priorNodesSkipped,
            Actions = stageActions,
            ModelValue = mctsRoot.ValueEstimates is { } values ? (double[])values.Value.Clone() : null,
            ValueTarget = ComputePlacementValueTarget(stageActions),
        };
    }

    private static StateRecord CreateStateRecord(
        CatanState state,
        string serialized,
        Kjarni.MCTS.Types.MCTSState mctsRoot,
        Kjarni.MCTS.Types.LogInfo logInfo,
        int selectedActionIndex,
        string? selectedOutcome)
    {
        var winCounts = mctsRoot.WinCounts is { Length: > 0 }
            ? (double[])mctsRoot.WinCounts.Clone()
            : Array.Empty<double>();
        var playerIndex = (int)state.PlayerTurn;
        var winRate = mctsRoot.Rollouts > 0 && playerIndex < winCounts.Length
            ? winCounts[playerIndex] / mctsRoot.Rollouts
            : 0.0;
        var resolvedTarget = tryResolveState(mctsRoot) is { } resolved
            ? (double[])resolved.Value.Clone()
            : null;

        return new StateRecord
        {
            PlayerTurn = state.CurrentPlayer,
            TurnNumber = state.TurnNumber,
            Stage = StateToken.EncodeTurnStage(state.Stage).ToString(),
            SerializedState = serialized,
            Scores = state.Scores(),
            Simulations = logInfo.simulations,
            ElapsedMs = (int)logInfo.elapsedTime.TotalMilliseconds,
            WinRate = winRate,
            Wins = winCounts,
            ValueTarget = resolvedTarget,
            Actions = CreateStateActionRecords(
                state.Actions(), mctsRoot, playerIndex, selectedActionIndex, selectedOutcome),
            ReachedTerminal = logInfo.reachedTerminal,
            PriorNodesRequested = logInfo.priorNodesRequested,
            PriorResponsesOrphaned = logInfo.priorResponsesOrphaned,
            PriorActionsApplied = logInfo.priorActionsApplied,
            PriorActionsRequested = logInfo.priorActionsRequested,
            PriorInferencesRequested = logInfo.priorInferencesRequested,
            PriorActionsPerDepth = logInfo.priorActionsPerDepth is { Count: > 0 }
                ? new Dictionary<int, int>(logInfo.priorActionsPerDepth)
                : null,
            PriorInferencesPerDepth = logInfo.priorInferencesPerDepth is { Count: > 0 }
                ? new Dictionary<int, int>(logInfo.priorInferencesPerDepth)
                : null,
            PriorNodesSkipped = logInfo.priorNodesSkipped,
        };
    }

    private static List<StateActionRecord> CreateStateActionRecords(
        CoreAction[] actions,
        Kjarni.MCTS.Types.MCTSState mctsRoot,
        int playerIndex,
        int selectedActionIndex,
        string? selectedOutcome)
    {
        var result = new List<StateActionRecord>(actions.Length);
        var flattenedPriorOffset = 0;
        for (var actionIndex = 0; actionIndex < actions.Length; actionIndex++)
        {
            result.Add(CreateStateActionRecord(
                mctsRoot,
                actions[actionIndex],
                actionIndex,
                playerIndex,
                actionIndex == selectedActionIndex,
                selectedOutcome,
                flattenedPriorOffset));
            var action = UnwrapCoreAction(actions[actionIndex]);
            flattenedPriorOffset += action is CatanStochasticAction stochastic
                ? stochastic.Outcomes().Length
                : 1;
        }
        return result;
    }

    private static StateRecord CreateUnsearchedStateRecord(
        CatanState state,
        CoreAction? forcedAction = null,
        string? selectedOutcome = null)
    {
        var wins = new double[state.PlayerCount];
        if (state.WinnerPlayer > 0)
            wins[state.WinnerPlayer - 1] = 1.0;
        else if (forcedAction is { IsStochastic: true })
        {
            var stochastic = (CatanStochasticAction)((CoreAction.Stochastic)forcedAction).Item;
            var outcomes = stochastic.Outcomes();
            if (outcomes.Length > 0 && outcomes.All(outcome => ((CatanState)outcome.Item2).WinnerPlayer > 0))
            {
                var totalWeight = outcomes.Sum(outcome => outcome.Item1);
                foreach (var outcome in outcomes)
                {
                    var winner = ((CatanState)outcome.Item2).WinnerPlayer;
                    wins[winner - 1] += outcome.Item1 / (double)totalWeight;
                }
            }
        }
        var resolved = wins.Sum() > 0.0;
        var actions = new List<StateActionRecord>();
        if (forcedAction is not null)
        {
            var coreAction = UnwrapCoreAction(forcedAction);
            var outcomes = new List<StateActionOutcomeRecord>();
            if (coreAction is CatanStochasticAction stochastic)
            {
                var ruleOutcomes = stochastic.Outcomes();
                for (var outcomeIndex = 0; outcomeIndex < ruleOutcomes.Length; outcomeIndex++)
                {
                    var outcomeState = (CatanState)ruleOutcomes[outcomeIndex].Item2;
                    var outcomeWins = new double[state.PlayerCount];
                    if (outcomeState.WinnerPlayer > 0)
                        outcomeWins[outcomeState.WinnerPlayer - 1] = 1.0;
                    outcomes.Add(new StateActionOutcomeRecord
                    {
                        Outcome = DescribeOutcome(stochastic, outcomeState, outcomeIndex),
                        Wins = outcomeWins.Sum() > 0 ? outcomeWins : [],
                        Visits = 0,
                        WinRate = outcomeWins[state.CurrentPlayer - 1],
                        ModelPrior = null,
                        Selected = selectedOutcome == DescribeOutcome(stochastic, outcomeState, outcomeIndex),
                    });
                }
            }
            actions.Add(new StateActionRecord
            {
                Action = DescribeAction(coreAction),
                Wins = resolved ? (double[])wins.Clone() : [],
                Visits = 0,
                WinRate = resolved ? wins[state.CurrentPlayer - 1] : 0.0,
                ModelPrior = null,
                Selected = true,
                Outcomes = outcomes,
            });
        }
        return new StateRecord
        {
            PlayerTurn = state.CurrentPlayer,
            TurnNumber = state.TurnNumber,
            Stage = StateToken.EncodeTurnStage(state.Stage).ToString(),
            SerializedState = state.SerializeStateOnly(),
            Scores = state.Scores(),
            Simulations = 0,
            ElapsedMs = 0,
            WinRate = resolved ? wins[state.CurrentPlayer - 1] : 0.0,
            Wins = wins,
            ValueTarget = resolved ? (double[])wins.Clone() : null,
            Actions = actions,
            ReachedTerminal = resolved,
        };
    }

    private GameResult RunSingleGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        IPriorClient? priorClient)
        => RunGame(config, playerCount, rng, gameSeed, gameNumber, priorClient, null, null);

    private CombinedGameResult RunSingleCombinedGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        IPriorClient? placementPriorClient,
        IPriorClient? statePriorClient)
    {
        var placementStates = new List<PlacementStateRecord>();
        var game = RunGame(config, playerCount, rng, gameSeed, gameNumber, statePriorClient,
            placementPriorClient, placementStates);
        return new CombinedGameResult
        {
            Game = game,
            PlacementStates = placementStates,
            PlacementSearchTimeMs = _options.PlacementSearchTimeMs,
            MainGameSearchTimeMs = _options.MainGameSearchTimeMs,
        };
    }

    private GameResult RunGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        IPriorClient? statePriorClient,
        IPriorClient? placementPriorClient,
        List<PlacementStateRecord>? placementStates)
    {
        var state = new CatanState(config, playerCount, rng);
        var combined = placementStates is not null;
        var leafBoundary = _options.PlacementOnly || combined
            ? Microsoft.FSharp.Core.FSharpOption<Microsoft.FSharp.Core.FSharpFunc<Kjarni.ICoreState, bool>>.Some(
                Microsoft.FSharp.Core.FuncConvert.FromFunc<Kjarni.ICoreState, bool>(IsPlacementLeafBoundary))
            : null;
        Microsoft.FSharp.Core.FSharpOption<IPriorClient>? PriorOption(IPriorClient? client) =>
            client is null ? null : Microsoft.FSharp.Core.FSharpOption<IPriorClient>.Some(client);

        Kjarni.MCTSConfig CreateMctsConfig(bool placement) => new(
            searchTime.NewMilliSeconds(combined
                ? placement ? _options.PlacementSearchTimeMs : _options.MainGameSearchTimeMs
                : _options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MaxRolloutDepth,
            System.Math.Sqrt(2.0),
            _options.ActionRolloutLimit,
            PriorOption(placement ? placementPriorClient : statePriorClient),
            _options.Prior && _options.ExportType != ExportType.InitialPlacement
                ? CatanStateLeafEvaluatorPool.Get(_options.NnUrl)
                : null,
            placement ? leafBoundary : null,
            int.MaxValue,
            _options.MaxPendingEvaluations,
            _options.LeafEvaluationTimeoutMs,
            _options.DrainTimeoutMs);
        var placementPhase = combined && state.TurnNumber == 0;
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(CreateMctsConfig(placementPhase));
        var states = new List<StateRecord>();
        var evaluationDiagnostics = new EvaluationDiagnostics();

        // Capture the board serialization once (invariant across turns).
        var boardSerialized = state.SerializeBoard();

        // Total action counter guards against infinite loops from non-advancing
        // actions (e.g., cyclic bank trades) where TurnNumber never increments.
        const int maxTotalActions = 10_000;
        var totalActions = 0;
        var lastReportedTurn = -1;

        // Persistent MCTS root for tree reuse across actions.
        Kjarni.MCTS.Types.MCTSState? mctsRoot = null;
        var placementActionSerializer = combined
            ? PlacementActionSerializer.ForTopology(config.Map.Topology)
            : null;

        while (state.WinnerPlayer == 0)
        {
            var isPlacementPhase = combined && state.TurnNumber == 0;
            if (placementPhase != isPlacementPhase)
            {
                placementPhase = isPlacementPhase;
                mctsRoot = null;
                mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(CreateMctsConfig(placementPhase));
            }
            var actions = state.Actions();
            if (actions.Length == 0) break;

            if (actions.Length == 1 && isPlacementPhase)
            {
                // Placement policy training only records genuine decisions.
                state = (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();
                mctsRoot = AdvanceMctsRoot(mctsRoot, 0, (ICoreState)state);
            }
            else
            {
                // Run MCTS for both decisions and forced normal-play states so every
                // exported state has value and stochastic-outcome diagnostics.
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
                evaluationDiagnostics.Add(logInfo);

                CatanState? selectedResult = null;
                string? selectedOutcome = null;
                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
                    selectedResult = (CatanState)UnwrapCoreAction(actions[bestPath.Head]).DoCoreAction();
                    if (UnwrapCoreAction(actions[bestPath.Head]) is CatanStochasticAction stochastic)
                        selectedOutcome = DescribeOutcome(stochastic, selectedResult, outcomeIndex: 0);
                }

                states.Add(CreateStateRecord(
                    state,
                    serialized,
                    mctsRoot,
                    logInfo,
                    bestPath.IsEmpty ? -1 : bestPath.Head,
                    selectedOutcome));

                if (isPlacementPhase
                    && state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceFirstRoad
                        or TurnStage.PlaceSecondSettlement or TurnStage.PlaceSecondRoad)
                {
                    placementStates!.Add(CreatePlacementStateRecord(
                        state, actions, mctsRoot, logInfo, placementActionSerializer!, bestPath.Head));
                }

                // Apply the best action from MCTS and advance the tree.
                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
                    // When the best action is a HorizonAction, the search has reached
                    // the expansion boundary — stop the game loop.
                    if (mctsRoot.Actions[bestPath.Head].IsHorizonAction && !combined)
                        break;

                    state = selectedResult!;
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

        if (state.WinnerPlayer != 0)
            states.Add(CreateUnsearchedStateRecord(state));

        // Aggregate per-depth prior stats across all MCTS decisions.
        Dictionary<int, int>? priorActionsPerDepth = null;
        Dictionary<int, int>? priorInferencesPerDepth = null;
        foreach (var s in states)
        {
            if (s.PriorActionsPerDepth is { Count: > 0 })
            {
                priorActionsPerDepth ??= new Dictionary<int, int>();
                foreach (var (depth, count) in s.PriorActionsPerDepth)
                {
                    priorActionsPerDepth.TryGetValue(depth, out var existing);
                    priorActionsPerDepth[depth] = existing + count;
                }
            }
            if (s.PriorInferencesPerDepth is { Count: > 0 })
            {
                priorInferencesPerDepth ??= new Dictionary<int, int>();
                foreach (var (depth, count) in s.PriorInferencesPerDepth)
                {
                    priorInferencesPerDepth.TryGetValue(depth, out var existing);
                    priorInferencesPerDepth[depth] = existing + count;
                }
            }
        }

        return new GameResult
        {
            Seed = gameSeed,
            Map = _options.MapConfig ?? "standard",
            Players = playerCount,
            Winner = state.WinnerPlayer,
            Turns = state.TurnNumber,
            SearchTimeMs = combined ? _options.MainGameSearchTimeMs : _options.SearchTimeMs,
            MaxSimulations = _options.MaxSimulations,
            MaxRolloutDepth = _options.MaxRolloutDepth,
            ActionRolloutLimit = _options.ActionRolloutLimit,
            BoardSerialized = boardSerialized,
            States = states,
            PriorActionsPerDepth = priorActionsPerDepth,
            PriorInferencesPerDepth = priorInferencesPerDepth,
            EvaluationDiagnostics = evaluationDiagnostics,
        };
    }

    internal static bool IsPlacementLeafBoundary(ICoreState state) =>
        state is CatanState { TurnNumber: 1, Stage: TurnStage.PreRoll };

    private PlacementGameResult RunSinglePlacementGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        IPriorClient? priorClient)
    {
        var state = new CatanState(config, playerCount, rng);
        var leafBoundary = Microsoft.FSharp.Core.FSharpOption<Microsoft.FSharp.Core.FSharpFunc<Kjarni.ICoreState, bool>>.Some(
            Microsoft.FSharp.Core.FuncConvert.FromFunc<Kjarni.ICoreState, bool>(IsPlacementLeafBoundary));
        var priorOption = priorClient is null
            ? null
            : Microsoft.FSharp.Core.FSharpOption<IPriorClient>.Some(priorClient);

        var mctsConfig = new Kjarni.MCTSConfig(
            searchTime.NewMilliSeconds(_options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MaxRolloutDepth,
            System.Math.Sqrt(2.0),
            _options.ActionRolloutLimit,
            priorOption,
            null,
            leafBoundary,
            int.MaxValue,
            _options.MaxPendingEvaluations,
            _options.LeafEvaluationTimeoutMs,
            _options.DrainTimeoutMs);
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(mctsConfig);
        var placementStates = new List<PlacementStateRecord>();
        var evaluationDiagnostics = new EvaluationDiagnostics();

        var boardSerialized = state.SerializeBoard();
        var actionSerializer = PlacementActionSerializer.ForTopology(config.Map.Topology);

        const int maxTotalActions = 1000;
        var totalActions = 0;

        Kjarni.MCTS.Types.MCTSState? mctsRoot = null;

        while (state.WinnerPlayer == 0)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;

            var isPlacementStage = state.Stage is TurnStage.PlaceFirstSettlement
                or TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondSettlement or TurnStage.PlaceSecondRoad;

            if (actions.Length == 1)
            {
                state = (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();
                mctsRoot = AdvanceMctsRoot(mctsRoot, 0, (ICoreState)state);
            }
            else if (isPlacementStage)
            {
                mctsRoot ??= new Kjarni.MCTS.Types.MCTSState((ICoreState)state);
                mcts.RunSimulation(mctsRoot);
                var bestPath = extractBestPath(mctsRoot);
                var logInfo = mcts.LatestLogInfo();
                evaluationDiagnostics.Add(logInfo);
                placementStates.Add(CreatePlacementStateRecord(
                    state, actions, mctsRoot, logInfo, actionSerializer, bestPath.Head));

                // Apply the best action and advance.
                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
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
                evaluationDiagnostics.Add(mcts.LatestLogInfo());

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

        // Placement records stop at the first normal-play state. Continue with
        // seeded random legal play only to provide the final outcome value target.
        while (state.WinnerPlayer == 0
               && state.TurnNumber > 0
               && totalActions < maxTotalActions)
        {
            var actions = state.Actions();
            if (actions.Length == 0) break;
            var action = actions[rng.Next(actions.Length)];
            state = (CatanState)UnwrapCoreAction(action).DoCoreAction();
            totalActions++;
        }

        // Aggregate per-depth prior stats across all MCTS decisions.
        Dictionary<int, int>? priorActionsPerDepth = null;
        Dictionary<int, int>? priorInferencesPerDepth = null;
        foreach (var s in placementStates)
        {
            if (s.PriorActionsPerDepth is { Count: > 0 })
            {
                priorActionsPerDepth ??= new Dictionary<int, int>();
                foreach (var (depth, count) in s.PriorActionsPerDepth)
                {
                    priorActionsPerDepth.TryGetValue(depth, out var existing);
                    priorActionsPerDepth[depth] = existing + count;
                }
            }
            if (s.PriorInferencesPerDepth is { Count: > 0 })
            {
                priorInferencesPerDepth ??= new Dictionary<int, int>();
                foreach (var (depth, count) in s.PriorInferencesPerDepth)
                {
                    priorInferencesPerDepth.TryGetValue(depth, out var existing);
                    priorInferencesPerDepth[depth] = existing + count;
                }
            }
        }

        return new PlacementGameResult
        {
            Seed = gameSeed,
            Map = _options.MapConfig ?? "standard",
            Players = playerCount,
            Winner = state.WinnerPlayer,
            SearchTimeMs = _options.SearchTimeMs,
            MaxSimulations = _options.MaxSimulations,
            MaxRolloutDepth = _options.MaxRolloutDepth,
            ActionRolloutLimit = _options.ActionRolloutLimit,
            BoardSerialized = boardSerialized,
            States = placementStates,
            PriorActionsPerDepth = priorActionsPerDepth,
            PriorInferencesPerDepth = priorInferencesPerDepth,
            EvaluationDiagnostics = evaluationDiagnostics,
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
            var totalPriorNodesRequested = stats.Games
                .SelectMany(g => g.States)
                .Sum(s => s.PriorNodesRequested);
            if (totalPriorNodesRequested > 0)
            {
                var totalPriorActionsApplied = stats.Games
                    .SelectMany(g => g.States)
                    .Sum(s => s.PriorActionsApplied);
                var totalPriorActionsRequested = stats.Games
                    .SelectMany(g => g.States)
                    .Sum(s => s.PriorActionsRequested);
                var totalPriorInferencesRequested = stats.Games
                    .SelectMany(g => g.States)
                    .Sum(s => s.PriorInferencesRequested);
                var avgNodesRequested = (double)totalPriorNodesRequested / mctsStates;
                var avgActionsApplied = (double)totalPriorActionsApplied / mctsStates;
                var avgActionsRequested = (double)totalPriorActionsRequested / mctsStates;
                var avgInferencesRequested = (double)totalPriorInferencesRequested / mctsStates;
                Console.WriteLine($"Prior nodes requested: {totalPriorNodesRequested} (avg {avgNodesRequested:F1}/decision)");
                Console.WriteLine($"Prior actions applied: {totalPriorActionsApplied} (avg {avgActionsApplied:F1}/decision)");
                Console.WriteLine($"Prior actions requested: {totalPriorActionsRequested} (avg {avgActionsRequested:F1}/decision)");
                Console.WriteLine($"Prior inferences requested: {totalPriorInferencesRequested} (avg {avgInferencesRequested:F1}/decision)");

                // Per-depth breakdown of action states sent to the NN.
                var aggregatedDepths = new SortedDictionary<int, int>();
                foreach (var game in stats.Games)
                {
                    if (game.PriorActionsPerDepth is not { Count: > 0 }) continue;
                    foreach (var (depth, count) in game.PriorActionsPerDepth)
                    {
                        aggregatedDepths.TryGetValue(depth, out var existing);
                        aggregatedDepths[depth] = existing + count;
                    }
                }

                if (aggregatedDepths.Count > 0)
                {
                    var depthParts = aggregatedDepths.Select(kv => $"{kv.Key}:{kv.Value}");
                    Console.WriteLine($"Prior actions by depth: {string.Join(", ", depthParts)}");
                }

                var aggregatedInferenceDepths = new SortedDictionary<int, int>();
                foreach (var game in stats.Games)
                {
                    if (game.PriorInferencesPerDepth is not { Count: > 0 }) continue;
                    foreach (var (depth, count) in game.PriorInferencesPerDepth)
                    {
                        aggregatedInferenceDepths.TryGetValue(depth, out var existing);
                        aggregatedInferenceDepths[depth] = existing + count;
                    }
                }

                if (aggregatedInferenceDepths.Count > 0)
                {
                    var depthParts = aggregatedInferenceDepths.Select(kv => $"{kv.Key}:{kv.Value}");
                    Console.WriteLine($"Prior inferences by depth: {string.Join(", ", depthParts)}");
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
    internal static object BuildGameJsonObject(
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
                s.TurnNumber,
                s.Stage,
                s.SerializedState,
                s.Scores,
                s.Simulations,
                s.ElapsedMs,
                 s.WinRate,
                s.Wins,
                s.ValueTarget,
                 actions = s.Actions.Select(a => new
                 {
                     a.Action,
                     a.Wins,
                     a.Visits,
                     a.WinRate,
                     a.ModelPrior,
                     a.Selected,
                     outcomes = a.Outcomes.Select(outcome => new
                     {
                         outcome.Outcome,
                         outcome.Wins,
                         outcome.Visits,
                         outcome.WinRate,
                         outcome.ModelPrior,
                         outcome.Selected,
                     }).ToArray(),
                     permutations = symmetryPerms.Select(permutation =>
                         TransformActionDescription(a.Action, permutation)).ToArray(),
                 }).ToArray(),
                 s.ReachedTerminal,
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                 s.PriorActionsRequested,
                 s.PriorInferencesRequested,
                 priorModelInvocationsPerDepth = s.PriorInferencesPerDepth != null
                    ? s.PriorInferencesPerDepth.OrderBy(kv => kv.Key)
                        .ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
                 s.PriorNodesSkipped,
                permutations = symmetryPerms.Length > 0
                    ? symmetryPerms.Select(p => BoardSymmetry.PermuteState(s.SerializedState, p)).ToArray()
                    : Array.Empty<string>(),
            }).ToArray(),
            priorActionsPerDepth = game.PriorActionsPerDepth != null
                ? game.PriorActionsPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            priorInferencesPerDepth = game.PriorInferencesPerDepth != null
                ? game.PriorInferencesPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            priorModelInvocationsPerDepth = game.PriorInferencesPerDepth != null
                ? game.PriorInferencesPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            evaluationDiagnostics = game.EvaluationDiagnostics,
        };
    }

    /// <summary>
    /// Builds the anonymous object representing a placement game result for JSON serialization.
    /// </summary>
    internal static object BuildPlacementGameJsonObject(
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
            game.Winner,
            exportType = "initialPlacement",
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
                s.Stage,
                s.SerializedState,
                s.Simulations,
                s.ElapsedMs,
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                s.PriorActionsRequested,
                s.PriorInferencesRequested,
                priorModelInvocationsPerDepth = s.PriorInferencesPerDepth != null
                    ? s.PriorInferencesPerDepth.OrderBy(kv => kv.Key)
                        .ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
                s.PriorNodesSkipped,
                s.ModelValue,
                s.ValueTarget,
                actions = s.Actions.Select(a => new
                {
                    a.PolicyIndex,
                    a.Wins,
                    a.Visits,
                     a.WinRate,
                     a.ModelPrior,
                     a.Selected,
                     permutations = symmetryPerms.Length > 0
                        ? symmetryPerms.Select(p => TransformPlacementPolicyIndex(s, a, p, actionSerializer)).ToArray()
                        : Array.Empty<int>(),
                }).ToArray(),
                permutations = symmetryPerms.Length > 0
                    ? symmetryPerms.Select(p => BoardSymmetry.PermutePlacementState(s.SerializedState, p)).ToArray()
                    : Array.Empty<string>(),
            }).ToArray(),
            priorActionsPerDepth = game.PriorActionsPerDepth != null
                ? game.PriorActionsPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            priorInferencesPerDepth = game.PriorInferencesPerDepth != null
                ? game.PriorInferencesPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            priorModelInvocationsPerDepth = game.PriorInferencesPerDepth != null
                ? game.PriorInferencesPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            evaluationDiagnostics = game.EvaluationDiagnostics,
        };
    }

    internal static object BuildCombinedGameJsonObject(
        CombinedGameResult combined,
        ImmutableArray<SymmetryPermutation> symmetryPerms)
    {
        var game = combined.Game;
        var topology = game.Map.ToLowerInvariant() switch
        {
            "mini" or "m" => BoardTopology.Mini,
            "small" or "sm" => BoardTopology.Small,
            _ => BoardTopology.Standard,
        };
        var actionSerializer = PlacementActionSerializer.ForTopology(topology);
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
            exportType = "placementAndState",
            constraints = new
            {
                placementSearchTimeMs = combined.PlacementSearchTimeMs,
                mainGameSearchTimeMs = combined.MainGameSearchTimeMs,
                game.MaxSimulations,
                game.MaxRolloutDepth,
                game.ActionRolloutLimit,
            },
            board = new { serialized = game.BoardSerialized, permutations = boardPermutations },
            placementStates = combined.PlacementStates.Select(s => new
            {
                s.PlayerTurn,
                s.Stage,
                s.SerializedState,
                s.Simulations,
                s.ElapsedMs,
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                s.PriorActionsRequested,
                s.PriorInferencesRequested,
                priorModelInvocationsPerDepth = s.PriorInferencesPerDepth != null
                    ? s.PriorInferencesPerDepth.OrderBy(kv => kv.Key)
                        .ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
                s.PriorNodesSkipped,
                s.ModelValue,
                s.ValueTarget,
                actions = s.Actions.Select(a => new
                {
                    a.PolicyIndex,
                    a.Wins,
                    a.Visits,
                     a.WinRate,
                     a.ModelPrior,
                     a.Selected,
                     permutations = symmetryPerms.Select(p =>
                        TransformPlacementPolicyIndex(s, a, p, actionSerializer)).ToArray(),
                }).ToArray(),
                permutations = symmetryPerms.Select(p =>
                    BoardSymmetry.PermutePlacementState(s.SerializedState, p)).ToArray(),
            }).ToArray(),
            states = game.States.Select(s => new
            {
                s.PlayerTurn,
                s.TurnNumber,
                s.Stage,
                s.SerializedState,
                s.Scores,
                s.Simulations,
                s.ElapsedMs,
                 s.WinRate,
                 s.Wins,
                 s.ValueTarget,
                 actions = s.Actions.Select(a => new
                 {
                     a.Action,
                     a.Wins,
                     a.Visits,
                     a.WinRate,
                     a.ModelPrior,
                     a.Selected,
                     outcomes = a.Outcomes.Select(outcome => new
                     {
                         outcome.Outcome,
                         outcome.Wins,
                         outcome.Visits,
                         outcome.WinRate,
                         outcome.ModelPrior,
                         outcome.Selected,
                     }).ToArray(),
                     permutations = symmetryPerms.Select(permutation =>
                         TransformActionDescription(a.Action, permutation)).ToArray(),
                  }).ToArray(),
                 s.ReachedTerminal,
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                s.PriorActionsRequested,
                s.PriorInferencesRequested,
                priorModelInvocationsPerDepth = s.PriorInferencesPerDepth != null
                    ? s.PriorInferencesPerDepth.OrderBy(kv => kv.Key)
                        .ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                    : null,
                s.PriorNodesSkipped,
                permutations = symmetryPerms.Select(p =>
                    BoardSymmetry.PermuteState(s.SerializedState, p)).ToArray(),
            }).ToArray(),
            priorModelInvocationsPerDepth = game.PriorInferencesPerDepth != null
                ? game.PriorInferencesPerDepth.OrderBy(kv => kv.Key)
                    .ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            evaluationDiagnostics = game.EvaluationDiagnostics,
        };
    }

    private static int TransformPlacementPolicyIndex(
        PlacementStateRecord state,
        PlacementActionRecord action,
        SymmetryPermutation permutation,
        PlacementActionSerializer serializer)
    {
        if (state.Stage is "a" or "f")
            return permutation.Vertices[action.PolicyIndex];
        if (state.Stage is "e" or "i" && state.PendingVertex is { } vertex && action.RoadEdge is { } edge)
            return serializer.TransformDirectionIndex(vertex, edge, permutation);
        throw new InvalidOperationException($"Invalid placement export stage/action data for stage {state.Stage}.");
    }

}
