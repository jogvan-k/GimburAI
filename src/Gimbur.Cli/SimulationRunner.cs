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
    /// <summary>Export placement policy roots and full-game states from one game.</summary>
    PlacementAndState,
}

internal static class SimulationRouting
{
    public static PriorMode PriorModeFor(ExportType exportType, bool placementPhase) =>
        exportType == ExportType.InitialPlacement
        || exportType == ExportType.PlacementAndState && placementPhase
            ? PriorMode.Placement
            : PriorMode.State;
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

    /// <summary>
    /// NN probability for this settlement-road composite. Null unless both the
    /// settlement marginal and conditional road policy were applied.
    /// </summary>
    public double? ModelPrior { get; init; }
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
    /// All candidate composite actions with per-action MCTS statistics.
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
    public int SimulationsPerAction { get; init; }
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
        PriorClient? priorClient = null;
        PriorClient? placementPriorClient = null;
        if (_options.Prior)
        {
            var priorMode = SimulationRouting.PriorModeFor(_options.ExportType, placementPhase: true);
            PlacementActionSerializer? priorActionSerializer = null;
            if (priorMode == PriorMode.Placement)
            {
                var priorConfig = ResolveGameConfig();
                priorActionSerializer = PlacementActionSerializer.ForTopology(priorConfig.Map.Topology);
            }
            priorClient = new PriorClient(_options.NnUrl, priorMode, priorActionSerializer);
            if (isCombinedExport)
            {
                placementPriorClient = priorClient;
                // The state model is a value-only leaf evaluator. Submitting an
                // additional child-state prior request duplicates inference and
                // can starve the shared leaf queue.
                priorClient = null;
            }
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
            priorClient?.Dispose();
            if (!ReferenceEquals(placementPriorClient, priorClient))
                placementPriorClient?.Dispose();
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

    private static double[]? ComputePlacementValueTarget(IReadOnlyList<PlacementActionRecord> actions)
    {
        var totalRollouts = actions.Sum(action => action.Rollouts);
        if (totalRollouts == 0)
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
            target[player] /= totalRollouts;
        return target;
    }

    private static PlacementStateRecord CreatePlacementStateRecord(
        CatanState state,
        CoreAction[] actions,
        Kjarni.MCTS.Types.MCTSState mctsRoot,
        Kjarni.MCTS.Types.LogInfo logInfo,
        PlacementActionSerializer actionSerializer)
    {
        var playerIndex = (int)state.PlayerTurn;
        var compositeActions = new List<PlacementActionRecord>();
        var densePriors = mctsRoot.DensePriors;
        for (var settlementIndex = 0; settlementIndex < actions.Length; settlementIndex++)
        {
            if (UnwrapCoreAction(actions[settlementIndex]) is not PlaceSettlementAction settlement)
                continue;
            Kjarni.MCTS.Types.MCTSState? settlementNode = null;
            var treeAction = mctsRoot.Actions[settlementIndex];
            if (treeAction.IsDeterministicAction)
                settlementNode = ((Kjarni.MCTS.Types.Action.DeterministicAction)treeAction).Item;
            else if (treeAction.IsHorizonAction)
                settlementNode = ((Kjarni.MCTS.Types.Action.HorizonAction)treeAction).Item;

            var roadState = (CatanState)settlement.DoCoreAction();
            var roadActions = roadState.Actions();
            for (var roadIndex = 0; roadIndex < roadActions.Length; roadIndex++)
            {
                if (UnwrapCoreAction(roadActions[roadIndex]) is not PlaceRoadAction road)
                    continue;
                var (wins, winRate, rollouts) = settlementNode is not null
                    && roadIndex < settlementNode.Actions.Length
                    ? GetActionWinData(settlementNode, roadIndex, playerIndex)
                    : (Array.Empty<double>(), 0.0, 0);
                var denseIndex = actionSerializer.IndexOf(settlement.VertexIndex, road.EdgeIndex);
                compositeActions.Add(new PlacementActionRecord
                {
                    Action = actionSerializer.Serialize(settlement.VertexIndex, road.EdgeIndex),
                    Vertex = settlement.VertexIndex,
                    Edge = road.EdgeIndex,
                    Wins = wins,
                    Rollouts = rollouts,
                    WinRate = winRate,
                    ModelPrior = densePriors is { } policy && denseIndex < policy.Value.Length
                        ? policy.Value[denseIndex]
                        : null,
                });
            }
        }

        return new PlacementStateRecord
        {
            PlayerTurn = state.CurrentPlayer,
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
            Actions = compositeActions,
            ModelValue = mctsRoot.ValueEstimates is { } values ? (double[])values.Value.Clone() : null,
            ValueTarget = ComputePlacementValueTarget(compositeActions),
        };
    }

    private static StateRecord CreateStateRecord(
        CatanState state,
        string serialized,
        Kjarni.MCTS.Types.MCTSState mctsRoot,
        Kjarni.MCTS.Types.LogInfo logInfo)
    {
        var winCounts = mctsRoot.WinCounts is { Length: > 0 }
            ? (double[])mctsRoot.WinCounts.Clone()
            : Array.Empty<double>();
        var playerIndex = (int)state.PlayerTurn;
        var winRate = mctsRoot.Rollouts > 0 && playerIndex < winCounts.Length
            ? winCounts[playerIndex] / mctsRoot.Rollouts
            : 0.0;

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

    private GameResult RunSingleGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        PriorClient? priorClient)
        => RunGame(config, playerCount, rng, gameSeed, gameNumber, priorClient, null, null);

    private CombinedGameResult RunSingleCombinedGame(
        GameConfig config,
        int playerCount,
        Random rng,
        int gameSeed,
        int gameNumber,
        PriorClient? placementPriorClient,
        PriorClient? statePriorClient)
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
        PriorClient? statePriorClient,
        PriorClient? placementPriorClient,
        List<PlacementStateRecord>? placementStates)
    {
        var state = new CatanState(config, playerCount, rng);
        var combined = placementStates is not null;
        var leafBoundary = _options.PlacementOnly || combined
            ? Microsoft.FSharp.Core.FSharpOption<Microsoft.FSharp.Core.FSharpFunc<Kjarni.ICoreState, bool>>.Some(
                Microsoft.FSharp.Core.FuncConvert.FromFunc<Kjarni.ICoreState, bool>(IsPlacementLeafBoundary))
            : null;

        Kjarni.MCTSConfig CreateMctsConfig(bool placement) => new(
            searchTime.NewMilliSeconds(combined
                ? placement ? _options.PlacementSearchTimeMs : _options.MainGameSearchTimeMs
                : _options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MaxRolloutDepth,
            System.Math.Sqrt(2.0),
            _options.ActionRolloutLimit,
            placement ? placementPriorClient : combined ? null : statePriorClient,
            _options.Prior && _options.ExportType != ExportType.InitialPlacement
                ? CatanStateLeafEvaluatorPool.Get(_options.NnUrl)
                : null,
            placement ? leafBoundary : null,
            _options.MaxPriorDepth,
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

            if (actions.Length == 1)
            {
                // A forced transition cannot benefit from search and used to consume
                // the full per-decision MCTS budget. It is also not a useful policy
                // training root because no action choice exists.
                state = (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();
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
                evaluationDiagnostics.Add(logInfo);

                states.Add(CreateStateRecord(state, serialized, mctsRoot, logInfo));

                if (isPlacementPhase
                    && state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
                {
                    placementStates!.Add(CreatePlacementStateRecord(
                        state, actions, mctsRoot, logInfo, placementActionSerializer!));
                }

                // Apply the best action from MCTS and advance the tree.
                if (!bestPath.IsEmpty && bestPath.Head < actions.Length)
                {
                    // When the best action is a HorizonAction, the search has reached
                    // the expansion boundary — stop the game loop.
                    if (mctsRoot.Actions[bestPath.Head].IsHorizonAction && !combined)
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
        PriorClient? priorClient)
    {
        var state = new CatanState(config, playerCount, rng);
        var leafBoundary = Microsoft.FSharp.Core.FSharpOption<Microsoft.FSharp.Core.FSharpFunc<Kjarni.ICoreState, bool>>.Some(
            Microsoft.FSharp.Core.FuncConvert.FromFunc<Kjarni.ICoreState, bool>(IsPlacementLeafBoundary));

        var mctsConfig = new Kjarni.MCTSConfig(
            searchTime.NewMilliSeconds(_options.SearchTimeMs),
            _options.MaxSimulations,
            _options.MaxRolloutDepth,
            System.Math.Sqrt(2.0),
            _options.ActionRolloutLimit,
            priorClient,
            null,
            leafBoundary,
            _options.MaxPriorDepth,
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
                var totalPriorNodesRequested = 0;
                var totalPriorActionsApplied = 0;
                var totalPriorActionsRequested = 0;
                var totalPriorInferencesRequested = 0;
                var totalPriorNodesSkipped = 0;
                var totalPriorResponsesOrphaned = 0;
                Dictionary<int, int>? totalPriorActionsPerDepth = null;
                Dictionary<int, int>? totalPriorInferencesPerDepth = null;

                // Create a per-action MCTS config: simulation-limited, no time limit.
                var perActionMctsConfig = new Kjarni.MCTSConfig(
                    searchTime.Unlimited,
                    _options.SimulationsPerAction,
                    _options.MaxRolloutDepth,
                    System.Math.Sqrt(2.0),
                    _options.ActionRolloutLimit,
                    priorClient,
                    null,
                    leafBoundary,
                    _options.MaxPriorDepth,
                    _options.MaxPendingEvaluations,
                    _options.LeafEvaluationTimeoutMs,
                    _options.DrainTimeoutMs);
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

                        // Accumulate prior stats from this per-action MCTS run.
                        var actionLogInfo = perActionMcts.LatestLogInfo();
                        evaluationDiagnostics.Add(actionLogInfo);
                        totalPriorNodesRequested += actionLogInfo.priorNodesRequested;
                        totalPriorActionsApplied += actionLogInfo.priorActionsApplied;
                        totalPriorActionsRequested += actionLogInfo.priorActionsRequested;
                        totalPriorInferencesRequested += actionLogInfo.priorInferencesRequested;
                        totalPriorNodesSkipped += actionLogInfo.priorNodesSkipped;
                        totalPriorResponsesOrphaned += actionLogInfo.priorResponsesOrphaned;
                        if (actionLogInfo.priorActionsPerDepth is { Count: > 0 })
                        {
                            totalPriorActionsPerDepth ??= new Dictionary<int, int>();
                            foreach (var (depth, count) in actionLogInfo.priorActionsPerDepth)
                            {
                                totalPriorActionsPerDepth.TryGetValue(depth, out var existing);
                                totalPriorActionsPerDepth[depth] = existing + count;
                            }
                        }
                        if (actionLogInfo.priorInferencesPerDepth is { Count: > 0 })
                        {
                            totalPriorInferencesPerDepth ??= new Dictionary<int, int>();
                            foreach (var (depth, count) in actionLogInfo.priorInferencesPerDepth)
                            {
                                totalPriorInferencesPerDepth.TryGetValue(depth, out var existing);
                                totalPriorInferencesPerDepth[depth] = existing + count;
                            }
                        }

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
                            ModelPrior = null,
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
                    PriorNodesRequested = totalPriorNodesRequested,
                    PriorActionsApplied = totalPriorActionsApplied,
                    PriorActionsRequested = totalPriorActionsRequested,
                    PriorInferencesRequested = totalPriorInferencesRequested,
                    PriorResponsesOrphaned = totalPriorResponsesOrphaned,
                    PriorActionsPerDepth = totalPriorActionsPerDepth,
                    PriorInferencesPerDepth = totalPriorInferencesPerDepth,
                    PriorNodesSkipped = totalPriorNodesSkipped,
                    Actions = compositeActions,
                    ModelValue = null,
                    ValueTarget = ComputePlacementValueTarget(compositeActions),
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
                evaluationDiagnostics.Add(logInfo);

                var playerIndex = (int)state.PlayerTurn;
                var stageChar = StateToken.EncodeTurnStage(state.Stage).ToString();
                var serializedState = state.SerializePlacementPhase();

                // Build composite actions by iterating settlement children and their road grandchildren.
                var compositeActions = new List<PlacementActionRecord>();
                var densePriors = mctsRoot.DensePriors;
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

                    // C# legality is authoritative even when this settlement is unexplored.
                    var roadState = (CatanState)coreSettlement.DoCoreAction();
                    var roadCoreActions = roadState.Actions();

                    for (var ri = 0; ri < roadCoreActions.Length; ri++)
                    {
                        var coreRoad = UnwrapCoreAction(roadCoreActions[ri]);
                        if (coreRoad is not PlaceRoadAction pra) continue;
                        var edge = pra.EdgeIndex;

                        var (wins, winRate, rollouts) = childMctsState is not null
                            && ri < childMctsState.Actions.Length
                            ? GetActionWinData(childMctsState, ri, playerIndex)
                            : (Array.Empty<double>(), 0.0, 0);
                        var denseIndex = actionSerializer.IndexOf(vertex, edge);
                        double? modelPrior = densePriors is { } modelPolicy
                            && denseIndex < modelPolicy.Value.Length
                            ? modelPolicy.Value[denseIndex]
                            : null;
                        var actionString = actionSerializer.Serialize(vertex, edge);

                        compositeActions.Add(new PlacementActionRecord
                        {
                            Action = actionString,
                            Vertex = vertex,
                            Edge = edge,
                            Wins = wins,
                            Rollouts = rollouts,
                            WinRate = winRate,
                            ModelPrior = modelPrior,
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
                    PriorNodesRequested = logInfo.priorNodesRequested,
                    PriorActionsApplied = logInfo.priorActionsApplied,
                    PriorActionsRequested = logInfo.priorActionsRequested,
                    PriorInferencesRequested = logInfo.priorInferencesRequested,
                    PriorResponsesOrphaned = logInfo.priorResponsesOrphaned,
                    PriorActionsPerDepth = logInfo.priorActionsPerDepth is { Count: > 0 }
                        ? new Dictionary<int, int>(logInfo.priorActionsPerDepth)
                        : null,
                    PriorInferencesPerDepth = logInfo.priorInferencesPerDepth is { Count: > 0 }
                        ? new Dictionary<int, int>(logInfo.priorInferencesPerDepth)
                        : null,
                    PriorNodesSkipped = logInfo.priorNodesSkipped,
                    Actions = compositeActions,
                    ModelValue = mctsRoot.ValueEstimates is { } modelValues
                        ? (double[])modelValues.Value.Clone()
                        : null,
                    ValueTarget = ComputePlacementValueTarget(compositeActions),
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
            SimulationsPerAction = _options.SimulationsPerAction,
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
                s.ReachedTerminal,
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                s.PriorActionsRequested,
                s.PriorInferencesRequested,
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
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                s.PriorActionsRequested,
                s.PriorInferencesRequested,
                s.PriorNodesSkipped,
                s.ModelValue,
                s.ValueTarget,
                actions = s.Actions.Select(a => new
                {
                    a.Action,
                    a.Wins,
                    a.Rollouts,
                    a.WinRate,
                    a.ModelPrior,
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
            priorActionsPerDepth = game.PriorActionsPerDepth != null
                ? game.PriorActionsPerDepth.OrderBy(kv => kv.Key).ToDictionary(kv => kv.Key.ToString(), kv => kv.Value)
                : null,
            priorInferencesPerDepth = game.PriorInferencesPerDepth != null
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
                s.PriorNodesSkipped,
                s.ModelValue,
                s.ValueTarget,
                actions = s.Actions.Select(a => new
                {
                    a.Action,
                    a.Wins,
                    a.Rollouts,
                    a.WinRate,
                    a.ModelPrior,
                    permutations = symmetryPerms.Select(p =>
                        actionSerializer.Serialize(p.Vertices[a.Vertex], p.Edges[a.Edge])).ToArray(),
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
                s.ReachedTerminal,
                s.PriorNodesRequested,
                s.PriorActionsApplied,
                s.PriorActionsRequested,
                s.PriorInferencesRequested,
                s.PriorNodesSkipped,
                permutations = symmetryPerms.Select(p =>
                    BoardSymmetry.PermuteState(s.SerializedState, p)).ToArray(),
            }).ToArray(),
            evaluationDiagnostics = game.EvaluationDiagnostics,
        };
    }
}
