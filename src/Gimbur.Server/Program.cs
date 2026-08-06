using System.Diagnostics;
using Gimbur;
using Gimbur.Rules;
using Kjarni;
using static Kjarni.MCTS.Algorithm;

var builder = WebApplication.CreateBuilder(args);

// Disable default ASP.NET request logging for cleaner output.
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.SetMinimumLevel(LogLevel.Warning);

var app = builder.Build();

app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

app.MapPost("/choose-action", (ChooseActionRequest req) =>
{
    // 1. Validate AI mode.
    var aiMode = req.AiMode?.ToLowerInvariant() ?? "mcts-ai";
    if (aiMode is not ("mcts-ai" or "mcts-nn-ai"))
        return Results.BadRequest(new { error = $"Unknown aiMode: '{req.AiMode}'. Supported: mcts-ai, mcts-nn-ai" });

    if (aiMode == "mcts-nn-ai" && string.IsNullOrWhiteSpace(req.NnUrl))
        return Results.BadRequest(new { error = "nnUrl is required when aiMode is 'mcts-nn-ai'" });

    if (req.PriorMode is not null && !req.PriorMode.Equals("state", StringComparison.OrdinalIgnoreCase))
        return Results.BadRequest(new { error = $"Unknown priorMode: '{req.PriorMode}'. Supported: state" });

    // 2. Select config preset.
    var config = (req.Config?.ToLowerInvariant()) switch
    {
        "mini" => GameConfig.Mini,
        "small" => GameConfig.Small,
        _ => GameConfig.Standard,
    };

    // 3. Deserialize the game state.
    CatanState state;
    try
    {
        state = CatanState.DeserializeHumanReadable(config, req.PlayerCount, req.State);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = "Failed to deserialize state", detail = ex.Message });
    }

    // 4. Get legal actions.
    var actions = state.Actions();
    if (actions.Length == 0)
        return Results.BadRequest(new { error = "No legal actions available (game may be over)" });

    // 5. If only one action, return it immediately (forced move).
    if (actions.Length == 1)
    {
        var forced = UnwrapCoreAction(actions[0]);
        return Results.Ok(new ChooseActionResponse
        {
            TypeTag = forced.TypeTag,
            Arg1 = forced.Arg1,
            Arg2 = forced.Arg2,
            ActionName = ActionName(forced.TypeTag),
            Visits = 0,
            WinRate = 0.0,
            TotalSimulations = 0,
            ElapsedMs = 0,
            AllActions =
            [
                new ActionInfo
                {
                    TypeTag = forced.TypeTag,
                    Arg1 = forced.Arg1,
                    Arg2 = forced.Arg2,
                    ActionName = ActionName(forced.TypeTag),
                    Visits = 0,
                    WinRate = 0.0,
                },
            ],
        });
    }

    // 6. Configure MCTS with optional NN prior client.
    var searchTimeMs = req.SearchTimeMs ?? 1000;
    var maxRolloutDepth = req.MaxRolloutDepth ?? 500;
    var maxPriorDepth = req.MaxPriorDepth ?? int.MaxValue;

    PriorClient? priorClient = null;
    if (aiMode == "mcts-nn-ai")
    {
        // Use a process-wide pool so the background poll thread stays warm
        // across HTTP requests; per-request PriorClients were being created
        // and disposed before any prior responses could be collected, leaving
        // server-mcts-nn effectively running without NN guidance.
        priorClient = PriorClientPool.Get(req.NnUrl!);
    }

    try
    {
        var mctsConfig = new MCTSConfig(
            searchTime.NewMilliSeconds(searchTimeMs),
            maxSimulations: int.MaxValue,
            maxRolloutDepth: maxRolloutDepth,
            explorationConstant: Math.Sqrt(2.0),
            actionRolloutLimit: int.MaxValue,
            priorClient: priorClient,
            leafEvaluator: aiMode == "mcts-nn-ai"
                ? CatanStateLeafEvaluatorPool.Get(req.NnUrl!)
                : null,
            leafBoundary: null,
            maxPriorDepth: maxPriorDepth,
            maxPendingEvaluations: 32,
            leafEvaluationTimeoutMs: 500,
            drainTimeoutMs: 1000);

        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(mctsConfig);
        var mctsRoot = new Kjarni.MCTS.Types.MCTSState((ICoreState)state);

        var sw = Stopwatch.StartNew();
        mcts.RunSimulation(mctsRoot);
        sw.Stop();

        // 7. Extract best action.
        var bestPath = extractBestPath(mctsRoot);
        if (bestPath.IsEmpty)
            return Results.Problem("MCTS returned no result");

        var bestActionIndex = bestPath.Head;
        var bestAction = UnwrapCoreAction(actions[bestActionIndex]);
        var playerIndex = (int)state.PlayerTurn;

        // 8. Gather per-action statistics.
        var allActions = new List<ActionInfo>();
        for (var i = 0; i < actions.Length; i++)
        {
            var catanAction = UnwrapCoreAction(actions[i]);
            var stats = mctsRoot.ActionStats[i];
            var rollouts = stats.CompletedVisits;
            var winRate = rollouts > 0 && playerIndex < stats.ValueSums.Length
                ? stats.ValueSums[playerIndex] / rollouts
                : 0.0;
            allActions.Add(new ActionInfo
            {
                TypeTag = catanAction.TypeTag,
                Arg1 = catanAction.Arg1,
                Arg2 = catanAction.Arg2,
                ActionName = ActionName(catanAction.TypeTag),
                Visits = rollouts,
                WinRate = Math.Round(winRate, 4),
            });
        }

        // Sort by visits descending for readability.
        allActions.Sort((a, b) => b.Visits.CompareTo(a.Visits));

        var logInfo = mcts.LatestLogInfo();

        return Results.Ok(new ChooseActionResponse
        {
            TypeTag = bestAction.TypeTag,
            Arg1 = bestAction.Arg1,
            Arg2 = bestAction.Arg2,
            ActionName = ActionName(bestAction.TypeTag),
            Visits = allActions.First(a =>
                a.TypeTag == bestAction.TypeTag &&
                a.Arg1 == bestAction.Arg1 &&
                a.Arg2 == bestAction.Arg2).Visits,
            WinRate = allActions.First(a =>
                a.TypeTag == bestAction.TypeTag &&
                a.Arg1 == bestAction.Arg1 &&
                a.Arg2 == bestAction.Arg2).WinRate,
            TotalSimulations = logInfo.simulations,
            ElapsedMs = (int)sw.ElapsedMilliseconds,
            PriorNodesRequested = logInfo.priorNodesRequested,
            PriorActionsApplied = logInfo.priorActionsApplied,
            PriorActionsRequested = logInfo.priorActionsRequested,
            PriorInferencesRequested = logInfo.priorInferencesRequested,
            LeafEvaluationBatches = logInfo.leafEvaluationBatches,
            LeafEvaluationStates = logInfo.leafEvaluationStates,
            AllActions = allActions,
        });
    }
    finally
    {
        // Pooled PriorClients are owned by PriorClientPool and must NOT be
        // disposed here.
    }
});

app.Run();

// --- Helper methods ---

static CatanAction UnwrapCoreAction(CoreAction coreAction)
{
    if (coreAction.IsDeterministic)
        return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
    if (coreAction.IsStochastic)
        return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
    throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
}

static string ActionName(byte tag) => tag switch
{
    0 => "PlaceSettlement",
    1 => "PlaceRoad",
    2 => "RollDice",
    3 => "ChooseRobberTile",
    4 => "BuildCity",
    5 => "BankTrade",
    6 => "BuyDevCard",
    7 => "PlayKnight",
    8 => "PlayRoadBuilding",
    9 => "PlayMonopoly",
    10 => "PlayYearOfPlenty",
    11 => "EndTurn",
    12 => "ChooseRobberVictim",
    _ => $"Unknown({tag})",
};

// --- Request/Response types ---

record ChooseActionRequest
{
    /// <summary>
    /// AI mode: "mcts-ai" (pure MCTS, default) or "mcts-nn-ai" (MCTS with NN prior guidance).
    /// </summary>
    public string? AiMode { get; init; }

    /// <summary>
    /// Game config preset: "mini", "small", or "standard" (default).
    /// </summary>
    public string? Config { get; init; }

    /// <summary>Number of players in the game.</summary>
    public int PlayerCount { get; init; }

    /// <summary>Human-readable serialized game state (11 pipe-delimited sections).</summary>
    public required string State { get; init; }

    /// <summary>MCTS search time limit in milliseconds (default: 1000).</summary>
    public int? SearchTimeMs { get; init; }

    /// <summary>Maximum rollout depth (default: 500).</summary>
    public int? MaxRolloutDepth { get; init; }

    /// <summary>Maximum tree depth for NN prior requests (default: unlimited).</summary>
    public int? MaxPriorDepth { get; init; }

    /// <summary>
    /// URL of the Python NN inference server. Required when aiMode is "mcts-nn-ai".
    /// </summary>
    public string? NnUrl { get; init; }

    /// <summary>Optional compatibility field. Only "state" is supported.</summary>
    public string? PriorMode { get; init; }

}

record ChooseActionResponse
{
    public byte TypeTag { get; init; }
    public int Arg1 { get; init; }
    public int Arg2 { get; init; }
    public required string ActionName { get; init; }
    public int Visits { get; init; }
    public double WinRate { get; init; }
    public int TotalSimulations { get; init; }
    public int ElapsedMs { get; init; }
    public int PriorNodesRequested { get; init; }
    public int PriorActionsApplied { get; init; }
    public int PriorActionsRequested { get; init; }
    public int PriorInferencesRequested { get; init; }
    public int LeafEvaluationBatches { get; init; }
    public int LeafEvaluationStates { get; init; }
    public required List<ActionInfo> AllActions { get; init; }
}

record ActionInfo
{
    public byte TypeTag { get; init; }
    public int Arg1 { get; init; }
    public int Arg2 { get; init; }
    public required string ActionName { get; init; }
    public int Visits { get; init; }
    public double WinRate { get; init; }
}
