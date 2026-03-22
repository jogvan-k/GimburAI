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
    // 1. Select config preset.
    var config = (req.Config?.ToLowerInvariant()) switch
    {
        "mini" => GameConfig.Mini,
        "small" => GameConfig.Small,
        _ => GameConfig.Standard,
    };

    // 2. Deserialize the game state.
    CatanState state;
    try
    {
        state = CatanState.DeserializeHumanReadable(config, req.PlayerCount, req.State);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = "Failed to deserialize state", detail = ex.Message });
    }

    // 3. Get legal actions.
    var actions = state.Actions();
    if (actions.Length == 0)
        return Results.BadRequest(new { error = "No legal actions available (game may be over)" });

    // 4. If only one action, return it immediately (forced move).
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

    // 5. Configure and run MCTS.
    var searchTimeMs = req.SearchTimeMs ?? 1000;
    var maxRolloutDepth = req.MaxRolloutDepth ?? 500;

    var mctsConfig = new MCTSConfig(
        searchTime.NewMilliSeconds(searchTimeMs),
        maxSimulations: int.MaxValue,
        maxRolloutDepth: maxRolloutDepth,
        explorationConstant: Math.Sqrt(2.0),
        actionRolloutLimit: int.MaxValue,
        priorClient: null);

    var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(mctsConfig);
    var mctsRoot = new Kjarni.MCTS.Types.MCTSState((ICoreState)state);

    var sw = Stopwatch.StartNew();
    mcts.RunSimulation(mctsRoot);
    sw.Stop();

    // 6. Extract best action.
    var bestPath = extractBestPath(mctsRoot);
    if (bestPath.IsEmpty)
        return Results.Problem("MCTS returned no result");

    var bestActionIndex = bestPath.Head;
    var bestAction = UnwrapCoreAction(actions[bestActionIndex]);
    var playerIndex = (int)state.PlayerTurn;

    // 7. Gather per-action statistics.
    var allActions = new List<ActionInfo>();
    for (var i = 0; i < actions.Length; i++)
    {
        var catanAction = UnwrapCoreAction(actions[i]);
        var (_, winRate, rollouts) = GetChildWinData(mctsRoot.Actions[i], playerIndex);
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
        AllActions = allActions,
    });
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

/// <summary>
/// Extracts win data from an MCTS child action node.
/// Copied from SimulationRunner — handles deterministic, stochastic, and terminal actions.
/// </summary>
static (double[] Wins, double WinRate, int Rollouts) GetChildWinData(
    Kjarni.MCTS.Types.Action childAction, int playerIndex)
{
    if (childAction.IsTerminal)
    {
        var outcome = ((Kjarni.MCTS.Types.Action.Terminal)childAction).Item;
        var wins = (double[])outcome.Clone();
        var wr = playerIndex < wins.Length ? wins[playerIndex] : 0.0;
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

    return (Array.Empty<double>(), 0.0, 0);
}

// --- Request/Response types ---

record ChooseActionRequest
{
    public string? Config { get; init; }
    public int PlayerCount { get; init; }
    public required string State { get; init; }
    public int? SearchTimeMs { get; init; }
    public int? MaxRolloutDepth { get; init; }
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
