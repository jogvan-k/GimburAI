using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Gimbur;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Benchmark player that delegates action selection to a Gimbur.Server instance
/// via the <c>/choose-action</c> HTTP endpoint. Supports both pure MCTS
/// (<c>mcts-ai</c>) and NN-guided MCTS (<c>mcts-nn-ai</c>) server modes.
/// </summary>
internal sealed class ServerPlayer : IBenchmarkPlayer, IPriorStatsProvider, IDisposable
{
    private readonly HttpClient _http;
    private readonly string _aiMode;
    private readonly string _mapConfig;
    private readonly int _playerCount;
    private readonly int _searchTimeMs;
    private readonly int _maxRolloutDepth;
    private readonly string? _nnUrl;
    private readonly string? _priorMode;
    private readonly int? _maxPriorDepth;

    public int TotalNnRequests { get; private set; }
    public int TotalNnStatesEvaluated { get; private set; }
    public int TotalPriorActionsApplied { get; private set; }
    public int TotalPriorActionsRequested { get; private set; }
    public int TotalPriorInferencesRequested => TotalNnStatesEvaluated;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <param name="serverUrl">Base URL of the Gimbur.Server (e.g. http://localhost:5123).</param>
    /// <param name="aiMode">"mcts-ai" or "mcts-nn-ai".</param>
    /// <param name="mapConfig">Map config name: "mini", "small", "standard".</param>
    /// <param name="playerCount">Number of players in the game.</param>
    /// <param name="searchTimeMs">MCTS search time limit in ms.</param>
    /// <param name="maxRolloutDepth">Maximum MCTS rollout depth.</param>
    /// <param name="nnUrl">NN inference server URL (required for mcts-nn-ai).</param>
    /// <param name="priorMode">Prior mode: "state" or "placement" (for mcts-nn-ai).</param>
    /// <param name="maxPriorDepth">Max prior depth (for mcts-nn-ai).</param>
    public ServerPlayer(
        string serverUrl,
        string aiMode,
        string mapConfig,
        int playerCount,
        int searchTimeMs,
        int maxRolloutDepth = 500,
        string? nnUrl = null,
        string? priorMode = null,
        int? maxPriorDepth = null)
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(serverUrl.TrimEnd('/') + "/"),
            Timeout = TimeSpan.FromSeconds(searchTimeMs / 1000.0 + 30), // generous timeout
        };
        _aiMode = aiMode;
        _mapConfig = mapConfig;
        _playerCount = playerCount;
        _searchTimeMs = searchTimeMs;
        _maxRolloutDepth = maxRolloutDepth;
        _nnUrl = nnUrl;
        _priorMode = priorMode;
        _maxPriorDepth = maxPriorDepth;
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var actions = state.Actions();
        if (actions.Length == 0) return null;

        // For forced moves, skip the server call entirely.
        if (actions.Length == 1)
        {
            return (CatanState)UnwrapCoreAction(actions[0]).DoCoreAction();
        }

        // Serialize state and call the server.
        var serialized = state.SerializeHumanReadable();

        var request = new ServerChooseActionRequest
        {
            AiMode = _aiMode,
            Config = _mapConfig,
            PlayerCount = _playerCount,
            State = serialized,
            SearchTimeMs = _searchTimeMs,
            MaxRolloutDepth = _maxRolloutDepth,
            NnUrl = _nnUrl,
            PriorMode = _priorMode,
            MaxPriorDepth = _maxPriorDepth,
        };

        var response = _http.PostAsJsonAsync("choose-action", request, JsonOptions)
            .GetAwaiter().GetResult();

        if (!response.IsSuccessStatusCode)
        {
            var body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            throw new InvalidOperationException(
                $"Server returned {(int)response.StatusCode}: {body}");
        }

        var result = response.Content
            .ReadFromJsonAsync<ServerChooseActionResponse>(JsonOptions)
            .GetAwaiter().GetResult();

        if (result is null)
            throw new InvalidOperationException("Server returned null response");

        TotalNnRequests += result.PriorNodesRequested;
        TotalNnStatesEvaluated += result.PriorInferencesRequested;
        TotalPriorActionsApplied += result.PriorActionsApplied;
        TotalPriorActionsRequested += result.PriorActionsRequested;

        // Find the matching action by (typeTag, arg1, arg2).
        for (var i = 0; i < actions.Length; i++)
        {
            var catanAction = UnwrapCoreAction(actions[i]);
            if (catanAction.TypeTag == result.TypeTag &&
                catanAction.Arg1 == result.Arg1 &&
                catanAction.Arg2 == result.Arg2)
            {
                return (CatanState)catanAction.DoCoreAction();
            }
        }

        // Build diagnostic: list all available actions.
        var available = new System.Text.StringBuilder();
        for (var i = 0; i < actions.Length; i++)
        {
            var a = UnwrapCoreAction(actions[i]);
            available.Append($"(tag={a.TypeTag}, arg1={a.Arg1}, arg2={a.Arg2}) ");
        }

        throw new InvalidOperationException(
            $"Server chose action (tag={result.TypeTag}, arg1={result.Arg1}, arg2={result.Arg2}) " +
            $"that does not match any of {actions.Length} legal actions: {available}" +
            $"Stage={state.Stage}, Player={state.CurrentPlayer}, Turn={state.TurnNumber}");
    }

    public void Dispose()
    {
        _http.Dispose();
    }

    private static CatanAction UnwrapCoreAction(CoreAction coreAction)
    {
        if (coreAction.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
        if (coreAction.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
    }

    // ── JSON types for server communication ──────────────────────────────

    private sealed class ServerChooseActionRequest
    {
        public string? AiMode { get; init; }
        public string? Config { get; init; }
        public int PlayerCount { get; init; }
        public required string State { get; init; }
        public int? SearchTimeMs { get; init; }
        public int? MaxRolloutDepth { get; init; }
        public int? MaxPriorDepth { get; init; }
        public string? NnUrl { get; init; }
        public string? PriorMode { get; init; }
    }

    private sealed class ServerChooseActionResponse
    {
        public byte TypeTag { get; init; }
        public int Arg1 { get; init; }
        public int Arg2 { get; init; }
        public string ActionName { get; init; } = "";
        public int Visits { get; init; }
        public double WinRate { get; init; }
        public int TotalSimulations { get; init; }
        public int ElapsedMs { get; init; }
        public int PriorNodesRequested { get; init; }
        public int PriorActionsApplied { get; init; }
        public int PriorActionsRequested { get; init; }
        public int PriorInferencesRequested { get; init; }
    }
}
