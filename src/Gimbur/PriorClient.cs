using System.Collections.Concurrent;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur;

/// <summary>
/// Mode of operation for the prior client, controlling which serialization
/// and server endpoint to use for prior requests.
/// </summary>
public enum PriorMode
{
    /// <summary>
    /// Game-state priors: serialize child states with
    /// <see cref="CatanStateSerializer.SerializeCompact"/> and call
    /// <c>/prior-enqueue</c> on the state-model server.
    /// </summary>
    State,

    /// <summary>
    /// Placement priors: serialize the parent placement state with
    /// <see cref="CatanState.SerializePlacementPhaseCompact"/> and enumerate
    /// composite (settlement, road) actions.  Calls <c>/prior-placement-enqueue</c>
    /// on the placement-model server.
    /// </summary>
    Placement,
}

/// <summary>
/// Asynchronous prior client for NN-guided MCTS search.
/// Implements <see cref="IPriorClient"/> by communicating with the Python
/// inference server via HTTP. Prior requests are fire-and-forget; completed
/// results are collected via a background polling thread that deposits
/// responses into a local mailbox.
///
/// Supports two modes:
/// <list type="bullet">
///   <item><see cref="PriorMode.State"/> — evaluates child states (standard game play).</item>
///   <item><see cref="PriorMode.Placement"/> — evaluates (parent state, composite action)
///     pairs at settlement decision points.</item>
/// </list>
/// </summary>
public sealed class PriorClient : IPriorClient, IDisposable
{
    private readonly HttpClient _http;
    private readonly ConcurrentQueue<PriorResponse> _mailbox = new();
    private readonly Thread _pollThread;
    private volatile bool _disposed;
    private readonly PriorMode _mode;
    private readonly PlacementActionSerializer? _actionSerializer;

    /// <summary>
    /// Minimum interval between server polls (milliseconds).
    /// Prevents hammering the server when the MCTS loop calls
    /// CollectPriors on every iteration.
    /// </summary>
    private const int PollIntervalMs = 20;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public PriorClient(string baseUrl, PriorMode mode = PriorMode.State, PlacementActionSerializer? actionSerializer = null)
    {
        _mode = mode;
        _actionSerializer = actionSerializer;
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };

        // Start background thread that polls the server for completed results.
        _pollThread = new Thread(PollLoop)
        {
            IsBackground = true,
            Name = "PriorClient-Poll",
        };
        _pollThread.Start();
    }

    /// <summary>
    /// Enqueue an async prior request. Dispatches to either state-based or
    /// placement-based serialization depending on the configured <see cref="PriorMode"/>.
    /// </summary>
    public void RequestPrior(long nodeId, ICoreState parentState, ICoreState[] states, int actingPlayer, int depth)
    {
        if (_mode == PriorMode.Placement)
        {
            RequestPlacementPrior(nodeId, parentState, states, actingPlayer, depth);
        }
        else
        {
            RequestStatePrior(nodeId, states, actingPlayer, depth);
        }
    }

    /// <summary>
    /// State-mode prior: serializes each child state via
    /// <see cref="CatanStateSerializer.SerializeCompact"/> and POSTs to /prior-enqueue.
    /// </summary>
    private void RequestStatePrior(long nodeId, ICoreState[] states, int actingPlayer, int depth)
    {
        var serialized = new string[states.Length];
        for (int i = 0; i < states.Length; i++)
        {
            serialized[i] = CatanStateSerializer.SerializeCompact((CatanState)states[i]);
        }

        var request = new PriorEnqueuePayload
        {
            Requests =
            [
                new PriorRequestItem
                {
                    Id = nodeId.ToString(),
                    States = serialized,
                    Player = actingPlayer,
                    Priority = depth,
                }
            ],
        };

        _ = Task.Run(async () =>
        {
            try
            {
                await _http.PostAsJsonAsync("prior-enqueue", request, JsonOptions);
            }
            catch
            {
                // Server unreachable — degrade gracefully (no priors for this node).
            }
        });
    }

    /// <summary>
    /// Placement-mode prior: at settlement decision points, serializes the parent
    /// placement state and enumerates all composite (settlement, road) actions from
    /// child states. For each child (road-stage state), discovers the legal roads
    /// and constructs composite action strings. Sends all (state, action) pairs to
    /// <c>/prior-placement-enqueue</c>. The response win probabilities are aggregated
    /// per settlement (max across roads) before being returned.
    ///
    /// At road stages and non-placement stages, no prior is requested.
    /// </summary>
    private void RequestPlacementPrior(long nodeId, ICoreState parentState, ICoreState[] states, int actingPlayer, int depth)
    {
        var parent = (CatanState)parentState;

        // Only provide priors at settlement decision points.
        if (parent.Stage is not (TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement))
        {
            return;
        }

        if (_actionSerializer is null)
        {
            return;
        }

        var placementState = parent.SerializePlacementPhaseCompact();

        // For each child state (road-stage state after placing settlement),
        // enumerate legal road actions and build composite action strings.
        // Track which composite actions belong to which settlement child (by index).
        var allStates = new List<string>();
        var allActions = new List<string>();
        var childBoundaries = new List<int>(); // start index in allActions for each child

        for (int ci = 0; ci < states.Length; ci++)
        {
            childBoundaries.Add(allActions.Count);
            var childState = (CatanState)states[ci];

            // Child should be in a road stage with a pending settlement vertex.
            if (childState.Stage is not (TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad))
            {
                continue;
            }

            if (childState.PendingSettlementVertex is not { } vertex)
            {
                continue;
            }

            var roadActions = childState.Actions();
            foreach (var roadCoreAction in roadActions)
            {
                CatanAction roadAction;
                if (roadCoreAction.IsDeterministic)
                    roadAction = (CatanDeterministicAction)((CoreAction.Deterministic)roadCoreAction).Item;
                else if (roadCoreAction.IsStochastic)
                    roadAction = (CatanStochasticAction)((CoreAction.Stochastic)roadCoreAction).Item;
                else
                    continue;

                if (roadAction is not PlaceRoadAction placeRoad)
                    continue;

                var actionString = _actionSerializer.Serialize(vertex, placeRoad.EdgeIndex);
                allStates.Add(placementState);
                allActions.Add(actionString);
            }
        }

        if (allActions.Count == 0)
        {
            return;
        }

        // Add a sentinel to simplify boundary calculation.
        childBoundaries.Add(allActions.Count);

        var request = new PlacementPriorEnqueuePayload
        {
            Requests =
            [
                new PlacementPriorRequestItem
                {
                    Id = nodeId.ToString(),
                    States = allStates.ToArray(),
                    Actions = allActions.ToArray(),
                    ChildBoundaries = childBoundaries.ToArray(),
                    Priority = depth,
                }
            ],
        };

        _ = Task.Run(async () =>
        {
            try
            {
                await _http.PostAsJsonAsync("prior-placement-enqueue", request, JsonOptions);
            }
            catch
            {
                // Server unreachable — degrade gracefully (no priors for this node).
            }
        });
    }

    /// <summary>
    /// Return all completed prior responses currently in the local mailbox.
    /// The mailbox is fed by the background polling thread; this method
    /// never makes HTTP calls itself and returns immediately.
    /// </summary>
    public PriorResponse[] CollectPriors()
    {
        var results = new List<PriorResponse>();
        while (_mailbox.TryDequeue(out var item))
        {
            results.Add(item);
        }
        return results.ToArray();
    }

    /// <summary>
    /// Clear the server queue and discard pending results.
    /// </summary>
    public void Flush()
    {
        try
        {
            var endpoint = _mode == PriorMode.Placement ? "prior-placement-flush" : "prior-flush";
            _http.PostAsync(endpoint, null).GetAwaiter().GetResult();
        }
        catch
        {
            // Server unreachable — nothing to flush.
        }

        // Clear local mailbox.
        while (_mailbox.TryDequeue(out _)) { }
    }

    public void Dispose()
    {
        _disposed = true;
        _pollThread.Join(timeout: TimeSpan.FromSeconds(2));
        _http.Dispose();
    }

    // ── Background polling ───────────────────────────────────────────────────

    private void PollLoop()
    {
        var endpoint = _mode == PriorMode.Placement ? "prior-placement-collect" : "prior-collect";

        while (!_disposed)
        {
            try
            {
                var response = _http.PostAsync(endpoint, null).GetAwaiter().GetResult();
                if (response.IsSuccessStatusCode)
                {
                    var result = response.Content
                        .ReadFromJsonAsync<PriorCollectPayload>(JsonOptions)
                        .GetAwaiter()
                        .GetResult();

                    if (result?.Responses != null)
                    {
                        foreach (var r in result.Responses)
                        {
                            if (long.TryParse(r.Id, out var nodeId))
                            {
                                var winProbs = new double[r.WinProbabilities.Length];
                                for (int i = 0; i < r.WinProbabilities.Length; i++)
                                    winProbs[i] = r.WinProbabilities[i];
                                _mailbox.Enqueue(new PriorResponse(nodeId, winProbs));
                            }
                        }
                    }
                }
            }
            catch
            {
                // Server unreachable — will retry on next poll.
            }

            Thread.Sleep(PollIntervalMs);
        }
    }

    // ── JSON payload types ───────────────────────────────────────────────────

    // State-mode payloads
    private sealed class PriorEnqueuePayload
    {
        public PriorRequestItem[] Requests { get; init; } = [];
    }

    private sealed class PriorRequestItem
    {
        public string Id { get; init; } = "";
        public string[] States { get; init; } = [];
        public int Player { get; init; }
        public int Priority { get; init; }
    }

    // Placement-mode payloads
    private sealed class PlacementPriorEnqueuePayload
    {
        public PlacementPriorRequestItem[] Requests { get; init; } = [];
    }

    private sealed class PlacementPriorRequestItem
    {
        public string Id { get; init; } = "";
        public string[] States { get; init; } = [];
        public string[] Actions { get; init; } = [];

        /// <summary>
        /// Boundary indices mapping composite actions back to settlement children.
        /// <c>ChildBoundaries[i]</c> is the start index of actions for child <c>i</c>;
        /// <c>ChildBoundaries[i+1]</c> is the exclusive end (sentinel at the end).
        /// </summary>
        [JsonPropertyName("child_boundaries")]
        public int[] ChildBoundaries { get; init; } = [];
        public int Priority { get; init; }
    }

    // Shared response payloads (same format for both modes)
    private sealed class PriorCollectPayload
    {
        public PriorCollectItem[] Responses { get; init; } = [];
    }

    private sealed class PriorCollectItem
    {
        public string Id { get; init; } = "";

        [JsonPropertyName("win_probabilities")]
        public float[] WinProbabilities { get; init; } = [];
    }
}
