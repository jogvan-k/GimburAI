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
    /// <c>/state/prior-enqueue</c> on the inference server.
    /// </summary>
    State,

    /// <summary>
    /// Placement priors: serialize the parent placement state with
    /// <see cref="CatanState.SerializePlacementPhaseCompact"/> and enumerate
    /// composite (settlement, road) actions.  Calls <c>/placement/prior-enqueue</c>
    /// on the inference server.
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
    /// When true, this client is owned by a pool and shared across many
    /// concurrent MCTS searches (typically one per HTTP request to
    /// Gimbur.Server). Enables the orphan soft cap in <see cref="CollectPriors"/>
    /// since a pooled client's mailbox can otherwise grow unboundedly
    /// from stale responses owned by completed searches.
    /// </summary>
    private readonly bool _pooled;

    /// <summary>
    /// Soft cap on the local mailbox size. When the mailbox grows beyond this,
    /// <see cref="CollectPriors"/> opportunistically drops the oldest stale
    /// entries (responses for NodeIds not in the active known set) to bound
    /// memory growth from orphan responses left by completed searches.
    /// Only relevant for pooled clients.
    /// </summary>
    private const int MailboxSoftCap = 4096;

    // ── Diagnostic counters (thread-safe via Interlocked) ────────────────
    private long _pollSuccessCount;
    private long _pollResponsesReceived;
    private long _pollErrorCount;
    private long _pollEmptyCount;
    private long _enqueueFireCount;
    private long _enqueueErrorCount;

    /// <summary>
    /// Limits the number of concurrent fire-and-forget HTTP enqueue requests
    /// to prevent exhausting file descriptors when the MCTS engine expands
    /// many nodes in parallel.
    /// </summary>
    private readonly SemaphoreSlim _enqueueThrottle = new(MaxConcurrentEnqueues);

    /// <summary>
    /// Maximum number of concurrent in-flight enqueue HTTP requests.
    /// </summary>
    private const int MaxConcurrentEnqueues = 20;

    /// <summary>
    /// Minimum interval between server polls (milliseconds).
    /// Prevents hammering the server when the MCTS loop calls
    /// CollectPriors on every iteration.
    /// </summary>
    private const int PollIntervalMs = 5;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public PriorClient(string baseUrl, PriorMode mode = PriorMode.State, PlacementActionSerializer? actionSerializer = null, bool pooled = false)
    {
        _mode = mode;
        _actionSerializer = actionSerializer;
        _pooled = pooled;
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };

        // Start background thread that polls the server for completed results.
        _pollThread = new Thread(PollLoop)
        {
            IsBackground = true,
            Name = pooled ? "PriorClient-Poll-Pooled" : "PriorClient-Poll",
        };
        _pollThread.Start();
    }

    /// <summary>
    /// Fast pre-check: returns <c>true</c> when the client can produce a
    /// meaningful prior for a node whose parent is in the given state.
    /// In <see cref="PriorMode.Placement"/>, only settlement decision points
    /// are eligible; in <see cref="PriorMode.State"/>, all states are eligible.
    /// </summary>
    public bool ShouldRequestPrior(ICoreState parentState)
    {
        if (_mode != PriorMode.Placement)
        {
            return true;
        }

        var stage = ((CatanState)parentState).Stage;
        return stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement;
    }

    /// <summary>
    /// Enqueue an async prior request. Dispatches to either state-based or
    /// placement-based serialization depending on the configured <see cref="PriorMode"/>.
    /// Returns the number of (state, action) inference pairs sent to the model.
    /// </summary>
    public int RequestPrior(long nodeId, ICoreState parentState, ICoreState[] states, int actingPlayer, int depth)
    {
        if (_mode == PriorMode.Placement)
        {
            return RequestPlacementPrior(nodeId, parentState, states, actingPlayer, depth);
        }
        else
        {
            return RequestStatePrior(nodeId, states, actingPlayer, depth);
        }
    }

    /// <summary>
    /// State-mode prior: serializes each child state via
    /// <see cref="CatanStateSerializer.SerializeCompact"/> and POSTs to /state/prior-enqueue.
    /// Returns the number of inference pairs sent (= states.Length).
    /// </summary>
    private int RequestStatePrior(long nodeId, ICoreState[] states, int actingPlayer, int depth)
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

        Interlocked.Increment(ref _enqueueFireCount);
        _ = Task.Run(async () =>
        {
            await _enqueueThrottle.WaitAsync();
            try
            {
                using var response = await _http.PostAsJsonAsync("state/prior-enqueue", request, JsonOptions);
            }
            catch
            {
                // Server unreachable — degrade gracefully (no priors for this node).
                Interlocked.Increment(ref _enqueueErrorCount);
            }
            finally
            {
                _enqueueThrottle.Release();
            }
        });

        return states.Length;
    }

    /// <summary>
    /// Placement-mode prior: at settlement decision points, serializes the parent
    /// placement state and enumerates all composite (settlement, road) actions from
    /// child states. For each child (road-stage state), discovers the legal roads
    /// and constructs composite action strings. Sends all (state, action) pairs to
    /// <c>/placement/prior-enqueue</c>. The response win probabilities are aggregated
    /// per settlement (max across roads) before being returned.
    ///
    /// At road stages and non-placement stages, no prior is requested.
    /// Returns the number of (state, action) inference pairs sent to the model
    /// (= total composite (settlement, road) pairs across all child states), or
    /// 0 when the request was declined.
    /// </summary>
    private int RequestPlacementPrior(long nodeId, ICoreState parentState, ICoreState[] states, int actingPlayer, int depth)
    {
        var parent = (CatanState)parentState;

        // Only provide priors at settlement decision points.
        if (parent.Stage is not (TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement))
        {
            return 0;
        }

        if (_actionSerializer is null)
        {
            return 0;
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
            return 0;
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

        Interlocked.Increment(ref _enqueueFireCount);
        _ = Task.Run(async () =>
        {
            await _enqueueThrottle.WaitAsync();
            try
            {
                using var response = await _http.PostAsJsonAsync("placement/prior-enqueue", request, JsonOptions);
            }
            catch
            {
                // Server unreachable — degrade gracefully (no priors for this node).
                Interlocked.Increment(ref _enqueueErrorCount);
            }
            finally
            {
                _enqueueThrottle.Release();
            }
        });

        return allActions.Count;
    }

    /// <summary>
    /// Return completed prior responses whose NodeId is in the given set.
    /// Responses for unknown node IDs are left in the mailbox so that other
    /// concurrent callers (e.g. parallel games sharing this client) can
    /// collect their own responses on a subsequent call.
    /// </summary>
    public PriorResponse[] CollectPriors(IReadOnlySet<long> knownNodeIds)
    {
        var results = new List<PriorResponse>();
        var putBack = new List<PriorResponse>();
        while (_mailbox.TryDequeue(out var item))
        {
            if (knownNodeIds.Contains(item.NodeId))
                results.Add(item);
            else
                putBack.Add(item);
        }

        // Re-enqueue responses that didn't match any known node ID, but bound
        // total mailbox size to prevent unbounded growth from orphan responses
        // left by completed pooled-client searches.
        if (_pooled && putBack.Count > MailboxSoftCap)
        {
            // Drop the oldest (front of the list, which corresponds to earlier
            // dequeue order) and keep only the most recent MailboxSoftCap orphans.
            putBack.RemoveRange(0, putBack.Count - MailboxSoftCap);
        }

        foreach (var item in putBack)
            _mailbox.Enqueue(item);
        return results.ToArray();
    }

    /// <summary>
    /// Drop pending responses belonging to a completed search, identified
    /// by the node IDs the caller still tracks. Responses for unknown
    /// node IDs (which may belong to other concurrent searches sharing
    /// this client) are preserved in the mailbox.
    ///
    /// Never clears the server-side queue: that queue is shared across
    /// all concurrent callers and clearing it would discard pending
    /// requests/responses owned by other searches.
    /// </summary>
    public void Flush(IReadOnlySet<long> knownNodeIds)
    {
        var keep = new List<PriorResponse>();
        while (_mailbox.TryDequeue(out var item))
        {
            if (!knownNodeIds.Contains(item.NodeId))
            {
                keep.Add(item);
            }
        }

        foreach (var item in keep)
        {
            _mailbox.Enqueue(item);
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _pollThread.Join(timeout: TimeSpan.FromSeconds(2));

        // Print diagnostic summary to help debug prior delivery issues.
        Console.WriteLine(
            $"[PriorClient] mode={_mode} | " +
            $"enqueued={Interlocked.Read(ref _enqueueFireCount)} " +
            $"enqueueErrors={Interlocked.Read(ref _enqueueErrorCount)} | " +
            $"pollSuccess={Interlocked.Read(ref _pollSuccessCount)} " +
            $"pollEmpty={Interlocked.Read(ref _pollEmptyCount)} " +
            $"pollErrors={Interlocked.Read(ref _pollErrorCount)} | " +
            $"responsesReceived={Interlocked.Read(ref _pollResponsesReceived)} " +
            $"mailboxSize={_mailbox.Count}");

        _http.Dispose();
        _enqueueThrottle.Dispose();
    }

    // ── Background polling ───────────────────────────────────────────────────

    private void PollLoop()
    {
        var endpoint = _mode == PriorMode.Placement ? "placement/prior-collect" : "state/prior-collect";
        var lastDiagnosticAt = DateTime.UtcNow;
        var diagnosticInterval = TimeSpan.FromSeconds(30);

        while (!_disposed)
        {
            try
            {
                using var response = _http.PostAsync(endpoint, null).GetAwaiter().GetResult();
                if (response.IsSuccessStatusCode)
                {
                    var result = response.Content
                        .ReadFromJsonAsync<PriorCollectPayload>(JsonOptions)
                        .GetAwaiter()
                        .GetResult();

                    Interlocked.Increment(ref _pollSuccessCount);

                    if (result?.Responses != null && result.Responses.Length > 0)
                    {
                        foreach (var r in result.Responses)
                        {
                            if (long.TryParse(r.Id, out var nodeId))
                            {
                                var winProbs = new double[r.WinProbabilities.Length];
                                for (int i = 0; i < r.WinProbabilities.Length; i++)
                                    winProbs[i] = r.WinProbabilities[i];
                                _mailbox.Enqueue(new PriorResponse(nodeId, winProbs));
                                Interlocked.Increment(ref _pollResponsesReceived);
                            }
                        }
                    }
                    else
                    {
                        Interlocked.Increment(ref _pollEmptyCount);
                    }
                }
            }
            catch
            {
                // Server unreachable — will retry on next poll.
                Interlocked.Increment(ref _pollErrorCount);
            }

            // Periodic diagnostic dump for pooled clients (which are never disposed
            // and so never reach the Dispose() summary). Helps verify the pool is
            // actually delivering priors to active MCTS searches.
            if (_pooled && DateTime.UtcNow - lastDiagnosticAt >= diagnosticInterval)
            {
                lastDiagnosticAt = DateTime.UtcNow;
                Console.WriteLine(
                    $"[PriorClient pooled] mode={_mode} | " +
                    $"enqueued={Interlocked.Read(ref _enqueueFireCount)} " +
                    $"enqueueErrors={Interlocked.Read(ref _enqueueErrorCount)} | " +
                    $"pollSuccess={Interlocked.Read(ref _pollSuccessCount)} " +
                    $"pollEmpty={Interlocked.Read(ref _pollEmptyCount)} " +
                    $"pollErrors={Interlocked.Read(ref _pollErrorCount)} | " +
                    $"responsesReceived={Interlocked.Read(ref _pollResponsesReceived)} " +
                    $"mailboxSize={_mailbox.Count}");
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
