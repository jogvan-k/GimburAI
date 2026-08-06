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
}

/// <summary>
/// Asynchronous prior client for NN-guided MCTS search.
/// Implements <see cref="IPriorClient"/> by communicating with the Python
/// inference server via HTTP. Prior requests are fire-and-forget; completed
/// results are collected via a background polling thread that deposits
/// responses into a local mailbox.
///
/// Uses the complete parent-state policy/value model for every game phase.
/// </summary>
public sealed class PriorClient : IPriorClient, IDisposable
{
    private readonly HttpClient _http;
    private readonly ConcurrentQueue<PriorResponse> _mailbox = new();
    private readonly Thread _pollThread;
    private volatile bool _disposed;
    private readonly PriorMode _mode;
    private readonly ConcurrentDictionary<long, PendingStateRequest> _pendingState = new();

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

    public PriorClient(string baseUrl, PriorMode mode = PriorMode.State, bool pooled = false)
    {
        _mode = mode;
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
    /// All Catan decision stages use the same complete policy vocabulary.
    /// </summary>
    public bool ShouldRequestPrior(ICoreState parentState)
    {
        return parentState is CatanState;
    }

    /// <summary>
    /// Enqueue an async prior request. Dispatches to either state-based or
    /// placement-based serialization depending on the configured <see cref="PriorMode"/>.
    /// Returns the number of (state, action) inference pairs sent to the model.
    /// </summary>
    public int RequestPrior(long nodeId, ICoreState parentState, ICoreState[] states, int actingPlayer, int depth)
    {
        return RequestStatePrior(nodeId, parentState, states, actingPlayer, depth);
    }

    /// <summary>
    /// State-mode prior: sends one canonical parent state and maps the returned
    /// complete policy into the flattened Kjarni action/outcome layout.
    /// </summary>
    private int RequestStatePrior(
        long nodeId,
        ICoreState parentState,
        ICoreState[] states,
        int actingPlayer,
        int depth)
    {
        var parent = (CatanState)parentState;
        var actions = parent.Actions().Select(UnwrapAction).ToArray();
        var serializer = new CatanPolicySerializer(parent.Board.Topology, parent.PlayerCount);
        var legalIndices = actions.Select(action => serializer.IndexOf(parent, action)).ToArray();
        var outcomeCounts = actions.Select(action => action is CatanStochasticAction stochastic
            ? stochastic.Outcomes().Length
            : 1).ToArray();
        if (outcomeCounts.Sum() != states.Length)
            return 0;

        var request = new PriorEnqueuePayload
        {
            Requests =
            [
                new PriorRequestItem
                {
                    Id = nodeId.ToString(),
                    ParentState = CatanStateSerializer.SerializeCompact((CatanState)parentState),
                    Priority = depth,
                }
            ],
        };

        Interlocked.Increment(ref _enqueueFireCount);
        _pendingState[nodeId] = new PendingStateRequest(
            legalIndices, outcomeCounts, parent.NumberOfPlayers, parent.CurrentPlayer, serializer);
        _ = EnqueueAsync(
            "state/prior-enqueue",
            request,
            () => _pendingState.TryRemove(nodeId, out _));

        return 1;
    }

    private async Task EnqueueAsync<T>(
        string endpoint,
        T request,
        Action? onFailure)
    {
        await _enqueueThrottle.WaitAsync();
        try
        {
            using var response = await _http.PostAsJsonAsync(endpoint, request, JsonOptions);
            response.EnsureSuccessStatusCode();
        }
        catch
        {
            onFailure?.Invoke();
            Interlocked.Increment(ref _enqueueErrorCount);
        }
        finally
        {
            _enqueueThrottle.Release();
        }
    }

    private static double[] NormalizePlayerValues(float[]? values, int playerCount)
    {
        if (values is null || values.Length != playerCount)
            return [];

        var normalized = new double[values.Length];
        var total = 0.0;
        for (var i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]) || values[i] < 0)
                return [];
            normalized[i] = values[i];
            total += values[i];
        }

        if (!double.IsFinite(total) || total <= 0)
            return [];

        for (var i = 0; i < normalized.Length; i++)
            normalized[i] /= total;
        return normalized;
    }

    internal static double[] RestoreAbsolutePlayerOrder(
        IReadOnlyList<double> canonicalValues,
        int actingPlayer)
    {
        if (canonicalValues.Count == 0)
            return [];
        var restored = new double[canonicalValues.Count];
        for (var canonicalIndex = 0; canonicalIndex < canonicalValues.Count; canonicalIndex++)
        {
            var absoluteIndex = (canonicalIndex + actingPlayer - 1) % canonicalValues.Count;
            restored[absoluteIndex] = canonicalValues[canonicalIndex];
        }
        return restored;
    }

    private static CatanAction UnwrapAction(CoreAction action)
    {
        if (action.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)action).Item;
        if (action.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)action).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {action.Tag}");
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

        foreach (var nodeId in knownNodeIds)
        {
            _pendingState.TryRemove(nodeId, out _);
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
        const string endpoint = "state/prior-collect";
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
                                var priors = Array.ConvertAll(r.Priors, value => (double)value);
                                if (_pendingState.TryRemove(nodeId, out var pendingState))
                                {
                                    var valueEstimates = NormalizePlayerValues(
                                        r.PlayerWinProbabilities, pendingState.PlayerCount);
                                    valueEstimates = RestoreAbsolutePlayerOrder(
                                        valueEstimates, pendingState.ActingPlayer);
                                    var flattened = MapFullStatePolicy(
                                        pendingState.Serializer, priors,
                                        pendingState.LegalPolicyIndices, pendingState.OutcomeCounts);
                                    _mailbox.Enqueue(new PriorResponse(
                                        nodeId, flattened, valueEstimates, priors));
                                }
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

    internal static double[] MapFullStatePolicy(
        CatanPolicySerializer serializer,
        IReadOnlyList<double> policy,
        IReadOnlyList<int> legalPolicyIndices,
        IReadOnlyList<int> outcomeCounts)
    {
        if (legalPolicyIndices.Count != outcomeCounts.Count)
            throw new ArgumentException("Each legal action must have an outcome count.", nameof(outcomeCounts));

        var actionPriors = serializer.MaskAndNormalize(policy, legalPolicyIndices);
        var flattened = new double[outcomeCounts.Sum()];
        var offset = 0;
        for (var actionIndex = 0; actionIndex < actionPriors.Length; actionIndex++)
        {
            Array.Fill(flattened, actionPriors[actionIndex], offset, outcomeCounts[actionIndex]);
            offset += outcomeCounts[actionIndex];
        }
        return flattened;
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

        [JsonPropertyName("parent_state")]
        public string ParentState { get; init; } = "";

        public int Priority { get; init; }
    }

    private sealed record PendingStateRequest(
        int[] LegalPolicyIndices,
        int[] OutcomeCounts,
        int PlayerCount,
        int ActingPlayer,
        CatanPolicySerializer Serializer);

    // Shared response payloads (same format for both modes)
    private sealed class PriorCollectPayload
    {
        public PriorCollectItem[] Responses { get; init; } = [];
    }

    private sealed class PriorCollectItem
    {
        public string Id { get; init; } = "";

        [JsonPropertyName("priors")]
        public float[] Priors { get; init; } = [];

        [JsonPropertyName("player_win_probabilities")]
        public float[]? PlayerWinProbabilities { get; init; }

    }
}
