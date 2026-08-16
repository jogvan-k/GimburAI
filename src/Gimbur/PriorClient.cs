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
/// results are collected via a background polling thread that routes responses
/// by globally unique node ID.
///
/// Uses the complete parent-state policy/value model for every game phase.
/// </summary>
public sealed class PriorClient : IPriorClient, IDisposable
{
    private readonly HttpClient _http;
    private readonly ConcurrentDictionary<long, PriorResponse> _completed = new();
    private readonly Thread _pollThread;
    private readonly Thread _senderThread;
    private readonly BlockingCollection<PriorRequestItem> _enqueueQueue = new(
        new ConcurrentQueue<PriorRequestItem>(), EnqueueQueueCapacity);
    private readonly CancellationTokenSource _shutdown = new();
    private volatile bool _disposed;
    private readonly PriorMode _mode;
    private readonly ConcurrentDictionary<long, PendingStateRequest> _pendingState = new();

    /// <summary>
    /// When true, this client is owned by a pool and shared across many
    /// concurrent MCTS searches (typically one per HTTP request to
    /// Gimbur.Server).
    /// </summary>
    /// <summary>
    // ── Diagnostic counters (thread-safe via Interlocked) ────────────────
    private long _pollSuccessCount;
    private long _pollResponsesReceived;
    private long _pollErrorCount;
    private long _pollEmptyCount;
    private long _enqueueFireCount;
    private long _enqueueErrorCount;

    /// <summary>
    private const int EnqueueQueueCapacity = 16384;
    private const int EnqueueBatchSize = 64;
    private const int EnqueueBatchWindowMs = 1;

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
        : this(baseUrl, new HttpClientHandler(), mode, pooled)
    {
    }

    internal PriorClient(
        string baseUrl,
        HttpMessageHandler handler,
        PriorMode mode = PriorMode.State,
        bool pooled = false)
    {
        _mode = mode;
        _http = new HttpClient(handler) { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };

        // Start background thread that polls the server for completed results.
        _pollThread = new Thread(PollLoop)
        {
            IsBackground = true,
            Name = pooled ? "PriorClient-Poll-Pooled" : "PriorClient-Poll",
        };
        _pollThread.Start();
        _senderThread = new Thread(SenderLoop)
        {
            IsBackground = true,
            Name = pooled ? "PriorClient-Send-Pooled" : "PriorClient-Send",
        };
        _senderThread.Start();
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

        var request = new PriorRequestItem
        {
            Id = nodeId.ToString(),
            ParentState = CatanStateSerializer.SerializeCompact((CatanState)parentState),
            LegalPolicyIndices = legalIndices,
            Priority = depth,
        };

        _pendingState[nodeId] = new PendingStateRequest(
            legalIndices, outcomeCounts, parent.NumberOfPlayers, parent.CurrentPlayer,
            serializer.PolicySize);
        if (!_enqueueQueue.TryAdd(request))
        {
            _pendingState.TryRemove(nodeId, out _);
            Interlocked.Increment(ref _enqueueErrorCount);
            return 0;
        }
        Interlocked.Increment(ref _enqueueFireCount);

        return 1;
    }

    private void SenderLoop()
    {
        var token = _shutdown.Token;
        while (!token.IsCancellationRequested)
        {
            try
            {
                if (!_enqueueQueue.TryTake(out var first, 5, token))
                    continue;
                if (token.WaitHandle.WaitOne(EnqueueBatchWindowMs))
                    break;
                var batch = new List<PriorRequestItem>(EnqueueBatchSize) { first };
                while (batch.Count < EnqueueBatchSize && _enqueueQueue.TryTake(out var next))
                    batch.Add(next);
                SendEnqueueBatch(batch, token);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (InvalidOperationException) when (_enqueueQueue.IsCompleted)
            {
                break;
            }
        }
    }

    private void SendEnqueueBatch(List<PriorRequestItem> batch, CancellationToken token)
    {
        try
        {
            using var response = _http.PostAsJsonAsync(
                "state/prior-enqueue",
                new PriorEnqueuePayload { Requests = batch.ToArray() },
                JsonOptions,
                token).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
        }
        catch (OperationCanceledException) when (_disposed)
        {
        }
        catch
        {
            Interlocked.Add(ref _enqueueErrorCount, batch.Count);
            foreach (var request in batch)
                if (long.TryParse(request.Id, out var nodeId))
                    _pendingState.TryRemove(nodeId, out _);
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
    /// Responses for unknown node IDs remain keyed for their owning search.
    /// </summary>
    public PriorResponse[] CollectPriors(IReadOnlySet<long> knownNodeIds)
    {
        return CollectMatching(_completed, knownNodeIds);
    }

    internal static PriorResponse[] CollectMatching(
        ConcurrentDictionary<long, PriorResponse> completed,
        IReadOnlySet<long> knownNodeIds)
    {
        var results = new List<PriorResponse>(knownNodeIds.Count);
        foreach (var nodeId in knownNodeIds)
        {
            if (completed.TryRemove(nodeId, out var response))
                results.Add(response);
        }
        return results.ToArray();
    }

    /// <summary>
    /// Drop pending responses belonging to a completed search. Responses for
    /// unknown node IDs remain available to other concurrent searches.
    ///
    /// Never clears the server-side queue: that queue is shared across
    /// all concurrent callers and clearing it would discard pending
    /// requests/responses owned by other searches.
    /// </summary>
    public void Flush(IReadOnlySet<long> knownNodeIds)
    {
        foreach (var nodeId in knownNodeIds)
        {
            _pendingState.TryRemove(nodeId, out _);
            _completed.TryRemove(nodeId, out _);
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _enqueueQueue.CompleteAdding();
        _shutdown.Cancel();
        _senderThread.Join(timeout: TimeSpan.FromSeconds(2));
        _pollThread.Join(timeout: TimeSpan.FromSeconds(2));

        _http.Dispose();
        _enqueueQueue.Dispose();
        _shutdown.Dispose();
    }

    // ── Background polling ───────────────────────────────────────────────────

    private void PollLoop()
    {
        const string endpoint = "state/prior-collect";
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
                                        priors, pendingState.OutcomeCounts);
                                    var densePriors = ToDensePolicy(
                                        priors, pendingState.LegalPolicyIndices,
                                        pendingState.PolicySize);
                                    _completed[nodeId] = new PriorResponse(
                                        nodeId, flattened, valueEstimates, densePriors);
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

            Thread.Sleep(PollIntervalMs);
        }
    }

    internal static double[] MapFullStatePolicy(
        IReadOnlyList<double> policy,
        IReadOnlyList<int> outcomeCounts)
    {
        var actionPriors = policy.Count == outcomeCounts.Count
            ? Normalize(policy)
            : Enumerable.Repeat(1.0 / outcomeCounts.Count, outcomeCounts.Count).ToArray();
        var flattened = new double[outcomeCounts.Sum()];
        var offset = 0;
        for (var actionIndex = 0; actionIndex < actionPriors.Length; actionIndex++)
        {
            Array.Fill(flattened, actionPriors[actionIndex], offset, outcomeCounts[actionIndex]);
            offset += outcomeCounts[actionIndex];
        }
        return flattened;
    }

    private static double[] Normalize(IReadOnlyList<double> policy)
    {
        var result = policy.Select(value => double.IsFinite(value) && value >= 0 ? value : 0).ToArray();
        var total = result.Sum();
        if (total > 0 && double.IsFinite(total))
        {
            for (var i = 0; i < result.Length; i++)
                result[i] /= total;
        }
        else if (result.Length > 0)
        {
            Array.Fill(result, 1.0 / result.Length);
        }
        return result;
    }

    private static double[] ToDensePolicy(
        IReadOnlyList<double> policy,
        IReadOnlyList<int> legalPolicyIndices,
        int policySize)
    {
        var dense = new double[policySize];
        for (var i = 0; i < policy.Count && i < legalPolicyIndices.Count; i++)
            dense[legalPolicyIndices[i]] += policy[i];
        return dense;
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

        [JsonPropertyName("legal_policy_indices")]
        public int[] LegalPolicyIndices { get; init; } = [];

        public int Priority { get; init; }
    }

    private sealed record PendingStateRequest(
        int[] LegalPolicyIndices,
        int[] OutcomeCounts,
        int PlayerCount,
        int ActingPlayer,
        int PolicySize);

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
