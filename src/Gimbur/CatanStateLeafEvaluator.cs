using System.Collections.Concurrent;
using System.Net.Http.Json;
using System.Text.Json;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur;

/// <summary>Shared asynchronous evaluator for Catan state-model value heads.</summary>
public sealed class CatanStateLeafEvaluator : ILeafEvaluator, IDisposable
{
    private const int DefaultQueueCapacity = 4096;
    private const int DefaultBatchSize = 64;
    private const int DefaultBatchWindowMs = 2;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    private readonly HttpClient _http;
    private readonly ConcurrentQueue<LeafEvaluationResponse> _mailbox = new();
    private readonly object _mailboxLock = new();
    private readonly ConcurrentDictionary<long, long> _submittedAt = new();
    private readonly BlockingCollection<QueuedLeafRequest> _pending;
    private readonly BlockingCollection<long[]> _pendingCancellations = new();
    private readonly AutoResetEvent _completed = new(false);
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Thread _senderThread;
    private readonly Thread _pollThread;
    private readonly int _batchSize;
    private readonly int _batchWindowMs;
    private int _disposeStarted;
    private volatile bool _disposed;

    public CatanStateLeafEvaluator(string baseUrl)
        : this(baseUrl, new HttpClientHandler())
    {
    }

    internal CatanStateLeafEvaluator(
        string baseUrl,
        HttpMessageHandler handler,
        int batchWindowMs = DefaultBatchWindowMs,
        int queueCapacity = DefaultQueueCapacity,
        int batchSize = DefaultBatchSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(batchWindowMs);
        ArgumentOutOfRangeException.ThrowIfLessThan(queueCapacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);

        _http = new HttpClient(handler) { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };
        _pending = new BlockingCollection<QueuedLeafRequest>(
            new ConcurrentQueue<QueuedLeafRequest>(), queueCapacity);
        _batchSize = batchSize;
        _batchWindowMs = batchWindowMs;
        _senderThread = new Thread(SenderLoop) { IsBackground = true, Name = "CatanLeafEvaluator-Send" };
        _pollThread = new Thread(PollLoop) { IsBackground = true, Name = "CatanLeafEvaluator-Poll" };
        _senderThread.Start();
        _pollThread.Start();
    }

    public bool Enqueue(long requestId, ICoreState[] states, int priority)
    {
        if (states.Length == 0 || _disposed)
            return false;

        var request = new QueuedLeafRequest
        {
            RequestId = requestId,
            Item = new LeafRequestItem
            {
                Id = requestId.ToString(),
                States = states.Select(state =>
                    CatanStateSerializer.SerializeCompact((CatanState)state)).ToArray(),
                Priority = priority,
            },
        };

        if (!_submittedAt.TryAdd(requestId, Environment.TickCount64))
            return false;
        try
        {
            if (_pending.TryAdd(request))
                return true;
        }
        catch (InvalidOperationException)
        {
            // Disposal completed the producer side between the initial check and add.
        }
        _submittedAt.TryRemove(requestId, out _);
        return false;
    }

    private void SenderLoop()
    {
        var token = _shutdown.Token;
        while (!token.IsCancellationRequested)
        {
            try
            {
                SendPendingCancellations(token);
                if (!_pending.TryTake(out var first, 5, token))
                    continue;
                if (_batchWindowMs > 0 && token.WaitHandle.WaitOne(_batchWindowMs))
                    break;

                var queued = new List<QueuedLeafRequest>(_batchSize) { first };
                while (queued.Count < _batchSize && _pending.TryTake(out var next))
                    queued.Add(next);

                var owned = queued.Where(item => _submittedAt.ContainsKey(item.RequestId)).ToArray();
                if (owned.Length > 0)
                    SendBatch(owned, token);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (InvalidOperationException) when (_pending.IsCompleted)
            {
                break;
            }
        }
    }

    private void SendPendingCancellations(CancellationToken cancellationToken)
    {
        if (!_pendingCancellations.TryTake(out var first))
            return;

        var ids = new HashSet<long>(first);
        while (_pendingCancellations.TryTake(out var next))
            ids.UnionWith(next);

        try
        {
            using var response = _http.PostAsJsonAsync(
                "state/leaf-cancel",
                new LeafCancelPayload { Ids = ids.Select(id => id.ToString()).ToArray() },
                JsonOptions,
                cancellationToken).GetAwaiter().GetResult();
        }
        catch (OperationCanceledException) when (_disposed)
        {
        }
        catch
        {
            // Cancellation is best effort; local ownership has already been removed.
        }
    }

    private void SendBatch(QueuedLeafRequest[] requests, CancellationToken cancellationToken)
    {
        try
        {
            var payload = new LeafEnqueuePayload { Requests = requests.Select(x => x.Item).ToArray() };
            using var response = _http.PostAsJsonAsync(
                "state/leaf-enqueue", payload, JsonOptions, cancellationToken).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            var acknowledgment = response.Content.ReadFromJsonAsync<LeafEnqueueResponse>(
                JsonOptions, cancellationToken).GetAwaiter().GetResult()
                ?? throw new JsonException("Missing leaf enqueue acknowledgment.");
            var acceptedIds = acknowledgment.AcceptedIds.ToHashSet(StringComparer.Ordinal);
            var droppedIds = acknowledgment.DroppedIds.ToHashSet(StringComparer.Ordinal);
            foreach (var id in droppedIds)
            {
                if (long.TryParse(id, out var requestId))
                    FailIfOwned(requestId);
            }
            foreach (var request in requests)
            {
                if (!acceptedIds.Contains(request.Item.Id) && !droppedIds.Contains(request.Item.Id))
                    FailIfOwned(request.RequestId);
            }
        }
        catch (OperationCanceledException) when (_disposed)
        {
        }
        catch
        {
            foreach (var request in requests)
                FailIfOwned(request.RequestId);
        }
    }

    private void FailIfOwned(long requestId)
    {
        lock (_mailboxLock)
        {
            if (!_submittedAt.TryRemove(requestId, out _))
                return;
            _mailbox.Enqueue(new LeafEvaluationResponse(requestId, [], 0));
            _completed.Set();
        }
    }

    public LeafEvaluationResponse[] Collect(IReadOnlySet<long> knownRequestIds)
    {
        lock (_mailboxLock)
        {
            var results = new List<LeafEvaluationResponse>();
            var keep = new List<LeafEvaluationResponse>();
            while (_mailbox.TryDequeue(out var response))
            {
                if (knownRequestIds.Contains(response.RequestId))
                    results.Add(response);
                else
                    keep.Add(response);
            }
            foreach (var response in keep)
                _mailbox.Enqueue(response);
            return results.ToArray();
        }
    }

    public bool WaitForResults(int timeoutMs) => _completed.WaitOne(Math.Max(0, timeoutMs));

    public void Cancel(IReadOnlySet<long> requestIds)
    {
        if (requestIds.Count == 0 || _disposed)
            return;

        lock (_mailboxLock)
        {
            foreach (var requestId in requestIds)
                _submittedAt.TryRemove(requestId, out _);

            var keep = new List<LeafEvaluationResponse>();
            while (_mailbox.TryDequeue(out var response))
            {
                if (!requestIds.Contains(response.RequestId))
                    keep.Add(response);
            }
            foreach (var response in keep)
                _mailbox.Enqueue(response);
        }

        try
        {
            _pendingCancellations.Add(requestIds.ToArray());
        }
        catch (InvalidOperationException)
        {
            // Disposal completed the producer side between the initial check and add.
        }
    }

    private void PollLoop()
    {
        while (!_disposed)
        {
            try
            {
                using var response = _http.PostAsync("state/leaf-collect", null).GetAwaiter().GetResult();
                if (response.IsSuccessStatusCode)
                {
                    var payload = response.Content.ReadFromJsonAsync<LeafCollectPayload>(JsonOptions)
                        .GetAwaiter().GetResult();
                    foreach (var item in payload?.Responses ?? [])
                    {
                        if (!long.TryParse(item.Id, out var requestId))
                            continue;
                        // A cancelled search no longer owns this response. Discard it
                        // instead of leaving an uncollectable mailbox entry forever.
                        lock (_mailboxLock)
                        {
                            if (!_submittedAt.TryRemove(requestId, out var submitted))
                                continue;
                            var values = item.Values.Select(vector =>
                                Array.ConvertAll(vector, value => (double)value)).ToArray();
                            _mailbox.Enqueue(new LeafEvaluationResponse(
                                requestId, values, Math.Max(0, Environment.TickCount64 - submitted)));
                            _completed.Set();
                        }
                    }
                }
            }
            catch
            {
                // Transient server failures are retried; request timeouts are owned by MCTS.
            }
            if (_shutdown.Token.WaitHandle.WaitOne(5))
                break;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeStarted, 1) != 0)
            return;
        _disposed = true;
        _pending.CompleteAdding();
        _pendingCancellations.CompleteAdding();
        _shutdown.Cancel();
        _http.CancelPendingRequests();
        _completed.Set();
        _senderThread.Join();
        _pollThread.Join();
        _pending.Dispose();
        _pendingCancellations.Dispose();
        _shutdown.Dispose();
        _completed.Dispose();
        _http.Dispose();
    }

    private sealed class QueuedLeafRequest
    {
        public long RequestId { get; init; }
        public LeafRequestItem Item { get; init; } = new();
    }

    private sealed class LeafEnqueuePayload
    {
        public LeafRequestItem[] Requests { get; init; } = [];
    }

    private sealed class LeafCancelPayload
    {
        public string[] Ids { get; init; } = [];
    }

    private sealed class LeafRequestItem
    {
        public string Id { get; init; } = "";
        public string[] States { get; init; } = [];
        public int Priority { get; init; }
    }

    private sealed class LeafEnqueueResponse
    {
        public string[] AcceptedIds { get; init; } = [];
        public string[] DroppedIds { get; init; } = [];
    }

    private sealed class LeafCollectPayload
    {
        public LeafCollectItem[] Responses { get; init; } = [];
    }

    private sealed class LeafCollectItem
    {
        public string Id { get; init; } = "";
        public float[][] Values { get; init; } = [];
    }
}
