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
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Thread _senderThread;
    private readonly int _batchSize;
    private readonly int _batchWindowMs;
    private long _completionVersion;
    private long _requestsQueued;
    private long _requestsSent;
    private long _requestsAcknowledged;
    private long _responsesPolled;
    private int _disposeStarted;
    private volatile bool _disposed;
    private long _lastDiagnosticAt = Environment.TickCount64;

    public (long Queued, long Sent, long Acknowledged, long Polled, int Owned, int Mailbox)
        Diagnostics => (
            Interlocked.Read(ref _requestsQueued),
            Interlocked.Read(ref _requestsSent),
            Interlocked.Read(ref _requestsAcknowledged),
            Interlocked.Read(ref _responsesPolled),
            _submittedAt.Count,
            _mailbox.Count);

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
        _senderThread.Start();
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
            {
                Interlocked.Increment(ref _requestsQueued);
                return true;
            }
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
                if (!_pending.TryTake(out var first, 5, token))
                {
                    LogDiagnosticsIfDue();
                    continue;
                }
                if (_batchWindowMs > 0 && token.WaitHandle.WaitOne(_batchWindowMs))
                    break;

                var queued = new List<QueuedLeafRequest>(_batchSize) { first };
                while (queued.Count < _batchSize && _pending.TryTake(out var next))
                    queued.Add(next);

                var owned = queued.Where(item => _submittedAt.ContainsKey(item.RequestId)).ToArray();
                if (owned.Length > 0)
                    SendBatch(owned, token);
                LogDiagnosticsIfDue();
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

    private void SendBatch(QueuedLeafRequest[] requests, CancellationToken cancellationToken)
    {
        try
        {
            var payload = new LeafEnqueuePayload { Requests = requests.Select(x => x.Item).ToArray() };
            Interlocked.Add(ref _requestsSent, requests.Length);
            using var response = _http.PostAsJsonAsync(
                "state/leaf-predict", payload, JsonOptions, cancellationToken).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            var result = response.Content.ReadFromJsonAsync<LeafCollectPayload>(
                JsonOptions, cancellationToken).GetAwaiter().GetResult()
                ?? throw new JsonException("Missing leaf prediction response.");
            var expectedIds = requests.Select(request => request.Item.Id).ToHashSet(StringComparer.Ordinal);
            var responseIds = result.Responses.Select(item => item.Id).ToArray();
            if (responseIds.Length != requests.Length ||
                responseIds.Distinct(StringComparer.Ordinal).Count() != responseIds.Length ||
                responseIds.Any(id => !expectedIds.Contains(id)))
                throw new JsonException("Leaf prediction response IDs did not match the request IDs.");

            Interlocked.Add(ref _requestsAcknowledged, result.Responses.Length);
            Interlocked.Add(ref _responsesPolled, result.Responses.Length);
            lock (_mailboxLock)
            {
                var addedResponse = false;
                foreach (var item in result.Responses)
                {
                    var requestId = long.Parse(item.Id);
                    if (!_submittedAt.TryRemove(requestId, out var submitted))
                        continue;
                    var values = item.Values.Select(vector =>
                        Array.ConvertAll(vector, value => (double)value)).ToArray();
                    _mailbox.Enqueue(new LeafEvaluationResponse(
                        requestId, values, Math.Max(0, Environment.TickCount64 - submitted)));
                    addedResponse = true;
                }
                if (addedResponse)
                    NotifyCompletion();
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
            NotifyCompletion();
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

    public bool WaitForResults(int timeoutMs)
    {
        lock (_mailboxLock)
        {
            // A response may arrive after the caller's Collect and before this
            // method acquires the lock. In that case do not wait for another
            // completion that may never come.
            if (!_mailbox.IsEmpty)
                return true;
            var observedVersion = _completionVersion;
            var remainingMs = Math.Max(0, timeoutMs);
            var deadline = Environment.TickCount64 + remainingMs;
            while (_completionVersion == observedVersion)
            {
                if (remainingMs == 0 || !Monitor.Wait(_mailboxLock, remainingMs))
                    return false;
                remainingMs = (int)Math.Min(int.MaxValue, Math.Max(0, deadline - Environment.TickCount64));
            }
            return true;
        }
    }

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
            NotifyCompletion();
        }

    }

    private void LogDiagnosticsIfDue()
    {
        if (Environment.TickCount64 - _lastDiagnosticAt < 30_000)
            return;
        _lastDiagnosticAt = Environment.TickCount64;
        var diagnostic = Diagnostics;
        Console.WriteLine(
            $"[LeafEvaluator] queued={diagnostic.Queued} sent={diagnostic.Sent} " +
            $"ack={diagnostic.Acknowledged} polled={diagnostic.Polled} " +
            $"owned={diagnostic.Owned} mailbox={diagnostic.Mailbox}");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeStarted, 1) != 0)
            return;
        _disposed = true;
        _pending.CompleteAdding();
        _shutdown.Cancel();
        lock (_mailboxLock)
            NotifyCompletion();
        _senderThread.Join();
        _pending.Dispose();
        _shutdown.Dispose();
        _http.Dispose();
    }

    private void NotifyCompletion()
    {
        _completionVersion++;
        Monitor.PulseAll(_mailboxLock);
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

    private sealed class LeafRequestItem
    {
        public string Id { get; init; } = "";
        public string[] States { get; init; } = [];
        public int Priority { get; init; }
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
