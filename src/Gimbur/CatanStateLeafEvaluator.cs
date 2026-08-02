using System.Collections.Concurrent;
using System.Net.Http.Json;
using System.Text.Json;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur;

/// <summary>Shared asynchronous evaluator for Catan state-model value heads.</summary>
public sealed class CatanStateLeafEvaluator : ILeafEvaluator, IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    private readonly HttpClient _http;
    private readonly ConcurrentQueue<LeafEvaluationResponse> _mailbox = new();
    private readonly ConcurrentDictionary<long, long> _submittedAt = new();
    private readonly AutoResetEvent _completed = new(false);
    private readonly Thread _pollThread;
    private volatile bool _disposed;

    public CatanStateLeafEvaluator(string baseUrl)
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };
        _pollThread = new Thread(PollLoop) { IsBackground = true, Name = "CatanLeafEvaluator-Poll" };
        _pollThread.Start();
    }

    public bool Enqueue(long requestId, ICoreState[] states, int priority)
    {
        if (states.Length == 0 || _disposed)
            return false;

        var payload = new LeafEnqueuePayload
        {
            Requests =
            [
                new LeafRequestItem
                {
                    Id = requestId.ToString(),
                    States = states.Select(state =>
                        CatanStateSerializer.SerializeCompact((CatanState)state)).ToArray(),
                    Priority = priority,
                },
            ],
        };
        _submittedAt[requestId] = Environment.TickCount64;
        _ = EnqueueAsync(requestId, payload);
        return true;
    }

    private async Task EnqueueAsync(long requestId, LeafEnqueuePayload payload)
    {
        try
        {
            using var response = await _http.PostAsJsonAsync("state/leaf-enqueue", payload, JsonOptions);
            response.EnsureSuccessStatusCode();
        }
        catch
        {
            _submittedAt.TryRemove(requestId, out _);
            _mailbox.Enqueue(new LeafEvaluationResponse(requestId, [], 0));
            _completed.Set();
        }
    }

    public LeafEvaluationResponse[] Collect(IReadOnlySet<long> knownRequestIds)
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

    public bool WaitForResults(int timeoutMs) => _completed.WaitOne(Math.Max(0, timeoutMs));

    public void Cancel(IReadOnlySet<long> requestIds)
    {
        foreach (var requestId in requestIds)
            _submittedAt.TryRemove(requestId, out _);
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
            catch
            {
                // Transient server failures are retried; request timeouts are owned by MCTS.
            }
            Thread.Sleep(5);
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _completed.Set();
        _pollThread.Join(TimeSpan.FromSeconds(2));
        _completed.Dispose();
        _http.Dispose();
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
