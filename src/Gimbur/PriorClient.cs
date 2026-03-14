using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Kjarni;

namespace Gimbur;

/// <summary>
/// Asynchronous prior client for NN-guided MCTS search.
/// Implements <see cref="IPriorClient"/> by communicating with the Python
/// inference server via HTTP. Prior requests are fire-and-forget; completed
/// results are collected via a background polling thread that deposits
/// responses into a local mailbox.
/// </summary>
public sealed class PriorClient : IPriorClient, IDisposable
{
    private readonly HttpClient _http;
    private readonly ConcurrentQueue<PriorResponse> _mailbox = new();
    private readonly Thread _pollThread;
    private volatile bool _disposed;

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

    public PriorClient(string baseUrl)
    {
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
    /// Enqueue an async prior request. Serializes each ICoreState via
    /// CatanStateSerializer and POSTs to /prior-enqueue. Non-blocking:
    /// fires the HTTP request without awaiting the response.
    /// </summary>
    public void RequestPrior(long nodeId, ICoreState[] states, int actingPlayer, int depth)
    {
        // Serialize each state to a compact string.
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

        // Fire-and-forget: send the request without blocking.
        _ = Task.Run(async () =>
        {
            try
            {
                await _http.PostAsJsonAsync("prior-enqueue", request, JsonOptions);
                // We don't need to check the response — the server returns 202.
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
            _http.PostAsync("prior-flush", null).GetAwaiter().GetResult();
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
        while (!_disposed)
        {
            try
            {
                var response = _http.PostAsync("prior-collect", null).GetAwaiter().GetResult();
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
