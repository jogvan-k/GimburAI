using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Gimbur;

/// <summary>
/// HTTP client for the Python GimburNet inference server.
/// Sends serialized game state strings to <c>/state/predict</c> and returns
/// per-state player win probability distributions.
/// Placement requests go to <c>/placement/predict</c>.
/// </summary>
public sealed class NnClient : IDisposable
{
    private readonly HttpClient _http;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public NnClient(string baseUrl)
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };
    }

    /// <summary>
    /// Sends one or more serialized game states to the inference server and
    /// returns the predicted per-player win distributions.
    /// </summary>
    /// <returns>
    /// An array of float arrays, one per input state.  Each inner array has
    /// one element per player, summing to approximately 1.0.
    /// </returns>
    public async Task<float[][]> PredictAsync(IReadOnlyList<string> compactStates)
    {
        var request = new PredictRequest { States = compactStates };
        using var response = await SendWithRetryAsync(
            () => _http.PostAsJsonAsync("state/predict", request, JsonOptions));
        var result = await response.Content.ReadFromJsonAsync<PredictResponse>(JsonOptions);
        return NormalizeDistributions(
            result?.PlayerWinProbabilities ?? [], compactStates.Count);
    }

    /// <summary>
    /// Convenience overload for a single state.
    /// </summary>
    public async Task<float[]> PredictSingleAsync(string compactState)
    {
        var results = await PredictAsync([compactState]);
        return results.Length > 0 ? results[0] : [];
    }

    /// <summary>
    /// Sends compact state strings together with target player numbers to
    /// the <c>/state/predict-player</c> endpoint.  The server rotates each state
    /// so that the target player becomes player 1, runs inference, and
    /// returns a scalar expected win probability for each state.
    /// </summary>
    /// <param name="compactStates">Compact serialized game states.</param>
    /// <param name="players">1-based target player for each state.</param>
    /// <returns>
    /// An array of floats in [0, 1], one per input state, representing
    /// the target player's expected win probability.
    /// </returns>
    public async Task<float[]> PredictPlayerAsync(
        IReadOnlyList<string> compactStates,
        IReadOnlyList<int> players)
    {
        var request = new PredictPlayerRequest { States = compactStates, Players = players };
        using var response = await SendWithRetryAsync(
            () => _http.PostAsJsonAsync("state/predict-player", request, JsonOptions));
        var result = await response.Content.ReadFromJsonAsync<PredictPlayerResponse>(JsonOptions);
        return result?.WinProbabilities ?? [];
    }

    /// <summary>
    /// Convenience overload: evaluates a single state for a single player.
    /// </summary>
    public async Task<float> PredictPlayerSingleAsync(string compactState, int player)
    {
        var results = await PredictPlayerAsync([compactState], [player]);
        return results.Length > 0 ? results[0] : 0f;
    }

    /// <summary>
    /// Sends compact placement states to <c>/placement/predict</c> and returns
    /// player values plus one dense policy over the full action vocabulary.
    /// </summary>
    /// <param name="compactStates">Compact serialized placement phase states.</param>
    public async Task<PlacementPrediction> PredictPlacementAsync(
        IReadOnlyList<string> compactStates)
    {
        var request = new PredictPlacementRequest { States = compactStates };
        using var response = await SendWithRetryAsync(
            () => _http.PostAsJsonAsync("placement/predict", request, JsonOptions));
        var result = await response.Content.ReadFromJsonAsync<PredictPlacementResponse>(JsonOptions);
        return new PlacementPrediction(
            NormalizeDistributions(
                result?.PlayerWinProbabilities ?? [], compactStates.Count),
            result?.PolicyProbabilities ?? []);
    }

    public sealed record PlacementPrediction(
        float[][] PlayerWinProbabilities,
        float[][] PolicyProbabilities);

    private static float[][] NormalizeDistributions(float[][] distributions, int expectedCount)
    {
        if (distributions.Length != expectedCount)
            throw new InvalidDataException("NN server returned the wrong number of player value distributions.");

        var playerCount = distributions.FirstOrDefault()?.Length ?? 0;
        foreach (var values in distributions)
        {
            if (values.Length == 0 || values.Length != playerCount)
                throw new InvalidDataException("NN server returned inconsistent player value distribution lengths.");

            var total = 0.0f;
            foreach (var value in values)
            {
                if (!float.IsFinite(value) || value < 0)
                    throw new InvalidDataException("NN server returned an invalid player value distribution.");
                total += value;
            }

            if (!float.IsFinite(total) || total <= 0)
                throw new InvalidDataException("NN server returned a zero-mass player value distribution.");

            for (var i = 0; i < values.Length; i++)
                values[i] /= total;
        }

        return distributions;
    }

    /// <summary>
    /// Maximum number of retry attempts for transient HTTP failures.
    /// </summary>
    private const int MaxRetries = 3;

    /// <summary>
    /// Base delay between retries. Each subsequent retry doubles this delay.
    /// </summary>
    private static readonly TimeSpan RetryBaseDelay = TimeSpan.FromMilliseconds(200);

    /// <summary>
    /// Sends an HTTP request with retry logic and exponential backoff for
    /// transient failures (5xx responses, timeouts, connection errors).
    /// Client errors (4xx) are not retried since they indicate a request problem.
    /// For non-2xx responses, the response body is included in the exception
    /// message for diagnostics.
    /// </summary>
    private async Task<HttpResponseMessage> SendWithRetryAsync(
        Func<Task<HttpResponseMessage>> sendFunc)
    {
        for (var attempt = 0; ; attempt++)
        {
            HttpResponseMessage? response = null;
            try
            {
                response = await sendFunc();
                if (!response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync();
                    var msg = $"NN server returned {(int)response.StatusCode} ({response.ReasonPhrase})";
                    if (!string.IsNullOrWhiteSpace(body))
                        msg += $": {body}";
                    throw new HttpRequestException(msg, null, response.StatusCode);
                }
                return response;
            }
            catch (HttpRequestException ex) when (attempt < MaxRetries && IsTransient(response, ex))
            {
                response?.Dispose();
                var delay = RetryBaseDelay * (1 << attempt); // 200ms, 400ms, 800ms
                await Task.Delay(delay);
            }
            catch (TaskCanceledException) when (attempt < MaxRetries)
            {
                response?.Dispose();
                var delay = RetryBaseDelay * (1 << attempt);
                await Task.Delay(delay);
            }
        }
    }

    /// <summary>
    /// Returns true for transient failures that are worth retrying:
    /// 5xx server errors, connection failures (no response), and timeouts.
    /// Returns false for 4xx client errors.
    /// </summary>
    private static bool IsTransient(HttpResponseMessage? response, HttpRequestException ex)
    {
        // No response at all — connection failure.
        if (response is null)
            return true;

        var code = (int)response.StatusCode;
        // 5xx server errors are transient; 4xx client errors are not.
        return code >= 500;
    }

    /// <summary>
    /// Checks whether the inference server is reachable.
    /// </summary>
    public async Task<bool> IsHealthyAsync()
    {
        try
        {
            using var response = await _http.GetAsync("health");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public void Dispose() => _http.Dispose();

    private sealed class PredictRequest
    {
        public IReadOnlyList<string> States { get; init; } = [];
    }

    private sealed class PredictResponse
    {
        [JsonPropertyName("player_win_probabilities")]
        public float[][] PlayerWinProbabilities { get; init; } = [];
    }

    private sealed class PredictPlayerRequest
    {
        public IReadOnlyList<string> States { get; init; } = [];
        public IReadOnlyList<int> Players { get; init; } = [];
    }

    private sealed class PredictPlayerResponse
    {
        [JsonPropertyName("win_probabilities")]
        public float[] WinProbabilities { get; init; } = [];
    }

    private sealed class PredictPlacementRequest
    {
        public IReadOnlyList<string> States { get; init; } = [];
    }

    private sealed class PredictPlacementResponse
    {
        [JsonPropertyName("player_win_probabilities")]
        public float[][] PlayerWinProbabilities { get; init; } = [];

        [JsonPropertyName("policy_probabilities")]
        public float[][] PolicyProbabilities { get; init; } = [];
    }
}
