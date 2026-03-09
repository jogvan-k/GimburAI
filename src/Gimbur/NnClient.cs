using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Gimbur;

/// <summary>
/// HTTP client for the Python GimburNet inference server.
/// Sends serialized game state strings to <c>/predict</c> and returns
/// per-state win probability distributions (128 buckets).
/// </summary>
public sealed class NnClient : IDisposable
{
    private readonly HttpClient _http;

    /// <summary>
    /// Number of buckets in the model's output distribution.
    /// </summary>
    public const int BucketCount = 128;

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
    /// returns the predicted probability distributions.
    /// </summary>
    /// <returns>
    /// An array of float arrays, one per input state.  Each inner array has
    /// <see cref="BucketCount"/> elements summing to ~1.0.
    /// </returns>
    public async Task<float[][]> PredictAsync(IReadOnlyList<string> compactStates)
    {
        var request = new PredictRequest { States = compactStates };
        var response = await _http.PostAsJsonAsync("predict", request, JsonOptions);
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<PredictResponse>(JsonOptions);
        return result?.Probabilities ?? [];
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
    /// the <c>/predict-player</c> endpoint.  The server rotates each state
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
        var response = await _http.PostAsJsonAsync("predict-player", request, JsonOptions);
        response.EnsureSuccessStatusCode();
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
    /// Converts a 128-bucket probability distribution into a single expected
    /// win probability in [0, 1].  Bucket centres are evenly spaced:
    /// centre(i) = (i + 0.5) / BucketCount.
    /// </summary>
    public static float ExpectedWinProbability(float[] buckets)
    {
        var expected = 0.0f;
        for (var i = 0; i < buckets.Length; i++)
        {
            var centre = (i + 0.5f) / BucketCount;
            expected += centre * buckets[i];
        }

        return expected;
    }

    /// <summary>
    /// Checks whether the inference server is reachable.
    /// </summary>
    public async Task<bool> IsHealthyAsync()
    {
        try
        {
            var response = await _http.GetAsync("health");
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
        public float[][] Probabilities { get; init; } = [];
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
}
