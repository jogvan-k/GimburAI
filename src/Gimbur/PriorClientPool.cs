using System.Collections.Concurrent;
namespace Gimbur;

/// <summary>
/// Process-wide pool of long-lived <see cref="PriorClient"/> instances keyed by
/// the inference server URL. Reusing clients across <c>/choose-action</c> requests keeps the
/// background polling thread warm so prior responses arrive in time to influence
/// MCTS search, instead of being created and torn down per request.
///
/// Pooled clients are <see cref="PriorClient"/> instances configured with
/// <see cref="PriorClient.PooledMode"/> = <c>true</c>, which suppresses the
/// per-search global server flush (which would otherwise discard priors for
/// concurrently-running searches sharing the same client).
/// </summary>
public static class PriorClientPool
{
    private static readonly ConcurrentDictionary<string, PriorClient> _clients = new();

    /// <summary>
    /// Get or create a pooled <see cref="PriorClient"/> for the given configuration.
    /// The returned client is owned by the pool and MUST NOT be disposed by the caller.
    /// </summary>
    public static PriorClient Get(string nnUrl)
    {
        var normalized = nnUrl.TrimEnd('/');
        return _clients.GetOrAdd(normalized, url => new PriorClient(url, pooled: true));
    }
}
