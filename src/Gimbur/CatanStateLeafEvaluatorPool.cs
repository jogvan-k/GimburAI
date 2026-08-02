using System.Collections.Concurrent;

namespace Gimbur;

public static class CatanStateLeafEvaluatorPool
{
    private static readonly ConcurrentDictionary<string, CatanStateLeafEvaluator> Evaluators = new();

    public static CatanStateLeafEvaluator Get(string baseUrl) =>
        Evaluators.GetOrAdd(baseUrl.TrimEnd('/'), url => new CatanStateLeafEvaluator(url));
}
