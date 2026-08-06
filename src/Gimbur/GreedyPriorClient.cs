using System.Collections.Concurrent;
using Kjarni;

namespace Gimbur;

/// <summary>
/// Supplies a local one-hot PUCT prior for the action selected by the greedy AI.
/// Used as the generation-zero teacher before a neural policy exists.
/// </summary>
public sealed class GreedyPriorClient : IPriorClient
{
    private readonly ConcurrentDictionary<long, PriorResponse> _responses = new();
    private readonly double _uniformMix;

    public GreedyPriorClient(double uniformMix = 0.25)
    {
        if (!double.IsFinite(uniformMix) || uniformMix is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(uniformMix));
        _uniformMix = uniformMix;
    }

    public bool ShouldRequestPrior(ICoreState parentState) => parentState is CatanState;

    public int RequestPrior(
        long nodeId,
        ICoreState parentState,
        ICoreState[] states,
        int actingPlayer,
        int depth)
    {
        if (parentState is not CatanState state)
            return 0;

        var actions = state.Actions();
        var greedyAction = new GreedyActionSelector().ChooseAction(
            state, new Random(HashCode.Combine(state.GetHashCode(), nodeId)));
        if (greedyAction is null)
            return 0;

        var priors = new double[states.Length];
        var densePriors = new double[actions.Length];
        var uniformPrior = _uniformMix / actions.Length;
        var stateIndex = 0;
        var found = false;
        for (var actionIndex = 0; actionIndex < actions.Length; actionIndex++)
        {
            var action = UnwrapAction(actions[actionIndex]);
            var outcomeCount = action is CatanStochasticAction stochastic
                ? stochastic.Outcomes().Length
                : 1;
            if (stateIndex + outcomeCount > priors.Length)
                return 0;

            if (action.Equals(greedyAction))
            {
                Array.Fill(priors, 1.0 - _uniformMix + uniformPrior, stateIndex, outcomeCount);
                densePriors[actionIndex] = 1.0;
                found = true;
            }
            else
            {
                Array.Fill(priors, uniformPrior, stateIndex, outcomeCount);
            }
            stateIndex += outcomeCount;
        }

        if (!found || stateIndex != states.Length)
            return 0;

        _responses[nodeId] = new PriorResponse(nodeId, priors, [], densePriors);
        return states.Length;
    }

    public PriorResponse[] CollectPriors(IReadOnlySet<long> knownNodeIds)
    {
        var results = new List<PriorResponse>();
        foreach (var nodeId in knownNodeIds)
        {
            if (_responses.TryRemove(nodeId, out var response))
                results.Add(response);
        }
        return results.ToArray();
    }

    public void Flush(IReadOnlySet<long> knownNodeIds)
    {
        foreach (var nodeId in knownNodeIds)
            _responses.TryRemove(nodeId, out _);
    }

    private static CatanAction UnwrapAction(CoreAction action)
    {
        if (action.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)action).Item;
        if (action.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)action).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {action.Tag}");
    }
}
