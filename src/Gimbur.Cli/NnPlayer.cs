using Gimbur;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Neural-network-based player that queries a Python inference server to
/// evaluate game states and picks the action leading to the highest win
/// probability for the current player.
///
/// For deterministic actions the evaluation is direct.  For stochastic
/// actions (dice rolls, robber placement, dev card draws) the expected
/// value is computed as a probability-weighted average across all outcomes.
///
/// The model returns one win probability per player; this player maximizes
/// the acting player's component.
/// </summary>
internal sealed class NnPlayer : IBenchmarkPlayer, INnStatsProvider
{
    private readonly NnClient _client;

    // INnStatsProvider
    public int TotalNnRequests { get; private set; }
    public int TotalNnStatesEvaluated { get; private set; }

    public NnPlayer(NnClient client)
    {
        _client = client;
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var coreActions = state.Actions();
        if (coreActions.Length == 0) return null;

        // Forced action — no decision to make.
        if (coreActions.Length == 1)
        {
            return (CatanState)UnwrapCoreAction(coreActions[0]).DoCoreAction();
        }

        var actions = coreActions.Select(UnwrapCoreAction).ToArray();
        var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
        var prediction = _client.PredictPolicyValueAsync([state.SerializeCompact()])
            .GetAwaiter().GetResult();
        TotalNnRequests++;
        TotalNnStatesEvaluated++;
        var densePolicy = prediction.PolicyProbabilities.Length == 1
            ? Array.ConvertAll(prediction.PolicyProbabilities[0], value => (double)value)
            : [];
        var legalIndices = actions.Select(action => serializer.IndexOf(state, action)).ToArray();
        var policy = serializer.MaskAndNormalize(densePolicy, legalIndices);
        var bestIndex = Array.IndexOf(policy, policy.Max());
        return (CatanState)actions[bestIndex].DoCoreAction();
    }

    private static CatanAction UnwrapCoreAction(CoreAction coreAction)
    {
        if (coreAction.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
        if (coreAction.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
    }

}
