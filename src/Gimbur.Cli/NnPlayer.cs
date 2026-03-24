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
/// The model always predicts player 1's win probability.  The server-side
/// <c>/predict-player</c> endpoint rotates the state so that the acting
/// player becomes player 1 before inference, returning a scalar win
/// probability that can be maximised directly.
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

        var actingPlayer = state.CurrentPlayer;

        // Collect all states we need to evaluate.
        // For deterministic actions: 1 resulting state.
        // For stochastic actions: multiple weighted outcomes.
        var allStates = new List<string>();
        var allPlayers = new List<int>();
        var actionDescriptors = new List<ActionDescriptor>();

        for (var i = 0; i < coreActions.Length; i++)
        {
            var coreAction = coreActions[i];

            if (coreAction.IsDeterministic)
            {
                var deterministicAction = (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
                var resultState = (CatanState)deterministicAction.State();
                var stateIndex = allStates.Count;
                allStates.Add(resultState.SerializeCompact());
                allPlayers.Add(actingPlayer);
                actionDescriptors.Add(new ActionDescriptor(i, stateIndex, 1, null));
            }
            else if (coreAction.IsStochastic)
            {
                var stochasticAction = (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
                var outcomes = stochasticAction.Outcomes();
                var stateIndex = allStates.Count;
                var weights = new int[outcomes.Length];
                for (var j = 0; j < outcomes.Length; j++)
                {
                    weights[j] = outcomes[j].Item1;
                    var outcomeState = (CatanState)outcomes[j].Item2;
                    allStates.Add(outcomeState.SerializeCompact());
                    allPlayers.Add(actingPlayer);
                }

                actionDescriptors.Add(new ActionDescriptor(i, stateIndex, outcomes.Length, weights));
            }
        }

        // Batch predict all states at once.  The server rotates each state
        // so that actingPlayer becomes player 1 and returns scalar win probs.
        var winProbs = _client.PredictPlayerAsync(allStates, allPlayers).GetAwaiter().GetResult();
        TotalNnRequests++;
        TotalNnStatesEvaluated += allStates.Count;

        // Score each action: for deterministic actions use the win prob directly;
        // for stochastic actions compute the expected value across outcomes.
        var bestActionIndex = 0;
        var bestScore = float.NegativeInfinity;

        foreach (var desc in actionDescriptors)
        {
            float score;
            if (desc.Weights is null)
            {
                // Deterministic: single state.
                score = winProbs[desc.StartIndex];
            }
            else
            {
                // Stochastic: probability-weighted average.
                var totalWeight = 0;
                var weightedSum = 0.0f;
                for (var j = 0; j < desc.Count; j++)
                {
                    var w = desc.Weights[j];
                    totalWeight += w;
                    weightedSum += w * winProbs[desc.StartIndex + j];
                }

                score = totalWeight > 0 ? weightedSum / totalWeight : 0;
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestActionIndex = desc.ActionIndex;
            }
        }

        return (CatanState)UnwrapCoreAction(coreActions[bestActionIndex]).DoCoreAction();
    }

    private static CatanAction UnwrapCoreAction(CoreAction coreAction)
    {
        if (coreAction.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
        if (coreAction.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
    }

    /// <summary>
    /// Describes how an action's resulting state(s) map into the flat
    /// prediction array.
    /// </summary>
    private readonly record struct ActionDescriptor(
        int ActionIndex,
        int StartIndex,
        int Count,
        int[]? Weights);
}
