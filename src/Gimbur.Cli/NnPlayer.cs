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

        var actingPlayer = state.CurrentPlayer;

        // Collect all states we need to evaluate.
        // For deterministic actions: 1 resulting state.
        // For stochastic actions: multiple weighted outcomes.
        var allStates = new List<string>();
        var actionDescriptors = new List<ActionDescriptor>();

        for (var i = 0; i < coreActions.Length; i++)
        {
            var coreAction = coreActions[i];

            if (coreAction.IsDeterministic)
            {
                var deterministicAction = (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
                var resultState = (CatanState)deterministicAction.State();
                if (resultState.WinnerPlayer != 0)
                {
                    actionDescriptors.Add(new ActionDescriptor(
                        i, [], [1], [resultState.WinnerPlayer == actingPlayer ? 1.0f : 0.0f]));
                    continue;
                }
                var stateIndex = allStates.Count;
                allStates.Add(resultState.SerializeCompact());
                actionDescriptors.Add(new ActionDescriptor(i, [stateIndex], [1], [float.NaN]));
            }
            else if (coreAction.IsStochastic)
            {
                var stochasticAction = (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
                var outcomes = stochasticAction.Outcomes();
                var weights = new int[outcomes.Length];
                var stateIndices = new int[outcomes.Length];
                var exactScores = new float[outcomes.Length];
                for (var j = 0; j < outcomes.Length; j++)
                {
                    weights[j] = outcomes[j].Item1;
                    var outcomeState = (CatanState)outcomes[j].Item2;
                    if (outcomeState.WinnerPlayer != 0)
                    {
                        stateIndices[j] = -1;
                        exactScores[j] = outcomeState.WinnerPlayer == actingPlayer ? 1.0f : 0.0f;
                    }
                    else
                    {
                        stateIndices[j] = allStates.Count;
                        exactScores[j] = float.NaN;
                        allStates.Add(outcomeState.SerializeCompact());
                    }
                }

                actionDescriptors.Add(new ActionDescriptor(i, stateIndices, weights, exactScores));
            }
        }

        var playerValues = allStates.Count > 0
            ? _client.PredictAsync(allStates).GetAwaiter().GetResult()
            : [];
        if (allStates.Count > 0)
        {
            TotalNnRequests++;
            TotalNnStatesEvaluated += allStates.Count;
        }

        // Score each action: for deterministic actions use the win prob directly;
        // for stochastic actions compute the expected value across outcomes.
        var bestActionIndex = 0;
        var bestScore = float.NegativeInfinity;

        foreach (var desc in actionDescriptors)
        {
            var totalWeight = 0;
            var weightedSum = 0.0f;
            for (var j = 0; j < desc.Weights.Length; j++)
            {
                var value = float.IsNaN(desc.ExactScores[j])
                    ? playerValues[desc.StateIndices[j]][actingPlayer - 1]
                    : desc.ExactScores[j];
                totalWeight += desc.Weights[j];
                weightedSum += desc.Weights[j] * value;
            }
            var score = totalWeight > 0 ? weightedSum / totalWeight : 0;

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
        int[] StateIndices,
        int[] Weights,
        float[] ExactScores);
}
