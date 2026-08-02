using Gimbur;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Hybrid player that uses a configurable fallback for the initial placement
/// phase and the neural network state model for the main game.
///
/// During placement stages (0–3), the player delegates to the fallback
/// (defaults to <see cref="GreedyPlayer"/>).  Once the main game begins
/// (stage ≥ 4), the player uses the same evaluation strategy as
/// <see cref="NnPlayer"/>: enumerating all actions, evaluating resulting
/// states via the <c>/state/predict</c> endpoint, and picking the action
/// with the highest expected win probability.
/// </summary>
internal sealed class NnStatePlayer : IBenchmarkPlayer, INnStatsProvider
{
    private readonly NnClient _client;
    private readonly IBenchmarkPlayer _placementFallback;

    // INnStatsProvider
    public int TotalNnRequests { get; private set; }
    public int TotalNnStatesEvaluated { get; private set; }

    public NnStatePlayer(NnClient client, IBenchmarkPlayer? placementFallback = null)
    {
        _client = client;
        _placementFallback = placementFallback ?? new GreedyPlayer();
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var isPlacement = state.Stage is TurnStage.PlaceFirstSettlement
                                     or TurnStage.PlaceFirstRoad
                                     or TurnStage.PlaceSecondSettlement
                                     or TurnStage.PlaceSecondRoad;

        if (isPlacement)
        {
            return _placementFallback.Act(state, rng);
        }

        var coreActions = state.Actions();
        if (coreActions.Length == 0) return null;

        // Forced action — no decision to make.
        if (coreActions.Length == 1)
        {
            return (CatanState)UnwrapCoreAction(coreActions[0]).DoCoreAction();
        }

        return ChooseByNn(state, coreActions);
    }

    /// <summary>
    /// Evaluates all actions via the state NN model (same logic as <see cref="NnPlayer"/>).
    /// </summary>
    private CatanState? ChooseByNn(CatanState state, CoreAction[] coreActions)
    {
        var actingPlayer = state.CurrentPlayer;

        var allStates = new List<string>();
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
                }

                actionDescriptors.Add(new ActionDescriptor(i, stateIndex, outcomes.Length, weights));
            }
        }

        var playerValues = _client.PredictAsync(allStates).GetAwaiter().GetResult();
        TotalNnRequests++;
        TotalNnStatesEvaluated += allStates.Count;

        var bestActionIndex = 0;
        var bestScore = float.NegativeInfinity;

        foreach (var desc in actionDescriptors)
        {
            float score;
            if (desc.Weights is null)
            {
                score = playerValues[desc.StartIndex][actingPlayer - 1];
            }
            else
            {
                var totalWeight = 0;
                var weightedSum = 0.0f;
                for (var j = 0; j < desc.Count; j++)
                {
                    var w = desc.Weights[j];
                    totalWeight += w;
                    weightedSum += w * playerValues[desc.StartIndex + j][actingPlayer - 1];
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

    private readonly record struct ActionDescriptor(
        int ActionIndex,
        int StartIndex,
        int Count,
        int[]? Weights);
}
