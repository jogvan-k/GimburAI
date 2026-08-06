using Gimbur;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Hybrid player that uses the neural network placement model for the initial
/// placement phase and falls back to greedy heuristics for the main game.
///
/// During placement, the player requests the current stage policy and chooses
/// among legal settlement vertices or road directions.
///
/// During the main game, the player delegates to a configurable fallback
/// (defaults to <see cref="GreedyPlayer"/>).
/// </summary>
internal sealed class NnPlacementPlayer : IBenchmarkPlayer, INnStatsProvider
{
    private readonly NnClient _client;
    private readonly PlacementActionSerializer _actionSerializer;
    private readonly IBenchmarkPlayer _fallback;

    // INnStatsProvider
    public int TotalNnRequests { get; private set; }
    public int TotalNnStatesEvaluated { get; private set; }

    public NnPlacementPlayer(NnClient client, PlacementActionSerializer actionSerializer, IBenchmarkPlayer? fallback = null)
    {
        _client = client;
        _actionSerializer = actionSerializer;
        _fallback = fallback ?? new GreedyPlayer();
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var isPlacement = state.Stage is TurnStage.PlaceFirstSettlement
                                     or TurnStage.PlaceFirstRoad
                                     or TurnStage.PlaceSecondSettlement
                                     or TurnStage.PlaceSecondRoad;

        if (!isPlacement)
        {
            return _fallback.Act(state, rng);
        }

        var coreActions = state.Actions();
        if (coreActions.Length == 0) return null;

        // Forced action — no decision to make.
        if (coreActions.Length == 1)
        {
            return (CatanState)UnwrapCoreAction(coreActions[0]).DoCoreAction();
        }

        return ChoosePlacementAction(state, coreActions);
    }

    /// <summary>
    /// Masks the current stage policy to legal actions and applies the best one.
    /// </summary>
    private CatanState? ChoosePlacementAction(CatanState state, CoreAction[] coreActions)
    {
        var placementState = state.SerializePlacementPhaseCompact();
        var actions = coreActions.Select(UnwrapCoreAction).ToArray();
        var legalIndices = state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement
            ? actions.Cast<PlaceSettlementAction>().Select(action => action.VertexIndex).ToArray()
            : actions.Cast<PlaceRoadAction>().Select(action => _actionSerializer.DirectionIndexOf(
                state.PendingSettlementVertex!.Value, action.EdgeIndex)).ToArray();

        var prediction = _client.PredictPlacementAsync([placementState]).GetAwaiter().GetResult();
        TotalNnRequests++;
        TotalNnStatesEvaluated++;

        var densePolicy = prediction.PolicyProbabilities.Length == 1
            ? Array.ConvertAll(prediction.PolicyProbabilities[0], value => (double)value)
            : [];
        var legalPolicy = _actionSerializer.MaskAndNormalizeStage(densePolicy, legalIndices);

        var bestIndex = 0;
        var bestScore = double.NegativeInfinity;
        for (var j = 0; j < actions.Length; j++)
        {
            var score = legalPolicy[j];
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = j;
            }
        }

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
