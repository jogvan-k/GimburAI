using Gimbur;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>
/// Hybrid player that uses the neural network placement model for the initial
/// placement phase and falls back to greedy heuristics for the main game.
///
/// During placement (settlement stages), the player enumerates all legal
/// (settlement, road) composite actions, sends them as (state, action) pairs
/// to the <c>/predict-placement</c> endpoint, and picks the pair with the
/// highest expected win probability.  Road stages apply the road that was
/// chosen as part of the best composite action.
///
/// During the main game, the player delegates to <see cref="GreedyPlayer"/>.
/// </summary>
internal sealed class NnPlacementPlayer : IBenchmarkPlayer
{
    private readonly NnClient _client;
    private readonly PlacementActionSerializer _actionSerializer;
    private readonly GreedyPlayer _greedy = new();

    /// <summary>
    /// When a settlement stage selects a (settlement, road) pair, the chosen
    /// road edge index is stored here and consumed on the following road stage.
    /// </summary>
    private int _pendingRoadEdge = -1;

    public NnPlacementPlayer(NnClient client, PlacementActionSerializer actionSerializer)
    {
        _client = client;
        _actionSerializer = actionSerializer;
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var isPlacement = state.Stage is TurnStage.PlaceFirstSettlement
                                     or TurnStage.PlaceFirstRoad
                                     or TurnStage.PlaceSecondSettlement
                                     or TurnStage.PlaceSecondRoad;

        if (!isPlacement)
        {
            _pendingRoadEdge = -1;
            return _greedy.Act(state, rng);
        }

        var coreActions = state.Actions();
        if (coreActions.Length == 0) return null;

        // Forced action — no decision to make.
        if (coreActions.Length == 1)
        {
            return (CatanState)UnwrapCoreAction(coreActions[0]).DoCoreAction();
        }

        // Road stage: apply the road selected during the preceding settlement stage.
        if (state.Stage is TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad)
        {
            return ApplyPendingRoad(state, coreActions, rng);
        }

        // Settlement stage: enumerate all (settlement, road) composite actions.
        return ChoosePlacement(state, coreActions, rng);
    }

    /// <summary>
    /// Applies the road that was chosen during the preceding settlement stage.
    /// Falls back to random if the chosen edge is not among the legal actions.
    /// </summary>
    private CatanState? ApplyPendingRoad(CatanState state, CoreAction[] coreActions, Random rng)
    {
        if (_pendingRoadEdge >= 0)
        {
            for (var i = 0; i < coreActions.Length; i++)
            {
                var action = UnwrapCoreAction(coreActions[i]);
                if (action is PlaceRoadAction roadAction && roadAction.EdgeIndex == _pendingRoadEdge)
                {
                    _pendingRoadEdge = -1;
                    return (CatanState)roadAction.DoCoreAction();
                }
            }
        }

        // Fallback: pending road edge was not found (shouldn't happen), pick randomly.
        _pendingRoadEdge = -1;
        var roll = rng.Next(coreActions.Length);
        return (CatanState)UnwrapCoreAction(coreActions[roll]).DoCoreAction();
    }

    /// <summary>
    /// Enumerates all legal (settlement, road) composite actions, evaluates them
    /// via the placement NN model, and applies the best settlement. Stores the
    /// chosen road edge for the following road stage.
    /// </summary>
    private CatanState? ChoosePlacement(CatanState state, CoreAction[] coreActions, Random rng)
    {
        var placementState = state.SerializePlacementPhaseCompact();

        // Build all composite (settlement, road) pairs.
        var compositeActions = new List<(int SettlementActionIndex, int RoadEdge, string ActionString)>();

        for (var i = 0; i < coreActions.Length; i++)
        {
            var settlementAction = UnwrapCoreAction(coreActions[i]);
            if (settlementAction is not PlaceSettlementAction placeSettlement)
                continue;

            // Simulate placing this settlement to discover legal road actions.
            var afterSettlement = (CatanState)placeSettlement.DoCoreAction();
            var roadActions = afterSettlement.Actions();

            foreach (var roadCoreAction in roadActions)
            {
                var roadAction = UnwrapCoreAction(roadCoreAction);
                if (roadAction is not PlaceRoadAction placeRoad)
                    continue;

                var actionString = _actionSerializer.Serialize(
                    placeSettlement.VertexIndex, placeRoad.EdgeIndex);
                compositeActions.Add((i, placeRoad.EdgeIndex, actionString));
            }
        }

        if (compositeActions.Count == 0)
        {
            // Unexpected: no composite actions found. Fall back to random settlement.
            var roll = rng.Next(coreActions.Length);
            return (CatanState)UnwrapCoreAction(coreActions[roll]).DoCoreAction();
        }

        // Batch predict all (state, action) pairs.
        var states = new List<string>(compositeActions.Count);
        var actions = new List<string>(compositeActions.Count);
        foreach (var (_, _, actionString) in compositeActions)
        {
            states.Add(placementState);
            actions.Add(actionString);
        }

        var buckets = _client.PredictPlacementAsync(states, actions).GetAwaiter().GetResult();

        // Find the composite action with the highest expected win probability.
        var bestIndex = 0;
        var bestScore = float.NegativeInfinity;
        for (var j = 0; j < compositeActions.Count; j++)
        {
            var score = j < buckets.Length
                ? NnClient.ExpectedWinProbability(buckets[j])
                : 0f;
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = j;
            }
        }

        var (bestSettlementIdx, bestRoadEdge, _) = compositeActions[bestIndex];

        // Store the chosen road edge for the road stage.
        _pendingRoadEdge = bestRoadEdge;

        // Apply the settlement action.
        return (CatanState)UnwrapCoreAction(coreActions[bestSettlementIdx]).DoCoreAction();
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
