using Gimbur.Rules;

namespace Gimbur.Cli;

/// <summary>
/// Hybrid benchmark player that delegates placement-phase decisions to
/// the Gimbur.Server (running either <c>mcts-ai</c> or <c>mcts-nn-ai</c>
/// mode) and falls back to a configurable simpler player for the main
/// game.
///
/// Used by the placement-only AI variants (<c>mcts-placement</c>,
/// <c>mcts-placement-random</c>, <c>nn-mcts-placement</c>,
/// <c>nn-mcts-placement-random</c>) so that the post-placement strategy
/// matches the opponent and the comparison isolates placement quality.
/// </summary>
internal sealed class ServerPlacementPlayer : IBenchmarkPlayer, IPriorStatsProvider, IDisposable
{
    private readonly ServerPlayer _placement;
    private readonly IBenchmarkPlayer _fallback;

    public int TotalNnRequests => _placement.TotalNnRequests;
    public int TotalNnStatesEvaluated => _placement.TotalNnStatesEvaluated;
    public int TotalPriorActionsApplied => _placement.TotalPriorActionsApplied;
    public int TotalPriorActionsRequested => _placement.TotalPriorActionsRequested;
    public int TotalPriorInferencesRequested => _placement.TotalPriorInferencesRequested;

    public ServerPlacementPlayer(ServerPlayer placement, IBenchmarkPlayer? fallback = null)
    {
        _placement = placement;
        _fallback = fallback ?? new GreedyPlayer();
    }

    public CatanState? Act(CatanState state, Random rng)
    {
        var isPlacement = state.Stage is TurnStage.PlaceFirstSettlement
                                     or TurnStage.PlaceFirstRoad
                                     or TurnStage.PlaceSecondSettlement
                                     or TurnStage.PlaceSecondRoad;

        return isPlacement
            ? _placement.Act(state, rng)
            : _fallback.Act(state, rng);
    }

    public void Dispose()
    {
        _placement.Dispose();
        if (_fallback is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
