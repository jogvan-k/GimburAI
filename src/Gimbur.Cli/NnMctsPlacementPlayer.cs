using Gimbur.Rules;

namespace Gimbur.Cli;

/// <summary>
/// Hybrid benchmark player that uses NN-guided MCTS (via the Gimbur.Server
/// <c>mcts-nn-ai</c> mode) for the initial placement phase, then falls back
/// to a configurable simpler player for the main game.
///
/// Mirrors <see cref="NnPlacementPlayer"/> but routes placement decisions
/// through MCTS+NN-priors instead of taking the NN's argmax directly.
/// Useful for measuring the value of MCTS search on top of the NN policy
/// during placement.
/// </summary>
internal sealed class NnMctsPlacementPlayer : IBenchmarkPlayer, IDisposable
{
    private readonly ServerPlayer _placement;
    private readonly IBenchmarkPlayer _fallback;

    public NnMctsPlacementPlayer(ServerPlayer placement, IBenchmarkPlayer? fallback = null)
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
