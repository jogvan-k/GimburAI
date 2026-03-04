using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Rules.Tests;

public class GreedyActionSelectorTests
{
    /// <summary>
    /// Unwraps CoreAction[] from Actions() into CatanAction instances.
    /// </summary>
    private static IEnumerable<Gimbur.CatanAction> GetCatanActions(Gimbur.CatanState state) =>
        state.Actions().Select(ca => ca.IsDeterministic
            ? (Gimbur.CatanAction)(Gimbur.CatanDeterministicAction)((CoreAction.Deterministic)ca).Item
            : (Gimbur.CatanAction)(Gimbur.CatanStochasticAction)((CoreAction.Stochastic)ca).Item);

    [Test]
    public void ChooseAction_ReturnsLegalAction()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(11));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(7);

        var action = greedy.ChooseAction(state, rng);
        Assert.That(action, Is.Not.Null);
        Assert.That(GetCatanActions(state), Has.Member(action));
    }

    [Test]
    public void CanPlayMultipleGreedyStepsWithoutInvalidAction()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(11));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(13);

        for (var i = 0; i < 80; i++)
        {
            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            var action = greedy.ChooseAction(state, rng);
            Assert.That(action, Is.Not.Null);
            Assert.That(actions, Has.Member(action));
            state = (Gimbur.CatanState)action.DoCoreAction();
        }
    }

    [Test]
    public void InitialSettlement_AvoidsSingleTileWhenBetterThreeTileOptionsExist()
    {
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(17);

        for (var seed = 1; seed <= 100; seed++)
        {
            var state = new Gimbur.CatanState(GameConfig.Standard, 3, new Random(seed));
            if (state.Stage != TurnStage.PlaceFirstSettlement)
            {
                continue;
            }

            var options = GetCatanActions(state).ToArray();
            var hasThreeTileOption = options.Any(a => state.Board.Topology.VertexTiles[a.Arg1].Length >= 3);
            var hasSingleTileOption = options.Any(a => state.Board.Topology.VertexTiles[a.Arg1].Length == 1);
            if (!hasThreeTileOption || !hasSingleTileOption)
            {
                continue;
            }

            var pick = greedy.ChooseAction(state, rng);
            Assert.That(pick, Is.Not.Null);
            Assert.That(pick, Is.InstanceOf<Gimbur.PlaceSettlementAction>());
            Assert.That(state.Board.Topology.VertexTiles[pick.Arg1].Length, Is.GreaterThanOrEqualTo(2));
            return;
        }

        Assert.Fail("Did not find a suitable seed with both 1-tile and 3-tile initial settlement options.");
    }
}
