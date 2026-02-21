using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

public class GreedyActionSelectorTests
{
    [Test]
    public void ChooseAction_ReturnsLegalAction()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(11));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(7);

        var action = greedy.ChooseAction(state, rng);
        Assert.That(action, Is.Not.Null);
        Assert.That(state.Actions().Cast<Gimbur.CatanAction>(), Has.Member(action));
    }

    [Test]
    public void CanPlayMultipleGreedyStepsWithoutInvalidAction()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(11));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(13);

        for (var i = 0; i < 80; i++)
        {
            var actions = state.Actions().Cast<Gimbur.CatanAction>().ToArray();
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

            var options = state.Actions().Cast<Gimbur.CatanAction>().ToArray();
            var hasThreeTileOption = options.Any(a => state.Board.Topology.VertexTiles[a.Arg1].Length >= 3);
            var hasSingleTileOption = options.Any(a => state.Board.Topology.VertexTiles[a.Arg1].Length == 1);
            if (!hasThreeTileOption || !hasSingleTileOption)
            {
                continue;
            }

            var pick = greedy.ChooseAction(state, rng);
            Assert.That(pick, Is.Not.Null);
            Assert.That(pick!.ActionType, Is.EqualTo(Gimbur.CatanActionType.PlaceSettlement));
            Assert.That(state.Board.Topology.VertexTiles[pick.Arg1].Length, Is.GreaterThanOrEqualTo(2));
            return;
        }

        Assert.Fail("Did not find a suitable seed with both 1-tile and 3-tile initial settlement options.");
    }
}
