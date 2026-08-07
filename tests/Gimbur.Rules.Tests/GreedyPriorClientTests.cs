using Gimbur;
using Kjarni;

namespace Gimbur.Rules.Tests;

[TestFixture]
internal sealed class GreedyPriorClientTests
{
    [Test]
    public void RequestPrior_ReturnsOneHotGreedyActionInFlattenedStateLayout()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        const long nodeId = 123;
        var actions = state.Actions().Select(Unwrap).ToArray();
        var greedy = new GreedyActionSelector().ChooseAction(
            state, new Random(HashCode.Combine(state.GetHashCode(), nodeId)))!;
        var flattenedStates = actions.SelectMany(ActionStates).ToArray();
        var client = new GreedyPriorClient(uniformMix: 0.25);

        var count = client.RequestPrior(nodeId, state, flattenedStates, state.CurrentPlayer, 0);
        var responses = client.CollectPriors(new HashSet<long> { nodeId });

        Assert.That(count, Is.EqualTo(flattenedStates.Length));
        Assert.That(responses, Has.Length.EqualTo(1));
        Assert.That(responses[0].ValueEstimates, Is.Empty);

        var offset = 0;
        foreach (var action in actions)
        {
            var length = ActionStates(action).Length;
            var expected = action.Equals(greedy)
                ? 0.75 + 0.25 / actions.Length
                : 0.25 / actions.Length;
            Assert.That(responses[0].Priors[offset..(offset + length)],
                Is.All.EqualTo(expected));
            offset += length;
        }
        var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
        Assert.That(responses[0].DensePriors, Has.Length.EqualTo(serializer.PolicySize));
        Assert.That(responses[0].DensePriors[serializer.IndexOf(state, greedy)], Is.EqualTo(1.0));
        Assert.That(responses[0].DensePriors.Count(value => value == 1.0), Is.EqualTo(1));
        Assert.That(responses[0].DensePriors.Count(value => value == 0.0),
            Is.EqualTo(serializer.PolicySize - 1));
    }

    [Test]
    public void CollectAndFlush_OnlyRemoveKnownNodes()
    {
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var states = state.Actions().SelectMany(action => ActionStates(Unwrap(action))).ToArray();
        var client = new GreedyPriorClient();
        client.RequestPrior(1, state, states, 1, 0);
        client.RequestPrior(2, state, states, 1, 0);

        client.Flush(new HashSet<long> { 1 });

        Assert.That(client.CollectPriors(new HashSet<long> { 1 }), Is.Empty);
        Assert.That(client.CollectPriors(new HashSet<long> { 2 }), Has.Length.EqualTo(1));
    }

    private static ICoreState[] ActionStates(CatanAction action) => action switch
    {
        CatanStochasticAction stochastic => stochastic.Outcomes().Select(outcome => outcome.Item2).ToArray(),
        _ => [action.DoCoreAction()],
    };

    private static CatanAction Unwrap(CoreAction action) => action.IsDeterministic
        ? (CatanDeterministicAction)((CoreAction.Deterministic)action).Item
        : (CatanStochasticAction)((CoreAction.Stochastic)action).Item;
}
