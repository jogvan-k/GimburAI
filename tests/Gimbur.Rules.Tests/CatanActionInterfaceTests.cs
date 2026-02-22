using Kjarni;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class CatanActionInterfaceTests
{
    [Test]
    public void ConcreteCatanActions_ImplementExactlyOneActionKindInterface()
    {
        var actionTypes = typeof(Gimbur.CatanAction).Assembly
            .GetTypes()
            .Where(t => typeof(Gimbur.CatanAction).IsAssignableFrom(t))
            .Where(t => !t.IsAbstract)
            .ToArray();

        Assert.That(actionTypes.Length, Is.GreaterThan(0));

        foreach (var actionType in actionTypes)
        {
            var deterministic = typeof(IDeterministicCoreAction).IsAssignableFrom(actionType);
            var stochastic = typeof(IStochasticCoreAction).IsAssignableFrom(actionType);
            Assert.That(deterministic ^ stochastic, Is.True, $"{actionType.Name} must implement exactly one action kind interface.");
        }
    }

    [Test]
    public void RollDiceAndBuyDevCard_AreStochasticOnly()
    {
        Assert.That(typeof(IStochasticCoreAction).IsAssignableFrom(typeof(Gimbur.RollDiceAction)), Is.True);
        Assert.That(typeof(IDeterministicCoreAction).IsAssignableFrom(typeof(Gimbur.RollDiceAction)), Is.False);

        Assert.That(typeof(IStochasticCoreAction).IsAssignableFrom(typeof(Gimbur.BuyDevCardAction)), Is.True);
        Assert.That(typeof(IDeterministicCoreAction).IsAssignableFrom(typeof(Gimbur.BuyDevCardAction)), Is.False);

        Assert.That(typeof(IStochasticCoreAction).IsAssignableFrom(typeof(Gimbur.ChooseRobberTileAction)), Is.True);
        Assert.That(typeof(IDeterministicCoreAction).IsAssignableFrom(typeof(Gimbur.ChooseRobberTileAction)), Is.False);

        Assert.That(typeof(IStochasticCoreAction).IsAssignableFrom(typeof(Gimbur.ChooseRobberVictimAction)), Is.True);
        Assert.That(typeof(IDeterministicCoreAction).IsAssignableFrom(typeof(Gimbur.ChooseRobberVictimAction)), Is.False);

        Assert.That(typeof(IStochasticCoreAction).IsAssignableFrom(typeof(Gimbur.PlayKnightAction)), Is.True);
        Assert.That(typeof(IDeterministicCoreAction).IsAssignableFrom(typeof(Gimbur.PlayKnightAction)), Is.False);
    }
}
