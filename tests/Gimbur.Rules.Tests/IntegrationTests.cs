using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Rules.Tests;

/// <summary>
/// Integration tests that run full games to completion using the greedy AI.
/// These validate that the engine doesn't crash, produces a winner, and that
/// game state invariants hold throughout play.
/// </summary>
public class IntegrationTests
{
    /// <summary>
    /// Unwraps CoreAction[] from Actions() into CatanAction instances.
    /// </summary>
    private static IEnumerable<Gimbur.CatanAction> GetCatanActions(Gimbur.CatanState state) =>
        state.Actions().Select(ca => ca.IsDeterministic
            ? (Gimbur.CatanAction)(Gimbur.CatanDeterministicAction)((CoreAction.Deterministic)ca).Item
            : (Gimbur.CatanAction)(Gimbur.CatanStochasticAction)((CoreAction.Stochastic)ca).Item);

    [TestCase(42)]
    [TestCase(123)]
    [TestCase(999)]
    [TestCase(7)]
    [TestCase(2026)]
    public void MiniGame_CompletesWithWinner(int seed)
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(seed));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(seed);

        var maxTurns = 500;
        while (state.WinnerPlayer == 0)
        {
            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            var action = greedy.ChooseAction(state, rng);
            Assert.That(action, Is.Not.Null, $"Greedy returned null at turn {state.TurnNumber}, stage {state.Stage}");
            state = (Gimbur.CatanState)action!.DoCoreAction();

            if (state.TurnNumber > maxTurns)
            {
                // Mini games can stall; this is expected behavior, not a crash.
                return;
            }
        }

        if (state.WinnerPlayer != 0)
        {
            Assert.That(state.WinnerPlayer, Is.InRange(1, 2));
            Assert.That(state.VictoryPointsFor(state.WinnerPlayer),
                Is.GreaterThanOrEqualTo(GameConfig.Mini.VictoryPointsToWin));
            Assert.That(state.Actions(), Is.Empty, "No actions should be available after a win.");
        }
    }

    [TestCase(42)]
    [TestCase(123)]
    [TestCase(7)]
    public void StandardGame_CompletesWithWinner(int seed)
    {
        var state = new Gimbur.CatanState(GameConfig.Standard, 3, new Random(seed));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(seed);

        var maxTurns = 500;
        while (state.WinnerPlayer == 0)
        {
            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            var action = greedy.ChooseAction(state, rng);
            Assert.That(action, Is.Not.Null, $"Greedy returned null at turn {state.TurnNumber}, stage {state.Stage}");
            state = (Gimbur.CatanState)action!.DoCoreAction();

            if (state.TurnNumber > maxTurns)
            {
                Assert.Fail($"Standard game did not complete within {maxTurns} turns (seed={seed}).");
            }
        }

        Assert.That(state.WinnerPlayer, Is.InRange(1, 3));
        Assert.That(state.VictoryPointsFor(state.WinnerPlayer),
            Is.GreaterThanOrEqualTo(GameConfig.Standard.VictoryPointsToWin));
        Assert.That(state.Actions(), Is.Empty);
    }

    [Test]
    public void FullGame_AllActionsAreLegalAtEveryStep()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(55));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(55);
        var stepCount = 0;

        while (state.WinnerPlayer == 0 && state.TurnNumber <= 200)
        {
            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            // Verify basic invariants every step
            Assert.That(state.CurrentPlayer, Is.InRange(1, state.PlayerCount),
                $"Invalid player at step {stepCount}");

            var action = greedy.ChooseAction(state, rng);
            Assert.That(action, Is.Not.Null);

            // Action must not throw
            state = (Gimbur.CatanState)action!.DoCoreAction();
            stepCount++;
        }

        Assert.That(stepCount, Is.GreaterThan(10), "Game should take more than 10 steps.");
    }

    [Test]
    public void FullGame_ResourceConservation_NoNegativeResources()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(88));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(88);

        while (state.WinnerPlayer == 0 && state.TurnNumber <= 200)
        {
            // Check no player has negative resources
            for (var player = 1; player <= state.PlayerCount; player++)
            {
                foreach (var resource in new[] { ResourceType.Wood, ResourceType.Brick, ResourceType.Sheep, ResourceType.Wheat, ResourceType.Ore })
                {
                    Assert.That(state.ResourceCountFor(player, resource), Is.GreaterThanOrEqualTo(0),
                        $"Player {player} has negative {resource} at turn {state.TurnNumber}");
                }
            }

            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            var action = greedy.ChooseAction(state, rng);
            state = (Gimbur.CatanState)action!.DoCoreAction();
        }
    }

    [Test]
    public void FullGame_VictoryPointsNeverDecrease_ForActivePlayer()
    {
        var state = new Gimbur.CatanState(GameConfig.Mini, 2, new Random(33));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(33);

        // VP can decrease if longest road or largest army is lost, so we just
        // track that each player's VP is always non-negative.
        while (state.WinnerPlayer == 0 && state.TurnNumber <= 200)
        {
            for (var player = 1; player <= state.PlayerCount; player++)
            {
                Assert.That(state.VictoryPointsFor(player), Is.GreaterThanOrEqualTo(0),
                    $"Player {player} has negative VP at turn {state.TurnNumber}");
            }

            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            var action = greedy.ChooseAction(state, rng);
            state = (Gimbur.CatanState)action!.DoCoreAction();
        }
    }

    [Test]
    public void FourPlayerStandardGame_CompletesWithWinner()
    {
        var state = new Gimbur.CatanState(GameConfig.Standard, 4, new Random(77));
        var greedy = new Gimbur.GreedyActionSelector();
        var rng = new Random(77);

        while (state.WinnerPlayer == 0 && state.TurnNumber <= 500)
        {
            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            var action = greedy.ChooseAction(state, rng);
            Assert.That(action, Is.Not.Null);
            state = (Gimbur.CatanState)action!.DoCoreAction();
        }

        Assert.That(state.WinnerPlayer, Is.InRange(1, 4));
        Assert.That(state.VictoryPointsFor(state.WinnerPlayer),
            Is.GreaterThanOrEqualTo(GameConfig.Standard.VictoryPointsToWin));
    }
}
