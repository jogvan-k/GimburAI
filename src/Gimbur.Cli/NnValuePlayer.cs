using Gimbur;
using Kjarni;

namespace Gimbur.Cli;

/// <summary>Ranks legal actions by expected successor-state value.</summary>
internal sealed class NnValuePlayer : IBenchmarkPlayer, INnStatsProvider
{
    private readonly NnClient _client;

    public NnValuePlayer(NnClient client)
    {
        _client = client;
    }

    public int TotalNnRequests { get; private set; }
    public int TotalNnStatesEvaluated { get; private set; }

    public CatanState? Act(CatanState state, Random rng)
    {
        var coreActions = state.Actions();
        if (coreActions.Length == 0)
            return null;
        if (coreActions.Length == 1)
            return (CatanState)Unwrap(coreActions[0]).DoCoreAction();

        var actingPlayer = state.CurrentPlayer;
        var states = new List<CatanState>();
        var actionOutcomes = new List<(int Weight, CatanState State)[]>(coreActions.Length);
        var descriptors = new List<ActionDescriptor>(coreActions.Length);
        for (var actionIndex = 0; actionIndex < coreActions.Length; actionIndex++)
        {
            var action = Unwrap(coreActions[actionIndex]);
            var outcomes = action is CatanStochasticAction stochastic
                ? stochastic.Outcomes()
                : [Tuple.Create(1, action.DoCoreAction())];
            actionOutcomes.Add(outcomes.Select(outcome => (
                outcome.Item1, (CatanState)outcome.Item2)).ToArray());
            var entries = new OutcomeDescriptor[outcomes.Length];
            for (var outcomeIndex = 0; outcomeIndex < outcomes.Length; outcomeIndex++)
            {
                var outcome = (CatanState)outcomes[outcomeIndex].Item2;
                if (outcome.WinnerPlayer != 0)
                {
                    entries[outcomeIndex] = new OutcomeDescriptor(
                        outcomes[outcomeIndex].Item1,
                        StateIndex: -1,
                        ExactValue: outcome.WinnerPlayer == actingPlayer ? 1.0f : 0.0f,
                        SuccessorPlayer: outcome.CurrentPlayer);
                }
                else
                {
                    entries[outcomeIndex] = new OutcomeDescriptor(
                        outcomes[outcomeIndex].Item1,
                        states.Count,
                        ExactValue: null,
                        SuccessorPlayer: outcome.CurrentPlayer);
                    states.Add(outcome);
                }
            }
            descriptors.Add(new ActionDescriptor(actionIndex, entries));
        }

        var predictions = states.Count > 0
            ? _client.PredictAsync(states.Select(state => state.SerializeCompact()).ToArray())
                .GetAwaiter().GetResult()
            : [];
        if (states.Count > 0)
        {
            TotalNnRequests++;
            TotalNnStatesEvaluated += states.Count;
        }

        var bestActions = new List<int>();
        var bestValue = float.NegativeInfinity;
        foreach (var descriptor in descriptors)
        {
            var weightedValue = 0.0f;
            var totalWeight = 0;
            foreach (var outcome in descriptor.Outcomes)
            {
                var value = outcome.ExactValue ?? predictions[outcome.StateIndex][
                    CanonicalPlayerSlot(actingPlayer, outcome.SuccessorPlayer, state.PlayerCount)];
                weightedValue += outcome.Weight * value;
                totalWeight += outcome.Weight;
            }
            var expectedValue = totalWeight > 0 ? weightedValue / totalWeight : 0.0f;
            if (expectedValue > bestValue)
            {
                bestValue = expectedValue;
                bestActions.Clear();
                bestActions.Add(descriptor.ActionIndex);
            }
            else if (Math.Abs(expectedValue - bestValue) < 1e-6f)
            {
                bestActions.Add(descriptor.ActionIndex);
            }
        }

        var bestAction = bestActions[rng.Next(bestActions.Count)];
        var outcomesToSample = actionOutcomes[bestAction];
        var roll = rng.Next(outcomesToSample.Sum(outcome => outcome.Weight));
        var cumulative = 0;
        foreach (var outcome in outcomesToSample)
        {
            cumulative += outcome.Weight;
            if (roll < cumulative)
                return outcome.State;
        }
        return outcomesToSample[^1].State;
    }

    internal static int CanonicalPlayerSlot(int absolutePlayer, int successorPlayer, int playerCount) =>
        (absolutePlayer - successorPlayer + playerCount) % playerCount;

    private static CatanAction Unwrap(CoreAction action) => action.IsDeterministic
        ? (CatanDeterministicAction)((CoreAction.Deterministic)action).Item
        : (CatanStochasticAction)((CoreAction.Stochastic)action).Item;

    private readonly record struct ActionDescriptor(int ActionIndex, OutcomeDescriptor[] Outcomes);
    private readonly record struct OutcomeDescriptor(
        int Weight,
        int StateIndex,
        float? ExactValue,
        int SuccessorPlayer);
}
