using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using Gimbur.Rules;
using Gimbur;
using Kjarni;

namespace Gimbur.Tui;

internal static class Program
{
    /// <summary>
    /// Unwraps CoreAction[] from Actions() into CatanAction instances.
    /// </summary>
    private static IEnumerable<CatanAction> GetCatanActions(CatanState state) =>
        state.Actions().Select(ca => ca.IsDeterministic
            ? (CatanAction)(CatanDeterministicAction)((CoreAction.Deterministic)ca).Item
            : (CatanAction)(CatanStochasticAction)((CoreAction.Stochastic)ca).Item);

    private const double Sqrt3 = 1.7320508075688772;

    private const string Reset = "\u001b[0m";
    private const string FgWhite = "\u001b[37m";
    private const string FgBrightWhite = "\u001b[97m";
    private const string FgBrightBlack = "\u001b[90m";
    private const string FgRed = "\u001b[31m";
    private const string FgBrightRed = "\u001b[91m";
    private const string FgYellow = "\u001b[33m";
    private const string FgBrightGreen = "\u001b[92m";
    private const string FgGreen = "\u001b[32m";
    private const string FgLightGreen = "\u001b[38;5;154m";
    private const string FgSilver = "\u001b[38;5;250m";
    private const string FgBeige = "\u001b[38;5;223m";
    private const string FgCyan = "\u001b[36m";
    private const int MaxVisibleLogEntries = 6;
    private static int _lastFrameLineCount;
    private static readonly Random UiRng = new();
    private static string _statusMessage = "";
    private static readonly List<LoggedAction> _actionLog = new();
    private static int[] _lastSeenLogIndexByPlayer = [];
    private static readonly StringBuilder _frameBuffer = new();
    private static NnClient? _nnClient;
    private static PlacementActionSerializer? _actionSerializer;

    private static readonly Regex AnsiEscapePattern = new(
        @"\u001b\[[0-9;]*m",
        RegexOptions.Compiled);

    private static int VisibleLength(string text) =>
        AnsiEscapePattern.Replace(text, "").Length;

    private static void Main(string[] args)
    {
        Console.WriteLine("Gimbur TUI");
        Console.WriteLine();

        var mapChoice = PromptMapTopology();
        var config = mapChoice switch
        {
            MapChoice.Mini => GameConfig.Mini,
            MapChoice.Small => GameConfig.Small,
            _ => GameConfig.Standard,
        };
        var players = PromptPlayerCount(config.MinPlayers, config.MaxPlayers);
        var controllers = PromptPlayerControllers(players);

        // If any player uses NN or NN-Value, set up the inference client.
        if (controllers.Any(c => c is PlayerController.NN or PlayerController.NNValue))
        {
            // Check for --nn-url command-line argument first.
            string? nnUrl = null;
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "--nn-url")
                {
                    nnUrl = args[i + 1];
                    break;
                }
            }

            nnUrl ??= PromptNnUrl();
            _nnClient = new NnClient(nnUrl);
            _actionSerializer = PlacementActionSerializer.ForTopology(config.Map.Topology);

            if (!_nnClient.IsHealthyAsync().GetAwaiter().GetResult())
            {
                Console.WriteLine($"Warning: NN server at {nnUrl} is not reachable.");
                Console.Write("Continue anyway? (y/n): ");
                var answer = Console.ReadLine()?.Trim().ToLowerInvariant();
                if (answer is not ("y" or "yes"))
                {
                    _nnClient.Dispose();
                    return;
                }
            }
        }

        var rng = new Random();
        var state = new CatanState(config, players, rng);

        RunGameLoop(state, controllers);
        _nnClient?.Dispose();
    }

    private const string DefaultNnUrl = "http://localhost:8000";

    private static string PromptNnUrl()
    {
        Console.Write($"NN server URL [{DefaultNnUrl}]: ");
        var input = Console.ReadLine()?.Trim();
        return string.IsNullOrEmpty(input) ? DefaultNnUrl : input;
    }

    private static void RunGameLoop(CatanState state, PlayerController[] controllers)
    {
        Console.Clear();
        _lastFrameLineCount = 0;
        Console.CursorVisible = false;
        _actionLog.Clear();
        _lastSeenLogIndexByPlayer = new int[state.PlayerCount + 1];
        var greedy = new GreedyActionSelector();

        while (true)
        {
            var actions = GetCatanActions(state).ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            if (controllers[state.CurrentPlayer] == PlayerController.Greedy)
            {
                state = ExecuteGreedyStep(state, greedy);
                continue;
            }

            if (controllers[state.CurrentPlayer] == PlayerController.MCTS)
            {
                state = ExecuteMctsStep(state);
                continue;
            }

            if (controllers[state.CurrentPlayer] == PlayerController.NN)
            {
                state = ExecuteNnStep(state, usePlacementModel: true);
                continue;
            }

            if (controllers[state.CurrentPlayer] == PlayerController.NNValue)
            {
                state = ExecuteNnStep(state, usePlacementModel: false);
                continue;
            }

            if (state.Stage is TurnStage.PlaceFirstSettlement
                or TurnStage.PlaceSecondSettlement
                or TurnStage.PlaceSettlementCommitted)
            {
                state = ExecuteSettlementPlacement(state);
                continue;
            }

            if (state.Stage is TurnStage.PlaceFirstRoad
                or TurnStage.PlaceSecondRoad
                or TurnStage.PlaceRoadBuildingFirst
                or TurnStage.PlaceRoadCommitted)
            {
                state = ExecuteRoadPlacement(state);
                continue;
            }

            if (state.Stage == TurnStage.PlaceCityCommitted)
            {
                state = ExecuteCityPlacement(state);
                continue;
            }

            if (state.Stage == TurnStage.ChooseRobberLocation)
            {
                state = ExecuteRobberPlacement(state);
                continue;
            }

            if (state.Stage == TurnStage.ChooseRobberVictim)
            {
                state = ExecuteRobberVictimSelection(state);
                continue;
            }

            state = ExecuteActionMenu(state, actions);
        }

        DrawFrame(() =>
        {
            RenderBoard(state.Board);
            WriteFixedLine();
            WriteFixedLine("Game finished.");
            WriteFixedLine($"Winner: Player {state.WinnerPlayer}");
            WriteFixedLine();
            for (var player = 1; player <= state.PlayerCount; player++)
            {
                WriteFixedLine(
                    $"Player {player}: VP={state.VictoryPointsFor(player)} settlements={state.Board.SettlementCount(player)} cities={state.Board.CityCount(player)} roads={state.Board.RoadCount(player)}");
            }
            WriteFixedLine();
            WriteFixedLine("Legend: tile text stack = resource name, number, robber(*); o empty vertex, s settlement, c city, |/\\/- road");
            WriteFixedLine("Ports: 3:1 generic plus Wood/Brick/Sheep/Wheat/Ore resource ports");
        });

        Console.CursorVisible = true;
    }

    private static CatanState ExecuteGreedyStep(CatanState state, GreedyActionSelector greedy)
    {
        var action = greedy.ChooseAction(state, UiRng);
        if (action is null)
        {
            return state;
        }

        return ApplyActionAndLog(state, action, aiControlled: true);
    }

    /// <summary>
    /// Runs a single MCTS simulation with a 5-second budget, then executes
    /// the entire best action sequence without re-running the search.
    /// </summary>
    private static CatanState ExecuteMctsStep(CatanState state)
    {
        var current = state;
        var startingPlayer = state.CurrentPlayer;

        // Apply forced actions (single-action states like dice rolls or
        // end-turn) immediately — no point running a 5-second search when
        // there is no decision to make.
        while (current.WinnerPlayer == 0 && current.CurrentPlayer == startingPlayer)
        {
            var forced = current.Actions();
            if (forced.Length != 1) break;

            var catanAction = UnwrapCoreAction(forced[0]);
            current = ApplyActionAndLog(current, catanAction, aiControlled: true);
        }

        // If the turn ended (player changed, game over, or no actions),
        // return without running MCTS.
        if (current.WinnerPlayer != 0
            || current.CurrentPlayer != startingPlayer
            || current.Actions().Length == 0)
        {
            return current;
        }

        // We have a real decision — run MCTS.
        var config = new Kjarni.MCTSConfig(
            searchTime.NewSeconds(5),
            System.Int32.MaxValue,
            500,
            System.Math.Sqrt(2.0),
            System.Int32.MaxValue,
            null,
            null,
            null,
            System.Int32.MaxValue,
            32,
            500,
            1000);
        var mcts = new Kjarni.MCTS.AI.MonteCarloTreeSearch(config);

        var mctsRoot = new Kjarni.MCTS.Types.MCTSState((ICoreState)current);
        mcts.RunSimulation(mctsRoot);

        var bestPath = Kjarni.MCTS.Algorithm.extractBestPath(mctsRoot);

        // Follow the best path, applying each action in sequence.
        // Stop when the acting player changes so we don't consume
        // another player's turn.
        foreach (var actionIndex in bestPath)
        {
            if (current.CurrentPlayer != startingPlayer) break;

            var actions = current.Actions();
            if (actionIndex >= actions.Length) break;

            var catanAction = UnwrapCoreAction(actions[actionIndex]);
            current = ApplyActionAndLog(current, catanAction, aiControlled: true);
        }

        // Continue applying any remaining forced actions on this player's turn.
        while (current.WinnerPlayer == 0 && current.CurrentPlayer == startingPlayer)
        {
            var actions = current.Actions();
            if (actions.Length != 1) break;

            var catanAction = UnwrapCoreAction(actions[0]);
            current = ApplyActionAndLog(current, catanAction, aiControlled: true);
        }

        return current;
    }

    /// <summary>
    /// Runs a single NN AI step.  Forced single-action states are applied
    /// immediately.  When a real decision is needed, actions are evaluated
    /// via the NN inference server and the best action is chosen.
    ///
    /// When <paramref name="usePlacementModel"/> is true, the placement
    /// model is used during initial placement stages and the value model
    /// for the rest of the game.  When false, the value model is used for
    /// all stages (greedy fallback during placement since the value model
    /// doesn't understand placement states).
    /// </summary>
    private static CatanState ExecuteNnStep(CatanState state, bool usePlacementModel)
    {
        var current = state;
        var startingPlayer = state.CurrentPlayer;

        // Apply forced actions immediately.
        while (current.WinnerPlayer == 0 && current.CurrentPlayer == startingPlayer)
        {
            var forced = current.Actions();
            if (forced.Length != 1) break;

            var catanAction = UnwrapCoreAction(forced[0]);
            current = ApplyActionAndLog(current, catanAction, aiControlled: true);
        }

        if (current.WinnerPlayer != 0
            || current.CurrentPlayer != startingPlayer
            || current.Actions().Length == 0)
        {
            return current;
        }

        // Check if we're in a placement stage.
        var isPlacement = current.Stage is TurnStage.PlaceFirstSettlement
                                        or TurnStage.PlaceFirstRoad
                                        or TurnStage.PlaceSecondSettlement
                                        or TurnStage.PlaceSecondRoad;

        if (isPlacement && usePlacementModel)
        {
            current = ExecuteNnPolicyStep(current);
        }
        else if (isPlacement)
        {
            // NNValue mode: use greedy during placement.
            var greedy = new GreedyActionSelector();
            while (current.WinnerPlayer == 0
                   && current.CurrentPlayer == startingPlayer
                   && current.Stage is TurnStage.PlaceFirstSettlement
                                    or TurnStage.PlaceFirstRoad
                                    or TurnStage.PlaceSecondSettlement
                                    or TurnStage.PlaceSecondRoad)
            {
                var action = greedy.ChooseAction(current, UiRng);
                if (action is null) break;
                current = ApplyActionAndLog(current, action, aiControlled: true);
            }
        }
        else
        {
            current = ExecuteNnPolicyStep(current);
        }

        // Continue applying forced actions on this player's turn.
        while (current.WinnerPlayer == 0 && current.CurrentPlayer == startingPlayer)
        {
            var actions = current.Actions();
            if (actions.Length != 1) break;

            var catanAction = UnwrapCoreAction(actions[0]);
            current = ApplyActionAndLog(current, catanAction, aiControlled: true);
        }

        // If the NN player still has a decision to make, recurse.
        if (current.WinnerPlayer == 0
            && current.CurrentPlayer == startingPlayer
            && current.Actions().Length > 1)
        {
            return ExecuteNnStep(current, usePlacementModel);
        }

        return current;
    }

    /// <summary>
    /// Handles one decision using the complete policy-value model.
    /// </summary>
    private static CatanState ExecuteNnPolicyStep(CatanState state)
    {
        var coreActions = state.Actions();
        if (coreActions.Length <= 1)
        {
            return coreActions.Length == 1
                ? ApplyActionAndLog(state, UnwrapCoreAction(coreActions[0]), aiControlled: true)
                : state;
        }

        var actions = coreActions.Select(UnwrapCoreAction).ToArray();
        var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
        var legalIndices = actions.Select(action => serializer.IndexOf(state, action)).ToArray();
        var prediction = _nnClient!.PredictPolicyValueAsync([state.SerializeCompact()])
            .GetAwaiter().GetResult();
        var densePolicy = prediction.PolicyProbabilities.Length == 1
            ? Array.ConvertAll(prediction.PolicyProbabilities[0], value => (double)value)
            : [];
        var legalPolicy = serializer.MaskAndNormalize(densePolicy, legalIndices);

        var bestIndex = 0;
        var bestScore = float.NegativeInfinity;
        for (var j = 0; j < actions.Length; j++)
        {
            var score = (float)legalPolicy[j];
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = j;
            }
        }

        return ApplyActionAndLog(state, actions[bestIndex], aiControlled: true);
    }

    private static CatanAction UnwrapCoreAction(CoreAction coreAction)
    {
        if (coreAction.IsDeterministic)
            return (CatanDeterministicAction)((CoreAction.Deterministic)coreAction).Item;
        if (coreAction.IsStochastic)
            return (CatanStochasticAction)((CoreAction.Stochastic)coreAction).Item;
        throw new InvalidOperationException($"Unknown CoreAction tag: {coreAction.Tag}");
    }

    private static CatanState ExecuteSettlementPlacement(CatanState state)
    {
        var actions = GetCatanActions(state)
            .Where(a => a is PlaceSettlementAction)
            .ToArray();

        var legal = actions.Select(a => a.TargetIndex).ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal settlement placements available.");
            return state;
        }

        // Compute per-candidate NN win rates when the inference server is connected.
        var candidateWinRates = ComputeCandidateWinRates(state, actions);

        var selected = SelectLocation(
            title: "Select settlement location",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            playerLines: BuildPlayerLines(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).VertexPoints,
            neighborProvider: vertex => state.Board.Topology.VertexNeighbors[vertex],
            mode: LocationSelectionMode.Vertex,
            candidateWinRates: candidateWinRates);

        if (selected is int vertex)
        {
            var action = new PlaceSettlementAction(state, vertex);
            return ApplyActionAndLog(state, action);
        }

        return state;
    }

    private static CatanState ExecuteRobberPlacement(CatanState state)
    {
        var actions = GetCatanActions(state)
            .Where(a => a is ChooseRobberTileAction)
            .ToArray();

        var legal = actions.Select(a => a.Arg1).ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal robber placements available.");
            return state;
        }

        // Compute per-candidate NN win rates when the inference server is connected.
        var candidateWinRates = ComputeCandidateWinRates(state, actions);

        var selected = SelectLocation(
            title: "Select robber destination tile",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            playerLines: BuildPlayerLines(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).TilePoints,
            neighborProvider: tile => state.Board.Topology.TileNeighbors[tile],
            mode: LocationSelectionMode.Tile,
            candidateWinRates: candidateWinRates);

        if (selected is int tile)
        {
            var action = new ChooseRobberTileAction(state, tile);
            return ApplyActionAndLog(state, action);
        }

        return state;
    }

    private static CatanState ExecuteRobberVictimSelection(CatanState state)
    {
        var victimActions = GetCatanActions(state)
            .Where(a => a is ChooseRobberVictimAction)
            .ToArray();
        if (victimActions.Length == 0)
        {
            return state;
        }

        var selectedIndex = 0;
        while (true)
        {
            DrawFrame(() =>
            {
                RenderBoard(state.Board);

                var leftCol = new List<string>
                {
                    $"Turn {state.TurnNumber} - Player {state.CurrentPlayer} | Stage: {StageLabel(state.Stage)}",
                };
                leftCol.AddRange(BuildPlayerLines(state));

                var rightCol = new List<string> { "Action log:" };
                rightCol.AddRange(ActionLogLinesForPlayer(state.CurrentPlayer));

                const int leftWidth = 48;
                WriteTwoColumnBlock(leftCol, rightCol, leftWidth);

                WriteFixedLine("Select player to rob (j/k or Up/Down + Enter):");
                for (var i = 0; i < victimActions.Length; i++)
                {
                    var prefix = i == selectedIndex ? ">" : " ";
                    WriteFixedLine($"{prefix} {DescribeAction(state, victimActions[i])}");
                }
            });

            var key = Console.ReadKey(intercept: true).Key;
            if (key is ConsoleKey.UpArrow or ConsoleKey.K)
            {
                selectedIndex = selectedIndex > 0 ? selectedIndex - 1 : victimActions.Length - 1;
                continue;
            }

            if (key is ConsoleKey.DownArrow or ConsoleKey.J)
            {
                selectedIndex = selectedIndex < victimActions.Length - 1 ? selectedIndex + 1 : 0;
                continue;
            }

            if (key == ConsoleKey.Enter)
            {
                return ApplyActionAndLog(state, victimActions[selectedIndex]);
            }
        }
    }

    /// <summary>
    /// Queries the NN server for the current player's win probability.
    /// </summary>
    private static float ComputeCurrentStateWinRate(CatanState state)
    {
        var playerValues = _nnClient!.PredictSingleAsync(
            state.SerializeCompact()).GetAwaiter().GetResult();
        return playerValues[CanonicalPlayerSlot(
            state.CurrentPlayer, state.CurrentPlayer, state.PlayerCount)];
    }

    /// <summary>
    /// Evaluates all actions via the NN server and returns a dictionary
    /// mapping each <see cref="CatanAction"/> to its expected win rate for
    /// the current player.
    /// For stochastic actions the expected value across outcomes is used.
    /// </summary>
    private static Dictionary<CatanAction, float> ComputeActionWinRates(
        CatanState state,
        IReadOnlyList<CatanAction> actions)
    {
        var actingPlayer = state.CurrentPlayer;
        var allStates = new List<string>();
        var descriptors = new List<(
            CatanAction Action,
            int StartIndex,
            int Count,
            int[]? Weights,
            int[] CanonicalSlots)>();

        foreach (var action in actions)
        {
            if (action is CatanDeterministicAction det)
            {
                var resultState = (CatanState)det.State();
                var idx = allStates.Count;
                allStates.Add(resultState.SerializeCompact());
                descriptors.Add((action, idx, 1, null,
                    [CanonicalPlayerSlot(actingPlayer, resultState.CurrentPlayer, state.PlayerCount)]));
            }
            else if (action is CatanStochasticAction stoch)
            {
                var outcomes = stoch.Outcomes();
                var idx = allStates.Count;
                var weights = new int[outcomes.Length];
                var canonicalSlots = new int[outcomes.Length];
                for (var j = 0; j < outcomes.Length; j++)
                {
                    weights[j] = outcomes[j].Item1;
                    var outcomeState = (CatanState)outcomes[j].Item2;
                    canonicalSlots[j] = CanonicalPlayerSlot(
                        actingPlayer, outcomeState.CurrentPlayer, state.PlayerCount);
                    allStates.Add(outcomeState.SerializeCompact());
                }
                descriptors.Add((action, idx, outcomes.Length, weights, canonicalSlots));
            }
        }

        if (allStates.Count == 0)
        {
            return new Dictionary<CatanAction, float>();
        }

        var playerValues = _nnClient!.PredictAsync(allStates).GetAwaiter().GetResult();

        var result = new Dictionary<CatanAction, float>();
        foreach (var desc in descriptors)
        {
            float score;
            if (desc.Weights is null)
            {
                score = playerValues[desc.StartIndex][desc.CanonicalSlots[0]];
            }
            else
            {
                var totalWeight = 0;
                var weightedSum = 0.0f;
                for (var j = 0; j < desc.Count; j++)
                {
                    var w = desc.Weights[j];
                    totalWeight += w;
                    weightedSum += w * playerValues[desc.StartIndex + j][desc.CanonicalSlots[j]];
                }
                score = totalWeight > 0 ? weightedSum / totalWeight : 0;
            }

            result[desc.Action] = score;
        }

        return result;
    }

    private static int CanonicalPlayerSlot(
        int absolutePlayer, int currentPlayer, int playerCount) =>
        (absolutePlayer - currentPlayer + playerCount) % playerCount;

    /// <summary>
    /// Computes NN win rates for a set of spatial placement actions and returns
    /// a dictionary mapping each candidate location index (vertex, edge, or tile)
    /// to the predicted win probability.  Returns null when the NN client is not
    /// connected.
    ///
    /// Uses the complete model's legal policy probabilities for spatial annotations.
    /// If inference is unavailable, the UI simply omits annotations.
    /// </summary>
    private static Dictionary<int, float>? ComputeCandidateWinRates(
        CatanState state, CatanAction[] actions)
    {
        if (_nnClient is null || actions.Length == 0)
            return null;

        try
        {
            var serializer = new CatanPolicySerializer(state.Board.Topology, state.PlayerCount);
            var prediction = _nnClient.PredictPolicyValueAsync([state.SerializeCompact()])
                .GetAwaiter().GetResult();
            var dense = prediction.PolicyProbabilities.Length == 1
                ? Array.ConvertAll(prediction.PolicyProbabilities[0], value => (double)value)
                : [];
            var legalIndices = actions.Select(action => serializer.IndexOf(state, action)).ToArray();
            var policy = serializer.MaskAndNormalize(dense, legalIndices);
            var result = new Dictionary<int, float>();
            for (var i = 0; i < actions.Length; i++)
                result[actions[i].TargetIndex] = (float)policy[i];
            return result;
        }
        catch (HttpRequestException)
        {
            return null;
        }
    }

    /// <summary>
    /// Annotates menu entry labels with NN win rate information.
    /// For direct actions the individual win rate is shown.
    /// For grouped entries (Place settlement, Trade, etc.) the best
    /// win rate among the grouped actions is shown.
    /// </summary>
    private static void AnnotateMenuEntries(
        List<MenuEntry> entries,
        Dictionary<CatanAction, float> winRates)
    {
        for (var i = 0; i < entries.Count; i++)
        {
            var entry = entries[i];

            if (entry.Action is not null && winRates.TryGetValue(entry.Action, out var rate))
            {
                entries[i] = entry with { Label = $"{entry.Label} {FgCyan}[{rate:P1}]{Reset}" };
                continue;
            }

            // For grouped modes, find the best win rate among matching actions.
            var matchingRates = entry.Mode switch
            {
                ActionSelectionMode.RollDice =>
                    winRates.Where(kv => kv.Key is RollDiceAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.BuyDevCardRandom =>
                    winRates.Where(kv => kv.Key is BuyDevCardAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.PlaceSettlement =>
                    winRates.Where(kv => kv.Key is PlaceSettlementAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.PlaceRoad =>
                    winRates.Where(kv => kv.Key is PlaceRoadAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.PlaceCity =>
                    winRates.Where(kv => kv.Key is PlaceCityAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.PlaceRobber =>
                    winRates.Where(kv => kv.Key is ChooseRobberTileAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.OpenTradeMenu =>
                    winRates.Where(kv => kv.Key is TradeWithBankAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.OpenYearOfPlentyMenu =>
                    winRates.Where(kv => kv.Key is PlayYearOfPlentyAction).Select(kv => kv.Value).ToArray(),
                ActionSelectionMode.OpenMonopolyMenu =>
                    winRates.Where(kv => kv.Key is PlayMonopolyAction).Select(kv => kv.Value).ToArray(),
                _ => [],
            };

            if (matchingRates.Length > 0)
            {
                var best = matchingRates.Max();
                entries[i] = entry with { Label = $"{entry.Label} {FgCyan}[best:{best:P1}]{Reset}" };
            }
        }
    }

    private static CatanState ExecuteActionMenu(CatanState state, IReadOnlyList<CatanAction> actions)
    {
        var context = ActionMenuContext.Root;
        var menuEntries = BuildMenuEntries(state, actions, context);
        var selectedIndex = 0;

        // Compute NN win rates for display if an inference server is connected.
        // Gracefully skip if the state model endpoint is not available (404).
        Dictionary<CatanAction, float>? winRates = null;
        float? currentWinRate = null;
        if (_nnClient is not null)
        {
            try
            {
                winRates = ComputeActionWinRates(state, actions);
                currentWinRate = ComputeCurrentStateWinRate(state);
            }
            catch (HttpRequestException)
            {
                // State model endpoint not available — skip annotations.
            }
        }
        if (winRates is not null)
        {
            AnnotateMenuEntries(menuEntries, winRates);
        }

        while (true)
        {
            DrawFrame(() =>
            {
                RenderBoard(state.Board);

                var leftCol = new List<string>
                {
                    $"Turn {state.TurnNumber} - Player {state.CurrentPlayer} | Stage: {StageLabel(state.Stage)}",
                    $"Last: {_statusMessage}",
                    $"LR: {(state.LongestRoadOwner == 0 ? "none" : $"Player {state.LongestRoadOwner}")}, LA: {(state.LargestArmyOwner == 0 ? "none" : $"Player {state.LargestArmyOwner}")}",
                };
                if (currentWinRate.HasValue)
                {
                    leftCol.Add($"NN win: {currentWinRate.Value:P1}");
                }
                leftCol.AddRange(BuildPlayerLines(state));

                var rightCol = new List<string> { "Action log:" };
                rightCol.AddRange(ActionLogLinesForPlayer(state.CurrentPlayer));

                const int leftWidth = 48;
                WriteTwoColumnBlock(leftCol, rightCol, leftWidth);

                WriteFixedLine($"Legal actions - {ContextLabel(context)} (j/k or Up/Down + Enter):");
                for (var i = 0; i < menuEntries.Count; i++)
                {
                    var prefix = i == selectedIndex ? ">" : " ";
                    WriteFixedLine($"{prefix} {menuEntries[i].Label}");
                }
            });

            var key = Console.ReadKey(intercept: true).Key;
            if (key is ConsoleKey.UpArrow or ConsoleKey.K)
            {
                selectedIndex = selectedIndex > 0 ? selectedIndex - 1 : menuEntries.Count - 1;
                continue;
            }

            if (key is ConsoleKey.DownArrow or ConsoleKey.J)
            {
                selectedIndex = selectedIndex < menuEntries.Count - 1 ? selectedIndex + 1 : 0;
                continue;
            }

            if (key != ConsoleKey.Enter)
            {
                continue;
            }

            var selected = menuEntries[selectedIndex];
            if (selected.Mode == ActionSelectionMode.OpenTradeMenu)
            {
                context = ActionMenuContext.Trade;
                menuEntries = BuildMenuEntries(state, actions, context);
                if (winRates is not null) AnnotateMenuEntries(menuEntries, winRates);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.OpenYearOfPlentyMenu)
            {
                context = ActionMenuContext.YearOfPlenty;
                menuEntries = BuildMenuEntries(state, actions, context);
                if (winRates is not null) AnnotateMenuEntries(menuEntries, winRates);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.OpenMonopolyMenu)
            {
                context = ActionMenuContext.Monopoly;
                menuEntries = BuildMenuEntries(state, actions, context);
                if (winRates is not null) AnnotateMenuEntries(menuEntries, winRates);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.Back)
            {
                context = ActionMenuContext.Root;
                menuEntries = BuildMenuEntries(state, actions, context);
                if (winRates is not null) AnnotateMenuEntries(menuEntries, winRates);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.Direct)
            {
                return ApplyActionAndLog(state, selected.Action!);
            }

            if (selected.Mode == ActionSelectionMode.RollDice)
            {
                return ExecuteRollDice(state);
            }

            if (selected.Mode == ActionSelectionMode.BuyDevCardRandom)
            {
                return ExecuteBuyDevCard(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceSettlement)
            {
                return ExecuteSettlementPlacement(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceRoad)
            {
                return ExecuteRoadPlacement(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceCity)
            {
                return ExecuteCityPlacement(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceRobber)
            {
                return ExecuteRobberPlacement(state);
            }
        }
    }

    private static List<MenuEntry> BuildMenuEntries(
        CatanState state,
        IReadOnlyList<CatanAction> actions,
        ActionMenuContext context)
    {
        var entries = new List<MenuEntry>();
        if (context == ActionMenuContext.Trade)
        {
            foreach (var action in actions.Where(a => a is TradeWithBankAction))
            {
                entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
            }

            entries.Add(new MenuEntry("Back", ActionSelectionMode.Back, null));
            return entries;
        }

        if (context == ActionMenuContext.YearOfPlenty)
        {
            foreach (var action in actions.Where(a => a is PlayYearOfPlentyAction))
            {
                entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
            }

            entries.Add(new MenuEntry("Back", ActionSelectionMode.Back, null));
            return entries;
        }

        if (context == ActionMenuContext.Monopoly)
        {
            foreach (var action in actions.Where(a => a is PlayMonopolyAction))
            {
                entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
            }

            entries.Add(new MenuEntry("Back", ActionSelectionMode.Back, null));
            return entries;
        }

        if (actions.Any(a => a is RollDiceAction))
        {
            entries.Add(new MenuEntry("Roll dice", ActionSelectionMode.RollDice, null));
        }

        if (actions.Any(a => a is PlaceSettlementAction))
        {
            entries.Add(new MenuEntry("Place settlement", ActionSelectionMode.PlaceSettlement, null));
        }

        if (actions.Any(a => a is PlaceRoadAction))
        {
            entries.Add(new MenuEntry("Place road", ActionSelectionMode.PlaceRoad, null));
        }

        if (actions.Any(a => a is PlaceCityAction))
        {
            entries.Add(new MenuEntry("Place city", ActionSelectionMode.PlaceCity, null));
        }

        if (actions.Any(a => a is ChooseRobberTileAction))
        {
            entries.Add(new MenuEntry("Place robber", ActionSelectionMode.PlaceRobber, null));
        }

        if (actions.Any(a => a is BuyDevCardAction))
        {
            entries.Add(new MenuEntry("Buy dev card", ActionSelectionMode.BuyDevCardRandom, null));
        }

        if (actions.Any(a => a is TradeWithBankAction))
        {
            entries.Add(new MenuEntry("Trade", ActionSelectionMode.OpenTradeMenu, null));
        }

        if (actions.Any(a => a is PlayYearOfPlentyAction))
        {
            entries.Add(new MenuEntry("Year of Plenty", ActionSelectionMode.OpenYearOfPlentyMenu, null));
        }

        if (actions.Any(a => a is PlayMonopolyAction))
        {
            entries.Add(new MenuEntry("Monopoly", ActionSelectionMode.OpenMonopolyMenu, null));
        }

        foreach (var action in actions)
        {
            if (action is
                RollDiceAction or
                BuyDevCardAction or
                TradeWithBankAction or
                PlayYearOfPlentyAction or
                PlayMonopolyAction or
                PlaceSettlementAction or
                PlaceRoadAction or
                PlaceCityAction or
                ChooseRobberTileAction)
            {
                continue;
            }

            entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
        }

        return entries;
    }

    private static string ContextLabel(ActionMenuContext context) =>
        context switch
        {
            ActionMenuContext.Root => "Root",
            ActionMenuContext.Trade => "Trade",
            ActionMenuContext.YearOfPlenty => "Year Of Plenty",
            ActionMenuContext.Monopoly => "Monopoly",
            _ => "Root",
        };

    private static CatanState ExecuteRollDice(CatanState state)
    {
        var action = GetCatanActions(state)
            .FirstOrDefault(a => a is RollDiceAction);

        if (action is null)
        {
            return state;
        }

        var next = (CatanState)action.DoCoreAction();
        return ApplyActionAndLog(state, action);
    }

    private static CatanState ExecuteBuyDevCard(CatanState state)
    {
        var action = GetCatanActions(state)
            .FirstOrDefault(a => a is BuyDevCardAction);

        if (action is null)
        {
            return state;
        }

        return ApplyActionAndLog(state, action);
    }

    private static string DescribeAction(CatanState state, CatanAction action)
    {
        return action switch
        {
            RollDiceAction => "Roll dice",
            PlaceCityAction city => $"Place city at vertex {city.VertexIndex}",
            BuyRoadAction => "Buy road",
            BuySettlementAction => "Buy settlement",
            UpgradeCityAction => "Upgrade city",
            ChooseRobberVictimAction => $"Rob player {action.Arg1}",
            TradeWithBankAction => "Trade with bank",
            ChooseBankTradeGiveAction give =>
                $"Give {state.Board.TradeRatio(state.CurrentPlayer, give.Resource)} {give.Resource}",
            ChooseBankTradeReceiveAction receive => $"Receive 1 {receive.Resource}",
            BuyDevCardAction => "Buy dev card",
            PlayKnightAction => "Play knight",
            PlayRoadBuildingAction => "Play road building",
            PlayMonopolyAction => "Play monopoly",
            ChooseMonopolyResourceAction choice => $"Choose {choice.Resource}",
            PlayYearOfPlentyAction => "Play year of plenty",
            ChooseYearOfPlentyResourceAction choice => $"Choose {choice.Resource}",
            EndTurnAction => "End turn",
            _ => action.GetType().Name,
        };
    }

    private static CatanState ExecuteRoadPlacement(CatanState state)
    {
        var actions = GetCatanActions(state)
            .Where(a => a is PlaceRoadAction)
            .ToArray();

        var legal = actions.Select(a => a.TargetIndex).ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal road placements available.");
            return state;
        }

        // Compute per-candidate NN win rates when the inference server is connected.
        var candidateWinRates = ComputeCandidateWinRates(state, actions);

        var selected = SelectLocation(
            title: "Select road location",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            playerLines: BuildPlayerLines(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).EdgePoints,
            neighborProvider: BuildEdgeNeighborProvider(state.Board.Topology),
            mode: LocationSelectionMode.Edge,
            candidateWinRates: candidateWinRates);

        if (selected is int edge)
        {
            var action = new PlaceRoadAction(state, edge);
            return ApplyActionAndLog(state, action);
        }

        return state;
    }

    private static CatanState ExecuteCityPlacement(CatanState state)
    {
        var actions = GetCatanActions(state)
            .Where(a => a is PlaceCityAction)
            .ToArray();

        var legal = actions.Select(a => a.Arg1).ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal city placements available.");
            return state;
        }

        // Compute per-candidate NN win rates when the inference server is connected.
        var candidateWinRates = ComputeCandidateWinRates(state, actions);

        var selected = SelectLocation(
            title: "Select city location",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            playerLines: BuildPlayerLines(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).VertexPoints,
            neighborProvider: vertex => state.Board.Topology.VertexNeighbors[vertex],
            mode: LocationSelectionMode.Vertex,
            candidateWinRates: candidateWinRates);

        if (selected is int vertex)
        {
            var action = new PlaceCityAction(state, vertex);
            return ApplyActionAndLog(state, action);
        }

        return state;
    }

    private static int? SelectLocation(
        string title,
        string stageLabel,
        int currentPlayer,
        Board board,
        IReadOnlyList<string> playerLines,
        int[] legalCandidates,
        IReadOnlyList<(int X, int Y)> pointProvider,
        Func<int, IEnumerable<int>> neighborProvider,
        LocationSelectionMode mode,
        Dictionary<int, float>? candidateWinRates = null)
    {
        var legalSet = legalCandidates.ToHashSet();

        // Start at the NN-best candidate when win rates are available,
        // otherwise default to the first legal candidate.
        var current = legalCandidates[0];
        if (candidateWinRates is { Count: > 0 })
        {
            var best = candidateWinRates.MaxBy(kv => kv.Value);
            current = best.Key;
        }

        while (true)
        {
            var displayCurrent = current;
            DrawFrame(() =>
            {
                if (mode == LocationSelectionMode.Vertex)
                {
                    RenderBoard(
                        board,
                        highlightedVertices: legalCandidates,
                        selectedVertex: displayCurrent);
                }
                else if (mode == LocationSelectionMode.Edge)
                {
                    RenderBoard(
                        board,
                        highlightedEdges: legalCandidates,
                        selectedEdge: displayCurrent);
                }
                else
                {
                    RenderBoard(
                        board,
                        highlightedTiles: legalCandidates,
                        selectedTile: displayCurrent);
                }

                var leftCol = new List<string>
                {
                    $"Player {currentPlayer} | Stage: {stageLabel}",
                };
                leftCol.AddRange(playerLines);

                if (candidateWinRates is not null
                    && candidateWinRates.TryGetValue(displayCurrent, out var winRate))
                {
                    leftCol.Add($"{title}: {FgCyan}NN win: {winRate:P1}{Reset}");
                }
                else
                {
                    leftCol.Add($"{title}:");
                }
                leftCol.Add("h/j/k/l or arrows, Enter, Esc");

                var rightCol = new List<string> { "Action log:" };
                rightCol.AddRange(ActionLogLinesForPlayer(currentPlayer));

                const int leftWidth = 48;
                WriteTwoColumnBlock(leftCol, rightCol, leftWidth);
            });

            var key = Console.ReadKey(intercept: true).Key;
            if (key == ConsoleKey.Enter)
            {
                return current;
            }

            if (key == ConsoleKey.Escape)
            {
                return null;
            }

            current = key switch
            {
                ConsoleKey.LeftArrow or ConsoleKey.H => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, -1, 0),
                ConsoleKey.RightArrow or ConsoleKey.L => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, 1, 0),
                ConsoleKey.UpArrow or ConsoleKey.K => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, 0, -1),
                ConsoleKey.DownArrow or ConsoleKey.J => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, 0, 1),
                _ => current,
            };
        }
    }

    private static int MoveSelection(
        int current,
        IReadOnlyList<int> legalCandidates,
        HashSet<int> legalSet,
        IReadOnlyList<(int X, int Y)> pointProvider,
        Func<int, IEnumerable<int>> neighborProvider,
        int dirX,
        int dirY)
    {
        var currentPoint = pointProvider[current];

        var neighborCandidates = neighborProvider(current).Where(legalSet.Contains).ToArray();
        var bestNeighbor = PickDirectionalBest(currentPoint, neighborCandidates, pointProvider, dirX, dirY);
        if (bestNeighbor.HasValue)
        {
            return bestNeighbor.Value;
        }

        var fallback = PickDirectionalBest(currentPoint, legalCandidates, pointProvider, dirX, dirY);
        return fallback ?? current;
    }

    private static int? PickDirectionalBest(
        (int X, int Y) currentPoint,
        IEnumerable<int> candidates,
        IReadOnlyList<(int X, int Y)> pointProvider,
        int dirX,
        int dirY)
    {
        var best = -1;
        var bestScore = double.MaxValue;
        var found = false;

        foreach (var candidate in candidates)
        {
            var point = pointProvider[candidate];
            var vx = point.X - currentPoint.X;
            var vy = point.Y - currentPoint.Y;

            var projection = (vx * dirX) + (vy * dirY);
            if (projection <= 0)
                continue;

            var perpendicular = Math.Abs((vx * dirY) - (vy * dirX));
            var score = (perpendicular * 1000.0) + projection;
            if (score < bestScore)
            {
                bestScore = score;
                best = candidate;
                found = true;
            }
        }

        return found ? best : null;
    }

    private static Func<int, IEnumerable<int>> BuildEdgeNeighborProvider(BoardTopology topology)
    {
        var neighbors = new List<int>[topology.EdgeCount];
        for (var i = 0; i < topology.EdgeCount; i++)
        {
            neighbors[i] = new List<int>();
        }

        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var edges = topology.VertexEdges[vi];
            for (var i = 0; i < edges.Length; i++)
            {
                for (var j = i + 1; j < edges.Length; j++)
                {
                    var a = edges[i];
                    var b = edges[j];
                    neighbors[a].Add(b);
                    neighbors[b].Add(a);
                }
            }
        }

        var frozen = neighbors.Select(n => n.Distinct().ToArray()).ToArray();
        return edge => frozen[edge];
    }

    private static string StageLabel(TurnStage stage) =>
        stage switch
        {
            TurnStage.PlaceFirstSettlement => "Place first settlement",
            TurnStage.PlaceFirstRoad => "Place first road",
            TurnStage.PlaceSecondSettlement => "Place second settlement",
            TurnStage.PlaceSecondRoad => "Place second road",
            TurnStage.PreRoll => "Pre-roll",
            TurnStage.ChooseRobberLocation => "Choose robber location",
            TurnStage.ChooseRobberVictim => "Choose robber victim",
            TurnStage.BuildTrade => "Build/trade",
            TurnStage.PlaceRoadBuildingFirst => "Place first Road Building road",
            TurnStage.PlaceRoadCommitted => "Place committed road",
            TurnStage.PlaceSettlementCommitted => "Place committed settlement",
            TurnStage.PlaceCityCommitted => "Place committed city",
            TurnStage.ChooseBankTradeGive => "Choose bank trade resource to give",
            TurnStage.ChooseBankTradeReceive => "Choose bank trade resource to receive",
            TurnStage.ChooseMonopolyResource => "Choose monopoly resource",
            TurnStage.ChooseYearOfPlentyFirst => "Choose first Year of Plenty resource",
            TurnStage.ChooseYearOfPlentySecond => "Choose second Year of Plenty resource",
            _ => stage.ToString(),
        };

    private static string BuildPlayerSummary(CatanState state)
    {
        var parts = new List<string>();
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            parts.Add(FormatPlayerInfo(state, player));
        }

        return string.Join(" | ", parts);
    }

    private static List<string> BuildPlayerLines(CatanState state)
    {
        var lines = new List<string>();
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            lines.Add(FormatPlayerInfo(state, player));
        }

        return lines;
    }

    private static string FormatPlayerInfo(CatanState state, int player)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append($"{PlayerColor(player)}P{player}{Reset} VP:{state.VictoryPointsFor(player)} K:{state.KnightsPlayedFor(player)}");
        sb.Append($" W:{state.ResourceCountFor(player, ResourceType.Wood)} B:{state.ResourceCountFor(player, ResourceType.Brick)} S:{state.ResourceCountFor(player, ResourceType.Sheep)} Wh:{state.ResourceCountFor(player, ResourceType.Wheat)} O:{state.ResourceCountFor(player, ResourceType.Ore)}");

        var devCards = new List<string>();
        AddDevCard(devCards, "Kn", state.DevCardsInHand(player, DevCardType.Knight));
        AddDevCard(devCards, "RB", state.DevCardsInHand(player, DevCardType.RoadBuilding));
        AddDevCard(devCards, "Mo", state.DevCardsInHand(player, DevCardType.Monopoly));
        AddDevCard(devCards, "YP", state.DevCardsInHand(player, DevCardType.YearOfPlenty));
        AddDevCard(devCards, "VP", state.DevCardsInHand(player, DevCardType.VictoryPoint));

        if (devCards.Count > 0)
        {
            sb.Append($" [{string.Join(",", devCards)}]");
        }

        return sb.ToString();
    }

    private static void AddDevCard(List<string> list, string label, int count)
    {
        if (count > 0)
        {
            list.Add($"{label}:{count}");
        }
    }

    private static CatanState ApplyActionAndLog(CatanState state, CatanAction action, bool aiControlled = false)
    {
        var actor = state.CurrentPlayer;
        var next = (CatanState)action.DoCoreAction();
        var description = action is RollDiceAction
            ? $"Roll dice ({next.LastDiceRoll})"
            : DescribeAction(state, action);
        var actorLabel = aiControlled ? $"P{actor}(AI)" : $"P{actor}";
        _statusMessage = $"{actorLabel}: {description}";
        _actionLog.Add(new LoggedAction(state.TurnNumber, actor, description));
        if (actor > 0 && actor < _lastSeenLogIndexByPlayer.Length)
        {
            _lastSeenLogIndexByPlayer[actor] = _actionLog.Count;
        }

        return next;
    }

    private static IEnumerable<string> ActionLogLinesForPlayer(int player)
    {
        var lines = new List<string>(MaxVisibleLogEntries);
        if (player <= 0 || player >= _lastSeenLogIndexByPlayer.Length)
        {
            lines.Add("  (action log unavailable)");
            while (lines.Count < MaxVisibleLogEntries)
            {
                lines.Add(string.Empty);
            }

            return lines;
        }

        var startIndex = Math.Clamp(_lastSeenLogIndexByPlayer[player], 0, _actionLog.Count);
        var unseen = _actionLog.Skip(startIndex).ToArray();
        if (unseen.Length == 0)
        {
            lines.Add("  (no actions yet)");
            while (lines.Count < MaxVisibleLogEntries)
            {
                lines.Add(string.Empty);
            }

            return lines;
        }

        if (unseen.Length > MaxVisibleLogEntries)
        {
            var hiddenCount = unseen.Length - (MaxVisibleLogEntries - 1);
            lines.Add($"  ... {hiddenCount} earlier action(s)");
            unseen = unseen.Skip(unseen.Length - (MaxVisibleLogEntries - 1)).ToArray();
        }
        else
        {
            unseen = unseen.Skip(Math.Max(0, unseen.Length - MaxVisibleLogEntries)).ToArray();
        }

        foreach (var entry in unseen)
        {
            lines.Add($"  T{entry.TurnNumber} P{entry.Player}: {entry.Description}");
        }

        while (lines.Count < MaxVisibleLogEntries)
        {
            lines.Add(string.Empty);
        }

        return lines;
    }

    private static MapChoice PromptMapTopology()
    {
        while (true)
        {
            Console.Write("Select map topology ([m]ini/[sm]all/[s]tandard): ");
            var input = Console.ReadLine()?.Trim().ToLowerInvariant();

            if (input is "m" or "mini")
            {
                return MapChoice.Mini;
            }

            if (input is "sm" or "small")
            {
                return MapChoice.Small;
            }

            if (input is "s" or "standard")
            {
                return MapChoice.Standard;
            }

            Console.WriteLine("Please enter 'mini' (or 'm'), 'small' (or 'sm'), or 'standard' (or 's').");
        }
    }

    private static int PromptPlayerCount(int minPlayers, int maxPlayers)
    {
        while (true)
        {
            if (minPlayers == maxPlayers)
            {
                Console.Write($"How many players should be included? ({minPlayers}): ");
            }
            else
            {
                Console.Write($"How many players should be included? ({minPlayers}-{maxPlayers}): ");
            }

            var input = Console.ReadLine()?.Trim();

            if (!int.TryParse(input, out var players))
            {
                Console.WriteLine("Please enter a valid whole number.");
                continue;
            }

            if (players < minPlayers || players > maxPlayers)
            {
                if (minPlayers == maxPlayers)
                {
                    Console.WriteLine($"Only {minPlayers} players are supported for this map.");
                }
                else
                {
                    Console.WriteLine($"Player count must be between {minPlayers} and {maxPlayers}.");
                }
                continue;
            }

            return players;
        }
    }

    private static PlayerController[] PromptPlayerControllers(int playerCount)
    {
        var controllers = new PlayerController[playerCount + 1];
        for (var player = 1; player <= playerCount; player++)
        {
            while (true)
            {
                Console.Write($"Player {player} controller ([h]uman/[g]reedy/[m]cts/[n]n/nn-[v]alue): ");
                var input = Console.ReadLine()?.Trim().ToLowerInvariant();
                if (string.IsNullOrEmpty(input) || input is "h" or "human")
                {
                    controllers[player] = PlayerController.Human;
                    break;
                }

                if (input is "g" or "greedy")
                {
                    controllers[player] = PlayerController.Greedy;
                    break;
                }

                if (input is "m" or "mcts" or "ai")
                {
                    controllers[player] = PlayerController.MCTS;
                    break;
                }

                if (input is "n" or "nn")
                {
                    controllers[player] = PlayerController.NN;
                    break;
                }

                if (input is "v" or "nn-value" or "nn-v")
                {
                    controllers[player] = PlayerController.NNValue;
                    break;
                }

                Console.WriteLine("Please enter 'h' for human, 'g' for greedy, 'm' for MCTS, 'n' for NN, or 'v' for NN-Value.");
            }
        }

        return controllers;
    }

    private static void RenderBoard(
        Board board,
        IEnumerable<int>? highlightedVertices = null,
        int? selectedVertex = null,
        IEnumerable<int>? highlightedEdges = null,
        int? selectedEdge = null,
        IEnumerable<int>? highlightedTiles = null,
        int? selectedTile = null)
    {
        var layout = BuildRenderLayout(board.Topology);
        var highlightedVertexSet = highlightedVertices is null ? null : highlightedVertices.ToHashSet();
        var highlightedEdgeSet = highlightedEdges is null ? null : highlightedEdges.ToHashSet();
        var highlightedTileSet = highlightedTiles is null ? null : highlightedTiles.ToHashSet();

        var canvas = new Canvas(layout.Width, layout.Height);

        for (var ei = 0; ei < board.Topology.EdgeCount; ei++)
        {
            var edge = board.Topology.Edges[ei];
            DrawEdge(
                canvas,
                layout.VertexPoints[edge.VertexA],
                layout.VertexPoints[edge.VertexB],
                board.EdgeOccupancy[ei].Player,
                highlighted: highlightedEdgeSet?.Contains(ei) ?? false,
                selected: selectedEdge == ei);
        }

        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            DrawVertex(
                canvas,
                layout.VertexPoints[vi],
                board.VertexOccupancy[vi],
                highlighted: highlightedVertexSet?.Contains(vi) ?? false,
                selected: selectedVertex == vi);
        }

        for (var ti = 0; ti < board.Topology.TileCount; ti++)
        {
            DrawTile(
                canvas,
                layout.TilePoints[ti],
                board,
                ti,
                highlighted: highlightedTileSet?.Contains(ti) ?? false,
                selected: selectedTile == ti);
        }

        for (var pi = 0; pi < board.Topology.PortCount; pi++)
        {
            var (va, vb) = board.Topology.Ports[pi];
            var port = board.PortType(pi);
            var portColor = PortColor(port);
            var mid = (
                X: (layout.VertexPixels[va].X + layout.VertexPixels[vb].X) / 2.0,
                Y: (layout.VertexPixels[va].Y + layout.VertexPixels[vb].Y) / 2.0);
            var outward = (
                X: mid.X - layout.BoardCenter.X,
                Y: mid.Y - layout.BoardCenter.Y);
            var length = Math.Sqrt((outward.X * outward.X) + (outward.Y * outward.Y));
            if (length < 0.0001)
            {
                length = 1.0;
                outward = (1.0, 0.0);
            }

            var portPos = (
                X: mid.X + (outward.X / length) * 1.15,
                Y: mid.Y + (outward.Y / length) * 1.15);
            var point = ToCanvasPoint(portPos, layout.MinX, layout.MinY, layout.ScaleX, layout.ScaleY, layout.MarginX, layout.MarginY);
            var label = PortLabel(port);
            var labelStartX = point.X - (label.Length / 2);
            var labelEndX = labelStartX + label.Length - 1;
            var leftAnchor = (X: labelStartX - 1, Y: point.Y);
            var rightAnchor = (X: labelEndX + 1, Y: point.Y);
            var aAnchor = layout.VertexPoints[va].X <= point.X ? leftAnchor : rightAnchor;
            var bAnchor = layout.VertexPoints[vb].X <= point.X ? leftAnchor : rightAnchor;

            DrawConnector(canvas, aAnchor, layout.VertexPoints[va], portColor);
            DrawConnector(canvas, bAnchor, layout.VertexPoints[vb], portColor);
            DrawString(canvas, labelStartX, point.Y, label, portColor);
        }

        canvas.Print(_frameBuffer);
    }

    private static BoardRenderLayout BuildRenderLayout(BoardTopology topology)
    {
        var tilePixels = new (double X, double Y)[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tilePixels[ti] = AxialToPixel(topology.Tiles[ti]);
        }

        var vertexPixels = new (double X, double Y)[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var key = topology.Vertices[vi];
            var a = AxialToPixel(key.A);
            var b = AxialToPixel(key.B);
            var c = AxialToPixel(key.C);
            vertexPixels[vi] = ((a.X + b.X + c.X) / 3.0, (a.Y + b.Y + c.Y) / 3.0);
        }

        var minX = Math.Min(tilePixels.Min(p => p.X), vertexPixels.Min(p => p.X));
        var maxX = Math.Max(tilePixels.Max(p => p.X), vertexPixels.Max(p => p.X));
        var minY = Math.Min(tilePixels.Min(p => p.Y), vertexPixels.Min(p => p.Y));
        var maxY = Math.Max(tilePixels.Max(p => p.Y), vertexPixels.Max(p => p.Y));

        const int marginX = 8;
        const int marginY = 4;
        const double scaleX = 8.0;
        const double scaleY = 4.0;

        var boardCenter = (
            X: tilePixels.Average(p => p.X),
            Y: tilePixels.Average(p => p.Y));

        var vertexPoints = new (int X, int Y)[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            vertexPoints[vi] = ToCanvasPoint(vertexPixels[vi], minX, minY, scaleX, scaleY, marginX, marginY);
        }

        var tilePoints = new (int X, int Y)[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tilePoints[ti] = ToCanvasPoint(tilePixels[ti], minX, minY, scaleX, scaleY, marginX, marginY);
        }

        var edgePoints = new (int X, int Y)[topology.EdgeCount];
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            var (va, vb) = topology.Edges[ei];
            edgePoints[ei] = ((vertexPoints[va].X + vertexPoints[vb].X) / 2, (vertexPoints[va].Y + vertexPoints[vb].Y) / 2);
        }

        var width = (int)Math.Ceiling((maxX - minX) * scaleX) + marginX * 2 + 20;
        var height = (int)Math.Ceiling((maxY - minY) * scaleY) + marginY * 2 + 2;

        return new BoardRenderLayout(
            tilePixels,
            vertexPixels,
            tilePoints,
            vertexPoints,
            edgePoints,
            minX,
            minY,
            scaleX,
            scaleY,
            marginX,
            marginY,
            width,
            height,
            boardCenter);
    }

    private static (double X, double Y) AxialToPixel(HexCoord c)
    {
        var x = Sqrt3 * (c.Q + c.R / 2.0);
        var y = -1.5 * c.R;
        return (x, y);
    }

    private static (int X, int Y) ToCanvasPoint(
        (double X, double Y) point,
        double minX,
        double minY,
        double scaleX,
        double scaleY,
        int marginX,
        int marginY)
    {
        var x = (int)Math.Round((point.X - minX) * scaleX) + marginX;
        var y = (int)Math.Round((point.Y - minY) * scaleY) + marginY;
        return (x, y);
    }

    private static void DrawEdge(
        Canvas canvas,
        (int X, int Y) p0,
        (int X, int Y) p1,
        int player,
        bool highlighted,
        bool selected)
    {
        var dx = p1.X - p0.X;
        var dy = p1.Y - p0.Y;
        var steps = Math.Max(Math.Abs(dx), Math.Abs(dy));
        if (steps <= 1)
        {
            return;
        }

        char edgeChar;
        if (Math.Abs(dy) <= 1)
        {
            edgeChar = '-';
        }
        else if (Math.Abs(dy) > Math.Abs(dx) * 2)
        {
            edgeChar = '|';
        }
        else if ((dx > 0 && dy > 0) || (dx < 0 && dy < 0))
        {
            edgeChar = '\\';
        }
        else
        {
            edgeChar = '/';
        }

        var color = player > 0
            ? PlayerColor(player)
            : selected
                ? FgBrightRed
                : highlighted
                    ? FgBrightWhite
                    : FgBrightBlack;

        for (var i = 1; i < steps; i++)
        {
            var x = p0.X + (dx * i) / steps;
            var y = p0.Y + (dy * i) / steps;
            canvas.Set(x, y, edgeChar, color);
        }
    }

    private static void DrawVertex(
        Canvas canvas,
        (int X, int Y) point,
        VertexOccupancy occupancy,
        bool highlighted,
        bool selected)
    {
        if (occupancy.IsEmpty)
        {
            var charToDraw = selected ? 'O' : 'o';
            var color = selected ? FgBrightRed : highlighted ? FgBrightWhite : FgBrightBlack;
            canvas.Set(point.X, point.Y, charToDraw, color);
            return;
        }

        var marker = occupancy.Building == BuildingType.City ? 'c' : 's';
        if (selected)
        {
            canvas.Set(point.X, point.Y, char.ToUpperInvariant(marker), FgBrightRed);
            return;
        }

        if (highlighted)
        {
            canvas.Set(point.X, point.Y, marker, FgBrightWhite);
            return;
        }

        canvas.Set(point.X, point.Y, marker, PlayerColor(occupancy.Player));
    }

    private static void DrawTile(
        Canvas canvas,
        (int X, int Y) center,
        Board board,
        int tileIndex,
        bool highlighted,
        bool selected)
    {
        var resource = board.TileResource(tileIndex);
        var resourceText = resource switch
        {
            ResourceType.Desert => "Desert",
            ResourceType.Wood => "Wood",
            ResourceType.Brick => "Brick",
            ResourceType.Sheep => "Sheep",
            ResourceType.Wheat => "Wheat",
            ResourceType.Ore => "Ore",
            _ => "Unknown",
        };

        var number = board.TileNumber(tileIndex);
        var numberText = number == 0 ? "" : number.ToString();
        var numberColor = number is 6 or 8 ? FgBrightRed : FgBrightWhite;

        DrawString(
            canvas,
            center.X - (resourceText.Length / 2),
            center.Y - 1,
            resourceText,
            ResourceStyle(resource));

        if (numberText.Length > 0)
        {
            DrawString(canvas, center.X - (numberText.Length / 2), center.Y, numberText, numberColor);
        }

        if (board.RobberTile == tileIndex)
        {
            DrawString(canvas, center.X, center.Y + 1, "*", FgRed);
        }

        if (selected || highlighted)
        {
            var marker = selected ? "@" : "+";
            var markerColor = selected ? FgBrightRed : FgCyan;
            DrawString(canvas, center.X, center.Y + 2, marker, markerColor);
        }
    }

    private static void DrawString(Canvas canvas, int x, int y, string text, string color)
    {
        for (var i = 0; i < text.Length; i++)
        {
            canvas.Set(x + i, y, text[i], color);
        }
    }

    private static void DrawConnector(Canvas canvas, (int X, int Y) p0, (int X, int Y) p1, string color)
    {
        var x0 = p0.X;
        var y0 = p0.Y;
        var x1 = p1.X;
        var y1 = p1.Y;
        var dx = Math.Abs(x1 - x0);
        var sx = x0 < x1 ? 1 : -1;
        var dy = -Math.Abs(y1 - y0);
        var sy = y0 < y1 ? 1 : -1;
        var err = dx + dy;

        while (!(x0 == x1 && y0 == y1))
        {
            var e2 = err * 2;
            var nextX = x0;
            var nextY = y0;
            if (e2 >= dy)
            {
                err += dy;
                nextX += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                nextY += sy;
            }

            if (!(nextX == x1 && nextY == y1))
            {
                var cx = nextX - x0;
                var cy = nextY - y0;
                var ch = PickLineChar(cx, cy);
                canvas.SetIfEmpty(nextX, nextY, ch, color);
            }

            x0 = nextX;
            y0 = nextY;
        }
    }

    private static char PickLineChar(int dx, int dy)
    {
        if (dx == 0)
        {
            return '|';
        }

        if (dy == 0)
        {
            return '-';
        }

        return (dx > 0 && dy > 0) || (dx < 0 && dy < 0) ? '\\' : '/';
    }

    private static string PortLabel(PortType port) =>
        port switch
        {
            PortType.Generic => "3:1",
            PortType.Wood => "Wood",
            PortType.Brick => "Brick",
            PortType.Sheep => "Sheep",
            PortType.Wheat => "Wheat",
            PortType.Ore => "Ore",
            _ => "??",
        };

    private static string PortColor(PortType port) =>
        port switch
        {
            PortType.Generic => FgBrightWhite,
            PortType.Wood => FgGreen,
            PortType.Brick => FgRed,
            PortType.Sheep => FgLightGreen,
            PortType.Wheat => FgYellow,
            PortType.Ore => FgSilver,
            _ => FgWhite,
        };

    private static string PlayerColor(int player) =>
        player switch
        {
            1 => FgRed,
            2 => FgBrightGreen,
            3 => FgYellow,
            4 => FgBrightWhite,
            _ => FgWhite,
        };

    private static string ResourceStyle(ResourceType resource) =>
        resource switch
        {
            ResourceType.Wood => FgGreen,
            ResourceType.Brick => FgRed,
            ResourceType.Sheep => FgLightGreen,
            ResourceType.Wheat => FgYellow,
            ResourceType.Ore => FgSilver,
            ResourceType.Desert => FgBeige,
            _ => Reset,
        };

    private static void Pause(string message)
    {
        DrawFrame(() =>
        {
            WriteFixedLine();
            WriteFixedLine(message);
            WriteFixedLine("Press any key to continue...");
        });
        Console.ReadKey(intercept: true);
    }

    private static void DrawFrame(Action draw)
    {
        _frameBuffer.Clear();
        draw();

        // Count how many lines the frame produced
        var lineCount = 0;
        foreach (var ch in _frameBuffer.ToString())
        {
            if (ch == '\n')
            {
                lineCount++;
            }
        }

        // Append blank lines to clear any leftover content from the previous frame
        var clearWidth = Math.Max(1, Console.WindowWidth - 1);
        for (var i = lineCount; i < _lastFrameLineCount; i++)
        {
            _frameBuffer.Append(' ', clearWidth);
            _frameBuffer.AppendLine();
        }

        // Flush everything in one write to avoid flicker
        Console.SetCursorPosition(0, 0);
        Console.Write(_frameBuffer);
        Console.SetCursorPosition(0, lineCount);
        _lastFrameLineCount = lineCount;
    }

    private static void WriteFixedLine(string text = "")
    {
        var width = Math.Max(1, Console.WindowWidth - 1);
        var visible = VisibleLength(text);
        if (visible > width)
        {
            text = TruncateAnsi(text, width);
            visible = width;
        }

        _frameBuffer.Append(text);
        _frameBuffer.Append(' ', Math.Max(0, width - visible));
        _frameBuffer.AppendLine();
    }

    /// <summary>
    /// Writes rows from two lists side by side. The left column occupies
    /// <paramref name="leftWidth"/> visible characters; the right column
    /// fills the rest. Rows beyond the shorter list are blank on that side.
    /// </summary>
    private static void WriteTwoColumnBlock(
        IReadOnlyList<string> leftLines,
        IReadOnlyList<string> rightLines,
        int leftWidth)
    {
        var totalWidth = Math.Max(1, Console.WindowWidth - 1);
        var rows = Math.Max(leftLines.Count, rightLines.Count);
        for (var i = 0; i < rows; i++)
        {
            var left = i < leftLines.Count ? leftLines[i] : "";
            var right = i < rightLines.Count ? rightLines[i] : "";
            var leftVisible = VisibleLength(left);

            // Pad or truncate left column to exactly leftWidth visible chars
            if (leftVisible > leftWidth)
            {
                left = TruncateAnsi(left, leftWidth);
                leftVisible = leftWidth;
            }

            _frameBuffer.Append(left);
            _frameBuffer.Append(' ', Math.Max(0, leftWidth - leftVisible));

            // Separator
            _frameBuffer.Append("  ");

            var rightVisible = VisibleLength(right);
            var rightWidth = Math.Max(0, totalWidth - leftWidth - 2);
            if (rightVisible > rightWidth)
            {
                right = TruncateAnsi(right, rightWidth);
                rightVisible = rightWidth;
            }

            _frameBuffer.Append(right);
            _frameBuffer.Append(' ', Math.Max(0, rightWidth - rightVisible));
            _frameBuffer.AppendLine();
        }
    }

    /// <summary>
    /// Truncates a string that may contain ANSI escape sequences to at most
    /// <paramref name="maxVisible"/> visible characters, preserving escape
    /// sequences and appending a Reset at the end.
    /// </summary>
    private static string TruncateAnsi(string text, int maxVisible)
    {
        var sb = new StringBuilder();
        var vis = 0;
        for (var i = 0; i < text.Length && vis < maxVisible;)
        {
            if (text[i] == '\u001b')
            {
                var end = text.IndexOf('m', i);
                if (end >= 0)
                {
                    sb.Append(text, i, end - i + 1);
                    i = end + 1;
                    continue;
                }
            }

            sb.Append(text[i]);
            vis++;
            i++;
        }

        sb.Append(Reset);
        return sb.ToString();
    }
}

internal enum ActionSelectionMode
{
    Direct,
    RollDice,
    BuyDevCardRandom,
    OpenTradeMenu,
    OpenYearOfPlentyMenu,
    OpenMonopolyMenu,
    Back,
    PlaceSettlement,
    PlaceRoad,
    PlaceCity,
    PlaceRobber,
}

internal enum PlayerController
{
    Human,
    Greedy,
    MCTS,
    NN,
    NNValue,
}

internal enum ActionMenuContext
{
    Root,
    Trade,
    YearOfPlenty,
    Monopoly,
}

internal enum LocationSelectionMode
{
    Vertex,
    Edge,
    Tile,
}

internal sealed record LoggedAction(int TurnNumber, int Player, string Description);

internal sealed record MenuEntry(string Label, ActionSelectionMode Mode, CatanAction? Action);

internal sealed record BoardRenderLayout(
    (double X, double Y)[] TilePixels,
    (double X, double Y)[] VertexPixels,
    (int X, int Y)[] TilePoints,
    (int X, int Y)[] VertexPoints,
    (int X, int Y)[] EdgePoints,
    double MinX,
    double MinY,
    double ScaleX,
    double ScaleY,
    int MarginX,
    int MarginY,
    int Width,
    int Height,
    (double X, double Y) BoardCenter);

internal sealed class Canvas
{
    private readonly char[,] _chars;
    private readonly string?[,] _colors;

    public int Width { get; }
    public int Height { get; }

    public Canvas(int width, int height)
    {
        Width = width;
        Height = height;
        _chars = new char[height, width];
        _colors = new string?[height, width];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                _chars[y, x] = ' ';
                _colors[y, x] = null;
            }
        }
    }

    public void Set(int x, int y, char ch, string color)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height)
        {
            return;
        }

        _chars[y, x] = ch;
        _colors[y, x] = color;
    }

    public void SetIfEmpty(int x, int y, char ch, string color)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height)
        {
            return;
        }

        if (_chars[y, x] != ' ')
        {
            return;
        }

        _chars[y, x] = ch;
        _colors[y, x] = color;
    }

    public void Print(StringBuilder buffer)
    {
        var termWidth = Math.Max(1, Console.WindowWidth - 1);

        // Find the last row that has non-space content to avoid trailing blank lines
        var lastUsedRow = Height - 1;
        while (lastUsedRow >= 0)
        {
            var rowEmpty = true;
            for (var x = 0; x < Width; x++)
            {
                if (_chars[lastUsedRow, x] != ' ')
                {
                    rowEmpty = false;
                    break;
                }
            }

            if (!rowEmpty)
            {
                break;
            }

            lastUsedRow--;
        }

        for (var y = 0; y <= lastUsedRow; y++)
        {
            string? activeColor = null;
            var visibleChars = 0;
            for (var x = 0; x < Width; x++)
            {
                var color = _colors[y, x];
                if (!string.Equals(color, activeColor, StringComparison.Ordinal))
                {
                    buffer.Append(color ?? "\u001b[0m");
                    activeColor = color;
                }

                buffer.Append(_chars[y, x]);
                visibleChars++;
            }

            if (activeColor is not null)
            {
                buffer.Append("\u001b[0m");
            }

            if (visibleChars < termWidth)
            {
                buffer.Append(' ', termWidth - visibleChars);
            }

            buffer.AppendLine();
        }
    }
}

internal enum MapChoice
{
    Mini,
    Small,
    Standard,
}
