module KjarniTest.MCTS.TerminalPropagationTest

open NUnit.Framework
open FsUnit

open Kjarni
open Kjarni.MCTS.Types
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.AI
open KjarniTest.TestTypes

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

/// A simple deterministic action wrapping an ICoreState.
type simple_det(target: ICoreState) =
    interface IDeterministicCoreAction with
        member _.State() = target

/// A state whose Actions() returns a single deterministic action to a target.
type det_state(playerTurn, hash, target: ICoreState) =
    interface ICoreState with
        member _.PlayerTurn = playerTurn
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member _.Actions() = [| Deterministic(simple_det target :> IDeterministicCoreAction) |]
        member _.Scores() = Array.zeroCreate<float> 2

    override _.GetHashCode() = hash
    override _.Equals other = hash = other.GetHashCode()

/// A state with no actions (terminal game state).
type terminal_game_state(playerTurn, hash) =
    interface ICoreState with
        member _.PlayerTurn = playerTurn
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member _.Actions() = Array.empty
        member _.Scores() =
            let scores = Array.zeroCreate<float> 2
            scores.[int playerTurn] <- 1.
            scores

    override _.GetHashCode() = hash
    override _.Equals other = hash = other.GetHashCode()

/// A state with multiple deterministic actions, each pointing to a target.
type multi_det_state(playerTurn, hash, targets: ICoreState list) =
    interface ICoreState with
        member _.PlayerTurn = playerTurn
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member _.Actions() =
            targets
            |> List.map (fun t -> Deterministic(simple_det t :> IDeterministicCoreAction))
            |> Array.ofList
        member _.Scores() = Array.zeroCreate<float> 2

    override _.GetHashCode() = hash
    override _.Equals other = hash = other.GetHashCode()

// ────────────────────────────────────────────────────────────────
// tryResolveState
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type TryResolveStateTests() =

    [<Test>]
    member _.NoTerminalActions_ReturnsNone() =
        let child = node_builder(p1, 0, 0, 1).build ()
        let root = MCTSState(node_builder(p1, 0, 0, 0, node_builder(p1, 0, 0, 1)).build ())
        // Expand the action so it's DeterministicAction, not Terminal
        let _ = expand None (root, 0)

        tryResolveState root |> should equal None

    [<Test>]
    member _.GuaranteedWin_Condition1() =
        // P1's turn, one action is Terminal with P1 winning 100%
        let root = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        // Manually set actions to include a Terminal and a non-Terminal
        root.Actions <- [|
            Terminal [| 1.0; 0.0 |]
            Unexplored (Deterministic(simple_det (terminal_game_state(p2, 1) :> ICoreState) :> IDeterministicCoreAction))
        |]

        let result = tryResolveState root
        result |> should not' (equal None)
        result.Value |> should equal [| 1.0; 0.0 |]

    [<Test>]
    member _.AllTerminal_Condition2_ActivePlayerPicksBest() =
        // P2's turn, two Terminal actions. P2 picks the one with higher P2 value.
        let stateNode = terminal_game_state(p2, 0)
        let root = MCTSState(stateNode :> ICoreState)
        root.Actions <- [|
            Terminal [| 0.5; 0.3 |]
            Terminal [| 0.1; 0.6 |]
        |]

        let result = tryResolveState root
        result |> should not' (equal None)
        // P2 picks action[1] because 0.6 > 0.3
        result.Value |> should equal [| 0.1; 0.6 |]

    [<Test>]
    member _.MixedTerminalAndNonTerminal_NoGuaranteedWin_ReturnsNone() =
        // P1's turn, one Terminal with P1 = 0.5, one DeterministicAction
        let root = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        root.Actions <- [|
            Terminal [| 0.5; 0.5 |]
            Unexplored (Deterministic(simple_det (terminal_game_state(p1, 1) :> ICoreState) :> IDeterministicCoreAction))
        |]

        let result = tryResolveState root
        // P1 = 0.5, not 1.0 → condition 1 not met
        // Not all actions are Terminal → condition 2 not met
        result |> should equal None

// ────────────────────────────────────────────────────────────────
// tryResolveStochasticAction
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type TryResolveStochasticActionTests() =

    [<Test>]
    member _.AllOutcomesTerminalGameStates_ReturnsWeightedAverage() =
        // Two terminal game states: P1 wins (weight 1), P2 wins (weight 3)
        let o1 = MCTSState(terminal_game_state(p1, 10) :> ICoreState)
        let o2 = MCTSState(terminal_game_state(p2, 11) :> ICoreState)

        let outcomes = [|
            { ProbabilityWeight = 1; State = o1 }
            { ProbabilityWeight = 3; State = o2 }
        |]

        let result = tryResolveStochasticAction outcomes
        result |> should not' (equal None)
        // Weighted avg: (1*[1,0] + 3*[0,1]) / 4 = [0.25, 0.75]
        result.Value.[0] |> should (equalWithin 0.001) 0.25
        result.Value.[1] |> should (equalWithin 0.001) 0.75

    [<Test>]
    member _.OneOutcomeUnresolved_ReturnsNone() =
        let o1 = MCTSState(terminal_game_state(p1, 10) :> ICoreState)
        // o2 has children (not terminal game state) and no Terminal actions
        let o2 = MCTSState(node_builder(p2, 0, 0, 11, node_builder(p1, 0, 0, 12)).build ())

        let outcomes = [|
            { ProbabilityWeight = 1; State = o1 }
            { ProbabilityWeight = 1; State = o2 }
        |]

        tryResolveStochasticAction outcomes |> should equal None

    [<Test>]
    member _.AllOutcomesFullyResolved_ActivePlayerPicksBest() =
        // Outcome 0 (P1's turn): has two Terminal actions, P1 picks the best
        let o1 = MCTSState(terminal_game_state(p1, 10) :> ICoreState)
        o1.Actions <- [|
            Terminal [| 0.8; 0.2 |]
            Terminal [| 0.3; 0.7 |]
        |]

        // Outcome 1 (P2's turn): has one Terminal action
        let o2 = MCTSState(terminal_game_state(p2, 11) :> ICoreState)
        o2.Actions <- [|
            Terminal [| 0.4; 0.6 |]
        |]

        let outcomes = [|
            { ProbabilityWeight = 1; State = o1 }
            { ProbabilityWeight = 1; State = o2 }
        |]

        let result = tryResolveStochasticAction outcomes
        result |> should not' (equal None)
        // o1: P1 picks [0.8, 0.2]; o2: P2 picks [0.4, 0.6]
        // Avg: ([0.8, 0.2] + [0.4, 0.6]) / 2 = [0.6, 0.4]
        result.Value.[0] |> should (equalWithin 0.001) 0.6
        result.Value.[1] |> should (equalWithin 0.001) 0.4

// ────────────────────────────────────────────────────────────────
// allActionsTerminal
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type AllActionsTerminalTests() =

    [<Test>]
    member _.EmptyActions_ReturnsFalse() =
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        allActionsTerminal s |> should be False

    [<Test>]
    member _.AllTerminal_ReturnsTrue() =
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        s.Actions <- [| Terminal [| 1.0; 0.0 |]; Terminal [| 0.0; 1.0 |] |]
        allActionsTerminal s |> should be True

    [<Test>]
    member _.MixedActions_ReturnsFalse() =
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        s.Actions <- [|
            Terminal [| 1.0; 0.0 |]
            Unexplored (Deterministic(simple_det (terminal_game_state(p1, 1) :> ICoreState) :> IDeterministicCoreAction))
        |]
        allActionsTerminal s |> should be False

// ────────────────────────────────────────────────────────────────
// isResolved
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type IsResolvedTests() =

    [<Test>]
    member _.NoTerminalActions_ReturnsFalse() =
        let s = MCTSState(node_builder(p1, 0, 0, 0, node_builder(p2, 0, 0, 1)).build ())
        isResolved s |> should be False

    [<Test>]
    member _.AllActionsTerminal_ReturnsTrue() =
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        s.Actions <- [| Terminal [| 0.5; 0.5 |]; Terminal [| 0.3; 0.7 |] |]
        isResolved s |> should be True

    [<Test>]
    member _.GuaranteedWin_MixedActions_ReturnsTrue() =
        // P1's turn: one Terminal with P1=1.0, one Unexplored → condition 1 fires
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        s.Actions <- [|
            Terminal [| 1.0; 0.0 |]
            Unexplored (Deterministic(simple_det (terminal_game_state(p1, 1) :> ICoreState) :> IDeterministicCoreAction))
        |]
        isResolved s |> should be True

    [<Test>]
    member _.NoGuaranteedWin_MixedActions_ReturnsFalse() =
        // P1's turn: one Terminal with P1=0.5, one Unexplored → neither condition met
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        s.Actions <- [|
            Terminal [| 0.5; 0.5 |]
            Unexplored (Deterministic(simple_det (terminal_game_state(p1, 1) :> ICoreState) :> IDeterministicCoreAction))
        |]
        isResolved s |> should be False

    [<Test>]
    member _.EmptyActions_ReturnsFalse() =
        let s = MCTSState(terminal_game_state(p1, 0) :> ICoreState)
        isResolved s |> should be False

// ────────────────────────────────────────────────────────────────
// propagateTerminals
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type PropagateTerminalsTests() =

    [<Test>]
    member _.DeterministicChild_AllActionsTerminal_PropagatesUp() =
        // Root → det child (P2's turn, all actions Terminal)
        let childState = terminal_game_state(p2, 1)
        let rootState = det_state(p1, 0, childState :> ICoreState)
        let root = MCTSState(rootState :> ICoreState)
        let child = expand None (root, 0)

        // Manually set child's actions to all Terminal
        child.Actions <- [|
            Terminal [| 0.3; 0.7 |]
            Terminal [| 0.6; 0.4 |]
        |]

        // Propagate: child is deepest, root is parent
        propagateTerminals [ child; root ]

        // Root's action[0] should now be Terminal with P2's best choice
        match root.Actions.[0] with
        | Terminal outcome ->
            // P2 picks [0.3, 0.7] because 0.7 > 0.4
            outcome |> should equal [| 0.3; 0.7 |]
        | a -> Assert.Fail $"Expected Terminal, got %A{a}"

    [<Test>]
    member _.DeterministicChild_NotResolvable_NoChange() =
        let childNode = node_builder(p2, 0, 0, 1, node_builder(p1, 0, 0, 2)).build ()
        let rootState = det_state(p1, 0, childNode :> ICoreState)
        let root = MCTSState(rootState :> ICoreState)
        let child = expand None (root, 0)

        // child has unexplored actions — not resolvable
        propagateTerminals [ child; root ]

        match root.Actions.[0] with
        | DeterministicAction _ -> () // Should remain unchanged
        | a -> Assert.Fail $"Expected DeterministicAction, got %A{a}"

    [<Test>]
    member _.CascadesPropagation_MultipleDepths() =
        // Root (P1) → child (P2) → grandchild (P1, terminal game state)
        // After grandchild is resolved, child becomes all-Terminal, then root cascades.
        let grandchild = terminal_game_state(p1, 2) :> ICoreState
        let childState = det_state(p2, 1, grandchild)
        let rootState = det_state(p1, 0, childState :> ICoreState)

        let root = MCTSState(rootState :> ICoreState)
        let child = expand None (root, 0)
        let grandchildMcts = expand None (child, 0)

        // grandchild has no actions → terminal game state
        // Replace child's action with Terminal for the grandchild (simulating what
        // the search loop does when it detects empty actions after expansion).
        let grandchildOutcome = oneHotOutcome(grandchildMcts.State.PlayerTurn, 2)
        child.Actions.[0] <- Terminal grandchildOutcome

        // Now propagate starting from child up to root.
        // child is fully resolved (all actions Terminal), so root should cascade.
        propagateTerminals [ child; root ]

        match root.Actions.[0] with
        | Terminal outcome ->
            // child (P2's turn) has one Terminal [1, 0]. P2's best is 0.0 but only option.
            outcome |> should equal [| 1.0; 0.0 |]
        | a -> Assert.Fail $"Expected Terminal, got %A{a}"

    [<Test>]
    member _.StopsWhenStateNotResolvable() =
        // Root → child (2 actions: one Terminal, one Unexplored) → doesn't propagate further
        let childNode =
            node_builder(p2, 0, 0, 1)
                .addChildren(
                    [ node_builder (p1, 0, 0, 2)
                      node_builder (p1, 0, 0, 3) ]
                )
                .build ()
        let rootState = det_state(p1, 0, childNode :> ICoreState)
        let root = MCTSState(rootState :> ICoreState)
        let child = expand None (root, 0)

        // child has 2 actions. Set the first one to Terminal, leave the second Unexplored.
        child.Actions.[0] <- Terminal [| 0.5; 0.5 |]
        // child.Actions.[1] is still Unexplored
        // Condition 1: not met (P2 = 0.5, not 1.0)
        // Condition 2: not met (not all actions Terminal)

        propagateTerminals [ child; root ]

        match root.Actions.[0] with
        | DeterministicAction _ -> () // Should remain unchanged
        | a -> Assert.Fail $"Expected DeterministicAction, got %A{a}"

// ────────────────────────────────────────────────────────────────
// Integration: search with terminal propagation
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type TerminalPropagationSearchTests() =

    [<Test>]
    member _.TerminalGameState_ImmediatelyResolved() =
        // Root with one deterministic action to a terminal game state
        let childState = terminal_game_state(p1, 1)
        let rootState = det_state(p1, 0, childState :> ICoreState)
        let mctsRoot = MCTSState(rootState :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with MaxSimulations = 100 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        logInfo.reachedTerminal |> should be True
        // Should terminate after very few rollouts
        mctsRoot.Rollouts |> should be (lessThan 10)

        match mctsRoot.Actions.[0] with
        | Terminal outcome ->
            outcome.[int Player.Player1] |> should equal 1.0
        | a -> Assert.Fail $"Expected Terminal, got %A{a}"

    [<Test>]
    member _.TwoTerminalChildren_BothResolved() =
        // Root (P1's turn) with two deterministic actions, both to terminal game states.
        // Child 0: P1 wins. Child 1: P2 wins.
        // After child 0 is expanded and resolved as Terminal with P1=1.0,
        // condition 1 fires (guaranteed win) and the search stops early.
        let child0 = terminal_game_state(p1, 1) :> ICoreState
        let child1 = terminal_game_state(p2, 2) :> ICoreState
        let rootState = multi_det_state(p1, 0, [ child0; child1 ])
        let mctsRoot = MCTSState(rootState :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with MaxSimulations = 100 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        logInfo.reachedTerminal |> should be True

        // Root is resolved via condition 1 (guaranteed win for P1)
        isResolved mctsRoot |> should be True

        // Best path should choose action 0 (P1 wins)
        let path = extractBestPath mctsRoot
        path |> should not' (be Empty)
        path.[0] |> should equal 0

    [<Test>]
    member _.DeepChain_PropagatesAllTheWayUp() =
        // Root → child → grandchild (terminal P1)
        // All deterministic, single-action chain
        let grandchild = terminal_game_state(p1, 2) :> ICoreState
        let child = det_state(p2, 1, grandchild) :> ICoreState
        let rootState = det_state(p1, 0, child)
        let mctsRoot = MCTSState(rootState :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with MaxSimulations = 100 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        logInfo.reachedTerminal |> should be True

        match mctsRoot.Actions.[0] with
        | Terminal outcome ->
            outcome.[int Player.Player1] |> should equal 1.0
        | a -> Assert.Fail $"Expected Terminal, got %A{a}"

    [<Test>]
    member _.StochasticWithTerminalOutcomes_Resolves() =
        // Root with a stochastic action: outcome A (P1 wins), outcome B (P2 wins)
        // Equal weights → Terminal should be [0.5, 0.5]
        let outcomeA = node (p1, 0, 0, 10)
        let outcomeB = node (p2, 0, 0, 11)
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (1, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with MaxSimulations = 100 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        logInfo.reachedTerminal |> should be True

        match mctsRoot.Actions.[0] with
        | Terminal outcome ->
            outcome.[0] |> should (equalWithin 0.001) 0.5
            outcome.[1] |> should (equalWithin 0.001) 0.5
        | a -> Assert.Fail $"Expected Terminal, got %A{a}"

    [<Test>]
    member _.NonTerminalTree_DoesNotTerminateEarly() =
        // A tree wide and deep enough that 50 simulations won't fully resolve it.
        // Each leaf at depth 2 has 3 children at depth 3, and each of those has
        // 3 children at depth 4. Total leaves = 3 * 3 * 3 * 3 = 81 leaf nodes.
        let tree =
            node_builder(p1, 0, 0, 0)
                .addChildren(
                    [ for i in 1..3 ->
                        node_builder(p2, 1, i, i)
                            .addChildren(
                                [ for j in 1..3 ->
                                    node_builder(p1, 2, i+j, 10*i+j)
                                        .addChildren(
                                            [ for k in 1..3 ->
                                                node_builder(p2, 3, i+j+k, 100*i+10*j+k)
                                                    .addChildren(
                                                        [ for l in 1..3 ->
                                                            node_builder(p1, 4, i+j+k+l, 1000*i+100*j+10*k+l) ]) ]) ]) ]
                )
                .build ()

        let mctsRoot = MCTSState tree

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with MaxSimulations = 50 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        // With 81 leaf nodes and only 50 simulations, not all branches can
        // be resolved, so the search should use the full budget.
        mctsRoot.Rollouts |> should equal 50
        logInfo.reachedTerminal |> should be False

    [<Test>]
    member _.GuaranteedWin_Condition1_TerminatesEarly() =
        // Root (P1's turn) with 3 actions:
        //   action 0: terminal game state where P1 wins → immediately Terminal
        //   action 1: node with children (not terminal)
        //   action 2: node with children (not terminal)
        // After action 0 is resolved as Terminal with P1=1.0 (condition 1 at root),
        // the root itself resolves.
        let child0 = terminal_game_state(p1, 1) :> ICoreState
        let child1 = node_builder(p2, 1, 0, 2, node_builder(p1, 2, 0, 5)).build () :> ICoreState
        let child2 = node_builder(p2, 1, 0, 3, node_builder(p1, 2, 0, 6)).build () :> ICoreState
        let rootState = multi_det_state(p1, 0, [ child0; child1; child2 ])
        let mctsRoot = MCTSState(rootState :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with MaxSimulations = 100 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        // P1 has a guaranteed win (action 0), so condition 1 should resolve the root
        logInfo.reachedTerminal |> should be True
        mctsRoot.Rollouts |> should be (lessThan 10)
