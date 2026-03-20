module KjarniTest.MCTS.StochasticTest

open NUnit.Framework
open FsUnit

open Kjarni
open Kjarni.MCTS.Types
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.AI
open KjarniTest.TestTypes

// ────────────────────────────────────────────────────────────────
// Helper: terminal node (no children) used as stochastic outcomes
// ────────────────────────────────────────────────────────────────
let terminalNode player hash =
    node (player, 0, 0, hash)

// ────────────────────────────────────────────────────────────────
// Test-local state that mixes deterministic and stochastic actions
// ────────────────────────────────────────────────────────────────

/// A simple deterministic action wrapping an ICoreState.
type simple_det_action(target: ICoreState) =
    interface IDeterministicCoreAction with
        member _.State() = target

/// A state whose Actions() array contains both deterministic and stochastic entries.
type mixed_state(playerTurn, hash, deterministicChildren: ICoreState list, stochasticOutcomes: (int * ICoreState) list) =
    interface ICoreState with
        member _.PlayerTurn = playerTurn
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0

        member _.Actions() =
            let detActions =
                deterministicChildren
                |> List.map (fun s -> Deterministic(simple_det_action s :> IDeterministicCoreAction))
                |> Array.ofList

            let stochAction =
                let outcomeArray = Array.ofList stochasticOutcomes
                [| Stochastic(stochastic_action outcomeArray :> IStochasticCoreAction) |]

            Array.append detActions stochAction

        member _.Scores() = Array.zeroCreate<float> 2

    override _.GetHashCode() = hash
    override _.Equals other = hash = other.GetHashCode()

/// A state whose Actions() contains a single deterministic action pointing to a target.
type single_det_state(playerTurn, hash, target: ICoreState) =
    interface ICoreState with
        member _.PlayerTurn = playerTurn
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member _.Actions() = [| Deterministic(simple_det_action target :> IDeterministicCoreAction) |]
        member _.Scores() = Array.zeroCreate<float> 2

    override _.GetHashCode() = hash
    override _.Equals other = hash = other.GetHashCode()

// ────────────────────────────────────────────────────────────────
// rollStochasticAction
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type RollStochasticActionTests() =

    [<Test>]
    member _.SingleOutcome_AlwaysReturnsZero() =
        for _ in 1 .. 100 do
            rollStochasticAction [| 1 |] |> should equal 0

    [<Test>]
    member _.Distribution_ConvergesOverManyTrials() =
        // Weights: 1, 3 → expected probabilities: 0.25, 0.75
        let counts = [| 0; 0 |]
        let trials = 10000

        for _ in 1 .. trials do
            let i = rollStochasticAction [| 1; 3 |]
            counts.[i] <- counts.[i] + 1

        let rate0 = float counts.[0] / float trials
        let rate1 = float counts.[1] / float trials

        rate0 |> should (equalWithin 0.05) 0.25
        rate1 |> should (equalWithin 0.05) 0.75

    [<Test>]
    member _.EqualWeights_DistributesEvenly() =
        let counts = [| 0; 0; 0 |]
        let trials = 9000

        for _ in 1 .. trials do
            let i = rollStochasticAction [| 1; 1; 1 |]
            counts.[i] <- counts.[i] + 1

        for c in counts do
            float c / float trials |> should (equalWithin 0.05) (1.0 / 3.0)

// ────────────────────────────────────────────────────────────────
// expand — Stochastic branch
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticExpansionTests() =

    let makeStochasticRoot () =
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (3, outcomeB) ])
        MCTSState(root :> ICoreState)

    [<Test>]
    member _.ExpandCreatesStochasticActionWithCorrectWeights() =
        let root = makeStochasticRoot ()
        // Before expansion the single action should be Unexplored
        match root.Actions.[0] with
        | Unexplored (Stochastic _) -> ()
        | a -> Assert.Fail $"Expected Unexplored Stochastic, got %A{a}"

        let _ = expand (root, 0)

        match root.Actions.[0] with
        | StochasticAction outcomes ->
            outcomes |> should haveLength 2
            outcomes.[0].ProbabilityWeight |> should equal 1
            outcomes.[1].ProbabilityWeight |> should equal 3
        | a -> Assert.Fail $"Expected StochasticAction, got %A{a}"

    [<Test>]
    member _.ExpandReturnsAnMCTSStateFromOutcomes() =
        let root = makeStochasticRoot ()
        let expanded = expand (root, 0)

        // The returned state should be one of the outcome states (hash 10 or 11)
        let hash = expanded.State.GetHashCode()
        hash |> should be (inRange 10 11)
        expanded.Rollouts |> should equal 0

    [<Test>]
    member _.ExpandAlreadyExpanded_Throws() =
        let root = makeStochasticRoot ()
        expand (root, 0) |> ignore

        (fun () -> expand (root, 0) |> ignore)
        |> should (throwWithMessage "Target action is already expanded") typeof<System.Exception>

// ────────────────────────────────────────────────────────────────
// select — Stochastic branches
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticselectTests() =

    let makeStochasticRoot () =
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (3, outcomeB) ])
        MCTSState(root :> ICoreState)

    [<Test>]
    member _.UnexpandedStochastic_ReturnsCandidate() =
        let root = makeStochasticRoot ()
        // The only action is Unexplored (Stochastic _), so select should
        // return Candidate for expansion.
        match select (sqrt 2.) None root with
        | Candidate (ancestors, idx) ->
            ancestors |> should haveLength 1
            idx |> should equal 0
        | r -> Assert.Fail $"Expected Candidate, got %A{r}"

    [<Test>]
    member _.ExpandedStochastic_AllUnvisited_ReturnsStochasticCandidate() =
        let root = makeStochasticRoot ()
        // Expand the stochastic action (creates StochasticAction with outcome states)
        let _ = expand (root, 0)
        // Give the root a rollout so it's no longer zero (otherwise actionEvaluator
        // treats StochasticAction with 0 total rollouts as "unexplored" score=10,
        // but since it's already expanded, select will follow the StochasticAction
        // branch, not the Unexplored branch).
        root.Rollouts <- 1
        root.WinCounts <- [| 1.; 0. |]

        let result = select (sqrt 2.) None root

        // Outcomes have 0 rollouts, so select should return StochasticCandidate
        match result with
        | StochasticCandidate (ancestors, actionIdx, outcomeIdx) ->
            ancestors |> should haveLength 1
            actionIdx |> should equal 0
            outcomeIdx |> should be (greaterThanOrEqualTo 0)
            outcomeIdx |> should be (lessThan 2)
        | r -> Assert.Fail $"Expected StochasticCandidate, got %A{r}"

    [<Test>]
    member _.ExpandedStochastic_VisitedOutcome_RecursesInto() =
        // Build a stochastic root where one outcome leads to a node with children
        let childA = node_builder(p2, 1, 0, 20, node_builder (p1, 2, 0, 30)).build ()
        let childB = terminalNode p2 21

        let root = stochastic_node (p1, 0, 0, 0, [ (1, childA); (1, childB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        // Expand the stochastic action
        let _ = expand (mctsRoot, 0)

        // Give all states rollouts so select recurses through the stochastic outcomes
        mctsRoot.Rollouts <- 10
        mctsRoot.WinCounts <- [| 5.; 5. |]

        match mctsRoot.Actions.[0] with
        | StochasticAction outcomes ->
            for o in outcomes do
                o.State.Rollouts <- 5
                o.State.WinCounts <- [| 2.; 3. |]
        | _ -> Assert.Fail "Expected StochasticAction"

        // select should recurse into one of the outcomes.
        // childA has children so it could yield Candidate.
        // childB is terminal so it could yield Exhausted.
        // Either way it should not be a StochasticCandidate since outcomes are visited.
        let result = select (sqrt 2.) None mctsRoot

        match result with
        | StochasticCandidate _ -> Assert.Fail "Expected recursion into visited outcome, not StochasticCandidate"
        | _ -> () // Candidate or Exhausted are both valid

// ────────────────────────────────────────────────────────────────
// sampledWinRate
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type SampledWinRateTests() =

    let makeOutcome weight rollouts p1Wins p2Wins =
        let dummyState = MCTSState(terminalNode p1 0)
        dummyState.Rollouts <- rollouts
        dummyState.WinCounts <- [| p1Wins; p2Wins |]
        { ProbabilityWeight = weight; State = dummyState }

    [<Test>]
    member _.AllUnvisited_ReturnsZero() =
        let outcomes =
            [| makeOutcome 1 0 0. 0.
               makeOutcome 3 0 0. 0. |]

        sampledWinRate outcomes Player.Player1 |> should equal 0.

    [<Test>]
    member _.SingleVisitedOutcome_ReturnsItsWinRate() =
        let outcomes =
            [| makeOutcome 1 10 8. 2. // WinRate P1 = 0.8
               makeOutcome 3 0 0. 0. |]

        sampledWinRate outcomes Player.Player1 |> should (equalWithin 0.001) 0.8

    [<Test>]
    member _.AllVisited_WeightedAverage() =
        // Outcome 0: weight=1, WinRate P1 = 1.0 (10/10)
        // Outcome 1: weight=3, WinRate P1 = 0.0 (0/10)
        // Expected: (1*1.0 + 3*0.0) / (1+3) = 0.25
        let outcomes =
            [| makeOutcome 1 10 10. 0.
               makeOutcome 3 10 0. 10. |]

        sampledWinRate outcomes Player.Player1 |> should (equalWithin 0.001) 0.25

    [<Test>]
    member _.EqualWeights_SimpleAverage() =
        // Outcome 0: weight=1, WinRate P1 = 0.5
        // Outcome 1: weight=1, WinRate P1 = 1.0
        // Expected: (1*0.5 + 1*1.0) / (1+1) = 0.75
        let outcomes =
            [| makeOutcome 1 10 5. 5.
               makeOutcome 1 10 10. 0. |]

        sampledWinRate outcomes Player.Player1 |> should (equalWithin 0.001) 0.75

// ────────────────────────────────────────────────────────────────
// actionEvaluator — StochasticAction branch
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticActionEvaluatorTests() =

    let dummyRoot () =
        let root = MCTSState(terminalNode p1 0)
        root.Rollouts <- 10
        root.WinCounts <- [| 5.; 5. |]
        root

    let makeOutcome weight rollouts p1Wins p2Wins =
        let s = MCTSState(terminalNode p1 0)
        s.Rollouts <- rollouts
        s.WinCounts <- [| p1Wins; p2Wins |]
        { ProbabilityWeight = weight; State = s }

    [<Test>]
    member _.ZeroRollouts_TreatedAsUnexplored() =
        let root = dummyRoot ()
        let outcomes =
            [| makeOutcome 1 0 0. 0.
               makeOutcome 3 0 0. 0. |]

        // Root has 2 actions (from terminalNode which has no children, but
        // dummyRoot wraps a terminalNode). We test the evaluator with a
        // StochasticAction as if it were at index 0 of a single-action node.
        // For PUCT with uniform prior (P=1/1=1.0 for single action), zero
        // rollouts: C * P * sqrt(N_parent) / 1 = sqrt(2) * 1.0 * sqrt(10) ≈ 4.47
        // But the root has no Actions array that matches this scenario, so we
        // just set up a node with one action to get the right prior.
        let stochRoot = MCTSState(terminalNode p1 0)
        stochRoot.Rollouts <- 10
        stochRoot.WinCounts <- [| 5.; 5. |]
        stochRoot.Actions <- [| StochasticAction outcomes |]

        let score = actionEvaluator (sqrt 2.) stochRoot 0 (StochasticAction outcomes)
        // Zero total rollouts → PUCT with uniform prior P=1.0 (single action):
        // C * P * sqrt(N_parent) / 1 = sqrt(2) * 1.0 * sqrt(10) ≈ 4.47
        score |> should (equalWithin 0.01) (sqrt 2. * sqrt 10.)

    [<Test>]
    member _.WithRollouts_CombinesWinRateAndExploration() =
        let root = dummyRoot ()
        let outcomes =
            [| makeOutcome 1 5 5. 0.   // WinRate P1 = 1.0
               makeOutcome 1 5 0. 5. |] // WinRate P1 = 0.0

        // Set up root with a single stochastic action for correct uniform prior
        root.Actions <- [| StochasticAction outcomes |]

        let score = actionEvaluator (sqrt 2.) root 0 (StochasticAction outcomes)
        // sampledWinRate = (1*1.0 + 1*0.0) / 2 = 0.5
        // totalRollouts = 10
        // PUCT: winRate + C * P * sqrt(N_parent) / (1 + N_action)
        //     = 0.5 + sqrt(2) * 1.0 * sqrt(10) / (1 + 10)
        //     = 0.5 + 4.47 / 11 ≈ 0.5 + 0.406 ≈ 0.906
        score |> should be (greaterThan 0.85)
        score |> should be (lessThan 0.97)

// ────────────────────────────────────────────────────────────────
// backPropagate through stochastic outcome states
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticBackPropagateTests() =

    [<Test>]
    member _.PropagatesThrough_StochasticOutcomeState() =
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (3, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        // Expand the stochastic action
        let expandedState = expand (mctsRoot, 0)

        // Simulate the search loop: backPropagate through expandedState and root
        let outcome = [| 1.; 0. |]
        backPropagate [ expandedState; mctsRoot ] outcome

        mctsRoot.Rollouts |> should equal 1
        expandedState.Rollouts |> should equal 1
        winRate mctsRoot Player.Player1 |> should equal 1.
        winRate expandedState Player.Player1 |> should equal 1.

    [<Test>]
    member _.MultipleSimulations_AccumulateCorrectly() =
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (3, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let _ = expand (mctsRoot, 0)

        match mctsRoot.Actions.[0] with
        | StochasticAction outcomes ->
            // Simulate P1 win through outcome 0
            backPropagate [ outcomes.[0].State; mctsRoot ] [| 1.; 0. |]
            // Simulate P2 win through outcome 1
            backPropagate [ outcomes.[1].State; mctsRoot ] [| 0.; 1. |]

            mctsRoot.Rollouts |> should equal 2
            outcomes.[0].State.Rollouts |> should equal 1
            outcomes.[1].State.Rollouts |> should equal 1
            winRate mctsRoot Player.Player1 |> should (equalWithin 0.001) 0.5
            winRate outcomes.[0].State Player.Player1 |> should equal 1.
            winRate outcomes.[1].State Player.Player1 |> should equal 0.
        | _ -> Assert.Fail "Expected StochasticAction"

// ────────────────────────────────────────────────────────────────
// extractBestPath — stops at stochastic actions
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticExtractBestPathTests() =

    [<Test>]
    member _.StopsAtStochasticAction() =
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (3, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let _ = expand (mctsRoot, 0)
        mctsRoot.Rollouts <- 10
        mctsRoot.WinCounts <- [| 5.; 5. |]

        // Give outcomes some rollouts for extractionEvaluator to work with
        match mctsRoot.Actions.[0] with
        | StochasticAction outcomes ->
            outcomes.[0].State.Rollouts <- 5
            outcomes.[0].State.WinCounts <- [| 5.; 0. |]
            outcomes.[1].State.Rollouts <- 5
            outcomes.[1].State.WinCounts <- [| 0.; 5. |]
        | _ -> Assert.Fail "Expected StochasticAction"

        let path = extractBestPath mctsRoot

        // Should contain just the stochastic action index (0) then stop
        path |> should haveLength 1
        path.[0] |> should equal 0

    [<Test>]
    member _.DeterministicThenStochastic_ExtractsCorrectPath() =
        // Root → det child → stochastic child
        // Use single_det_state so the root has a deterministic action to the stochastic middle node
        let leafA = terminalNode p1 100
        let leafB = terminalNode p2 101

        let middleNode = stochastic_node (p2, 1, 0, 50, [ (1, leafA); (1, leafB) ])
        let rootNode = single_det_state (p1, 0, middleNode :> ICoreState)

        let mctsRoot = MCTSState(rootNode :> ICoreState)

        // Expand the deterministic action at index 0
        let middleMCTS = expand (mctsRoot, 0)
        mctsRoot.Rollouts <- 10
        mctsRoot.WinCounts <- [| 5.; 5. |]
        middleMCTS.Rollouts <- 10
        middleMCTS.WinCounts <- [| 5.; 5. |]

        // Expand the stochastic action in the middle node
        let _ = expand (middleMCTS, 0)
        match middleMCTS.Actions.[0] with
        | StochasticAction outcomes ->
            outcomes.[0].State.Rollouts <- 5
            outcomes.[0].State.WinCounts <- [| 5.; 0. |]
            outcomes.[1].State.Rollouts <- 5
            outcomes.[1].State.WinCounts <- [| 0.; 5. |]
        | _ -> Assert.Fail "Expected StochasticAction"

        let path = extractBestPath mctsRoot

        // Should be [0; 0] — deterministic at root, then stochastic stops
        path |> should haveLength 2
        path.[0] |> should equal 0
        path.[1] |> should equal 0

// ────────────────────────────────────────────────────────────────
// simulate — stochastic rollout
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticSimulateTests() =

    [<Test>]
    member _.SimulateStochastic_ReturnsOneHotOutcome() =
        // A root with a stochastic action leading to terminal outcomes
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (1, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let result = simulate defaultMaxRolloutDepth mctsRoot

        // Should be a one-hot outcome (either [1,0] or [0,1])
        result |> should haveLength 2
        let sum = result.[0] + result.[1]
        sum |> should (equalWithin 0.001) 1.0
        // One entry should be 1, the other 0
        (result.[0] = 1. || result.[1] = 1.) |> should be True

    [<Test>]
    member _.SimulateStochastic_DistributionConverges() =
        // Weights 1:3 → P1 wins 25%, P2 wins 75%
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (3, outcomeB) ])

        let mutable p1Wins = 0.
        let trials = 2000

        for _ in 1 .. trials do
            let result = simulate defaultMaxRolloutDepth (MCTSState(root :> ICoreState))
            p1Wins <- p1Wins + result.[int Player.Player1]

        let p1Rate = p1Wins / float trials
        p1Rate |> should (equalWithin 0.05) 0.25

// ────────────────────────────────────────────────────────────────
// search — full integration with stochastic actions
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type StochasticSearchIntegrationTests() =

    [<Test>]
    member _.SearchWithStochasticAction_Completes() =
        // Root with a stochastic action: outcome A is P1 terminal, outcome B is P2 terminal.
        // Weights are equal, so the expected result is a 50/50 win rate.
        // Because both outcomes are terminal game states, terminal propagation
        // resolves the tree after just 2 rollouts (one per outcome).
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (1, outcomeA); (1, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with SearchTime = Seconds 5; MaxSimulations = 100 })
        let result = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        // Search terminates early via terminal propagation
        logInfo.reachedTerminal |> should be True
        result.Rollouts |> should be (lessThan 100)
        mctsRoot.Rollouts |> should equal result.Rollouts

    [<Test>]
    member _.SearchWithMixedActions_PrefersWinningPath() =
        // Root has two deterministic children and one stochastic action.
        // Det child 0: terminal, P2 turn → P2 wins (bad for P1)
        // Det child 1: terminal, P1 turn → P1 wins (good for P1)
        // Stochastic action: 50/50 between P1 win and P2 win
        // MCTS should prefer the deterministic P1-win path (index 1).
        let detChildBad = node_builder(p2, 1, 0, 10).build ()
        let detChildGood = node_builder(p1, 1, 0, 11).build ()
        let stochOutcomeA = terminalNode p1 20
        let stochOutcomeB = terminalNode p2 21

        let root =
            mixed_state (
                p1, 0,
                [ detChildBad :> ICoreState; detChildGood :> ICoreState ],
                [ (1, stochOutcomeA :> ICoreState); (1, stochOutcomeB :> ICoreState) ]
            )

        let mctsRoot = MCTSState(root :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with SearchTime = Seconds 5; MaxSimulations = 500 })
        let _ = mcts.RunSimulation(mctsRoot)

        let bestPath = extractBestPath mctsRoot

        // Best first action should be 1 (the deterministic P1-win)
        bestPath |> should not' (be Empty)
        bestPath.[0] |> should equal 1

    [<Test>]
    member _.SearchWithHeavilyBiasedStochastic_ConvergesCorrectly() =
        // Root has a single stochastic action with:
        // weight 9 → P1 wins, weight 1 → P2 wins
        // Both outcomes are terminal game states, so terminal propagation
        // resolves the tree with the weighted average outcome [0.9, 0.1].
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node (p1, 0, 0, 0, [ (9, outcomeA); (1, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let mcts = MonteCarloTreeSearch({ MCTSConfig.Default with SearchTime = Seconds 5; MaxSimulations = 1000 })
        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        // Search terminates early via terminal propagation
        logInfo.reachedTerminal |> should be True

        // The root's only action should now be Terminal with the weighted average
        match mctsRoot.Actions.[0] with
        | Terminal outcome ->
            outcome.[int Player.Player1] |> should (equalWithin 0.001) 0.9
            outcome.[int Player.Player2] |> should (equalWithin 0.001) 0.1
        | a -> Assert.Fail $"Expected Terminal, got %A{a}"
