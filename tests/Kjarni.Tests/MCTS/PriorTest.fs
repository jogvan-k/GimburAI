module KjarniTest.MCTS.PriorTest

open System
open System.Collections.Generic
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open NUnit.Framework
open FsUnit

open Kjarni
open Kjarni.MCTS.Types
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.AI
open KjarniTest.TestTypes

// ────────────────────────────────────────────────────────────────
// Helper: local helpers (some mirror StochasticTest helpers)
// ────────────────────────────────────────────────────────────────

let terminalNode player hash = node(player, 0, 0, hash)

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

// ────────────────────────────────────────────────────────────────
// Helper types
// ────────────────────────────────────────────────────────────────

/// A mock IPriorClient that records requests and returns pre-configured
/// responses on CollectPriors.
type MockPriorClient() =
    let _requests = ResizeArray<int64 * ICoreState * ICoreState[] * int * int>()
    let _responses = Queue<PriorResponse>()
    let mutable _flushed = false

    /// All prior requests that were enqueued (nodeId, parentState, states, actingPlayer, depth).
    member _.Requests = _requests |> Seq.toList

    /// Schedule a response to be returned on the next CollectPriors call.
    member _.EnqueueResponse(nodeId: int64, winProbs: float[]) =
        _responses.Enqueue(PriorResponse(nodeId, winProbs, Array.empty))

    /// Whether Flush was called at least once.
    member _.WasFlushed = _flushed

    interface IPriorClient with
        member _.ShouldRequestPrior(_parentState) = true

        member _.RequestPrior(nodeId, parentState, states, actingPlayer, depth) =
            _requests.Add((nodeId, parentState, states, actingPlayer, depth))
            states.Length

        member _.CollectPriors(knownNodeIds: IReadOnlySet<int64>) =
            let matched = _responses |> Seq.filter (fun r -> knownNodeIds.Contains(r.NodeId)) |> Seq.toArray
            let remaining = _responses |> Seq.filter (fun r -> not (knownNodeIds.Contains(r.NodeId))) |> Seq.toList
            _responses.Clear()
            for r in remaining do _responses.Enqueue(r)
            matched

        member _.Flush(_knownNodeIds: IReadOnlySet<int64>) =
            _flushed <- true
            _responses.Clear()

/// A mock prior client that immediately enqueues a response for every request
/// using a callback that produces win probabilities from the states.
type AutoRespondPriorClient(winProbFn: ICoreState[] -> float[]) =
    let _requests = ResizeArray<int64 * ICoreState * ICoreState[] * int * int>()
    let _responses = Queue<PriorResponse>()
    let mutable _flushed = false

    member _.Requests = _requests |> Seq.toList
    member _.WasFlushed = _flushed

    interface IPriorClient with
        member _.ShouldRequestPrior(_parentState) = true

        member _.RequestPrior(nodeId, parentState, states, actingPlayer, depth) =
            _requests.Add((nodeId, parentState, states, actingPlayer, depth))
            let winProbs = winProbFn states
            _responses.Enqueue(PriorResponse(nodeId, winProbs, Array.empty))
            states.Length

        member _.CollectPriors(knownNodeIds: IReadOnlySet<int64>) =
            let matched = _responses |> Seq.filter (fun r -> knownNodeIds.Contains(r.NodeId)) |> Seq.toArray
            let remaining = _responses |> Seq.filter (fun r -> not (knownNodeIds.Contains(r.NodeId))) |> Seq.toList
            _responses.Clear()
            for r in remaining do _responses.Enqueue(r)
            matched

        member _.Flush(_knownNodeIds: IReadOnlySet<int64>) =
            _flushed <- true
            _responses.Clear()

type DenseRespondPriorClient(densePriors: float[]) =
    let _responses = Queue<PriorResponse>()

    interface IPriorClient with
        member _.ShouldRequestPrior(_parentState) = true

        member _.RequestPrior(nodeId, _parentState, states, _actingPlayer, _depth) =
            _responses.Enqueue(
                PriorResponse(nodeId, Array.create states.Length 1., Array.empty, densePriors))
            states.Length

        member _.CollectPriors(knownNodeIds: IReadOnlySet<int64>) =
            _responses |> Seq.filter (fun r -> knownNodeIds.Contains(r.NodeId)) |> Seq.toArray

        member _.Flush(_knownNodeIds: IReadOnlySet<int64>) =
            _responses.Clear()

type ValueRespondPriorClient(valueEstimates: float[]) =
    let _responses = Queue<PriorResponse>()

    interface IPriorClient with
        member _.ShouldRequestPrior(_parentState) = true

        member _.RequestPrior(nodeId, _parentState, states, _actingPlayer, _depth) =
            _responses.Enqueue(PriorResponse(nodeId, Array.create states.Length 1., valueEstimates))
            states.Length

        member _.CollectPriors(knownNodeIds: IReadOnlySet<int64>) =
            _responses |> Seq.filter (fun r -> knownNodeIds.Contains(r.NodeId)) |> Seq.toArray

        member _.Flush(_knownNodeIds: IReadOnlySet<int64>) =
            _responses.Clear()

type CombinedRespondPriorClient(valueEstimates: float[]) =
    let responses = Queue<PriorResponse>()
    let mutable requests = 0

    member _.Requests = requests

    interface IPriorClient with
        member _.ShouldRequestPrior(_parentState) = true
        member _.RequestPrior(nodeId, _parentState, states, _actingPlayer, _depth) =
            requests <- requests + 1
            responses.Enqueue(PriorResponse(nodeId, Array.create states.Length 1., valueEstimates))
            1
        member _.CollectPriors(knownNodeIds: IReadOnlySet<int64>) =
            let matched = responses |> Seq.filter (fun r -> knownNodeIds.Contains(r.NodeId)) |> Seq.toArray
            let remaining = responses |> Seq.filter (fun r -> not (knownNodeIds.Contains(r.NodeId))) |> Seq.toArray
            responses.Clear()
            for response in remaining do responses.Enqueue(response)
            matched
        member _.Flush(_knownNodeIds: IReadOnlySet<int64>) = responses.Clear()

type RejectingLeafEvaluator() =
    let mutable enqueues = 0
    member _.Enqueues = enqueues
    interface ILeafEvaluator with
        member _.Enqueue(_, _, _) =
            enqueues <- enqueues + 1
            false
        member _.Collect(_) = Array.empty
        member _.WaitForResults(_) = false
        member _.Cancel(_) = ()

type DelayedCombinedPriorClient(valueEstimates: float[]) =
    let requests = ConcurrentDictionary<int64, int>()
    let responses = ConcurrentQueue<PriorResponse>()
    let release = new ManualResetEventSlim(false)

    member _.RequestCount = requests.Count
    member _.Release() = release.Set()

    interface IPriorClient with
        member _.ShouldRequestPrior(_parentState) = true
        member _.RequestPrior(nodeId, _parentState, states, _actingPlayer, _depth) =
            requests.[nodeId] <- states.Length
            1
        member _.CollectPriors(knownNodeIds: IReadOnlySet<int64>) =
            if release.IsSet then
                for KeyValue(nodeId, stateCount) in requests do
                    if knownNodeIds.Contains(nodeId) && (requests.TryRemove(nodeId) |> fst) then
                        responses.Enqueue(
                            PriorResponse(nodeId, Array.create stateCount 1., valueEstimates))
            let matched = ResizeArray<PriorResponse>()
            let keep = ResizeArray<PriorResponse>()
            let mutable response = Unchecked.defaultof<PriorResponse>
            while responses.TryDequeue(&response) do
                if knownNodeIds.Contains(response.NodeId) then matched.Add(response)
                else keep.Add(response)
            for item in keep do responses.Enqueue(item)
            matched.ToArray()
        member _.Flush(knownNodeIds: IReadOnlySet<int64>) =
            for nodeId in knownNodeIds do requests.TryRemove(nodeId) |> ignore


// ────────────────────────────────────────────────────────────────
// MCTSState.NodeId uniqueness
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type NodeIdTests() =

    [<Test>]
    member _.NodeIds_AreUnique() =
        let s1 = MCTSState(node(p1, 0, 0, 0))
        let s2 = MCTSState(node(p1, 0, 0, 1))
        let s3 = MCTSState(node(p2, 1, 0, 2))

        s1.NodeId |> should not' (equal s2.NodeId)
        s2.NodeId |> should not' (equal s3.NodeId)
        s1.NodeId |> should not' (equal s3.NodeId)

    [<Test>]
    member _.NodeIds_ArePositive() =
        let s = MCTSState(node(p1, 0, 0, 0))
        s.NodeId |> should be (greaterThan 0L)

    [<Test>]
    member _.NodeIds_AreMonotonicallyIncreasing() =
        let s1 = MCTSState(node(p1, 0, 0, 0))
        let s2 = MCTSState(node(p1, 0, 0, 1))
        s2.NodeId |> should be (greaterThan s1.NodeId)

// ────────────────────────────────────────────────────────────────
// actionEvaluator — PUCT with explicit priors
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type PUCTActionEvaluatorTests() =

    let termNode player hash = node(player, 0, 0, hash)

    [<Test>]
    member _.UniformPrior_SingleAction_Unexplored() =
        // Single action → uniform prior P = 1/1 = 1.0
        // PUCT: C * P * sqrt(N_parent) / (1 + 0)
        let root = MCTSState(termNode p1 0)
        root.Rollouts <- 16
        root.WinCounts <- [| 8.; 8. |]
        root.Actions <- [| Unexplored(Deterministic(action(termNode p1 0, termNode p2 1))) |]

        let score = actionEvaluator (sqrt 2.) root 0 root.Actions.[0]
        // C * 1.0 * sqrt(16) / 1 = sqrt(2) * 4 ≈ 5.657
        score |> should (equalWithin 0.001) (sqrt 2. * 4.)

    [<Test>]
    member _.UniformPrior_TwoActions_Unexplored() =
        // Two actions → uniform prior P = 1/2 = 0.5
        let child1 = termNode p2 1
        let child2 = termNode p2 2
        let root = MCTSState(termNode p1 0)
        root.Rollouts <- 16
        root.WinCounts <- [| 8.; 8. |]
        root.Actions <-
            [| Unexplored(Deterministic(action(termNode p1 0, child1)))
               Unexplored(Deterministic(action(termNode p1 0, child2))) |]

        let score = actionEvaluator (sqrt 2.) root 0 root.Actions.[0]
        // C * 0.5 * sqrt(16) / 1 = sqrt(2) * 0.5 * 4 ≈ 2.828
        score |> should (equalWithin 0.001) (sqrt 2. * 0.5 * 4.)

    [<Test>]
    member _.ExplicitPriors_OverrideUniform() =
        // Two actions with priors [0.8, 0.2]
        let child1 = termNode p2 1
        let child2 = termNode p2 2
        let root = MCTSState(termNode p1 0)
        root.Rollouts <- 100
        root.WinCounts <- [| 50.; 50. |]
        root.Actions <-
            [| Unexplored(Deterministic(action(termNode p1 0, child1)))
               Unexplored(Deterministic(action(termNode p1 0, child2))) |]
        root.Priors <- Some [| 0.8; 0.2 |]

        let score0 = actionEvaluator (sqrt 2.) root 0 root.Actions.[0]
        let score1 = actionEvaluator (sqrt 2.) root 1 root.Actions.[1]

        // action 0: C * 0.8 * sqrt(100) / 1 = sqrt(2) * 0.8 * 10 = 11.314
        score0 |> should (equalWithin 0.001) (sqrt 2. * 0.8 * 10.)
        // action 1: C * 0.2 * sqrt(100) / 1 = sqrt(2) * 0.2 * 10 = 2.828
        score1 |> should (equalWithin 0.001) (sqrt 2. * 0.2 * 10.)

        // Higher prior → higher score
        score0 |> should be (greaterThan score1)

    [<Test>]
    member _.ExplicitPriors_WithVisitedAction() =
        // Two actions with priors [0.7, 0.3]
        // Action 0 is expanded with 10 rollouts, action 1 is unexplored
        let child1State = MCTSState(termNode p2 1)
        child1State.Rollouts <- 10
        child1State.WinCounts <- [| 6.; 4. |] // winRate for P1 = 0.6

        let root = MCTSState(termNode p1 0)
        root.Rollouts <- 20
        root.WinCounts <- [| 12.; 8. |]
        root.Actions <-
            [| DeterministicAction child1State
               Unexplored(Deterministic(action(termNode p1 0, termNode p2 2))) |]
        root.Priors <- Some [| 0.7; 0.3 |]
        root.ActionStats.[0].CompletedVisits <- 10
        root.ActionStats.[0].ValueSums.[0] <- 6.
        root.ActionStats.[0].ValueSums.[1] <- 4.

        let score0 = actionEvaluator (sqrt 2.) root 0 root.Actions.[0]
        // Q = 0.6, C * P * sqrt(N_parent) / (1 + N_action) = sqrt(2) * 0.7 * sqrt(20) / 11
        let expectedExploration0 = sqrt 2. * 0.7 * sqrt 20. / 11.
        score0 |> should (equalWithin 0.001) (0.6 + expectedExploration0)

        let score1 = actionEvaluator (sqrt 2.) root 1 root.Actions.[1]
        // Unexplored: C * P * sqrt(N_parent) / 1 = sqrt(2) * 0.3 * sqrt(20)
        let expectedExploration1 = sqrt 2. * 0.3 * sqrt 20.
        score1 |> should (equalWithin 0.001) expectedExploration1

    [<Test>]
    member _.ExplicitPrior_BreaksInitialSelectionTie() =
        let root = MCTSState(termNode p1 0)
        root.Actions <-
            [| Unexplored(Deterministic(action(termNode p1 0, termNode p2 1)))
               Unexplored(Deterministic(action(termNode p1 0, termNode p2 2))) |]
        root.Priors <- Some [| 0.875; 0.125 |]

        match select (sqrt 2.) root with
        | Candidate (_, index) -> index |> should equal 0
        | result -> Assert.Fail $"Expected Candidate, got {result}"

    [<Test>]
    member _.ExtractBestPath_UsesPriorWhenActionsAreUnvisited() =
        let root = MCTSState(termNode p1 0)
        root.Actions <-
            [| Unexplored(Deterministic(action(termNode p1 0, termNode p2 1)))
               Unexplored(Deterministic(action(termNode p1 0, termNode p2 2))) |]
        root.Priors <- Some [| 0.9; 0.1 |]

        extractBestPath root |> should equal [ 0 ]

    [<Test>]
    member _.ExtractBestPath_PrefersVisitsOverHigherQ() =
        let root = MCTSState(termNode p1 0)
        root.Actions <-
            [| Unexplored(Deterministic(action(termNode p1 0, termNode p2 1)))
               Unexplored(Deterministic(action(termNode p1 0, termNode p2 2))) |]
        root.ActionStats.[0].CompletedVisits <- 10
        root.ActionStats.[0].ValueSums.[0] <- 4.
        root.ActionStats.[1].CompletedVisits <- 1
        root.ActionStats.[1].ValueSums.[0] <- 1.

        extractBestPath root |> should equal [ 0 ]

// ────────────────────────────────────────────────────────────────
// explorationRate — PUCT formula
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type ExplorationRateTests() =

    [<Test>]
    member _.ZeroChildVisits_MaxExploration() =
        // C * P * sqrt(N_parent) / (1 + 0)
        let rate = explorationRate 2.0 100 0 0.5
        rate |> should (equalWithin 0.001) (2.0 * 0.5 * sqrt 100.)

    [<Test>]
    member _.HighChildVisits_LowExploration() =
        // C * P * sqrt(N_parent) / (1 + 100)
        let rate = explorationRate 2.0 100 100 0.5
        rate |> should (equalWithin 0.001) (2.0 * 0.5 * sqrt 100. / 101.)

    [<Test>]
    member _.PriorAffectsExploration() =
        let rateHigh = explorationRate 2.0 100 5 0.9
        let rateLow = explorationRate 2.0 100 5 0.1
        rateHigh |> should be (greaterThan rateLow)
        // Ratio should be 9:1
        (rateHigh / rateLow) |> should (equalWithin 0.001) 9.0

// ────────────────────────────────────────────────────────────────
// collectActionStates
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type CollectActionStatesTests() =

    [<Test>]
    member _.DeterministicActions_OneStatePerAction() =
        let child1 = node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20)).build()
        let child2 = node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21)).build()
        let rootNode = node_builder(p1, 0, 0, 0).addChildren([node_builder(p2, 1, 0, 10); node_builder(p2, 1, 0, 11)]).build()

        let root = MCTSState(rootNode :> ICoreState)

        let (states, layout, weights) = collectActionStates root

        // Two deterministic actions → two states
        states |> should haveLength 2
        layout |> should haveLength 2
        layout.[0] |> should equal 1
        layout.[1] |> should equal 1
        weights.[0] |> should equal [| 1 |]
        weights.[1] |> should equal [| 1 |]

    [<Test>]
    member _.StochasticAction_MultipleStatesPerAction() =
        let outcomeA = terminalNode p1 10
        let outcomeB = terminalNode p2 11
        let root = stochastic_node(p1, 0, 0, 0, [ (2, outcomeA); (3, outcomeB) ])
        let mctsRoot = MCTSState(root :> ICoreState)

        let (states, layout, weights) = collectActionStates mctsRoot

        // One stochastic action with 2 outcomes → 2 states
        states |> should haveLength 2
        layout |> should haveLength 1
        layout.[0] |> should equal 2
        weights.[0] |> should equal [| 2; 3 |]

    [<Test>]
    member _.MixedActions_CorrectLayout() =
        let detChild = node(p2, 1, 0, 10)
        let stochOutA = terminalNode p1 20
        let stochOutB = terminalNode p2 21

        let root =
            mixed_state(
                p1, 0,
                [ detChild :> ICoreState ],
                [ (1, stochOutA :> ICoreState); (1, stochOutB :> ICoreState) ]
            )
        let mctsRoot = MCTSState(root :> ICoreState)

        let (states, layout, weights) = collectActionStates mctsRoot

        // 1 deterministic (1 state) + 1 stochastic (2 states) = 3 states
        states |> should haveLength 3
        layout |> should haveLength 2
        layout.[0] |> should equal 1  // deterministic
        layout.[1] |> should equal 2  // stochastic

    [<Test>]
    member _.TerminalAction_ZeroStates() =
        let root = MCTSState(terminalNode p1 0)
        root.Actions <- [| Terminal [| 1.; 0. |] |]

        let (states, layout, weights) = collectActionStates root

        states |> should haveLength 0
        layout |> should haveLength 1
        layout.[0] |> should equal 0
        weights.[0] |> should equal Array.empty<int>

    [<Test>]
    member _.ExpandedDeterministicAction_ReadsChildState() =
        let childNode = node_builder(p2, 1, 0, 10).build()
        let rootNode = node_builder(p1, 0, 0, 0, node_builder(p2, 1, 0, 10)).build()
        let root = MCTSState(rootNode :> ICoreState)

        // Expand the action
        let expanded = expand None (root, 0)

        let (states, layout, _) = collectActionStates root

        states |> should haveLength 1
        layout.[0] |> should equal 1
        // The state should be the expanded child's ICoreState
        Object.ReferenceEquals(states.[0], expanded.State) |> should be True

// ────────────────────────────────────────────────────────────────
// computePriorPolicy
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type ComputePriorPolicyTests() =

    [<Test>]
    member _.TwoDeterministic_NormalizesToOne() =
        // Two deterministic actions: winProbs = [0.6, 0.4]
        let winProbs = [| 0.6; 0.4 |]
        let layout = [| 1; 1 |]
        let weights = [| [| 1 |]; [| 1 |] |]

        let policy = computePriorPolicy winProbs layout weights

        policy |> should haveLength 2
        policy.[0] |> should (equalWithin 0.001) 0.6
        policy.[1] |> should (equalWithin 0.001) 0.4

    [<Test>]
    member _.StochasticAction_WeightedAverage() =
        // One stochastic action with 2 outcomes: weights [1, 3], winProbs [0.8, 0.2]
        // Weighted avg = (1*0.8 + 3*0.2) / (1+3) = 1.4/4 = 0.35
        let winProbs = [| 0.8; 0.2 |]
        let layout = [| 2 |]
        let weights = [| [| 1; 3 |] |]

        let policy = computePriorPolicy winProbs layout weights

        policy |> should haveLength 1
        policy.[0] |> should (equalWithin 0.001) 1.0 // only action, normalized

    [<Test>]
    member _.MixedActions_CorrectNormalization() =
        // Action 0: deterministic, winProb = 0.6
        // Action 1: stochastic with 2 outcomes, weights [1, 1], winProbs [0.8, 0.2]
        //   → weighted avg = (1*0.8 + 1*0.2) / 2 = 0.5
        // Raw: [0.6, 0.5] → normalized: [0.6/1.1, 0.5/1.1] ≈ [0.545, 0.455]
        let winProbs = [| 0.6; 0.8; 0.2 |]
        let layout = [| 1; 2 |]
        let weights = [| [| 1 |]; [| 1; 1 |] |]

        let policy = computePriorPolicy winProbs layout weights

        policy |> should haveLength 2
        let total = policy.[0] + policy.[1]
        total |> should (equalWithin 0.001) 1.0
        policy.[0] |> should (equalWithin 0.001) (0.6 / 1.1)
        policy.[1] |> should (equalWithin 0.001) (0.5 / 1.1)

    [<Test>]
    member _.TerminalAction_GetsZeroPrior() =
        // Action 0: deterministic, winProb = 0.6
        // Action 1: terminal (0 states in layout)
        // Raw: [0.6, 0.0] → normalized: [1.0, 0.0]
        let winProbs = [| 0.6 |]
        let layout = [| 1; 0 |]
        let weights = [| [| 1 |]; Array.empty |]

        let policy = computePriorPolicy winProbs layout weights

        policy |> should haveLength 2
        policy.[0] |> should (equalWithin 0.001) 1.0
        policy.[1] |> should (equalWithin 0.001) 0.0

    [<Test>]
    member _.AllZeroWinProbs_FallsBackToUniform() =
        let winProbs = [| 0.0; 0.0 |]
        let layout = [| 1; 1 |]
        let weights = [| [| 1 |]; [| 1 |] |]

        let policy = computePriorPolicy winProbs layout weights

        policy |> should haveLength 2
        policy.[0] |> should (equalWithin 0.001) 0.5
        policy.[1] |> should (equalWithin 0.001) 0.5

    [<Test>]
    member _.SingleAction_NormalizesToOne() =
        let winProbs = [| 0.3 |]
        let layout = [| 1 |]
        let weights = [| [| 1 |] |]

        let policy = computePriorPolicy winProbs layout weights

        policy |> should haveLength 1
        policy.[0] |> should (equalWithin 0.001) 1.0

    [<Test>]
    member _.MalformedPriors_FallBackToUniform() =
        let malformed =
            [| [| 0.5 |]
               [| 0.5; 0.3; 0.2 |]
               [| Double.NaN; 0.5 |]
               [| Double.PositiveInfinity; 0.5 |]
               [| -0.1; 0.5 |] |]

        for winProbs in malformed do
            let policy = computePriorPolicy winProbs [| 1; 1 |] [| [| 1 |]; [| 1 |] |]
            policy |> should equal [| 0.5; 0.5 |]

// ────────────────────────────────────────────────────────────────
// Search integration with mock IPriorClient
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type PriorSearchIntegrationTests() =

    [<Test>]
    member _.SearchWithPriorClient_RequestsAreFired() =
        // Tree must be deep enough that expanded children have their own actions
        // (requestPrior skips terminal nodes with no actions).
        // root (P1) → child A (P2) → grandchild (P1, terminal)
        //           → child B (P2) → grandchild (P1, terminal)
        let rootNode =
            node_builder(p1, 0, 0, 0)
                .addChildren([
                    node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
                    node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21))
                ])
                .build()

        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let mockClient = MockPriorClient()

        let mcts = MonteCarloTreeSearch(
            { MCTSConfig.Default with
                SearchTime = Seconds 5
                MaxSimulations = 10
                PriorClient = Some (mockClient :> IPriorClient) })

        let _ = mcts.RunSimulation(mctsRoot)

        // At least one prior request should have been fired
        mockClient.Requests.Length |> should be (greaterThan 0)

    [<Test>]
    member _.SearchWithPriorClient_FlushIsCalledAfterSearch() =
        let rootNode = node_builder(p1, 0, 0, 0, node_builder(p2, 1, 0, 10)).build()
        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let mockClient = MockPriorClient()

        let mcts = MonteCarloTreeSearch(
            { MCTSConfig.Default with
                SearchTime = Seconds 5
                MaxSimulations = 5
                PriorClient = Some (mockClient :> IPriorClient) })

        let _ = mcts.RunSimulation(mctsRoot)

        mockClient.WasFlushed |> should be True

    [<Test>]
    member _.SearchWithAutoRespondClient_PriorsAreApplied() =
        // Build a tree deep enough that expanded children have actions.
        // root (P1) → child 0 (P2) → grandchild (P1, terminal)
        //           → child 1 (P2) → grandchild (P1, terminal)
        //           → child 2 (P2) → grandchild (P1, terminal)
        let rootNode =
            node_builder(p1, 0, 0, 0)
                .addChildren([
                    node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
                    node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21))
                    node_builder(p2, 1, 0, 12, node_builder(p1, 2, 0, 22))
                ])
                .build()

        let mctsRoot = MCTSState(rootNode :> ICoreState)

        // NN says: all grandchild states get a win probability, but we just
        // need any non-zero response so priors get applied.
        let autoClient = AutoRespondPriorClient(fun states ->
            states |> Array.map (fun _ -> 0.5)
        )

        let mcts = MonteCarloTreeSearch(
            { MCTSConfig.Default with
                SearchTime = Seconds 5
                MaxSimulations = 200
                PriorClient = Some (autoClient :> IPriorClient) })

        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        // Priors should have been requested and applied
        logInfo.priorActionsRequested |> should be (greaterThan 0)
        logInfo.priorActionsApplied |> should be (greaterThan 0)
        logInfo.priorNodesApplied |> should be (greaterThan 0)

    [<Test>]
    member _.SearchWithDenseResponse_RetainsDensePolicyOnRoot() =
        let rootNode =
            node_builder(p1, 0, 0, 0)
                .addChildren([
                    node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
                    node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21))
                ])
                .build()
        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let dense = [| 0.1; 0.2; 0.7 |]
        let client = DenseRespondPriorClient(dense)
        let mcts =
            MonteCarloTreeSearch(
                { MCTSConfig.Default with
                    MaxSimulations = 2
                    PriorClient = Some (client :> IPriorClient) })

        mcts.RunSimulation(mctsRoot) |> ignore

        mctsRoot.DensePriors |> should equal (Some dense)

    [<Test>]
    member _.SearchWithValueResponse_NormalizesAndStoresPerPlayerValues() =
        let rootNode =
            node_builder(p1, 0, 0, 0)
                .addChildren([
                    node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
                    node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21))
                ])
                .build()
        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let client = ValueRespondPriorClient([| 1.; 3. |])
        let mcts =
            MonteCarloTreeSearch(
                { MCTSConfig.Default with
                    MaxSimulations = 2
                    PriorClient = Some (client :> IPriorClient) })

        mcts.RunSimulation(mctsRoot) |> ignore

        mctsRoot.ValueEstimates |> should equal (Some [| 0.25; 0.75 |])

    [<Test>]
    member _.CombinedPriorResponseSuppliesExpandedNodePolicyAndValueWithoutLeafRequest() =
        let child = node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
        let root = MCTSState(node_builder(p1, 0, 0, 0, child).build())
        let client = CombinedRespondPriorClient([| 0.7; 0.3 |])
        let leaf = RejectingLeafEvaluator()
        let mcts =
            MonteCarloTreeSearch(
                { MCTSConfig.Default with
                    MaxSimulations = 1
                    PriorClient = Some (client :> IPriorClient)
                    LeafEvaluator = Some (leaf :> ILeafEvaluator) })

        mcts.RunSimulation(root) |> ignore

        leaf.Enqueues |> should equal 0
        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].ValueSums |> should equal [| 0.7; 0.3 |]
        match root.Actions.[0] with
        | DeterministicAction expanded ->
            expanded.Priors.IsSome |> should equal true
            expanded.ValueEstimates |> should equal (Some [| 0.7; 0.3 |])
        | _ -> Assert.Fail "Expected deterministic child expansion"

    [<Test>]
    member _.CombinedPriorSearchQueuesSeveralPendingNodesBeforeWaiting() =
        let root =
            MCTSState(
                node_builder(p1, 0, 0, 0, [
                    node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
                    node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21))
                    node_builder(p2, 1, 0, 12, node_builder(p1, 2, 0, 22))
                ]).build())
        let client = DelayedCombinedPriorClient([| 0.6; 0.4 |])
        let mcts =
            MonteCarloTreeSearch(
                { MCTSConfig.Default with
                    MaxSimulations = 3
                    MaxPendingEvaluations = 3
                    PriorClient = Some (client :> IPriorClient)
                    LeafEvaluationTimeoutMs = 1000 })
        let task = Task.Run(fun () -> mcts.RunSimulation(root) |> ignore)

        let queued = SpinWait.SpinUntil((fun () -> client.RequestCount >= 4), 1000)
        queued |> should equal true // root plus three expanded children
        root.ActionStats |> Array.sumBy (fun stats -> stats.PendingVisits) |> should equal 3
        client.Release()
        task.Wait(2000) |> should equal true

        root.Rollouts |> should equal 3
        root.ActionStats |> Array.sumBy (fun stats -> stats.PendingVisits) |> should equal 0

    [<TestCaseSource("InvalidValueEstimates")>]
    member _.SearchWithInvalidValueResponse_RejectsValues(valueEstimates: float[]) =
        let rootNode =
            node_builder(p1, 0, 0, 0)
                .addChildren([
                    node_builder(p2, 1, 0, 10, node_builder(p1, 2, 0, 20))
                    node_builder(p2, 1, 0, 11, node_builder(p1, 2, 0, 21))
                ])
                .build()
        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let client = ValueRespondPriorClient(valueEstimates)
        let mcts =
            MonteCarloTreeSearch(
                { MCTSConfig.Default with
                    MaxSimulations = 2
                    PriorClient = Some (client :> IPriorClient) })

        mcts.RunSimulation(mctsRoot) |> ignore

        mctsRoot.ValueEstimates |> should equal None

    static member InvalidValueEstimates =
        [| [| 1. |]
           [| 0.; 0. |]
           [| -1.; 2. |]
           [| Double.NaN; 1. |]
           [| Double.PositiveInfinity; 1. |] |]

    [<Test>]
    member _.SearchWithoutPriorClient_PriorStatsAreZero() =
        let rootNode = node_builder(p1, 0, 0, 0, node_builder(p2, 1, 0, 10)).build()
        let mctsRoot = MCTSState(rootNode :> ICoreState)

        let mcts = MonteCarloTreeSearch(
            { MCTSConfig.Default with
                SearchTime = Seconds 5
                MaxSimulations = 10
                PriorClient = None })

        let _ = mcts.RunSimulation(mctsRoot)
        let logInfo = mcts.LatestLogInfo()

        logInfo.priorActionsRequested |> should equal 0
        logInfo.priorActionsApplied |> should equal 0
        logInfo.priorNodesApplied |> should equal 0

    [<Test>]
    member _.PriorRequest_IncludesCorrectActingPlayer() =
        let rootNode =
            node_builder(p1, 0, 0, 0)
                .addChildren([
                    node_builder(p2, 1, 0, 10)
                    node_builder(p2, 1, 0, 11)
                ])
                .build()

        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let mockClient = MockPriorClient()

        let mcts = MonteCarloTreeSearch(
            { MCTSConfig.Default with
                SearchTime = Seconds 5
                MaxSimulations = 5
                PriorClient = Some (mockClient :> IPriorClient) })

        let _ = mcts.RunSimulation(mctsRoot)

        // The first request should be for the expanded child.
        // The root's playerTurn is P1, children are P2. When a child is expanded,
        // the prior request is for the child's actions, and the actingPlayer
        // should be the child's PlayerTurn + 1 (1-indexed).
        if mockClient.Requests.Length > 0 then
            let (_, _, _, actingPlayer, _) = mockClient.Requests.[0]
            // actingPlayer is 1-indexed player whose turn it is at the expanded node
            actingPlayer |> should be (greaterThanOrEqualTo 1)

    [<Test>]
    member _.PriorRequest_DepthIsCorrect() =
        // Build a deeper tree: root → child → grandchild → leaf (terminal)
        // so that the expanded child and grandchild both have actions.
        let rootNode =
            node_builder(p1, 0, 0, 0,
                node_builder(p2, 1, 0, 10,
                    node_builder(p1, 2, 0, 20,
                        node_builder(p2, 3, 0, 30)))).build()

        let mctsRoot = MCTSState(rootNode :> ICoreState)
        let mockClient = MockPriorClient()

        let mcts = MonteCarloTreeSearch(
            { MCTSConfig.Default with
                SearchTime = Seconds 5
                MaxSimulations = 20
                PriorClient = Some (mockClient :> IPriorClient) })

        let _ = mcts.RunSimulation(mctsRoot)

        // The root is requested at depth 0 and its expanded child at depth 1.
        mockClient.Requests.Length |> should be (greaterThan 0)
        let depths = mockClient.Requests |> List.map (fun (_, _, _, _, d) -> d)
        depths |> should contain 0
        depths |> should contain 1

// ────────────────────────────────────────────────────────────────
// Priors field on MCTSState
// ────────────────────────────────────────────────────────────────

[<TestFixture>]
type MCTSStatePriorsTests() =

    [<Test>]
    member _.Priors_DefaultsToNone() =
        let s = MCTSState(node(p1, 0, 0, 0))
        s.Priors |> should equal None

    [<Test>]
    member _.Priors_CanBeSet() =
        let s = MCTSState(node(p1, 0, 0, 0))
        s.Priors <- Some [| 0.5; 0.3; 0.2 |]

        match s.Priors with
        | Some p ->
            p |> should haveLength 3
            p.[0] |> should (equalWithin 0.001) 0.5
        | None -> Assert.Fail "Expected Some priors"

    [<Test>]
    member _.Priors_CanBeCleared() =
        let s = MCTSState(node(p1, 0, 0, 0))
        s.Priors <- Some [| 0.5; 0.5 |]
        s.Priors <- None
        s.Priors |> should equal None

    [<Test>]
    member _.ValueEstimates_DefaultToNoneAndStorePerPlayerValues() =
        let s = MCTSState(node(p1, 0, 0, 0))
        s.ValueEstimates |> should equal None

        s.ValueEstimates <- Some [| 0.25; 0.75 |]

        s.ValueEstimates |> should equal (Some [| 0.25; 0.75 |])
