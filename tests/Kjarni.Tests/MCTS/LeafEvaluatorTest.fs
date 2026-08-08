module KjarniTest.MCTS.LeafEvaluatorTest

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open FsUnit
open Kjarni
open Kjarni.MCTS.AI
open Kjarni.MCTS.Types
open KjarniTest.TestTypes
open NUnit.Framework

type DelayedLeafEvaluator(values: float[][], ?autoRelease: bool) =
    let requests = ConcurrentDictionary<int64, ICoreState[]>()
    let ready = ConcurrentQueue<LeafEvaluationResponse>()
    let enqueued = new ManualResetEventSlim(false)
    let released = new ManualResetEventSlim(defaultArg autoRelease false)
    let mutable waitCalls = 0
    let mutable cancelled = 0

    member _.Enqueued = enqueued
    member _.RequestCount = requests.Count
    member _.WaitCalls = waitCalls
    member _.Cancelled = cancelled
    member _.Release() = released.Set()

    interface ILeafEvaluator with
        member _.Enqueue(requestId, states, _) =
            requests.[requestId] <- states
            enqueued.Set()
            true

        member _.Collect(knownIds) =
            if released.IsSet then
                for KeyValue(requestId, _) in requests do
                    if requests.TryRemove(requestId) |> fst then
                        ready.Enqueue(LeafEvaluationResponse(requestId, values, 2L))
            let results = ResizeArray<LeafEvaluationResponse>()
            let keep = ResizeArray<LeafEvaluationResponse>()
            let mutable response = Unchecked.defaultof<LeafEvaluationResponse>
            while ready.TryDequeue(&response) do
                if knownIds.Contains(response.RequestId) then results.Add(response)
                else keep.Add(response)
            for item in keep do ready.Enqueue(item)
            results.ToArray()

        member _.WaitForResults(timeoutMs) =
            Interlocked.Increment(&waitCalls) |> ignore
            released.Wait(timeoutMs)

        member _.Cancel(requestIds) =
            for requestId in requestIds do
                if requests.TryRemove(requestId) |> fst then
                    Interlocked.Increment(&cancelled) |> ignore

let private config evaluator =
    { MCTSConfig.Default with
        MaxSimulations = 1
        LeafEvaluator = Some evaluator
        LeafEvaluationTimeoutMs = 1000
        DrainTimeoutMs = 100 }

let private oneBranchRoot () =
    node_builder(p1, 0, 0, 1, node_builder(p2, 1, 0, 2, node_builder(p1, 2, 1, 3))).build()

[<TestFixture>]
type LeafEvaluatorTests() =
    [<Test>]
    member _.SelectionReservesPendingAndCompletionCommitsOneVisit() =
        let evaluator = DelayedLeafEvaluator([| [| 0.25; 0.75 |] |])
        let root = MCTSState(oneBranchRoot())
        let mcts = MonteCarloTreeSearch(config (evaluator :> ILeafEvaluator))
        let task = Task.Run(fun () -> mcts.RunSimulation(root) |> ignore)

        evaluator.Enqueued.Wait(1000) |> should equal true
        root.ActionStats.[0].PendingVisits |> should equal 1
        root.ActionStats.[0].CompletedVisits |> should equal 0
        evaluator.Release()
        task.Wait(1000) |> should equal true

        root.ActionStats.[0].PendingVisits |> should equal 0
        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].ValueSums |> should equal [| 0.25; 0.75 |]
        mcts.LatestLogInfo().leafEvaluationsApplied |> should equal 1

    [<Test>]
    member _.AllPendingWaitsInsteadOfSpinning() =
        let evaluator = DelayedLeafEvaluator([| [| 0.4; 0.6 |] |])
        let root = MCTSState(oneBranchRoot())
        let mcts = MonteCarloTreeSearch(config (evaluator :> ILeafEvaluator))
        let task = Task.Run(fun () -> mcts.RunSimulation(root) |> ignore)
        evaluator.Enqueued.Wait(1000) |> should equal true
        Thread.Sleep(30)
        evaluator.WaitCalls |> should be (greaterThan 0)
        evaluator.WaitCalls |> should be (lessThan 10)
        evaluator.Release()
        task.Wait(1000) |> should equal true

    [<Test>]
    member _.DeadlineCancelsAndRemovesReservationsWithoutVisit() =
        let evaluator = DelayedLeafEvaluator([| [| 0.5; 0.5 |] |])
        let root = MCTSState(oneBranchRoot())
        let cfg =
            { config (evaluator :> ILeafEvaluator) with
                SearchTime = MilliSeconds 20
                MaxSimulations = Int32.MaxValue
                DrainTimeoutMs = 20 }
        let mcts = MonteCarloTreeSearch(cfg)
        mcts.RunSimulation(root) |> ignore

        root.ActionStats.[0].PendingVisits |> should equal 0
        root.ActionStats.[0].CompletedVisits |> should equal 0
        evaluator.Cancelled |> should equal 1
        mcts.LatestLogInfo().leafEvaluationsCancelled |> should equal 1

    [<Test>]
    member _.ExactTerminalBypassesEvaluator() =
        let evaluator = DelayedLeafEvaluator([| [| 0.5; 0.5 |] |], true)
        let terminal = node_builder(p1, 0, 0, 10, node_builder(p2, 1, 0, 11)).build()
        let root = MCTSState(terminal)
        let mcts = MonteCarloTreeSearch(config (evaluator :> ILeafEvaluator))
        mcts.RunSimulation(root) |> ignore

        evaluator.RequestCount |> should equal 0
        root.ActionStats.[0].CompletedVisits |> should equal 1
        mcts.LatestLogInfo().leafEvaluationsSubmitted |> should equal 0

    [<Test>]
    member _.StochasticOutcomesUseOneBatchAndOneWeightedActionVisit() =
        let outcomeA = node_builder(p2, 1, 0, 21, node_builder(p1, 2, 1, 22)).build()
        let outcomeB = node_builder(p2, 1, 0, 23, node_builder(p1, 2, 1, 24)).build()
        let state = stochastic_node(p1, 0, 0, 20, [ 1, outcomeA; 3, outcomeB ])
        let evaluator = DelayedLeafEvaluator([| [| 0.8; 0.2 |]; [| 0.2; 0.8 |] |], true)
        let root = MCTSState(state)
        let mcts = MonteCarloTreeSearch(config (evaluator :> ILeafEvaluator))
        mcts.RunSimulation(root) |> ignore

        evaluator.RequestCount |> should equal 0
        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].ValueSums.[0] |> should (equalWithin 0.0001) 0.35
        root.ActionStats.[0].ValueSums.[1] |> should (equalWithin 0.0001) 0.65
        mcts.LatestLogInfo().leafEvaluationStates |> should equal 2

    [<Test>]
    member _.InvalidResponseFallsBackToOneRollout() =
        let evaluator = DelayedLeafEvaluator([| [| Double.NaN; 1. |] |], true)
        let root = MCTSState(oneBranchRoot())
        let mcts = MonteCarloTreeSearch(config (evaluator :> ILeafEvaluator))
        mcts.RunSimulation(root) |> ignore

        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].PendingVisits |> should equal 0
        mcts.LatestLogInfo().leafEvaluationsInvalid |> should equal 1
        mcts.LatestLogInfo().leafEvaluationFallbacks |> should equal 1

    [<Test>]
    member _.EvaluationTimeoutFallsBackBeforeSearchDeadline() =
        let evaluator = DelayedLeafEvaluator([| [| 0.5; 0.5 |] |])
        let root = MCTSState(oneBranchRoot())
        let cfg = { config (evaluator :> ILeafEvaluator) with LeafEvaluationTimeoutMs = 10 }
        let mcts = MonteCarloTreeSearch(cfg)
        mcts.RunSimulation(root) |> ignore

        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].PendingVisits |> should equal 0
        mcts.LatestLogInfo().leafEvaluationTimeouts |> should equal 1
        mcts.LatestLogInfo().leafEvaluationFallbacks |> should equal 1

    [<Test>]
    member _.PendingRequestsDoNotOvershootSimulationLimit() =
        let evaluator = DelayedLeafEvaluator([| [| 0.5; 0.5 |] |], true)
        let childA = node_builder(p2, 1, 0, 31, node_builder(p1, 2, 1, 32))
        let childB = node_builder(p2, 1, 0, 33, node_builder(p1, 2, 1, 34))
        let root = MCTSState(node_builder(p1, 0, 0, 30, [ childA; childB ]).build())
        let mcts = MonteCarloTreeSearch(config (evaluator :> ILeafEvaluator))
        mcts.RunSimulation(root) |> ignore

        root.Rollouts |> should equal 1
        root.ActionStats |> Array.sumBy (fun stats -> stats.CompletedVisits) |> should equal 1

    [<Test>]
    member _.MaxTreeDepthOneEvaluatesDeterministicChildrenWithoutExpansion() =
        let evaluator = DelayedLeafEvaluator([| [| 0.7; 0.3 |] |], true)
        let child = node_builder(p2, 1, 0, 41, node_builder(p1, 2, 1, 42))
        let root = MCTSState(node_builder(p1, 0, 0, 40, child).build())
        let cfg =
            { config (evaluator :> ILeafEvaluator) with
                MaxTreeDepth = 1
                MaxSimulations = 1 }
        let mcts = MonteCarloTreeSearch(cfg)

        mcts.RunSimulation(root) |> ignore

        root.Actions.[0] |> should be (ofCase <@ Unexplored @>)
        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].ValueSums.[0] |> should (equalWithin 0.0001) 0.7

    [<Test>]
    member _.MaxTreeDepthOneEvaluatesAllStochasticOutcomesWithoutExpansion() =
        let outcomeA = node_builder(p2, 1, 0, 51, node_builder(p1, 2, 1, 52)).build()
        let outcomeB = node_builder(p2, 1, 0, 53, node_builder(p1, 2, 1, 54)).build()
        let evaluator = DelayedLeafEvaluator([| [| 0.8; 0.2 |]; [| 0.2; 0.8 |] |], true)
        let root = MCTSState(stochastic_node(p1, 0, 0, 50, [ 1, outcomeA; 3, outcomeB ]))
        let cfg =
            { config (evaluator :> ILeafEvaluator) with
                MaxTreeDepth = 1
                MaxSimulations = 1 }
        let mcts = MonteCarloTreeSearch(cfg)

        mcts.RunSimulation(root) |> ignore

        root.Actions.[0] |> should be (ofCase <@ Unexplored @>)
        root.ActionStats.[0].CompletedVisits |> should equal 1
        root.ActionStats.[0].ValueSums.[0] |> should (equalWithin 0.0001) 0.35
