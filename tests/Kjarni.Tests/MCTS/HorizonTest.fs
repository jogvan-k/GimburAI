module KjarniTest.MCTS.HorizonTest

open System
open NUnit.Framework
open Kjarni
open Kjarni.MCTS.Types
open Kjarni.MCTS.Algorithm
open KjarniTest.TestTypes

open FsUnit

[<TestFixture>]
type horizonTests() =
    let twoChildNode =
        node_builder(p1, 0, 0, 100)
            .addChildren(
                [ node_builder (p2, 0, 0, 101)
                  node_builder (p2, 0, 0, 102) ]
            )
            .build ()

    [<Test>]
    member _.DeterministicResultAtBoundary_BecomesExactHorizonState() =
        let root = MCTSState twoChildNode
        let expectedState =
            match root.Actions.[0] with
            | Unexplored (Deterministic action) -> action.State()
            | _ -> failwith "Expected deterministic action"

        let expandedState = expand (Some (fun state -> Object.ReferenceEquals(state, expectedState))) (root, 0)
        let result = select (sqrt 2.) root

        result |> should be (ofCase <@ Horizon @>)

        match root.Actions.[0] with
        | HorizonAction horizonState ->
            Object.ReferenceEquals(horizonState, expandedState) |> should equal true
            Object.ReferenceEquals(horizonState.State, expectedState) |> should equal true
        | other -> Assert.Fail $"Expected HorizonAction but got %A{other}"

    [<Test>]
    member _.Boundary_SubsequentVisit_ReturnsHorizon() =
        let rootNode = node_builder(p1, 0, 0, 110, node_builder(p2, 0, 0, 111)).build()
        let root = MCTSState rootNode

        expand (Some (fun _ -> true)) (root, 0) |> ignore
        let firstResult = select (sqrt 2.) root
        firstResult |> should be (ofCase <@ Horizon @>)

        // Give the horizon state and root some rollouts so selection revisits
        match root.Actions.[0] with
        | HorizonAction hs ->
            hs.Rollouts <- 5
            hs.WinCounts <- [| 3.; 2. |]
        | _ -> Assert.Fail "Expected HorizonAction after first select"

        root.Rollouts <- 5
        root.WinCounts <- [| 3.; 2. |]

        // Second select should still return Horizon (from the HorizonAction match arm)
        let secondResult = select (sqrt 2.) root
        secondResult |> should be (ofCase <@ Horizon @>)

    [<Test>]
    member _.NonBoundaryResult_RemainsDeterministic() =
        let root = MCTSState twoChildNode
        expand (Some (fun _ -> false)) (root, 0) |> ignore

        root.Actions.[0] |> should be (ofCase <@ DeterministicAction @>)

    [<Test>]
    member _.NoGuard_ReturnsCandidate() =
        let root = MCTSState twoChildNode
        let result = select (sqrt 2.) root

        result |> should be (ofCase <@ Candidate @>)

    [<Test>]
    member _.StochasticExpansion_DoesNotApplyBoundaryToFirstOutcome() =
        let first = node (p1, 1, 0, 801)
        let second = node (p2, 2, 0, 802)
        let rootNode = stochastic_node(p1, 0, 0, 800, [ (1, first); (1, second) ])
        let root = MCTSState rootNode
        let mutable predicateCalls = 0
        let predicate (_: ICoreState) =
            predicateCalls <- predicateCalls + 1
            true

        expand (Some predicate) (root, 0) |> ignore

        predicateCalls |> should equal 0
        match root.Actions.[0] with
        | StochasticAction outcomes ->
            outcomes |> should haveLength 2
            Object.ReferenceEquals(outcomes.[0].State.State, first) |> should equal true
            Object.ReferenceEquals(outcomes.[1].State.State, second) |> should equal true
        | other -> Assert.Fail $"Expected StochasticAction but got %A{other}"

    [<Test>]
    member _.ActionEvaluator_HorizonAction_SameAsDeterministic() =
        let rootNode =
            node_builder(p1, 0, 0, 200)
                .addChildren(
                    [ node_builder (p2, 0, 0, 201)
                      node_builder (p2, 0, 0, 202) ]
                )
                .build ()

        let root = MCTSState rootNode
        root.Rollouts <- 10
        root.WinCounts <- [| 5.; 5. |]

        // Create two identical child states with known stats
        let childNodeA = node (p2, 0, 0, 301)
        let childA = MCTSState(childNodeA)
        childA.Rollouts <- 4
        childA.WinCounts <- [| 2.; 2. |]

        let childNodeB = node (p2, 0, 0, 302)
        let childB = MCTSState(childNodeB)
        childB.Rollouts <- 4
        childB.WinCounts <- [| 2.; 2. |]

        let horizonAction = HorizonAction childA
        let deterministicAction = DeterministicAction childB

        let horizonValue = actionEvaluator (sqrt 2.) root 0 horizonAction
        let deterministicValue = actionEvaluator (sqrt 2.) root 0 deterministicAction

        horizonValue |> should equal deterministicValue

    [<Test>]
    member _.ExtractionEvaluator_HorizonAction_ReturnsWinRate() =
        let childNode = node (p1, 0, 0, 400)
        let child = MCTSState(childNode)
        child.Rollouts <- 10
        child.WinCounts <- [| 6.; 4. |]

        let result = extractionEvaluator (Player.Player1, HorizonAction child)

        result |> should equal 0.6

    [<Test>]
    member _.ActionRollouts_HorizonAction_ReturnsRollouts() =
        let childNode = node (p1, 0, 0, 500)
        let child = MCTSState(childNode)
        child.Rollouts <- 42

        let result = actionRollouts (HorizonAction child)

        result |> should equal 42

    [<Test>]
    member _.ExtractBestPath_StopsAtHorizon() =
        let rootNode =
            node_builder(p1, 0, 0, 600)
                .addChildren(
                    [ node_builder (p2, 0, 0, 601)
                      node_builder (p2, 0, 0, 602) ]
                )
                .build ()

        let root = MCTSState rootNode

        // Set up first action as HorizonAction with more rollouts (best)
        let horizonChildNode = node (p2, 0, 0, 701)
        let horizonChild = MCTSState(horizonChildNode)
        horizonChild.Rollouts <- 10
        horizonChild.WinCounts <- [| 8.; 2. |]
        root.Actions.[0] <- HorizonAction horizonChild

        // Set up second action as DeterministicAction with fewer rollouts
        let detChildNode = node (p2, 0, 0, 702)
        let detChild = MCTSState(detChildNode)
        detChild.Rollouts <- 5
        detChild.WinCounts <- [| 2.; 3. |]
        root.Actions.[1] <- DeterministicAction detChild

        let path = extractBestPath root

        // Horizon is best for Player1 (0.8 vs 0.4), path stops at horizon
        path |> should equal [ 0 ]
