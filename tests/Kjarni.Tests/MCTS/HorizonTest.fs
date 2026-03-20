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
    member _.GuardBlocks_ReturnsHorizon() =
        let root = MCTSState twoChildNode
        let guard (_state: ICoreState) (_action: CoreAction) = true
        let result = select (sqrt 2.) (Some guard) root

        result |> should be (ofCase <@ Horizon @>)

        match root.Actions.[0] with
        | HorizonAction _ -> ()
        | other -> Assert.Fail $"Expected HorizonAction but got %A{other}"

    [<Test>]
    member _.GuardBlocks_SubsequentVisit_ReturnsHorizon() =
        let root = MCTSState twoChildNode
        let guard (_state: ICoreState) (_action: CoreAction) = true

        // First select converts Unexplored -> HorizonAction
        let firstResult = select (sqrt 2.) (Some guard) root
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
        let secondResult = select (sqrt 2.) (Some guard) root
        secondResult |> should be (ofCase <@ Horizon @>)

    [<Test>]
    member _.GuardDoesNotBlock_ReturnsCandidate() =
        let root = MCTSState twoChildNode
        let guard (_state: ICoreState) (_action: CoreAction) = false
        let result = select (sqrt 2.) (Some guard) root

        result |> should be (ofCase <@ Candidate @>)

    [<Test>]
    member _.NoGuard_ReturnsCandidate() =
        let root = MCTSState twoChildNode
        let result = select (sqrt 2.) None root

        result |> should be (ofCase <@ Candidate @>)

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
