module KjarniTest.MCTS.AlgorithmTest

open System
open NUnit.Framework
open Kjarni
open Kjarni.MCTS.Types
open Kjarni.MCTS.Algorithm
open KjarniTest.TestTypes

open FsUnit

[<TestFixture>]
type selectionTests() =
    let branchingNode =
        node_builder(p1, 0, 0, 0)
            .addChildren(
                [ node_builder (p2, 0, 0, 1, node_builder (p2, 0, 0, 4, node_builder (p2, 0, 0, 7)))
                  node_builder (p1, 0, 0, 2, node_builder (p1, 0, 0, 5, node_builder (p2, 0, 0, 8)))
                  node_builder (p2, 0, 0, 3, node_builder (p2, 0, 0, 6, node_builder (p2, 0, 0, 9))) ]
            )
            .build ()

    let stateHash (s: MCTSState) = (s.State :> Object).GetHashCode()

    [<Test>]
    member _.TerminalNode([<Values(Player.Player1, Player.Player2)>] playerTurn) =
        let terminalNode = MCTSState(node (playerTurn, 0, 0, 0))
        let result = selection (sqrt 2.) terminalNode

        result |> should be (ofCase <@ Exhausted @>)

    [<Test>]
    member _.AllUnexploredLeaves_SelectsFirst() =
        let root = MCTSState branchingNode
        let result = selection (sqrt 2.) root

        match result with
        | Candidate (ancestors, i) ->
            // selection starts with [root] so the visited states list contains just the root
            ancestors |> should haveLength 1
            ancestors.[0] |> should equal root
            i |> should equal 0
        | _ -> Assert.Fail()

    [<Test>]
    member _.ExpandAndSelectUnexplored() =
        let root = MCTSState branchingNode

        // Expand first action to make it non-Unexplored
        let expandedState = expand (root, 0)
        expandedState.Rollouts <- 1
        expandedState.WinCounts <- [| 0.; 1. |]
        root.Rollouts <- 1
        root.WinCounts <- [| 0.; 1. |]

        let result = selection (sqrt 2.) root

        match result with
        | Candidate (_, i) ->
            // Should select one of the remaining unexplored leaves (index 1 or 2)
            i |> should be (greaterThanOrEqualTo 1)
        | _ -> Assert.Fail()

    [<Test>]
    member _.AllExpanded_SelectsByEvaluation() =
        let root = MCTSState branchingNode

        // Expand all actions
        for i in 0..2 do
            let expanded = expand (root, i)
            expanded.Rollouts <- 1
            expanded.WinCounts <- [| 0.; 0. |]

        root.Rollouts <- 3
        root.WinCounts <- [| 0.; 0. |]

        let result = selection (sqrt 2.) root

        // When all are expanded with equal stats, selection should still return a candidate
        result |> should be (ofCase <@ Candidate @>)
