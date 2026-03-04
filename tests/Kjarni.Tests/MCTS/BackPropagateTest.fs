module KjarniTest.MCTS.BackPropagatingTest

open Kjarni
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.Types

open KjarniTest.TestTypes

open NUnit.Framework

open FsUnit

[<TestFixture>]
type BackPropagateTest() =

    [<Test>]
    member _.PropagatesOutcomeToAllStates([<Values(Player.Player1, Player.Player2)>] playerWin) =
        let rootNode =
            node_builder(p1, 0, 0, 0, node_builder (p2, 1, 0, 1, node_builder (p1, 2, 0, 2)))
                .build ()

        let root = MCTSState rootNode
        let child = MCTSState(rootNode.children.[0])
        let grandchild = MCTSState(rootNode.children.[0].children.[0])

        let visitedStates = [ grandchild; child; root ]

        let outcome =
            if playerWin = Player.Player1 then
                [| 1.; 0. |]
            else
                [| 0.; 1. |]

        backPropagate visitedStates outcome

        root.Rollouts |> should equal 1
        child.Rollouts |> should equal 1
        grandchild.Rollouts |> should equal 1

        winRate root playerWin |> should equal 1.
        winRate child playerWin |> should equal 1.
        winRate grandchild playerWin |> should equal 1.

    [<Test>]
    member _.PropagatesMultipleTimes() =
        let rootNode =
            node_builder(p1, 0, 0, 0, node_builder (p2, 1, 0, 1))
                .build ()

        let root = MCTSState rootNode
        let child = MCTSState(rootNode.children.[0])

        let visitedStates = [ child; root ]

        backPropagate visitedStates [| 1.; 0. |]
        backPropagate visitedStates [| 1.; 0. |]

        root.Rollouts |> should equal 2
        child.Rollouts |> should equal 2

        winRate root Player.Player1 |> should equal 1.
        winRate child Player.Player1 |> should equal 1.
