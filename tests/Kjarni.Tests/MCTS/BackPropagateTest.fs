module KjarniTest.MCTS.BackPropagatingTest

open Kjarni
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.Types

open KjarniTest.TestTypes

open NUnit.Framework

open FsUnit

[<TestFixture>]
type BackPropagateTest() =

    let branchingNode =
        node_builder (
            p1,
            0,
            0,
            0,
            [ node_builder(p1, 1, 0, 1).addChild(node_builder (p2, 2, 0, 11)).addChild (node_builder (p1, 3, 0, 111))
              node_builder(p2, 1, 0, 2).addChild(node_builder (p1, 2, 0, 22)).addChild (node_builder (p2, 3, 0, 222))
              node_builder(p1, 1, 0, 3).addChild(node_builder (p2, 2, 0, 33)).addChild (node_builder (p1, 3, 0, 333)) ]
        )

    let constructSut (nodes: ICoreState) =
        let root = State nodes

        root.leaves <-
            [| 0; 1; 2 |]
            |> Array.map (fun i ->
                let childNode = nodes.Actions().[i].DoCoreAction()
                let s1 = State childNode
                let grandchild = childNode.Actions().[0].DoCoreAction()
                let s2 = State grandchild
                s1.leaves.[0] <- Leaf(Action(grandchild.PlayerTurn, s2))
                Leaf(Action(root.playerTurn, s1)))

        root

    let getLeaf l =
        match l with
        | Leaf a -> a.state
        | _ -> failwith "not a leaf"

    let assertWinRate (state: State) expectWinningPlayer =
        state.winRate
        |> should
            equal
            (if state.state.PlayerTurn = expectWinningPlayer then
                 1.
             else
                 0.)

    [<Test>]
    member _.BranchingTree([<Values(0, 1, 2)>] branch, [<Values(Player.Player1, Player.Player2)>] playerWin) =
        let root = constructSut (branchingNode.build ())
        let state2 = getLeaf root.leaves.[branch]
        let action1 = Action(root.playerTurn, state2)

        let action2 = Action(state2.playerTurn, getLeaf state2.leaves.[0])

        let visitedActions = [ action1; action2 ]

        let outcome =
            if playerWin = Player.Player1 then
                [| 1.; 0.; 0.; 0. |]
            else
                [| 0.; 1.; 0.; 0. |]

        backPropagate root visitedActions outcome
        backPropagate root visitedActions outcome

        root.visitCount |> should equal 2
        assertWinRate root playerWin

        action1.visitCount |> should equal 2
        action1.state.visitCount |> should equal 2
        assertWinRate action1.state playerWin

        action2.visitCount |> should equal 2
        action2.state.visitCount |> should equal 2
        assertWinRate action2.state playerWin

        for i in [| 0; 1; 2 |] |> Array.where (fun i -> i <> branch) do
            let leaf = getLeaf root.leaves.[i]
            leaf.winRate |> should equal 0.
            leaf.visitCount |> should equal 1.
