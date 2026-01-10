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

    let stateHash (s: State) = (s.state :> Object).GetHashCode()

    [<Test>]
    member _.TerminalNode([<Values(Player.Player1, Player.Player2)>] playerTurn) =
        let terminalNode = State(node (playerTurn, 0, 0, 0))
        let result = selection (terminalNode, leafEvaluator)

        result |> should be (ofCase <@ Exhausted @>)

    [<Test>]
    member _.AllUnexploredLeaves_SelectsFirst() =
        let root = State branchingNode
        let result = selection (root, leafEvaluator)

        match result with
        | Candidate (a, i) ->
            a |> should haveLength 0
            i |> should equal 0
        | _ -> Assert.Fail()

    [<Test>]
    member _.ExploredAndUnexploredLeaves_SelectUnexplored() =
        let root = State branchingNode

        root.leaves <-
            [| Leaf(Action(root.playerTurn, State(branchingNode.children.[0])))
               Terminal p2
               Unexplored(root.state.Actions().[2]) |]

        let result = selection (root, leafEvaluator)

        match result with
        | Candidate (a, i) ->
            a |> should haveLength 0
            i |> should equal 2
        | _ -> Assert.Fail()

    [<Test>]
    member _.AllExploredOnRoot_SelectHighestEvaluated([<Range(1, 3)>] highestEvaluated: int) =
        let root = State branchingNode

        root.leaves <-
            (branchingNode :> ICoreState).Actions()
            |> Array.mapi
                (fun i -> fun _ -> Leaf(Action(branchingNode.playerTurn, State(branchingNode.children.[i]))))

        let leafEvaluator =
            fun (_, i) ->
                match i with
                | Leaf s ->
                    if stateHash s.state = highestEvaluated then
                        0.1
                    else
                        0.
                | _ -> 1.

        let result = selection (root, leafEvaluator)

        match result with
        | Candidate (a, i) ->
            a
            |> List.map (fun a -> stateHash a.state)
            |> should equal [ highestEvaluated ]

            i |> should equal 0
        | _ -> Assert.Fail()

    [<TestCase(1, 4)>]
    [<TestCase(2, 5)>]
    [<TestCase(3, 6)>]
    member _.AllExploredOnRootAndOneLevelDown_CandidateRecursivelyChosen
        (
            highestEvaluated: int,
            expectedCandidate: int
        ) =
        let root = State branchingNode

        root.leaves <-
            (branchingNode :> ICoreState).Actions()
            |> Array.mapi
                (fun i -> fun _ -> Leaf(Action(branchingNode.playerTurn, State(branchingNode.children.[i]))))

        let mutable leaves = List.empty

        for l in (branchingNode :> ICoreState).Actions() do
            let state = l.DoCoreAction()
            let leafState = State state

            leafState.leaves <-
                Array.singleton (Leaf(Action(state.PlayerTurn, State(state.Actions().[0].DoCoreAction()))))

            let leaf =
                Leaf(Action(state.PlayerTurn, leafState))

            leaves <- leaf :: leaves

        root.leaves <- leaves |> List.rev |> List.toArray

        let leafEvaluator =
            fun (_, i) ->
                match i with
                | Leaf a ->
                    if stateHash a.state = highestEvaluated then
                        0.1
                    else
                        0.
                | _ -> 1.

        let result = selection (root, leafEvaluator)

        match result with
        | Candidate (a, i) ->
            a
            |> List.map (fun a -> stateHash a.state)
            |> should equal [ expectedCandidate; highestEvaluated ]

            i |> should equal 0
        | _ -> Assert.Fail()
