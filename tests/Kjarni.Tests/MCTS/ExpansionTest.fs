module KjarniTest.MCTS.ExpansionTest

open System
open NUnit.Framework

open FsUnit
open KjarniTest.TestTypes
open Kjarni
open Kjarni.MCTS.Types
open Kjarni.MCTS.Algorithm

[<TestFixture>]
type ExpansionTest() =
    let branchingNode =
        node_builder(p1, 0, 0, 0)
            .addChildren(
                [ node_builder (p2, 0, 0, 1, node_builder (p2, 0, 0, 4))
                  node_builder (p1, 0, 0, 2, node_builder (p1, 0, 0, 5))
                  node_builder (p2, 0, 0, 3, node_builder (p2, 0, 0, 6)) ]
            )
            .build ()

    let constructSut () = MCTSState branchingNode

    let stateHash (s: MCTSState) = (s.State :> Object).GetHashCode()

    [<Test>]
    member _.ExpandUnexplored([<Range(0, 2)>] expandTo) =
        let sut = constructSut ()
        let expandedState = expand None (sut, expandTo)

        // The expanded state should be an MCTSState wrapping the child node
        expandedState |> should not' (be Null)
        expandedState.State |> should not' (be Null)

    [<Test>]
    member _.ExpandToTerminal() =
        let node =
            node_builder(p1, 0, 0, 0, node_builder (p2, 0, 0, 2)).build ()

        let sut = MCTSState node

        // The child node (hash=2) has no children, so it's terminal.
        // expand should still return the MCTSState for it.
        let expandedState = expand None (sut, 0)
        expandedState |> should not' (be Null)

    [<Test>]
    member _.ExpandAlreadyExpanded() =
        let sut = constructSut ()
        expand None (sut, 0) |> ignore

        (fun () -> expand None (sut, 0) |> ignore)
        |> should (throwWithMessage "Target action is already expanded") typeof<Exception>
