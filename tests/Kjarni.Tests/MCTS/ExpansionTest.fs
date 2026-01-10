module KjarniTest.MCTS.ExpansionTest

open System
open NUnit.Framework

open FsUnit
open KjarniTest.TestTypes
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

    let constructSut () = State branchingNode

    let stateHash (s: State) = (s.state :> Object).GetHashCode()

    let assertIsState leaf expandTo =
        match leaf with
        | Leaf a -> stateHash a.state |> should equal expandTo
        | _ -> Assert.Fail()

    let assertIsTerminal terminal win =
        match terminal with
        | Terminal w -> w |> should equal win
        | _ -> Assert.Fail()

    [<Test>]
    member _.ExpandUnexplored([<Range(1, 3)>] expandTo) =
        let sut = constructSut ()

        let result =
            expansion (sut, expandTo - 1, None)

        assertIsState result expandTo

        sut.leaves.[expandTo - 1]
        |> should be (ofCase <@ Leaf @>)

    [<Test>]
    member _.ExpandToTerminal() =
        let node =
            node_builder(p1, 0, 0, 0, node_builder (p2, 0, 0, 2)).build ()

        let sut = State node
        let result = expansion (sut, 0, None)
        assertIsTerminal result p2

        sut.leaves.[0]
        |> should be (ofCase <@ Terminal @>)

    [<Test>]
    member _.ExpandExplored() =
        let sut = constructSut ()
        sut.leaves.[0] <- Leaf(Action(p1, sut))

        (fun () -> expansion (sut, 0, None) |> ignore)
        |> should (throwWithMessage "Target leaf is already expanded") typeof<Exception>

    [<Test>]
    member _.ExpandWithTranspositionTable([<Range(1, 3)>] expandTo) =
        let sut = constructSut ()
        let tTable = TranspositionTable()
        tTable.Add(0, sut)

        let result =
            expansion (sut, expandTo - 1, Some tTable)

        assertIsState result expandTo

        sut.leaves.[expandTo - 1]
        |> should be (ofCase <@ Leaf @>)

        tTable.SuccessfulLookups |> should equal 0
        tTable.Count |> should equal 2

    [<Test>]
    member _.ExpandToValueInTranspositionTable([<Range(1, 3)>] expandTo) =
        let sut = constructSut ()

        let tTable = TranspositionTable()
        tTable.Add(0, sut)

        tTable.Add(
            expandTo,
            State(
                node_builder(p1, 99, 0, expandTo, node_builder (p2, 0, 0, 10))
                    .build ()
            )
        )

        let result =
            expansion (sut, expandTo - 1, Some tTable)

        match result with
        | Leaf a -> a.state.state.TurnNumber |> should equal 99
        | _ -> Assert.Fail()

        tTable.SuccessfulLookups |> should equal 1
        tTable.Count |> should equal 2
