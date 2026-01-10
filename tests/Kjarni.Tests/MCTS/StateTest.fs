module KjarniTest.MCTS.StateTest

open NUnit.Framework
open Kjarni.MCTS.Types
open KjarniTest.TestTypes
open FsUnit

[<TestFixture>]
type StateTest() =

    let sampleState () = State(node (p1, 0, 0, 0))

    [<Test>]
    member _.OnlyWin() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            state.registerWin ()

        state.visitCount |> should equal 10
        state.winRate |> should equal 1.

    [<Test>]
    member _.OnlyLoss() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            state.registerLoss ()

        state.visitCount |> should equal 10
        state.winRate |> should equal 0.

    [<Test>]
    member _.WinThenLosses() =
        let state = sampleState ()

        state.registerWin ()

        Seq.iter (fun _ -> state.registerLoss ()) [1..99]

        state.visitCount |> should equal 100

        state.winRate
        |> should (equalWithin 0.0000001) 0.01
