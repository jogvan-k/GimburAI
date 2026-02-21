module KjarniTest.MCTS.StateTest

open System
open Kjarni
open Kjarni.MCTS.Algorithm
open NUnit.Framework
open Kjarni.MCTS.Types
open KjarniTest.TestTypes
open FsUnit

type terminalState(winner: Player) =
    interface ICoreState with
        member _.PlayerTurn = winner
        member _.TurnNumber = 0
        member _.Actions() = Array.empty

type stochasticRootAction(origin: ICoreState) =
    interface ICoreAction with
        member _.Origin = origin
        member _.DoCoreAction() = terminalState(Player.Player2) :> ICoreState
    interface IComparable with
        member _.CompareTo _ = 0
    interface IStochasticCoreAction with
        member _.Outcomes() =
            [| (terminalState(Player.Player1) :> ICoreState, 0.25)
               (terminalState(Player.Player2) :> ICoreState, 0.75) |]
    override _.Equals _ = true
    override _.GetHashCode() = 0

type rootState() =
    interface ICoreState with
        member _.PlayerTurn = Player.Player1
        member _.TurnNumber = 0
        member this.Actions() =
            [| stochasticRootAction(this :> ICoreState) :> ICoreAction |]

[<TestFixture>]
type StateTest() =

    let sampleState () = State(node (p1, 0, 0, 0))

    [<Test>]
    member _.OnlyWin() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            state.registerOutcome [| 0.; 1.; 0.; 0.; 0. |]

        state.visitCount |> should equal 10
        state.winRate |> should equal 1.

    [<Test>]
    member _.OnlyLoss() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            state.registerOutcome [| 0.; 0.; 1.; 0.; 0. |]

        state.visitCount |> should equal 10
        state.winRate |> should equal 0.

    [<Test>]
    member _.WinThenLosses() =
        let state = sampleState ()

        state.registerOutcome [| 0.; 1.; 0.; 0.; 0. |]

        Seq.iter (fun _ -> state.registerOutcome [| 0.; 0.; 1.; 0.; 0. |]) [1..99]

        state.visitCount |> should equal 100

        state.winRate
        |> should (equalWithin 0.0000001) 0.01

    [<Test>]
    member _.StochasticSimulation_UsesWeightedAverage() =
        let result = simulation (State(rootState() :> ICoreState))
        result.[int Player.Player1] |> should (equalWithin 0.0000001) 0.25
        result.[int Player.Player2] |> should (equalWithin 0.0000001) 0.75
