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
        member _.Scores() =
            let scores = Array.zeroCreate<float> 5
            let i = int winner
            if i > 0 && i < 5 then scores.[i] <- 1.
            scores

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
        member _.Scores() = Array.zeroCreate<float> 5

[<TestFixture>]
type StateTest() =

    let sampleState () = State(node (p1, 0, 0, 0))

    [<Test>]
    member _.OnlyWin() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            state.registerOutcome [| 1.; 0.; 0.; 0. |]

        state.visitCount |> should equal 10
        state.winRate |> should equal 1.

    [<Test>]
    member _.OnlyLoss() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            state.registerOutcome [| 0.; 1.; 0.; 0. |]

        state.visitCount |> should equal 10
        state.winRate |> should equal 0.

    [<Test>]
    member _.WinThenLosses() =
        let state = sampleState ()

        state.registerOutcome [| 1.; 0.; 0.; 0. |]

        Seq.iter (fun _ -> state.registerOutcome [| 0.; 1.; 0.; 0. |]) [1..99]

        state.visitCount |> should equal 100

        state.winRate
        |> should (equalWithin 0.0000001) 0.01

    [<Test>]
    member _.StochasticSimulation_SamplesOutcome() =
        // Each simulation samples a single stochastic outcome. Over many runs
        // the distribution should converge to the expected probabilities
        // (P1=0.25, P2=0.75). A single call returns a one-hot outcome.
        let mutable p1Wins = 0.
        let mutable p2Wins = 0.
        let trials = 1000

        for _ in 1 .. trials do
            let result = simulate defaultMaxRolloutDepth (State(rootState() :> ICoreState))
            p1Wins <- p1Wins + result.[int Player.Player1 - 1]
            p2Wins <- p2Wins + result.[int Player.Player2 - 1]

        let p1Rate = p1Wins / float trials
        let p2Rate = p2Wins / float trials

        // With 1000 trials, rates should be within ~0.05 of expected values.
        p1Rate |> should (equalWithin 0.1) 0.25
        p2Rate |> should (equalWithin 0.1) 0.75
