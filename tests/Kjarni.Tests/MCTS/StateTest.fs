module KjarniTest.MCTS.StateTest

open System
open Kjarni
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.AI
open NUnit.Framework
open Kjarni.MCTS.Types
open KjarniTest.TestTypes
open FsUnit

type terminalState(winner: Player) =
    interface ICoreState with
        member _.PlayerTurn = winner
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member _.Actions() = Array.empty
        member _.Scores() =
            let scores = Array.zeroCreate<float> 2
            scores.[int winner] <- 1.
            scores

type stochasticRootAction() =
    interface IStochasticCoreAction with
        member _.Outcomes() =
            [| (1, terminalState(Player.Player1) :> ICoreState)
               (3, terminalState(Player.Player2) :> ICoreState) |]

type rootState() =
    interface ICoreState with
        member _.PlayerTurn = Player.Player1
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member this.Actions() =
            [| Stochastic (stochasticRootAction() :> IStochasticCoreAction) |]
        member _.Scores() = Array.zeroCreate<float> 2

type forcedLoopAction(state: ICoreState) =
    interface IDeterministicCoreAction with
        member _.State() = state

type forcedLoopState() as this =
    interface ICoreState with
        member _.PlayerTurn = Player.Player1
        member _.NumberOfPlayers = 2
        member _.TurnNumber = 0
        member _.Actions() =
            [| Deterministic(forcedLoopAction(this :> ICoreState) :> IDeterministicCoreAction) |]
        member _.Scores() = [| 1.; 0. |]

let registerOutcome (state: MCTSState) (outcome: float array) =
    state.Rollouts <- state.Rollouts + 1
    state.WinCounts <- Array.map2 (+) state.WinCounts outcome

[<TestFixture>]
type StateTest() =

    let sampleState () = MCTSState(node (p1, 0, 0, 0))

    [<Test>]
    member _.OnlyWin() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            registerOutcome state [| 1.; 0. |]

        state.Rollouts |> should equal 10
        winRate state state.State.PlayerTurn |> should equal 1.

    [<Test>]
    member _.OnlyLoss() =
        let state = sampleState ()

        for _ in [ 1 .. 10 ] do
            registerOutcome state [| 0.; 1. |]

        state.Rollouts |> should equal 10
        winRate state state.State.PlayerTurn |> should equal 0.

    [<Test>]
    member _.WinThenLosses() =
        let state = sampleState ()

        registerOutcome state [| 1.; 0. |]

        Seq.iter (fun _ -> registerOutcome state [| 0.; 1. |]) [1..99]

        state.Rollouts |> should equal 100

        winRate state state.State.PlayerTurn
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
            let result = simulate defaultMaxRolloutDepth (MCTSState(rootState() :> ICoreState))
            p1Wins <- p1Wins + result.[int Player.Player1]
            p2Wins <- p2Wins + result.[int Player.Player2]

        let p1Rate = p1Wins / float trials
        let p2Rate = p2Wins / float trials

        // With 1000 trials, rates should be within ~0.05 of expected values.
        p1Rate |> should (equalWithin 0.1) 0.25
        p2Rate |> should (equalWithin 0.1) 0.75

    [<Test>]
    member _.RunSimulation_TopsUpInheritedForcedRootToTarget() =
        let root = MCTSState(forcedLoopState() :> ICoreState)
        root.Rollouts <- 4
        root.WinCounts <- [| 2.; 2. |]
        let mcts =
            MonteCarloTreeSearch(
                { MCTSConfig.Default with
                    MaxSimulations = 10
                    MaxRolloutDepth = 2 })

        mcts.RunSimulation(root) |> ignore

        root.Rollouts |> should equal 10
        Array.sum root.WinCounts |> should equal 10.
