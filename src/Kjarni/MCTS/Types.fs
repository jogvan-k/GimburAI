module Kjarni.MCTS.Types

open System
open Kjarni

type Leaf =
    | Unexplored of ICoreAction
    | Leaf of Action
    | Terminal of Player

and State(state: ICoreState) =
    let mutable _leaves = state.Actions() |> Array.map Unexplored
    let mutable _visitCount = 0
    let maxTrackedPlayers = int Player.Player4
    let mutable _winRates = Array.zeroCreate<float> (maxTrackedPlayers + 1)
    let incrementVisitCount () = _visitCount <- _visitCount + 1
    member _.state = state

    member _.leaves
        with get () = _leaves
        and set value = _leaves <- value

    member _.registerOutcome(outcome: float array) =
        incrementVisitCount ()

        for i in 1 .. maxTrackedPlayers do
            let target =
                if i < outcome.Length then
                    outcome.[i]
                else
                    0.

            _winRates.[i] <- _winRates.[i] + (target - _winRates.[i]) / float _visitCount

    member _.winRateFor(player: Player) =
        let i = int player
        if i < 0 || i >= _winRates.Length then
            0.
        else
            _winRates.[i]

    member this.winRate = this.winRateFor state.PlayerTurn

    member _.winCounts =
        let vc = float (Math.Max(1, _visitCount))
        _winRates |> Array.map (fun r -> r * vc)

    member _.visitCount = Math.Max(1, _visitCount)
    member _.playerTurn = state.PlayerTurn

and Action(activePlayer: Player, state: State) =
    let mutable _visitCount = 0
    member _.incrementVisitCount() = _visitCount <- _visitCount + 1
    member _.state = state
    member _.visitCount = Math.Max(_visitCount, 1)

    member _.winRate =
        state.winRateFor activePlayer

type SelectionResult =
    | Exhausted of (Action list * Player)
    | Candidate of (Action list * int)

type TranspositionTable() =
    let mutable _map = Map.empty
    let mutable _successfulLookups = 0

    member _.Add(h: int, s: State) = _map <- _map.Add(h, s)

    member _.Lookup h =
        let result = _map.TryFind h

        if result.IsSome then
            _successfulLookups <- _successfulLookups + 1

        result

    member _.SuccessfulLookups = _successfulLookups
    member _.Count = _map.Count

type LogInfo =
    struct
        val mutable simulations: int
        val mutable elapsedTime: TimeSpan
        val mutable estimatedAiWinChance: float
        val mutable winCounts: float array
        val mutable successfulTranspositionTableLookup: int
        val mutable transpositionTableSize: int
    end
