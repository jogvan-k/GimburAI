module Kjarni.MCTS.Types

open System
open Kjarni

type MCTSState(state: ICoreState) =
    let mutable _rollouts = 0
    let mutable _winCounts = Array.zeroCreate<float> state.NumberOfPlayers
    let mutable _actions = state.Actions() |> Array.map Unexplored
    member _.Rollouts
      with get () = _rollouts
      and set value = _rollouts <- value
    member _.WinCounts
      with get () = _winCounts
      and set value = _winCounts <- value
    member _.State = state
    member _.Actions
      with get () = _actions
      and set value = _actions <- value

and Action =
    | Unexplored of CoreAction
    | DeterministicAction of MCTSState
    | StochasticAction of StochasticOutcome []
    | Terminal of float[]

and StochasticOutcome = { ProbabilityWeight: int; State: MCTSState }

type SelectionResult =
    | Candidate of (MCTSState list * int)
    | StochasticCandidate of (MCTSState list * int * int)
    | Exhausted of (MCTSState list * float array)

type LogInfo =
    struct
        val mutable simulations: int
        val mutable elapsedTime: TimeSpan
        val mutable estimatedAiWinChance: float
        val mutable winCounts: float array
        val mutable successfulTranspositionTableLookup: int
        val mutable transpositionTableSize: int
    end
