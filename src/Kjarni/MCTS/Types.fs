module Kjarni.MCTS.Types

open System
open System.Collections.Generic
open System.Threading
open Kjarni

type MCTSState(state: ICoreState) =
    static let mutable _nextNodeId = 0L
    let _nodeId = Interlocked.Increment(&_nextNodeId)
    let mutable _rollouts = 0
    let mutable _winCounts = Array.zeroCreate<float> state.NumberOfPlayers
    let mutable _actions = state.Actions() |> Array.map Unexplored
    let mutable _priors: float[] option = None
    /// Unique identifier for this node, used to correlate prior responses.
    member _.NodeId = _nodeId
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
    /// Optional NN prior policy over actions. When Some, actionEvaluator uses
    /// P(action_i) = Priors[i] in the PUCT formula. When None, uniform prior.
    member _.Priors
      with get () = _priors
      and set value = _priors <- value

and Action =
    | Unexplored of CoreAction
    | DeterministicAction of MCTSState
    | StochasticAction of StochasticOutcome []
    | Terminal of float[]
    | HorizonAction of MCTSState

and StochasticOutcome = { ProbabilityWeight: int; State: MCTSState }

type SelectionResult =
    | Candidate of (MCTSState list * int)
    | StochasticCandidate of (MCTSState list * int * int)
    | Exhausted of (MCTSState list * float array)
    | Horizon of (MCTSState list * MCTSState)

type LogInfo =
    struct
        val mutable simulations: int
        val mutable elapsedTime: TimeSpan
        val mutable estimatedAiWinChance: float
        val mutable winCounts: float array
        val mutable successfulTranspositionTableLookup: int
        val mutable transpositionTableSize: int
        val mutable reachedTerminal: bool
        val mutable priorStatesRequested: int
        /// Number of tree nodes that had prior policies successfully applied.
        val mutable priorNodesApplied: int
        /// Number of individual action states covered by applied priors.
        val mutable priorsApplied: int
        val mutable priorStatesEvaluated: int
        /// Per-depth count of prior states evaluated (depth → state count).
        val mutable priorStatesPerDepth: Dictionary<int, int>
        val mutable horizonSkips: int
        /// Number of nodes skipped by the ShouldRequestPrior pre-check.
        val mutable priorsSkipped: int
        val mutable stateNotFound: int
    end
