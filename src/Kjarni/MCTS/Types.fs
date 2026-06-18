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
    let mutable _valueEstimate: float = System.Double.NaN
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
    /// NN value estimate for this node's state (from the acting player's
    /// perspective). NaN when no estimate is available.
    member _.ValueEstimate
      with get () = _valueEstimate
      and set value = _valueEstimate <- value

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
        // ── Prior stats (mirrors Kjarni.MCTS.Algorithm.PriorStats) ─────────
        /// Number of nodes for which a prior request was issued.
        val mutable priorNodesRequested: int
        /// Number of MCTS-level action states sent to the client across all
        /// requested nodes.
        val mutable priorActionsRequested: int
        /// Number of (state, action) inference pairs actually sent to the model.
        /// In placement mode this counts post-fan-out composite (settlement, road)
        /// pairs; in state mode it equals priorActionsRequested.
        val mutable priorInferencesRequested: int
        /// Number of nodes whose prior policy was successfully attached.
        val mutable priorNodesApplied: int
        /// Number of action states whose prior probabilities were applied.
        val mutable priorActionsApplied: int
        /// Per-depth count of MCTS-level action states sent to the client (depth → count).
        val mutable priorActionsPerDepth: Dictionary<int, int>
        /// Per-depth count of model inference pairs (depth → count).
        val mutable priorInferencesPerDepth: Dictionary<int, int>
        /// Number of nodes refused by the client's ShouldRequestPrior pre-check.
        val mutable priorNodesSkipped: int
        /// Number of responses returned for nodes the search no longer tracks.
        val mutable priorResponsesOrphaned: int
        /// Number of selection paths that hit the maxPriorDepth horizon.
        val mutable horizonSkips: int
    end
