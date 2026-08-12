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
    let mutable _actionStats = _actions |> Array.map (fun _ -> ActionStats(state.NumberOfPlayers))
    let mutable _priors: float[] option = None
    let mutable _densePriors: float[] option = None
    let mutable _flattenedPriors: float[] option = None
    let mutable _valueEstimates: float[] option = None
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
      and set value =
          _actions <- value
          _actionStats <- value |> Array.map (fun _ -> ActionStats(state.NumberOfPlayers))
    member _.ActionStats = _actionStats
    /// Optional NN prior policy over actions. When Some, actionEvaluator uses
    /// P(action_i) = Priors[i] in the PUCT formula. When None, uniform prior.
    member _.Priors
      with get () = _priors
      and set value = _priors <- value
    /// Optional client-specific dense policy retained for export and diagnostics.
    /// It is not used by MCTS selection.
    member _.DensePriors
      with get () = _densePriors
      and set value = _densePriors <- value
    /// Raw child-state prior scores in collectActionStates order, retained for diagnostics.
    member _.FlattenedPriors
      with get () = _flattenedPriors
      and set value = _flattenedPriors <- value
    /// Optional normalized NN value distribution for this node's state,
    /// indexed by Player enum value.
    member _.ValueEstimates
      with get () = _valueEstimates
      and set value = _valueEstimates <- value

and ActionStats(numberOfPlayers: int) =
    let mutable _completedVisits = 0
    let mutable _pendingVisits = 0
    let _valueSums = Array.zeroCreate<float> numberOfPlayers

    member _.CompletedVisits
      with get () = _completedVisits
      and set value = _completedVisits <- value
    member _.PendingVisits
      with get () = _pendingVisits
      and set value = _pendingVisits <- value
    member _.ValueSums = _valueSums

and Action =
    | Unexplored of CoreAction
    | DeterministicAction of MCTSState
    | StochasticAction of StochasticOutcome []
    | Terminal of float[]
    | HorizonAction of MCTSState

and StochasticOutcome = { ProbabilityWeight: int; State: MCTSState }

type SelectedAction = { Parent: MCTSState; ActionIndex: int }

type SelectionPath =
    { States: MCTSState list
      Edges: SelectedAction list }

type SelectionResult =
    | Candidate of (SelectionPath * int)
    | StochasticCandidate of (SelectionPath * int)
    | Exhausted of (SelectionPath * float array)
    | Horizon of (SelectionPath * MCTSState)
    | Blocked

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
        /// Number of selection paths that hit a structural evaluation horizon.
        val mutable horizonSkips: int
        val mutable leafEvaluationsSubmitted: int
        val mutable leafEvaluationsApplied: int
        val mutable leafEvaluationTimeouts: int
        val mutable leafEvaluationsInvalid: int
        val mutable leafEvaluationsCancelled: int
        val mutable leafEvaluationFallbacks: int
        val mutable leafEvaluationOrphans: int
        val mutable leafEvaluationBatches: int
        val mutable leafEvaluationStates: int
        val mutable leafEvaluationLatencyMs: int64
    end
