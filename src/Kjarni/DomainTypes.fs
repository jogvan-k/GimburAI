namespace Kjarni

open System
open System.Collections.Generic

type Player =
    | Player1 = 0
    | Player2 = 1
    | Player3 = 2
    | Player4 = 3

type ICoreState =
    abstract PlayerTurn : Player
    abstract NumberOfPlayers : int
    abstract TurnNumber : int
    abstract Actions : unit -> CoreAction []
    /// Returns per-player scores (indexed by Player enum value, size == NumberOfPlayers)
    /// Used by the MCTS rollout to evaluate positions when max depth is reached.
    abstract Scores : unit -> float []

and CoreAction =
    | Deterministic of IDeterministicCoreAction
    | Stochastic of IStochasticCoreAction

and IDeterministicCoreAction =
    abstract State : unit -> ICoreState

and IStochasticCoreAction =
    abstract Outcomes : unit -> (int * ICoreState) []

type IEvaluator =
    abstract Evaluate : ICoreState -> int

/// A completed prior response from the inference server.
/// NodeId is an opaque identifier used to correlate the response back to the
/// parent MCTSState. Priors contains per-action prior policy weights in the
/// same order as the node's actions. ValueEstimate is a scalar value for the
/// node's state from the acting player's perspective (NaN if unavailable).
type PriorResponse =
  struct
    val NodeId: int64
    val Priors: float[]
    val ValueEstimate: float
    new(nodeId, priors, valueEstimate) = { NodeId = nodeId; Priors = priors; ValueEstimate = valueEstimate }
  end

/// Asynchronous prior client for NN-guided MCTS search.
/// Fires non-blocking prior requests on node expansion and collects completed
/// responses after backpropagation.
type IPriorClient =
    /// Fast pre-check: returns true when the client can produce a meaningful
    /// prior for a node whose parent is in the given state.  Called before the
    /// expensive collectActionStates step so that the engine can skip nodes
    /// that the implementation would discard anyway (e.g. road-stage nodes
    /// in placement mode).
    abstract ShouldRequestPrior : parentState: ICoreState -> bool

    /// Enqueue an async prior request for the given node.
    /// nodeId — opaque identifier to correlate response back to the MCTSState.
    /// parentState — the state at the node being expanded (before any action).
    /// states — result states for each action (one per deterministic action,
    ///          one per stochastic outcome). The implementation is responsible
    ///          for serialization.
    /// actingPlayer — the player whose turn it is at the parent node.
    /// depth — depth from root (lower = higher priority).
    /// Returns the number of (state, action) inference pairs actually sent to
    /// the model. In state mode this equals states.Length; in placement mode
    /// it is the total number of (settlement, road) composite actions enqueued
    /// across all child states. Returns 0 when the implementation declines
    /// to send a request (e.g. non-placement stage in placement mode).
    abstract RequestPrior : nodeId: int64 * parentState: ICoreState * states: ICoreState[] * actingPlayer: int * depth: int -> int

    /// Collect completed prior responses matching the given set of node IDs.
    /// Only responses whose NodeId is in knownNodeIds are returned; others
    /// remain in the mailbox for subsequent calls (e.g. from other games
    /// sharing the same client).
    abstract CollectPriors : knownNodeIds: IReadOnlySet<int64> -> PriorResponse[]

    /// Drop pending responses belonging to a completed search, identified
    /// by the node IDs the caller still tracks. Entries whose NodeId is in
    /// knownNodeIds are removed; all other entries (which may belong to
    /// other concurrent searches sharing this client) are preserved.
    /// Implementations must NOT clear server-side queues, since those are
    /// shared with other concurrent callers.
    abstract Flush : knownNodeIds: IReadOnlySet<int64> -> unit

type SimulationResult = 
  struct
    val mutable Rollouts: int
    val mutable ActionValues: float[]
    val mutable ElapsedTime: TimeSpan
  end

type IGameAI =
    // Calculates the best path for a given state.
    abstract RunSimulation : state: ICoreState -> SimulationResult

type IGameAIWithVariationPath =
    // Calculates the best path for a given state. A previously calculated best path can be provided from previous calculations to speed up
    // calculation time.
    abstract DetermineActionWithVariation : state: (ICoreState) -> variation: int [] -> int []

type searchTime =
    | Unlimited
    | Minutes of int
    | Seconds of int
    | MilliSeconds of int

type MCTSConfig =
    { SearchTime: searchTime
      MaxSimulations: int
      MaxRolloutDepth: int
      ExplorationConstant: float
      ActionRolloutLimit: int
      PriorClient: IPriorClient option
      ExpansionGuard: (ICoreState -> CoreAction -> bool) option
      MaxPriorDepth: int }

    static member Default =
        { SearchTime = Unlimited
          MaxSimulations = System.Int32.MaxValue
          MaxRolloutDepth = 500
          ExplorationConstant = sqrt 2.
          ActionRolloutLimit = System.Int32.MaxValue
          PriorClient = None
          ExpansionGuard = None
          MaxPriorDepth = System.Int32.MaxValue }
