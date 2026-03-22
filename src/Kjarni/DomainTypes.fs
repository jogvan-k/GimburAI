namespace Kjarni

open System

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
/// parent MCTSState. WinProbabilities contains per-action-state win probabilities
/// in the same order as the request's states array.
type PriorResponse =
  struct
    val NodeId: int64
    val WinProbabilities: float[]
    new(nodeId, winProbabilities) = { NodeId = nodeId; WinProbabilities = winProbabilities }
  end

/// Asynchronous prior client for NN-guided MCTS search.
/// Fires non-blocking prior requests on node expansion and collects completed
/// responses after backpropagation.
type IPriorClient =
    /// Enqueue an async prior request for the given node.
    /// nodeId — opaque identifier to correlate response back to the MCTSState.
    /// parentState — the state at the node being expanded (before any action).
    /// states — result states for each action (one per deterministic action,
    ///          one per stochastic outcome). The implementation is responsible
    ///          for serialization.
    /// actingPlayer — the player whose turn it is at the parent node.
    /// depth — depth from root (lower = higher priority).
    abstract RequestPrior : nodeId: int64 * parentState: ICoreState * states: ICoreState[] * actingPlayer: int * depth: int -> unit

    /// Drain all completed prior responses from the mailbox. Non-blocking.
    abstract CollectPriors : unit -> PriorResponse[]

    /// Clear the server queue and discard pending results.
    abstract Flush : unit -> unit

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
      ExpansionGuard: (ICoreState -> CoreAction -> bool) option }

    static member Default =
        { SearchTime = Unlimited
          MaxSimulations = System.Int32.MaxValue
          MaxRolloutDepth = 500
          ExplorationConstant = sqrt 2.
          ActionRolloutLimit = System.Int32.MaxValue
          PriorClient = None
          ExpansionGuard = None }
