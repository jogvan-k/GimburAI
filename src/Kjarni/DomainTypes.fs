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
      ExplorationConstant: float }

    static member Default =
        { SearchTime = Unlimited
          MaxSimulations = System.Int32.MaxValue
          MaxRolloutDepth = 500
          ExplorationConstant = sqrt 2. }
