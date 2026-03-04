module Kjarni.MCTS.Algorithm

open System

open System.Diagnostics
open Kjarni
open Kjarni.MCTS.Types

let emptyOutcome maxTrackedPlayers = Array.zeroCreate<float> maxTrackedPlayers

let oneHotOutcome (winner: Player, maxTrackedPlayers) =
    let outcome = emptyOutcome maxTrackedPlayers
    outcome.[int winner] <- 1.
    outcome

let explorationRate (explorationConstant: float) (stateVisitCount: int) (actionVisitCount: int) =
    explorationConstant
    * sqrt (
        log (float stateVisitCount)
        / float actionVisitCount
    )

let winRate (state: MCTSState) (player: Player) =
    state.WinCounts[int player] / float state.Rollouts

let sampledWinRate outcomes player =
    let sampledOutcomes = outcomes |> Array.filter (fun o -> o.State.Rollouts > 0)
    let denominator = Array.sumBy (fun o -> float o.ProbabilityWeight) sampledOutcomes
    if denominator = 0
    then
        0.
    else
        Array.sumBy (fun o -> float o.ProbabilityWeight * winRate o.State player) sampledOutcomes / denominator

let actionEvaluator (explorationConstant: float) (state: MCTSState) (l: Action) =
    let actingPlayer = state.State.PlayerTurn

    match l with
    | Unexplored _ -> 10.
    | DeterministicAction resState ->
        let winRate = winRate resState actingPlayer
        let explorationRate = explorationRate explorationConstant state.Rollouts resState.Rollouts
        winRate + explorationRate
    | StochasticAction outcomes ->
        let totalRollouts = Array.sumBy (fun i -> i.State.Rollouts) outcomes
        if totalRollouts = 0 then 10. // treat as unexplored
        else
            let winRate = sampledWinRate outcomes actingPlayer
            let explorationRate = explorationRate explorationConstant state.Rollouts totalRollouts
            winRate + explorationRate
    | Terminal win -> win.[int actingPlayer]

let rollStochasticAction(probWeights: int array) =
    let totalWeight = Array.sum probWeights
    let roll = Random.Shared.Next totalWeight // [0, totalWeight)
    let mutable cumulative = 0
    let mutable i = 0
    while cumulative + probWeights.[i] <= roll do
      cumulative <- cumulative + probWeights.[i]
      i <- i + 1
    i


let rec recSelect (explorationConstant: float) (s: MCTSState, visitedStates: MCTSState list) =
    if Array.isEmpty s.Actions then
        Exhausted(visitedStates, oneHotOutcome(s.State.PlayerTurn, s.State.NumberOfPlayers))
    else
        let selectedAction =
            s.Actions
              |> Array.indexed
              |> Array.maxBy (fun a -> actionEvaluator explorationConstant s (snd a))
        match snd selectedAction with
        | Unexplored _ -> Candidate(visitedStates, fst selectedAction)
        | DeterministicAction ds -> recSelect explorationConstant (ds,  ds :: visitedStates)
        | StochasticAction so ->
            let i = rollStochasticAction (Array.map (fun o -> o.ProbabilityWeight) so)
            let state = so.[i].State
            if state.Rollouts = 0 // Unexplored outcome, return state
            then StochasticCandidate(visitedStates, fst selectedAction, i)
            else recSelect explorationConstant (state, state :: visitedStates)
        | Terminal outcome -> Exhausted(visitedStates, outcome)

let select (explorationConstant: float) (root: MCTSState) =
    recSelect explorationConstant (root, [root])

let expand (s: MCTSState, i) =
      match s.Actions.[i] with
      | Unexplored a ->
          match a with
          | Deterministic da ->
              let expandedState = MCTSState(da.State())
              s.Actions.[i] <- DeterministicAction(expandedState)
              expandedState
          | Stochastic sa ->
              let stochasticActions = sa.Outcomes() |> Array.map (fun i -> { ProbabilityWeight = fst i; State = MCTSState(snd i)})
              s.Actions.[i] <- StochasticAction(stochasticActions)
              stochasticActions.[rollStochasticAction (Array.map (fun i -> i.ProbabilityWeight) stochasticActions)].State
      | _ -> raise (Exception "Target action is already expanded")


let defaultMaxRolloutDepth = 500

let scoreBasedOutcome (state: ICoreState) =
    let scores = state.Scores()
    let outcome = emptyOutcome state.NumberOfPlayers

    // Find the maximum score among all players.
    let mutable maxScore = Double.NegativeInfinity
    for i in 0 .. state.NumberOfPlayers - 1 do
        let s = if i < scores.Length then scores.[i] else 0.
        if s > maxScore then
            maxScore <- s

    if maxScore <= 0. then
        // No one has any score; return draw (empty outcome).
        outcome
    else
        // Count how many players share the top score.
        let mutable tiedCount = 0
        for i in 0 .. state.NumberOfPlayers - 1 do
            let s = if i < scores.Length then scores.[i] else 0.
            if s = maxScore then
                tiedCount <- tiedCount + 1

        // Split the win equally among tied leaders.
        let share = 1. / float tiedCount
        for i in 0 .. state.NumberOfPlayers - 1 do
            let s = if i < scores.Length then scores.[i] else 0.
            if s = maxScore then
                outcome.[i] <- share

        outcome

let rec simulationDistribution (maxRolloutDepth: int) (state: ICoreState) (depth: int) =
    let actions = state.Actions()
    if Array.isEmpty actions then
        oneHotOutcome(state.PlayerTurn, state.NumberOfPlayers)
    elif depth >= maxRolloutDepth then
        scoreBasedOutcome state
    else
        // TODO: implement policy based rollout
        // Random rollout
        let roll = Random.Shared.Next actions.Length
        let nextMove = actions.[roll]
        match nextMove with
        | Stochastic stochasticAction->
            let outcomes = stochasticAction.Outcomes()
            if Array.isEmpty outcomes then
                oneHotOutcome(state.PlayerTurn, state.NumberOfPlayers)
            else
                let i = rollStochasticAction (Array.map fst outcomes)
                let sampled = snd outcomes.[i]
                simulationDistribution maxRolloutDepth sampled (depth + 1)
        | Deterministic deterministicAction ->
            simulationDistribution maxRolloutDepth (deterministicAction.State()) (depth + 1)

let simulate (maxRolloutDepth: int) (s: MCTSState) = simulationDistribution maxRolloutDepth s.State 0

let backPropagate (visitedStates: MCTSState list) (outcome: float array) =
    for state in visitedStates do
        state.Rollouts <- state.Rollouts + 1
        for j in 0 .. state.WinCounts.Length - 1 do
            state.WinCounts.[j] <- state.WinCounts.[j] + outcome.[j]

let extractionEvaluator (p: Player, l: Action) =
    match l with
    | Terminal outcome -> outcome.[int p]
    | DeterministicAction da -> winRate da p
    | StochasticAction outcomes -> sampledWinRate outcomes p
    | Unexplored _ -> 0.

let extractBestPath (root: MCTSState) =
    let mutable path = List.empty
    let mutable currentState = root
    let mutable stopExtraction = false

    while not stopExtraction do
        if Array.isEmpty currentState.Actions then
            stopExtraction <- true
        else
            let bestAction =
                currentState.Actions
                |> Array.indexed
                |> Array.maxBy (fun l -> extractionEvaluator (currentState.State.PlayerTurn, snd l))

            path <- fst bestAction :: path

            match snd bestAction with
            | DeterministicAction state -> currentState <- state
            | _ -> stopExtraction <- true

    path |> List.rev

let actionRollouts (a: Action) =
    match a with
    | Unexplored _ -> 0
    | DeterministicAction s -> s.Rollouts
    | StochasticAction outcomes -> Array.sumBy (fun o -> o.State.Rollouts) outcomes
    | Terminal _ -> 0

let maxActionRollouts (root: MCTSState) =
    if Array.isEmpty root.Actions then 0
    else root.Actions |> Array.map actionRollouts |> Array.max

let search (root: MCTSState, maxSimulationCount, timer: Stopwatch, evaluateUntil: Int64 option, maxRolloutDepth: int, explorationConstant: float, minActionRollouts: int) =
    while root.Rollouts < maxSimulationCount
          && (not evaluateUntil.IsSome
              || timer.ElapsedTicks < evaluateUntil.Value)
          && maxActionRollouts root < minActionRollouts do
        match select explorationConstant root with
        | Exhausted (stateHistory, outcome) -> backPropagate stateHistory outcome
        | Candidate (stateHistory, i) ->
            let mostRecentState = stateHistory.[0]

            let expandedState = expand (mostRecentState, i)
            let outcome = simulate maxRolloutDepth expandedState
            backPropagate (expandedState :: stateHistory) outcome
        | StochasticCandidate (stateHistory, ia, is) ->
            match stateHistory.[0].Actions.[ia] with
            | StochasticAction stochasticOutcomes ->
                let expandedState = stochasticOutcomes.[is].State
                let outcome = simulate maxRolloutDepth expandedState
                backPropagate (expandedState :: stateHistory) outcome
            | _ -> failwith "unreachable"

    extractBestPath root |> List.toArray
