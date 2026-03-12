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

/// Returns true when every action in the state is Terminal.
let allActionsTerminal (s: MCTSState) =
    not (Array.isEmpty s.Actions)
    && s.Actions |> Array.forall (fun a -> match a with Terminal _ -> true | _ -> false)

/// Given a state whose actions include at least one Terminal, find the best
/// outcome for the active player. Returns Some(outcome) when the state can be
/// resolved (condition 1: guaranteed win, or condition 2: all actions Terminal).
let tryResolveState (s: MCTSState) =
    let player = int s.State.PlayerTurn
    let terminals = s.Actions |> Array.choose (fun a -> match a with Terminal o -> Some o | _ -> None)

    if Array.isEmpty terminals then
        None
    else
        // Condition 1 — guaranteed win: any Terminal has 1.0 for the active player
        let bestTerminal =
            terminals |> Array.maxBy (fun o -> o.[player])

        if bestTerminal.[player] >= 1.0 then
            Some bestTerminal
        // Condition 2 — fully resolved: all actions are Terminal
        elif terminals.Length = s.Actions.Length then
            Some bestTerminal
        else
            None

/// Resolve a stochastic action when all outcome states are fully resolved.
/// Returns the weighted-average outcome across outcomes, where each outcome
/// state picks the best Terminal for its active player.
let tryResolveStochasticAction (outcomes: StochasticOutcome[]) =
    // Check that every outcome state is fully resolved (all actions Terminal)
    let resolvedOutcomes =
        outcomes
        |> Array.choose (fun o ->
            if Array.isEmpty o.State.Actions then
                // Terminal game state (no actions) — one-hot for current player
                Some (o.ProbabilityWeight, oneHotOutcome (o.State.State.PlayerTurn, o.State.State.NumberOfPlayers))
            else
                match tryResolveState o.State with
                | Some outcome -> Some (o.ProbabilityWeight, outcome)
                | None -> None)

    if resolvedOutcomes.Length <> outcomes.Length then
        None
    else
        let totalWeight = resolvedOutcomes |> Array.sumBy fst |> float
        let numPlayers = (snd resolvedOutcomes.[0]).Length
        let avg = Array.zeroCreate<float> numPlayers

        for (w, outcome) in resolvedOutcomes do
            for j in 0 .. numPlayers - 1 do
                avg.[j] <- avg.[j] + float w * outcome.[j]

        for j in 0 .. numPlayers - 1 do
            avg.[j] <- avg.[j] / totalWeight

        Some avg

/// Returns true when a state is resolved: either all actions are Terminal
/// (condition 2), or the active player has a guaranteed win via a Terminal
/// action (condition 1). Used as the root early-termination check.
let isResolved (s: MCTSState) =
    tryResolveState s |> Option.isSome

/// Walk the selection path bottom-up, replacing actions with Terminal when
/// their subtree is fully resolved. The path is ordered [deepest; ...; root].
/// In the visitedStates list from selection, index 0 is the most recent
/// (deepest) state — which is what we want for bottom-up traversal.
let propagateTerminals (visitedStates: MCTSState list) =
    // Walk pairs: each state checks if its parent (the next state in the list)
    // should replace the action pointing to it.
    // visitedStates is [deepest; ...; root], so we iterate pairs from front.
    let rec propagate (states: MCTSState list) =
        match states with
        | child :: parent :: rest ->
            // Try to resolve the child state
            let childResolved =
                if Array.isEmpty child.Actions then
                    // Terminal game state
                    Some (oneHotOutcome (child.State.PlayerTurn, child.State.NumberOfPlayers))
                else
                    tryResolveState child

            match childResolved with
            | Some outcome ->
                // Find the action in parent that points to child and replace it
                let mutable replaced = false

                for i in 0 .. parent.Actions.Length - 1 do
                    if not replaced then
                        match parent.Actions.[i] with
                        | DeterministicAction ds when Object.ReferenceEquals(ds, child) ->
                            parent.Actions.[i] <- Terminal outcome
                            replaced <- true
                        | StochasticAction outcomes ->
                            // Check if any outcome state is the child
                            let hasChild = outcomes |> Array.exists (fun o -> Object.ReferenceEquals(o.State, child))

                            if hasChild then
                                // For stochastic: try to resolve the entire stochastic action
                                match tryResolveStochasticAction outcomes with
                                | Some avg ->
                                    parent.Actions.[i] <- Terminal avg
                                    replaced <- true
                                | None -> ()
                        | _ -> ()

                if replaced then
                    propagate (parent :: rest)
            | None -> ()
        | _ -> ()

    propagate visitedStates

let search (root: MCTSState, maxSimulationCount, timer: Stopwatch, evaluateUntil: Int64 option, maxRolloutDepth: int, explorationConstant: float, actionRolloutLimit: int) =
    while root.Rollouts < maxSimulationCount
          && (not evaluateUntil.IsSome
              || timer.ElapsedTicks < evaluateUntil.Value)
          && maxActionRollouts root < actionRolloutLimit
          && not (isResolved root) do
        match select explorationConstant root with
        | Exhausted (stateHistory, outcome) ->
            backPropagate stateHistory outcome
            propagateTerminals stateHistory
        | Candidate (stateHistory, i) ->
            let mostRecentState = stateHistory.[0]

            let expandedState = expand (mostRecentState, i)
            let visitedStates = expandedState :: stateHistory

            // Check if the expanded state is a terminal game state (no actions).
            // For deterministic actions, replace the parent's action with Terminal
            // immediately. For stochastic actions (which expand all outcomes at
            // once), let propagateTerminals handle resolution after all outcomes
            // are visited.
            if Array.isEmpty expandedState.Actions then
                let outcome = oneHotOutcome (expandedState.State.PlayerTurn, expandedState.State.NumberOfPlayers)
                backPropagate visitedStates outcome
                match mostRecentState.Actions.[i] with
                | DeterministicAction _ ->
                    mostRecentState.Actions.[i] <- Terminal outcome
                | _ -> ()
                propagateTerminals visitedStates
            else
                let outcome = simulate maxRolloutDepth expandedState
                backPropagate visitedStates outcome
                propagateTerminals visitedStates
        | StochasticCandidate (stateHistory, ia, is) ->
            match stateHistory.[0].Actions.[ia] with
            | StochasticAction stochasticOutcomes ->
                let expandedState = stochasticOutcomes.[is].State
                let visitedStates = expandedState :: stateHistory

                if Array.isEmpty expandedState.Actions then
                    let outcome = oneHotOutcome (expandedState.State.PlayerTurn, expandedState.State.NumberOfPlayers)
                    backPropagate visitedStates outcome
                    propagateTerminals visitedStates
                else
                    let outcome = simulate maxRolloutDepth expandedState
                    backPropagate visitedStates outcome
                    propagateTerminals visitedStates
            | _ -> failwith "unreachable"

    extractBestPath root |> List.toArray
