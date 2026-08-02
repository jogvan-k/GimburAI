module Kjarni.MCTS.Algorithm

open System
open System.Collections.Generic
open System.Diagnostics
open System.Threading
open Kjarni
open Kjarni.MCTS.Types

let emptyOutcome maxTrackedPlayers = Array.zeroCreate<float> maxTrackedPlayers

let oneHotOutcome (winner: Player, maxTrackedPlayers) =
    let outcome = emptyOutcome maxTrackedPlayers
    outcome.[int winner] <- 1.
    outcome

let explorationRate (explorationConstant: float) (parentVisitCount: int) (actionVisitCount: int) (prior: float) =
    explorationConstant
    * prior
    * sqrt (float parentVisitCount)
    / (1. + float actionVisitCount)

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

let actionEvaluator (explorationConstant: float) (state: MCTSState) (actionIndex: int) (l: Action) =
    let actingPlayer = state.State.PlayerTurn
    let prior =
        match state.Priors with
        | Some p -> p.[actionIndex]
        | None -> 1. / float state.Actions.Length

    match l with
    | Unexplored _ ->
        explorationRate explorationConstant state.Rollouts 0 prior
    | DeterministicAction resState | HorizonAction resState ->
        let winRate = winRate resState actingPlayer
        let explorationRate = explorationRate explorationConstant state.Rollouts resState.Rollouts prior
        winRate + explorationRate
    | StochasticAction outcomes ->
        let totalRollouts = Array.sumBy (fun i -> i.State.Rollouts) outcomes
        if totalRollouts = 0 then
            explorationRate explorationConstant state.Rollouts 0 prior
        else
            let winRate = sampledWinRate outcomes actingPlayer
            let explorationRate = explorationRate explorationConstant state.Rollouts totalRollouts prior
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
              |> Array.maxBy (fun (i, a) -> actionEvaluator explorationConstant s i a)
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
        | HorizonAction hs -> Horizon(visitedStates, hs)

let select (explorationConstant: float) (root: MCTSState) =
    recSelect explorationConstant (root, [root])

let expand (leafBoundary: (ICoreState -> bool) option) (s: MCTSState, i) =
      match s.Actions.[i] with
      | Unexplored a ->
          match a with
          | Deterministic da ->
              let expandedState = MCTSState(da.State())
              s.Actions.[i] <-
                  match leafBoundary with
                  | Some predicate when predicate expandedState.State -> HorizonAction expandedState
                  | _ -> DeterministicAction expandedState
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
    | HorizonAction ha -> winRate ha p
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
    | HorizonAction s -> s.Rollouts
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

// ──────────────────────────────────────────────────────────────────────────────
// Prior support: helpers for enqueuing requests and applying responses
// ──────────────────────────────────────────────────────────────────────────────

/// Collect the result ICoreStates for every action of a node by peeking at the
/// underlying CoreActions. For deterministic actions the result is a single
/// state from action.State(). For stochastic actions each outcome is a separate
/// state. Returns the flat array of states, a layout descriptor that records
/// how many states each action contributed, and (for stochastic actions) the
/// outcome weights so win probabilities can be mapped back.
///
/// This works on Unexplored actions (calling the CoreAction interface) as well
/// as already-expanded actions (reading the MCTSState wrappers). The typical
/// use case is right after a node is expanded — the expanded node's own actions
/// are still Unexplored.
let collectActionStates (s: MCTSState) : (ICoreState[] * int[] * int[][]) =
    let states = ResizeArray<ICoreState>()
    let layout = ResizeArray<int>()
    let weights = ResizeArray<int[]>()

    for action in s.Actions do
        match action with
        | Unexplored coreAction ->
            match coreAction with
            | Deterministic da ->
                states.Add(da.State())
                layout.Add(1)
                weights.Add([| 1 |])
            | Stochastic sa ->
                let outcomes = sa.Outcomes()
                for (_, resultState) in outcomes do
                    states.Add(resultState)
                layout.Add(outcomes.Length)
                weights.Add(outcomes |> Array.map fst)
        | DeterministicAction child | HorizonAction child ->
            states.Add(child.State)
            layout.Add(1)
            weights.Add([| 1 |])
        | StochasticAction outcomes ->
            for o in outcomes do
                states.Add(o.State.State)
            layout.Add(outcomes.Length)
            weights.Add(outcomes |> Array.map (fun o -> o.ProbabilityWeight))
        | Terminal _ ->
            layout.Add(0)
            weights.Add(Array.empty)

    (states.ToArray(), layout.ToArray(), weights.ToArray())

/// Given an array of per-state win probabilities (from the NN, in the same
/// order as collectActionStates produced), the layout descriptor, and the
/// outcome weights, compute a normalised prior policy over the node's actions.
///
/// For deterministic actions: prior = winProb directly.
/// For stochastic actions: prior = weighted average of outcome winProbs.
/// Actions with 0 states in the layout (Terminal) get zero prior.
/// The result is normalised so it sums to 1.
let computePriorPolicy (winProbs: float[]) (layout: int[]) (outcomeWeights: int[][]) : float[] =
    let n = layout.Length
    let rawPriors = Array.zeroCreate<float> n
    let mutable probIdx = 0
    let expectedCount = Array.sum layout
    let valid =
        winProbs.Length = expectedCount
        && winProbs |> Array.forall (fun value -> Double.IsFinite(value) && value >= 0.)

    if valid then
        for i in 0 .. n - 1 do
            let count = layout.[i]
            if count = 0 then
                rawPriors.[i] <- 0.
            elif count = 1 then
                rawPriors.[i] <- winProbs.[probIdx]
                probIdx <- probIdx + 1
            else
                // Stochastic: weighted average across outcomes
                let weights = outcomeWeights.[i]
                let mutable weightedSum = 0.
                let mutable totalWeight = 0.
                for j in 0 .. count - 1 do
                    let w = float weights.[j]
                    weightedSum <- weightedSum + w * winProbs.[probIdx + j]
                    totalWeight <- totalWeight + w
                rawPriors.[i] <- if totalWeight > 0. then weightedSum / totalWeight else 0.
                probIdx <- probIdx + count

    // Normalise to sum to 1
    let total = Array.sum rawPriors
    if total > 0. then
        for i in 0 .. n - 1 do
            rawPriors.[i] <- rawPriors.[i] / total
    else
        // All zero — fall back to uniform
        let uniform = 1. / float n
        for i in 0 .. n - 1 do
            rawPriors.[i] <- uniform

    rawPriors

/// Counts collected during prior request/application in the search loop.
///
/// Terminology:
///   * "node" = an MCTS tree node where a prior was considered.
///   * "action" = an MCTS-level child action (one per legal action enumerated
///     by the engine; for stochastic actions, one per outcome).
///   * "inference" = an individual (state, action) pair actually evaluated by
///     the model. In state mode this equals the action count; in placement
///     mode each MCTS child (settlement) fans out into multiple composite
///     (settlement, road) inferences, so inferences > actions.
type PriorStats =
  struct
    /// Number of nodes for which a prior request was issued to the client.
    val mutable priorNodesRequested: int
    /// Number of MCTS-level action states sent to the client across all
    /// requested nodes.
    val mutable priorActionsRequested: int
    /// Number of (state, action) inference pairs actually sent to the model.
    /// Equals priorActionsRequested in state mode; in placement mode it counts
    /// the post-fan-out composite (settlement, road) pairs.
    val mutable priorInferencesRequested: int
    /// Number of nodes whose prior policy was successfully attached.
    val mutable priorNodesApplied: int
    /// Number of action states whose prior probabilities were applied.
    val mutable priorActionsApplied: int
    /// Per-depth count of MCTS-level action states sent to the client (depth → count).
    val mutable priorActionsPerDepth: Dictionary<int, int>
    /// Per-depth count of model inference pairs (depth → count). Differs from
    /// priorActionsPerDepth in placement mode where each MCTS child fans out.
    val mutable priorInferencesPerDepth: Dictionary<int, int>
    /// Number of nodes refused by the client's ShouldRequestPrior pre-check.
    val mutable priorNodesSkipped: int
    /// Number of responses returned for nodes the search no longer tracks.
    val mutable priorResponsesOrphaned: int
    /// Number of selection paths that hit the maxPriorDepth horizon.
    val mutable horizonSkips: int
  end

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

let search (root: MCTSState, maxSimulationCount, timer: Stopwatch, evaluateUntil: Int64 option, maxRolloutDepth: int, explorationConstant: float, actionRolloutLimit: int, priorClient: IPriorClient option, leafBoundary: (ICoreState -> bool) option, maxPriorDepth: int) =
    // Normalise the option to guard against C# passing Some(null).
    let priorClient =
        match priorClient with
        | Some client when not (isNull (box client)) -> Some client
        | _ -> None

    // Node registry and layout registry for correlating prior responses.
    // Only allocated when a prior client is configured.
    let nodeRegistry =
        match priorClient with
        | Some _ -> Some (Dictionary<int64, MCTSState>())
        | None -> None
    let layoutRegistry =
        match priorClient with
        | Some _ -> Some (Dictionary<int64, int[] * int[][]>())
        | None -> None

    let mutable priorStats = PriorStats()
    priorStats.priorActionsPerDepth <- Dictionary<int, int>()
    priorStats.priorInferencesPerDepth <- Dictionary<int, int>()

    let normalizeValueEstimates playerCount (values: float[]) =
        if values.Length <> playerCount
           || values |> Array.exists (fun value -> not (Double.IsFinite(value)) || value < 0.) then
            None
        else
            let total = Array.sum values
            if not (Double.IsFinite(total)) || total <= 0. then
                None
            else
                Some (values |> Array.map (fun value -> value / total))

    /// Fire a prior request for the given node (non-blocking).
    /// Skips if the node already has priors or is already registered
    /// (pending response from an earlier request in this search).
    let requestPrior (node: MCTSState) (depth: int) =
        match priorClient, nodeRegistry, layoutRegistry with
        | Some client, Some nodeReg, Some layoutReg ->
            if depth <= maxPriorDepth && not (Array.isEmpty node.Actions) && node.Priors.IsNone && not (nodeReg.ContainsKey(node.NodeId)) then
                if not (client.ShouldRequestPrior(node.State)) then
                    priorStats.priorNodesSkipped <- priorStats.priorNodesSkipped + 1
                else
                    let (actionStates, layout, outcomeWeights) = collectActionStates node
                    if actionStates.Length > 0 then
                        nodeReg.[node.NodeId] <- node
                        layoutReg.[node.NodeId] <- (layout, outcomeWeights)
                        let inferenceCount = client.RequestPrior(node.NodeId, node.State, actionStates, int node.State.PlayerTurn + 1, depth)
                        if inferenceCount > 0 then
                            priorStats.priorNodesRequested <- priorStats.priorNodesRequested + 1
                            priorStats.priorActionsRequested <- priorStats.priorActionsRequested + actionStates.Length
                            priorStats.priorInferencesRequested <- priorStats.priorInferencesRequested + inferenceCount
                            let aCount =
                                match priorStats.priorActionsPerDepth.TryGetValue(depth) with
                                | true, v -> v
                                | _ -> 0
                            priorStats.priorActionsPerDepth.[depth] <- aCount + actionStates.Length
                            let iCount =
                                match priorStats.priorInferencesPerDepth.TryGetValue(depth) with
                                | true, v -> v
                                | _ -> 0
                            priorStats.priorInferencesPerDepth.[depth] <- iCount + inferenceCount
                        else
                            nodeReg.Remove(node.NodeId) |> ignore
                            layoutReg.Remove(node.NodeId) |> ignore
                            priorStats.priorNodesSkipped <- priorStats.priorNodesSkipped + 1
        | _ -> ()

    /// Collect completed prior responses and apply them to tree nodes.
    let collectPriors () =
        match priorClient, nodeRegistry, layoutRegistry with
        | Some client, Some nodeReg, Some layoutReg ->
            let knownIds = HashSet<int64>(nodeReg.Keys) :> IReadOnlySet<int64>
            let responses = client.CollectPriors(knownIds)
            for resp in responses do
                match nodeReg.TryGetValue(resp.NodeId) with
                | true, node ->
                    match layoutReg.TryGetValue(resp.NodeId) with
                    | true, (layout, outcomeWeights) ->
                        if resp.Priors.Length > 0 then
                            let policy = computePriorPolicy resp.Priors layout outcomeWeights
                            node.Priors <- Some policy
                            priorStats.priorActionsApplied <- priorStats.priorActionsApplied + resp.Priors.Length
                        if resp.DensePriors.Length > 0 then
                            node.DensePriors <- Some resp.DensePriors
                        match normalizeValueEstimates node.State.NumberOfPlayers resp.ValueEstimates with
                        | Some values -> node.ValueEstimates <- Some values
                        | None -> ()
                        priorStats.priorNodesApplied <- priorStats.priorNodesApplied + 1
                        // Remove from registries once applied
                        nodeReg.Remove(resp.NodeId) |> ignore
                        layoutReg.Remove(resp.NodeId) |> ignore
                    | _ -> ()
                | _ ->
                    // Stale response (node no longer in registry) — discard
                    priorStats.priorResponsesOrphaned <- priorStats.priorResponsesOrphaned + 1
                    ()
        | _ -> ()

    // Re-submit prior requests for already-expanded nodes along the
    // selection path that are missing priors (e.g. from tree reuse where
    // the fresh registries don't know about them yet).
    let resubmitPriors (stateHistory: MCTSState list) =
        let len = List.length stateHistory
        // stateHistory is [deepest; ...; root], depths are [len-1; ...; 0].
        let mutable depth = len - 1
        for node in stateHistory do
            requestPrior node depth
            depth <- depth - 1


    // ── Root prior wait ──────────────────────────────────────────────────
    // Proactively request the root node's prior and block briefly until
    // it arrives.  The root prior is the single most valuable piece of NN
    // guidance: it steers the very first action selection and every
    // subsequent UCB comparison.  Without this wait the search typically
    // finishes all rollouts before the HTTP round-trip completes.
    let rootPriorWaitTimeoutMs = 50L
    requestPrior root 0
    if priorClient.IsSome && root.Priors.IsNone then
        let waitStart = timer.ElapsedMilliseconds
        while root.Priors.IsNone
              && (timer.ElapsedMilliseconds - waitStart) < rootPriorWaitTimeoutMs do
            collectPriors ()
            Thread.Sleep(1)
        // One final collection attempt after the sleep loop.
        collectPriors ()

    while root.Rollouts < maxSimulationCount
          && (not evaluateUntil.IsSome
              || timer.ElapsedTicks < evaluateUntil.Value)
          && maxActionRollouts root < actionRolloutLimit
          && not (isResolved root) do
        match select explorationConstant root with
        | Exhausted (stateHistory, outcome) ->
            resubmitPriors stateHistory
            backPropagate stateHistory outcome
            propagateTerminals stateHistory
            collectPriors ()
        | Candidate (stateHistory, i) ->
            resubmitPriors stateHistory
            let mostRecentState = stateHistory.[0]
            let depth = stateHistory.Length

            let expandedState = expand leafBoundary (mostRecentState, i)
            let visitedStates = expandedState :: stateHistory

            match mostRecentState.Actions.[i] with
            | HorizonAction _ ->
                let outcome = simulate maxRolloutDepth expandedState
                backPropagate visitedStates outcome
                priorStats.horizonSkips <- priorStats.horizonSkips + 1
                collectPriors ()
            // Check if the expanded state is a terminal game state (no actions).
            // For deterministic actions, replace the parent's action with Terminal
            // immediately. For stochastic actions (which expand all outcomes at
            // once), let propagateTerminals handle resolution after all outcomes
            // are visited.
            | _ when Array.isEmpty expandedState.Actions ->
                let outcome = oneHotOutcome (expandedState.State.PlayerTurn, expandedState.State.NumberOfPlayers)
                backPropagate visitedStates outcome
                match mostRecentState.Actions.[i] with
                | DeterministicAction _ ->
                    mostRecentState.Actions.[i] <- Terminal outcome
                | _ -> ()
                propagateTerminals visitedStates
                collectPriors ()
            | _ ->
                // Phase 3 — Prior request: enqueue async NN evaluation for the
                // expanded node's actions.
                requestPrior expandedState depth
                let outcome = simulate maxRolloutDepth expandedState
                backPropagate visitedStates outcome
                propagateTerminals visitedStates
                collectPriors ()
        | StochasticCandidate (stateHistory, ia, is) ->
            resubmitPriors stateHistory
            match stateHistory.[0].Actions.[ia] with
            | StochasticAction stochasticOutcomes ->
                let expandedState = stochasticOutcomes.[is].State
                let visitedStates = expandedState :: stateHistory

                if Array.isEmpty expandedState.Actions then
                    let outcome = oneHotOutcome (expandedState.State.PlayerTurn, expandedState.State.NumberOfPlayers)
                    backPropagate visitedStates outcome
                    propagateTerminals visitedStates
                    collectPriors ()
                else
                    let outcome = simulate maxRolloutDepth expandedState
                    backPropagate visitedStates outcome
                    propagateTerminals visitedStates
                    collectPriors ()
            | _ -> failwith "unreachable"
        | Horizon (stateHistory, horizonState) ->
            resubmitPriors stateHistory
            let outcome = simulate maxRolloutDepth horizonState
            backPropagate (horizonState :: stateHistory) outcome
            priorStats.horizonSkips <- priorStats.horizonSkips + 1
            collectPriors ()

    // Final collection: apply any priors that arrived during the last
    // iterations of the search loop, before the caller flushes.
    collectPriors ()

    // Drop this search's pending responses from the shared client
    // mailbox. Responses for other concurrent searches are preserved.
    // The server-side queue is never cleared (it is shared too).
    match priorClient, nodeRegistry with
    | Some client, Some nodeReg ->
        let knownIds = HashSet<int64>(nodeReg.Keys) :> IReadOnlySet<int64>
        client.Flush(knownIds)
    | _ -> ()

    (extractBestPath root |> List.toArray, priorStats)
