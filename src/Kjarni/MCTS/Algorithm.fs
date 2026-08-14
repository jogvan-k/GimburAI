module Kjarni.MCTS.Algorithm

open System
open System.Collections.Generic
open System.Diagnostics
open System.Threading
open Kjarni
open Kjarni.MCTS.Types

let mutable private nextLeafRequestId = 0L

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

let edgeQ (stats: ActionStats) (player: Player) =
    if stats.CompletedVisits = 0 then 0.
    else stats.ValueSums.[int player] / float stats.CompletedVisits

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
    let stats = state.ActionStats.[actionIndex]
    let prior =
        match state.Priors with
        | Some p -> p.[actionIndex]
        | None -> 1. / float state.Actions.Length

    match l with
    | Unexplored _ | DeterministicAction _ | HorizonAction _ | StochasticAction _ ->
        edgeQ stats actingPlayer
        + explorationRate
            explorationConstant
            state.Rollouts
            (stats.CompletedVisits + stats.PendingVisits)
            prior
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


let rec recSelect (explorationConstant: float) (s: MCTSState, path: SelectionPath) =
    if Array.isEmpty s.Actions then
        Exhausted(path, oneHotOutcome(s.State.PlayerTurn, s.State.NumberOfPlayers))
    else
        let available =
            s.Actions
            |> Array.indexed
            |> Array.filter (fun (i, _) -> s.ActionStats.[i].PendingVisits = 0)
        if Array.isEmpty available then
            Blocked
        else
            let selectedAction =
                available
                |> Array.maxBy (fun (i, a) ->
                    let prior =
                        match s.Priors with
                        | Some priors -> priors.[i]
                        | None -> 1. / float s.Actions.Length
                    actionEvaluator explorationConstant s i a, prior)
            match snd selectedAction with
            | Unexplored _ -> Candidate(path, fst selectedAction)
            | DeterministicAction ds ->
                let edge = { Parent = s; ActionIndex = fst selectedAction }
                recSelect explorationConstant (ds, { States = ds :: path.States; Edges = edge :: path.Edges })
            | StochasticAction so ->
                let i = rollStochasticAction (Array.map (fun o -> o.ProbabilityWeight) so)
                let state = so.[i].State
                let edge = { Parent = s; ActionIndex = fst selectedAction }
                let nextPath = { States = state :: path.States; Edges = edge :: path.Edges }
                if state.Rollouts = 0 // Unexplored outcome, return state
                then StochasticCandidate(nextPath, i)
                else recSelect explorationConstant (state, nextPath)
            | Terminal outcome ->
                let edge = { Parent = s; ActionIndex = fst selectedAction }
                Exhausted({ path with Edges = edge :: path.Edges }, outcome)
            | HorizonAction hs ->
                let edge = { Parent = s; ActionIndex = fst selectedAction }
                Horizon({ path with Edges = edge :: path.Edges }, hs)

let select (explorationConstant: float) (root: MCTSState) =
    recSelect explorationConstant (root, { States = [root]; Edges = [] })

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

let backPropagatePath (path: SelectionPath) (outcome: float array) =
    backPropagate path.States outcome
    for edge in path.Edges do
        let stats = edge.Parent.ActionStats.[edge.ActionIndex]
        stats.CompletedVisits <- stats.CompletedVisits + 1
        for j in 0 .. stats.ValueSums.Length - 1 do
            stats.ValueSums.[j] <- stats.ValueSums.[j] + outcome.[j]

let weightedOutcome (weightedOutcomes: (int * float array) array) =
    let totalWeight = weightedOutcomes |> Array.sumBy (fst >> float)
    let result = Array.zeroCreate<float> (snd weightedOutcomes.[0]).Length
    for weight, outcome in weightedOutcomes do
        for j in 0 .. result.Length - 1 do
            result.[j] <- result.[j] + float weight * outcome.[j] / totalWeight
    result

let evaluateStochasticOutcomes (evaluate: MCTSState -> float array) (outcomes: StochasticOutcome array) =
    outcomes
    |> Array.map (fun stochasticOutcome ->
        stochasticOutcome.ProbabilityWeight, evaluate stochasticOutcome.State)
    |> weightedOutcome

let extractionEvaluator (state: MCTSState) actionIndex =
    match state.Actions.[actionIndex] with
    | Terminal outcome -> outcome.[int state.State.PlayerTurn]
    | _ -> edgeQ state.ActionStats.[actionIndex] state.State.PlayerTurn

let extractionKey (state: MCTSState) actionIndex =
    let prior =
        match state.Priors with
        | Some priors when actionIndex < priors.Length -> priors.[actionIndex]
        | _ -> 1. / float state.Actions.Length
    let stats = state.ActionStats.[actionIndex]
    stats.CompletedVisits + stats.PendingVisits, extractionEvaluator state actionIndex, prior

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
                |> Array.maxBy (fun (i, _) -> extractionKey currentState i)

            path <- fst bestAction :: path

            match snd bestAction with
            | DeterministicAction state -> currentState <- state
            | _ -> stopExtraction <- true

    path |> List.rev

let actionRollouts (state: MCTSState) actionIndex =
    let stats = state.ActionStats.[actionIndex]
    stats.CompletedVisits + stats.PendingVisits

let maxActionRollouts (root: MCTSState) =
    if Array.isEmpty root.ActionStats then 0
    else root.Actions |> Array.mapi (fun i _ -> actionRollouts root i) |> Array.max

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
    /// Number of selection paths that hit a structural evaluation horizon.
    val mutable horizonSkips: int
  end

type LeafEvaluationStats =
  struct
    val mutable submitted: int
    val mutable applied: int
    val mutable timeouts: int
    val mutable invalid: int
    val mutable cancelled: int
    val mutable fallback: int
    val mutable orphan: int
    val mutable batches: int
    val mutable states: int
    val mutable latencyMs: int64
  end

type private PendingEvaluation =
    { Path: SelectionPath
      States: MCTSState[]
      Weights: int[]
      ExactOutcomes: (int * float[])[]
      SubmittedAtMs: int64 }

type private PendingPriorEvaluation =
    { Path: SelectionPath
      States: MCTSState[]
      Weights: int[]
      ExactOutcomes: (int * float[])[]
      SubmittedAtMs: int64 }

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

let search (root: MCTSState, maxSimulationCount, timer: Stopwatch, evaluateUntil: Int64 option, maxRolloutDepth: int, explorationConstant: float, actionRolloutLimit: int, priorClient: IPriorClient option, leafEvaluator: ILeafEvaluator option, leafBoundary: (ICoreState -> bool) option, maxTreeDepth: int, maxPendingEvaluations: int, leafEvaluationTimeoutMs: int, drainTimeoutMs: int) =
    // Normalise the option to guard against C# passing Some(null).
    let priorClient =
        match priorClient with
        | Some client when not (isNull (box client)) -> Some client
        | _ -> None
    let leafEvaluator =
        match leafEvaluator with
        | Some evaluator when not (isNull (box evaluator)) -> Some evaluator
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
    let mutable leafStats = LeafEvaluationStats()
    let pendingEvaluations = Dictionary<int64, PendingEvaluation>()
    let pendingPriorEvaluations = Dictionary<int64, PendingPriorEvaluation>()

    let mutable effectiveEvaluateUntil = evaluateUntil
    let beforeDeadline () =
        not effectiveEvaluateUntil.IsSome || timer.ElapsedTicks < effectiveEvaluateUntil.Value

    let reservePath (path: SelectionPath) delta =
        for edge in path.Edges do
            let stats = edge.Parent.ActionStats.[edge.ActionIndex]
            stats.PendingVisits <- max 0 (stats.PendingVisits + delta)

    let validValues playerCount (values: float[][]) expectedCount =
        values.Length = expectedCount
        && values
           |> Array.forall (fun vector ->
               let total = Array.sum vector
               vector.Length = playerCount
               && (vector |> Array.forall (fun value -> Double.IsFinite(value) && value >= 0.))
               && Double.IsFinite(total)
               && total > 0.)

    let normalizeValues (values: float[]) =
        let total = Array.sum values
        values |> Array.map (fun value -> value / total)

    let applyPending (pending: PendingEvaluation) (values: float[][]) =
        reservePath pending.Path -1
        let normalized = values |> Array.map normalizeValues
        for i in 0 .. pending.States.Length - 1 do
            backPropagate [ pending.States.[i] ] normalized.[i]
        let outcome =
            Array.append
                (Array.map2 (fun weight value -> weight, value) pending.Weights normalized)
                pending.ExactOutcomes
            |> weightedOutcome
        backPropagatePath pending.Path outcome
        propagateTerminals ((pending.States |> Array.toList) @ pending.Path.States)
        leafStats.applied <- leafStats.applied + 1

    let fallbackPending (pending: PendingEvaluation) =
        reservePath pending.Path -1
        let values = pending.States |> Array.map (simulate maxRolloutDepth)
        for i in 0 .. pending.States.Length - 1 do
            backPropagate [ pending.States.[i] ] values.[i]
        let outcome =
            Array.append
                (Array.map2 (fun weight value -> weight, value) pending.Weights values)
                pending.ExactOutcomes
            |> weightedOutcome
        backPropagatePath pending.Path outcome
        leafStats.fallback <- leafStats.fallback + 1

    let applyPendingPrior (pending: PendingPriorEvaluation) =
        reservePath pending.Path -1
        let values = pending.States |> Array.map (fun state -> state.ValueEstimates.Value)
        for i in 0 .. pending.States.Length - 1 do
            backPropagate [ pending.States.[i] ] values.[i]
        Array.append
            (Array.map2 (fun weight value -> weight, value) pending.Weights values)
            pending.ExactOutcomes
        |> weightedOutcome
        |> backPropagatePath pending.Path
        propagateTerminals ((pending.States |> Array.toList) @ pending.Path.States)

    let fallbackPendingPrior (pending: PendingPriorEvaluation) =
        reservePath pending.Path -1
        let values = pending.States |> Array.map (simulate maxRolloutDepth)
        for i in 0 .. pending.States.Length - 1 do
            backPropagate [ pending.States.[i] ] values.[i]
        Array.append
            (Array.map2 (fun weight value -> weight, value) pending.Weights values)
            pending.ExactOutcomes
        |> weightedOutcome
        |> backPropagatePath pending.Path

    let collectLeaves allowFallback =
        match leafEvaluator with
        | Some evaluator when pendingEvaluations.Count > 0 ->
            let knownIds = HashSet<int64>(pendingEvaluations.Keys) :> IReadOnlySet<int64>
            for response in evaluator.Collect(knownIds) do
                match pendingEvaluations.TryGetValue(response.RequestId) with
                | true, pending ->
                    pendingEvaluations.Remove(response.RequestId) |> ignore
                    leafStats.batches <- leafStats.batches + 1
                    leafStats.latencyMs <- leafStats.latencyMs + response.LatencyMs
                    if validValues pending.States.[0].State.NumberOfPlayers response.Values pending.States.Length then
                        applyPending pending response.Values
                    else
                        leafStats.invalid <- leafStats.invalid + 1
                        if allowFallback then fallbackPending pending
                        else
                            reservePath pending.Path -1
                            leafStats.cancelled <- leafStats.cancelled + 1
                | _ -> leafStats.orphan <- leafStats.orphan + 1

            let timedOut =
                pendingEvaluations
                |> Seq.filter (fun pair -> timer.ElapsedMilliseconds - pair.Value.SubmittedAtMs >= int64 leafEvaluationTimeoutMs)
                |> Seq.map (fun pair -> pair.Key)
                |> Seq.toArray
            for requestId in timedOut do
                let pending = pendingEvaluations.[requestId]
                pendingEvaluations.Remove(requestId) |> ignore
                evaluator.Cancel(HashSet<int64>([ requestId ]) :> IReadOnlySet<int64>)
                leafStats.timeouts <- leafStats.timeouts + 1
                if allowFallback then fallbackPending pending
                else
                    reservePath pending.Path -1
                    leafStats.cancelled <- leafStats.cancelled + 1
        | _ -> ()

    let enqueueLeaf (path: SelectionPath) (states: MCTSState[]) (weights: int[]) exactOutcomes depth =
        match leafEvaluator with
        | Some evaluator ->
            let requestId = Interlocked.Increment(&nextLeafRequestId)
            reservePath path 1
            if evaluator.Enqueue(requestId, states |> Array.map (fun state -> state.State), depth) then
                pendingEvaluations.[requestId] <-
                    { Path = path
                      States = states
                      Weights = weights
                      ExactOutcomes = exactOutcomes
                      SubmittedAtMs = timer.ElapsedMilliseconds }
                leafStats.submitted <- leafStats.submitted + 1
                leafStats.states <- leafStats.states + states.Length
                true
            else
                reservePath path -1
                false
        | None -> false

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
        let mutable requested = false
        match priorClient, nodeRegistry, layoutRegistry with
        | Some client, Some nodeReg, Some layoutReg ->
            if not (Array.isEmpty node.Actions) && node.Priors.IsNone && not (nodeReg.ContainsKey(node.NodeId)) then
                if not (client.ShouldRequestPrior(node.State)) then
                    priorStats.priorNodesSkipped <- priorStats.priorNodesSkipped + 1
                else
                    let (actionStates, layout, outcomeWeights) = collectActionStates node
                    if actionStates.Length > 0 then
                        nodeReg.[node.NodeId] <- node
                        layoutReg.[node.NodeId] <- (layout, outcomeWeights)
                        let inferenceCount = client.RequestPrior(node.NodeId, node.State, actionStates, int node.State.PlayerTurn + 1, depth)
                        if inferenceCount > 0 then
                            requested <- true
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
        requested

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
                            node.FlattenedPriors <- Some (Array.copy resp.Priors)
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

    let collectPendingPriors allowFallback =
        match nodeRegistry with
        | Some registry when pendingPriorEvaluations.Count > 0 ->
            let completed =
                pendingPriorEvaluations
                |> Seq.filter (fun pair ->
                    pair.Value.States
                    |> Array.forall (fun state -> not (registry.ContainsKey(state.NodeId))))
                |> Seq.map (fun pair -> pair.Key)
                |> Seq.toArray
            for requestId in completed do
                let pending = pendingPriorEvaluations.[requestId]
                pendingPriorEvaluations.Remove(requestId) |> ignore
                if pending.States |> Array.forall (fun state -> state.ValueEstimates.IsSome) then
                    applyPendingPrior pending
                elif allowFallback then
                    fallbackPendingPrior pending
                else
                    reservePath pending.Path -1
            let timedOut =
                pendingPriorEvaluations
                |> Seq.filter (fun pair ->
                    timer.ElapsedMilliseconds - pair.Value.SubmittedAtMs >= int64 leafEvaluationTimeoutMs)
                |> Seq.map (fun pair -> pair.Key)
                |> Seq.toArray
            for requestId in timedOut do
                let pending = pendingPriorEvaluations.[requestId]
                pendingPriorEvaluations.Remove(requestId) |> ignore
                if allowFallback then fallbackPendingPrior pending
                else reservePath pending.Path -1
        | _ -> ()

    let enqueuePriorEvaluation
        (path: SelectionPath)
        (states: MCTSState[])
        (weights: int[])
        (exactOutcomes: (int * float[])[])
        depth =
        reservePath path 1
        let requested =
            states
            |> Array.map (fun state ->
                state.ValueEstimates.IsSome || requestPrior state depth)
        if requested |> Array.forall id then
            let requestId = Interlocked.Increment(&nextLeafRequestId)
            pendingPriorEvaluations.[requestId] <-
                { Path = path
                  States = states
                  Weights = weights
                  ExactOutcomes = exactOutcomes
                  SubmittedAtMs = timer.ElapsedMilliseconds }
            true
        else
            reservePath path -1
            false

    // Re-submit prior requests for already-expanded nodes along the
    // selection path that are missing priors (e.g. from tree reuse where
    // the fresh registries don't know about them yet).
    let resubmitPriors (stateHistory: MCTSState list) =
        let len = List.length stateHistory
        // stateHistory is [deepest; ...; root], depths are [len-1; ...; 0].
        let mutable depth = len - 1
        for node in stateHistory do
            requestPrior node depth |> ignore
            depth <- depth - 1


    // ── Root prior wait ──────────────────────────────────────────────────
    // Proactively request the root node's prior and block briefly until
    // it arrives.  The root prior is the single most valuable piece of NN
    // guidance: it steers the very first action selection and every
    // subsequent UCB comparison.  Without this wait the search typically
    // finishes all rollouts before the HTTP round-trip completes.
    let rootPriorWaitTimeoutMs = 250L
    requestPrior root 0 |> ignore
    if priorClient.IsSome && root.Priors.IsNone then
        let waitStartTicks = timer.ElapsedTicks
        let waitStart = timer.ElapsedMilliseconds
        while root.Priors.IsNone
              && (timer.ElapsedMilliseconds - waitStart) < rootPriorWaitTimeoutMs do
            collectPriors ()
            Thread.Sleep(1)
        // One final collection attempt after the sleep loop.
        collectPriors ()
        match effectiveEvaluateUntil with
        | Some deadline ->
            effectiveEvaluateUntil <- Some (deadline + timer.ElapsedTicks - waitStartTicks)
        | None -> ()

    while root.Rollouts < maxSimulationCount
          && beforeDeadline ()
          && maxActionRollouts root < actionRolloutLimit
          && not (isResolved root) do
        let selection =
            let pendingCount = pendingEvaluations.Count + pendingPriorEvaluations.Count
            if pendingCount >= max 1 maxPendingEvaluations
               || pendingCount >= maxSimulationCount - root.Rollouts then Blocked
            else select explorationConstant root
        match selection with
        | Blocked ->
            collectPriors ()
            collectPendingPriors true
            collectLeaves true
            if pendingEvaluations.Count + pendingPriorEvaluations.Count > 0 then
                match leafEvaluator with
                | Some evaluator ->
                    evaluator.WaitForResults(min 10 leafEvaluationTimeoutMs) |> ignore
                | None -> Thread.Sleep(1)
                collectPriors ()
                collectPendingPriors true
                collectLeaves true
        | Exhausted (path, outcome) ->
            resubmitPriors path.States
            backPropagatePath path outcome
            propagateTerminals path.States
            collectPriors ()
        | Candidate (path, i) ->
            resubmitPriors path.States
            let mostRecentState = path.States.[0]
            let depth = path.States.Length

            let selectedEdge = { Parent = mostRecentState; ActionIndex = i }
            let coreAction =
                match mostRecentState.Actions.[i] with
                | Unexplored action -> action
                | _ -> failwith "Candidate action was already expanded"
            if depth >= maxTreeDepth then
                let evaluationPath = { path with Edges = selectedEdge :: path.Edges }
                match coreAction with
                | Deterministic action ->
                    let state = MCTSState(action.State())
                    if Array.isEmpty state.Actions then
                        backPropagatePath evaluationPath
                            (oneHotOutcome(state.State.PlayerTurn, state.State.NumberOfPlayers))
                    elif priorClient.IsSome then
                        if not (enqueuePriorEvaluation evaluationPath [| state |] [| 1 |] Array.empty depth) then
                            backPropagatePath evaluationPath (simulate 0 state)
                    elif not (enqueueLeaf evaluationPath [| state |] [| 1 |] Array.empty depth) then
                        backPropagatePath evaluationPath (simulate 0 state)
                | Stochastic action ->
                    let outcomes = action.Outcomes()
                    let states = outcomes |> Array.map (snd >> MCTSState)
                    let nonterminal =
                        states
                        |> Array.mapi (fun index state -> index, state)
                        |> Array.filter (fun (_, state) -> not (Array.isEmpty state.Actions))
                    let exact =
                        states
                        |> Array.mapi (fun index state -> index, state)
                        |> Array.choose (fun (index, state) ->
                            if Array.isEmpty state.Actions then
                                Some(fst outcomes.[index], oneHotOutcome(state.State.PlayerTurn, state.State.NumberOfPlayers))
                            else None)
                    if nonterminal.Length > 0 && priorClient.IsSome then
                        let nonterminalStates = nonterminal |> Array.map snd
                        if not (enqueuePriorEvaluation
                                    evaluationPath
                                    nonterminalStates
                                    (nonterminal |> Array.map (fun (index, _) -> fst outcomes.[index]))
                                    exact depth) then
                            let evaluated =
                                nonterminal
                                |> Array.map (fun (index, state) -> fst outcomes.[index], simulate 0 state)
                            Array.append evaluated exact
                            |> weightedOutcome
                            |> backPropagatePath evaluationPath
                    elif nonterminal.Length > 0
                         && enqueueLeaf evaluationPath
                              (nonterminal |> Array.map snd)
                              (nonterminal |> Array.map (fun (index, _) -> fst outcomes.[index]))
                              exact depth then ()
                    else
                        let weighted =
                            states
                            |> Array.mapi (fun index state ->
                                fst outcomes.[index],
                                if Array.isEmpty state.Actions then
                                    oneHotOutcome(state.State.PlayerTurn, state.State.NumberOfPlayers)
                                else simulate 0 state)
                            |> weightedOutcome
                        backPropagatePath evaluationPath weighted
                collectPriors ()
            else
                let expandedState = expand leafBoundary (mostRecentState, i)
                let expandedPath =
                    { States = expandedState :: path.States
                      Edges = selectedEdge :: path.Edges }

                match mostRecentState.Actions.[i] with
                | HorizonAction _ ->
                    priorStats.horizonSkips <- priorStats.horizonSkips + 1
                    let evaluationPath = { path with Edges = selectedEdge :: path.Edges }
                    if priorClient.IsSome then
                        if not (enqueuePriorEvaluation evaluationPath [| expandedState |] [| 1 |] Array.empty depth) then
                            backPropagatePath evaluationPath (simulate maxRolloutDepth expandedState)
                    elif not (enqueueLeaf evaluationPath [| expandedState |] [| 1 |] Array.empty depth) then
                        let outcome = simulate maxRolloutDepth expandedState
                        backPropagatePath expandedPath outcome
                    collectPriors ()
                | StochasticAction stochasticOutcomes ->
                    let evaluationPath =
                        { States = path.States
                          Edges = selectedEdge :: path.Edges }
                    let nonterminal =
                        stochasticOutcomes |> Array.filter (fun outcome -> not (Array.isEmpty outcome.State.Actions))
                    let exactOutcomes =
                        stochasticOutcomes
                        |> Array.choose (fun outcome ->
                            if Array.isEmpty outcome.State.Actions then
                                Some (
                                    outcome.ProbabilityWeight,
                                    oneHotOutcome (outcome.State.State.PlayerTurn, outcome.State.State.NumberOfPlayers))
                            else None)
                    if nonterminal.Length > 0 && priorClient.IsSome then
                        let outcomeStates = nonterminal |> Array.map (fun outcome -> outcome.State)
                        if not (enqueuePriorEvaluation
                                    evaluationPath outcomeStates
                                    (nonterminal |> Array.map (fun outcome -> outcome.ProbabilityWeight))
                                    exactOutcomes depth) then
                            let evaluated =
                                nonterminal
                                |> Array.map (fun outcome ->
                                    outcome.ProbabilityWeight, simulate maxRolloutDepth outcome.State)
                            Array.append evaluated exactOutcomes
                            |> weightedOutcome
                            |> backPropagatePath evaluationPath
                    elif nonterminal.Length > 0
                         && enqueueLeaf
                              evaluationPath
                              (nonterminal |> Array.map (fun outcome -> outcome.State))
                              (nonterminal |> Array.map (fun outcome -> outcome.ProbabilityWeight))
                              exactOutcomes
                              depth then
                        ()
                    else
                        let evaluateOutcome (outcomeState: MCTSState) =
                            let outcome =
                                if Array.isEmpty outcomeState.Actions then
                                    oneHotOutcome (outcomeState.State.PlayerTurn, outcomeState.State.NumberOfPlayers)
                                else
                                    simulate maxRolloutDepth outcomeState
                            backPropagate [ outcomeState ] outcome
                            outcome
                        let outcome = evaluateStochasticOutcomes evaluateOutcome stochasticOutcomes
                        backPropagatePath evaluationPath outcome
                        for stochasticOutcome in stochasticOutcomes do
                            propagateTerminals (stochasticOutcome.State :: path.States)
                    collectPriors ()
                // Check if the expanded state is a terminal game state (no actions).
                | _ when Array.isEmpty expandedState.Actions ->
                    let outcome = oneHotOutcome (expandedState.State.PlayerTurn, expandedState.State.NumberOfPlayers)
                    backPropagatePath expandedPath outcome
                    match mostRecentState.Actions.[i] with
                    | DeterministicAction _ ->
                        mostRecentState.Actions.[i] <- Terminal outcome
                    | _ -> ()
                    propagateTerminals expandedPath.States
                    collectPriors ()
                | _ ->
                    let evaluationPath = { path with Edges = selectedEdge :: path.Edges }
                    if priorClient.IsSome then
                        if not (enqueuePriorEvaluation
                                    evaluationPath [| expandedState |] [| 1 |] Array.empty depth) then
                            let outcome = simulate maxRolloutDepth expandedState
                            backPropagatePath expandedPath outcome
                            propagateTerminals expandedPath.States
                    elif enqueueLeaf evaluationPath [| expandedState |] [| 1 |] Array.empty depth then
                        ()
                    else
                        let outcome = simulate maxRolloutDepth expandedState
                        backPropagatePath expandedPath outcome
                        propagateTerminals expandedPath.States
        | StochasticCandidate (path, _) ->
            resubmitPriors path.States
            let expandedState = path.States.[0]
            if Array.isEmpty expandedState.Actions then
                let outcome = oneHotOutcome (expandedState.State.PlayerTurn, expandedState.State.NumberOfPlayers)
                backPropagatePath path outcome
                propagateTerminals path.States
            else
                let evaluationPath = { path with States = List.tail path.States }
                if priorClient.IsSome then
                    if not (enqueuePriorEvaluation
                                evaluationPath [| expandedState |] [| 1 |] Array.empty path.States.Length) then
                        let outcome = simulate maxRolloutDepth expandedState
                        backPropagatePath path outcome
                        propagateTerminals path.States
                elif not (enqueueLeaf evaluationPath [| expandedState |] [| 1 |] Array.empty path.States.Length) then
                    let outcome = simulate maxRolloutDepth expandedState
                    backPropagatePath path outcome
                    propagateTerminals path.States
            collectPriors ()
        | Horizon (path, horizonState) ->
            resubmitPriors path.States
            priorStats.horizonSkips <- priorStats.horizonSkips + 1
            if priorClient.IsSome then
                if not (enqueuePriorEvaluation path [| horizonState |] [| 1 |] Array.empty path.States.Length) then
                    let outcome = simulate maxRolloutDepth horizonState
                    backPropagatePath { path with States = horizonState :: path.States } outcome
            elif not (enqueueLeaf path [| horizonState |] [| 1 |] Array.empty path.States.Length) then
                let outcome = simulate maxRolloutDepth horizonState
                backPropagatePath { path with States = horizonState :: path.States } outcome
            collectPriors ()

        collectLeaves true
        collectPendingPriors true

        if pendingEvaluations.Count + pendingPriorEvaluations.Count >= max 1 maxPendingEvaluations then
            match leafEvaluator with
            | Some evaluator -> evaluator.WaitForResults(min 10 leafEvaluationTimeoutMs) |> ignore
            | None -> ()
            collectLeaves true
            collectPriors ()
            collectPendingPriors true

    // Final collection: apply any priors that arrived during the last
    // iterations of the search loop, before the caller flushes.
    collectPriors ()
    collectPendingPriors true

    let drainDeadline = timer.ElapsedMilliseconds + int64 (max 0 drainTimeoutMs)
    while pendingEvaluations.Count + pendingPriorEvaluations.Count > 0
          && timer.ElapsedMilliseconds < drainDeadline do
        collectPriors ()
        collectPendingPriors false
        collectLeaves false
        if pendingEvaluations.Count + pendingPriorEvaluations.Count > 0 then
            match leafEvaluator with
            | Some evaluator ->
                evaluator.WaitForResults(min 10 (int (drainDeadline - timer.ElapsedMilliseconds))) |> ignore
            | None -> Thread.Sleep(1)
    collectPriors ()
    collectPendingPriors false
    collectLeaves false
    if pendingEvaluations.Count > 0 then
        match leafEvaluator with
        | Some evaluator ->
            let ids = HashSet<int64>(pendingEvaluations.Keys)
            evaluator.Cancel(ids :> IReadOnlySet<int64>)
        | None -> ()
        for pending in pendingEvaluations.Values do
            reservePath pending.Path -1
            leafStats.cancelled <- leafStats.cancelled + 1
        pendingEvaluations.Clear()
    if pendingPriorEvaluations.Count > 0 then
        for pending in pendingPriorEvaluations.Values do
            reservePath pending.Path -1
        pendingPriorEvaluations.Clear()

    // Drop this search's pending responses from the shared client
    // mailbox. Responses for other concurrent searches are preserved.
    // The server-side queue is never cleared (it is shared too).
    match priorClient, nodeRegistry with
    | Some client, Some nodeReg ->
        let knownIds = HashSet<int64>(nodeReg.Keys) :> IReadOnlySet<int64>
        client.Flush(knownIds)
    | _ -> ()

    (extractBestPath root |> List.toArray, priorStats, leafStats)
