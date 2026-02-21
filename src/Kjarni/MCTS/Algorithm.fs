module Kjarni.MCTS.Algorithm

open System

open System.Diagnostics
open Kjarni
open Kjarni.MCTS.Types

let isUnexplored l =
    match l with
    | Unexplored _ -> true
    | _ -> false

let isLeaf l =
    match l with
    | Leaf _ -> true
    | _ -> false

let explorationConstant = sqrt 2.
let maxTrackedPlayers = int Player.Player4

let emptyOutcome() = Array.zeroCreate<float> (maxTrackedPlayers + 1)

let oneHotOutcome (winner: Player) =
    let outcome = emptyOutcome ()
    let i = int winner
    if i > 0 && i <= maxTrackedPlayers then
        outcome.[i] <- 1.
    outcome

let addScaledOutcome (target: float array) (source: float array) (scale: float) =
    for i in 1 .. maxTrackedPlayers do
        let v = if i < source.Length then source.[i] else 0.
        target.[i] <- target.[i] + v * scale

let explorationRate (stateVisitCount: int, actionVisitCount: int) =
    explorationConstant
    * sqrt (
        log (float stateVisitCount)
        / float actionVisitCount
    )

let leafEvaluator (state: State, l: Leaf) =
    let actingPlayer = state.playerTurn

    match l with
    | Terminal win -> if actingPlayer = win then 100. else 0.
    | Unexplored _ -> 10.
    | Leaf a ->
        a.winRate
        + explorationRate (state.visitCount, a.visitCount)


let rec recSelection (s: State, actionHistory: Action list, leafEvaluator) =
    if Array.isEmpty s.leaves then
        Exhausted(actionHistory, s.playerTurn)
    else
        match s.leaves
              |> Array.indexed
              |> Array.maxBy (fun i -> leafEvaluator (s, snd i)) with
        | _, Terminal win -> Exhausted(actionHistory, win)
        | i, Unexplored _ -> Candidate(actionHistory, i)
        | _, Leaf ls -> recSelection (ls.state, ls :: actionHistory, leafEvaluator)

let selection (s: State, leafEvaluator) = recSelection (s, [], leafEvaluator)

let expandUnexplored (parent: State, i, state: State) =
    let leaf =
        if Array.isEmpty state.leaves then
            Terminal state.playerTurn
        else
            let a = Action(parent.playerTurn, state)
            Leaf a

    parent.leaves.[i] <- leaf
    leaf


let expansion (s: State, i, tTable: TranspositionTable Option) =
    match s.leaves.[i] with
    | Unexplored a ->
        let nextState = a.DoCoreAction()

        match tTable with
        | Some t ->
            let stateHash = nextState.GetHashCode()

            match t.Lookup stateHash with
            | Some r -> expandUnexplored (s, i, r)
            | None ->
                let ex = State nextState
                t.Add(stateHash, ex)
                expandUnexplored (s, i, ex)
        | None -> expandUnexplored (s, i, State nextState)

    | _ -> raise (Exception "Target leaf is already expanded")

let rec simulationDistribution (state: ICoreState) =
    let actions = state.Actions()
    if Array.isEmpty actions then
        oneHotOutcome state.PlayerTurn
    else
        let nextMove = actions |> Seq.sort |> Seq.head
        match nextMove with
        | :? IStochasticCoreAction as stochastic ->
            let outcomes = stochastic.Outcomes()
            if Array.isEmpty outcomes then
                oneHotOutcome state.PlayerTurn
            else
                let result = emptyOutcome ()
                for nextState, probability in outcomes do
                    if probability > 0. then
                        addScaledOutcome result (simulationDistribution nextState) probability
                result
        | _ -> simulationDistribution (nextMove.DoCoreAction())

let simulation (s: State) = simulationDistribution s.state

let registerResult (s: State) (outcome: float array) = s.registerOutcome outcome

let backPropagating (root: State) (a: Action list) (outcome: float array) =
    for a1 in a do
        a1.incrementVisitCount ()
        registerResult a1.state outcome

    registerResult root outcome

let extractionEvaluator (p: Player, l: Leaf) =
    match l with
    | Terminal win -> if p = win then 1. else 0.
    | Leaf a -> a.winRate
    | Unexplored _ -> 0.

let extractBestPath (s: State) =
    let mutable path = List.empty
    let mutable currentState = ref s
    let mutable endReached = false

    while not endReached do
        if Array.isEmpty s.leaves then
            endReached <- true
        else
            let bestAction =
                currentState.Value.leaves
                |> Array.indexed
                |> Array.maxBy (fun l -> extractionEvaluator (currentState.Value.state.PlayerTurn, snd l))

            path <- fst bestAction :: path

            match snd bestAction with
            | Leaf action -> currentState <- ref action.state
            | _ -> endReached <- true

    path |> List.rev

let search (root: State, maxSimulationCount, timer: Stopwatch, tTable, evaluateUntil: Int64 option) =
    while root.visitCount < maxSimulationCount
          && (not evaluateUntil.IsSome
              || timer.ElapsedTicks < evaluateUntil.Value) do
        match selection (root, leafEvaluator) with
        | Exhausted (actionHistory, win) -> backPropagating root actionHistory (oneHotOutcome win)
        | Candidate (actionHistory, a) ->
            let s =
                if List.isEmpty actionHistory then
                    root
                else
                    actionHistory.[0].state

            match expansion (s, a, tTable) with
            | Leaf a ->
                let outcome = simulation a.state
                backPropagating root (a :: actionHistory) outcome
            | Terminal win -> backPropagating root actionHistory (oneHotOutcome win)
            | _ -> raise (Exception "Expanded to unexpected leaf type")

    extractBestPath root |> List.toArray

let parallelSearch (root: State, maxSimulationCount, tTable, evaluateUntil: int) =
    let expression =
        async {
            let leaf, ah =
                lock
                    root
                    (fun () ->
                        match selection (root, leafEvaluator) with
                        | Exhausted a -> Terminal(snd a), fst a
                        | Candidate (ah, i) ->
                            let s =
                                if List.isEmpty ah then
                                    root
                                else
                                    ah.[0].state

                            expansion (s, i, tTable), ah)

            let win, actionHistory =
                match leaf with
                | Leaf a -> simulation a.state, a :: ah
                | Terminal win -> oneHotOutcome win, ah
                | Unexplored _ -> failwith "Not Implemented"

            lock root (fun () -> backPropagating root actionHistory win)
        }

    try
        let tasks =
            Async.Parallel [ for _ in 1 .. maxSimulationCount -> expression ]

        Async.RunSynchronously(tasks, evaluateUntil)
        |> ignore
    with
    | :? TimeoutException -> ()

    extractBestPath root |> List.toArray
