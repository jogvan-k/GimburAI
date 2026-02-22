module Kjarni.MCTS.AI

open System
open System.Diagnostics
open Kjarni
open Kjarni.Algorithms
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.Types

[<Flags>]
type configuration =
    | None = 0x0
    | TranspositionTable = 0x1
    | AsyncExecution = 0x2
    | All = 0x3

let tTable (config: configuration) =
    if config.HasFlag configuration.TranspositionTable then
        Some(TranspositionTable())
    else
        None

type MonteCarloTreeSearch(st: searchTime, maxSimulationCount: int, config: configuration, maxRolloutDepth: int) =
    let mutable _logInfos: LogInfo list = List.empty

    let extractWinChance (s: State) =
        let aiPlayer = s.state.PlayerTurn

        if s.leaves |> Array.isEmpty then
            s.winRate
        else
            s.leaves
            |> Array.map (fun i -> extractionEvaluator (aiPlayer, i))
            |> Array.max

    new(st: searchTime, maxSimulationCount: int, config: configuration) =
        MonteCarloTreeSearch(st, maxSimulationCount, config, defaultMaxRolloutDepth)

    interface IGameAI with
        /// <summary></summary>
        /// <param name="state"></param>
        /// <returns></returns>
        member _.DetermineAction state =
            let timer = Stopwatch.StartNew()
            let root = State state
            let tTable = tTable config

            let result =
                if config.HasFlag configuration.AsyncExecution then
                    parallelSearch (root, maxSimulationCount, tTable, Utility.toMilliseconds st, maxRolloutDepth)
                else
                    search (root, maxSimulationCount, timer, tTable, Utility.toStopwatchTics st, maxRolloutDepth)

            let mutable logInfo = LogInfo()
            logInfo.simulations <- root.visitCount
            logInfo.elapsedTime <- timer.Elapsed
            logInfo.estimatedAiWinChance <- extractWinChance root
            logInfo.winCounts <- root.winCounts

            match tTable with
            | Some t ->
                logInfo.successfulTranspositionTableLookup <- t.SuccessfulLookups
                logInfo.transpositionTableSize <- t.Count
            | None -> ()

            _logInfos <- logInfo :: _logInfos
            result

    member _.LatestLogInfo() = List.head _logInfos
