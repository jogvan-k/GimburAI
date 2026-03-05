module Kjarni.MCTS.AI

open System.Diagnostics
open Kjarni
open Kjarni.Algorithms
open Kjarni.MCTS.Algorithm
open Kjarni.MCTS.Types

type MonteCarloTreeSearch(config: MCTSConfig) =
    let mutable _logInfos: LogInfo list = List.empty

    member _.RunSimulation (root : MCTSState) =
        let timer = Stopwatch.StartNew()

        let result =
            search (
                root,
                config.MaxSimulations,
                timer,
                Utility.toStopwatchTics config.SearchTime,
                config.MaxRolloutDepth,
                config.ExplorationConstant,
                config.ActionRolloutLimit
            )

        let mutable logInfo = LogInfo()
        logInfo.simulations <- root.Rollouts
        logInfo.elapsedTime <- timer.Elapsed

        _logInfos <- logInfo :: _logInfos

        let mutable simulationResult = SimulationResult()
        simulationResult.Rollouts <- root.Rollouts
        simulationResult

    member _.LatestLogInfo() = List.head _logInfos
