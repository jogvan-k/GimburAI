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

        let (result, priorStats) =
            search (
                root,
                config.MaxSimulations,
                timer,
                Utility.toStopwatchTics config.SearchTime,
                config.MaxRolloutDepth,
                config.ExplorationConstant,
                config.ActionRolloutLimit,
                config.PriorClient,
                config.ExpansionGuard,
                config.MaxPriorDepth
            )

        // Flush the prior queue after search completes
        match config.PriorClient with
        | Some client when not (isNull (box client)) -> client.Flush()
        | _ -> ()

        let mutable logInfo = LogInfo()
        logInfo.simulations <- root.Rollouts
        logInfo.elapsedTime <- timer.Elapsed
        logInfo.reachedTerminal <- isResolved root
        logInfo.priorStatesRequested <- priorStats.priorStatesRequested
        logInfo.priorNodesApplied <- priorStats.priorNodesApplied
        logInfo.priorsApplied <- priorStats.priorActionsApplied
        logInfo.priorStatesEvaluated <- priorStats.priorActionsEvaluated
        logInfo.priorStatesPerDepth <- priorStats.priorStatesPerDepth
        logInfo.horizonSkips <- priorStats.horizonSkips
        logInfo.priorsSkipped <- priorStats.priorsSkipped
        logInfo.stateNotFound <- priorStats.stateNotFound

        _logInfos <- logInfo :: _logInfos

        let mutable simulationResult = SimulationResult()
        simulationResult.Rollouts <- root.Rollouts
        simulationResult

    member _.LatestLogInfo() = List.head _logInfos
