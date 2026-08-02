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
                config.LeafBoundary,
                config.MaxPriorDepth
            )

        // search() handles per-search Flush internally so that it can
        // hand the prior client the local set of node IDs it owned.

        let mutable logInfo = LogInfo()
        logInfo.simulations <- root.Rollouts
        logInfo.elapsedTime <- timer.Elapsed
        logInfo.reachedTerminal <- isResolved root
        logInfo.priorNodesRequested <- priorStats.priorNodesRequested
        logInfo.priorActionsRequested <- priorStats.priorActionsRequested
        logInfo.priorInferencesRequested <- priorStats.priorInferencesRequested
        logInfo.priorNodesApplied <- priorStats.priorNodesApplied
        logInfo.priorActionsApplied <- priorStats.priorActionsApplied
        logInfo.priorActionsPerDepth <- priorStats.priorActionsPerDepth
        logInfo.priorInferencesPerDepth <- priorStats.priorInferencesPerDepth
        logInfo.priorNodesSkipped <- priorStats.priorNodesSkipped
        logInfo.priorResponsesOrphaned <- priorStats.priorResponsesOrphaned
        logInfo.horizonSkips <- priorStats.horizonSkips

        _logInfos <- logInfo :: _logInfos

        let mutable simulationResult = SimulationResult()
        simulationResult.Rollouts <- root.Rollouts
        simulationResult

    member _.LatestLogInfo() = List.head _logInfos
