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

        let (result, priorStats, leafStats) =
            search (
                root,
                config.MaxSimulations,
                timer,
                Utility.toStopwatchTics config.SearchTime,
                config.MaxRolloutDepth,
                config.ExplorationConstant,
                config.ActionRolloutLimit,
                config.PriorClient,
                config.LeafEvaluator,
                config.LeafBoundary,
                config.MaxPriorDepth,
                config.MaxPendingEvaluations,
                config.LeafEvaluationTimeoutMs,
                config.DrainTimeoutMs
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
        logInfo.leafEvaluationsSubmitted <- leafStats.submitted
        logInfo.leafEvaluationsApplied <- leafStats.applied
        logInfo.leafEvaluationTimeouts <- leafStats.timeouts
        logInfo.leafEvaluationsInvalid <- leafStats.invalid
        logInfo.leafEvaluationsCancelled <- leafStats.cancelled
        logInfo.leafEvaluationFallbacks <- leafStats.fallback
        logInfo.leafEvaluationOrphans <- leafStats.orphan
        logInfo.leafEvaluationBatches <- leafStats.batches
        logInfo.leafEvaluationStates <- leafStats.states
        logInfo.leafEvaluationLatencyMs <- leafStats.latencyMs

        _logInfos <- logInfo :: _logInfos

        let mutable simulationResult = SimulationResult()
        simulationResult.Rollouts <- root.Rollouts
        simulationResult

    member _.LatestLogInfo() = List.head _logInfos
