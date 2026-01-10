module Kjarni.AI.PvsAI

open Kjarni
open Kjarni.AI.AIBase
open Kjarni.AI.MinimaxTypes
open Kjarni.Algorithms.PVS

type PvsAI
    (
        evaluator: IEvaluator,
        depth,
        ?searchConfig0: SearchConfiguration,
        ?loggingConfiguration0: LoggingConfiguration
    ) =
    inherit BaseAI(evaluator,
                   depth,
                   defaultArg searchConfig0 SearchConfiguration.NoRestrictions,
                   defaultArg loggingConfiguration0 LoggingConfiguration.LogAll)

    override this.AICall d s acc pv = pvs d s acc pv
