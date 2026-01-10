namespace Kjarni.AITypes

open Kjarni
open Kjarni.AI.AIBase
open Kjarni.AI.MinimaxTypes
open Kjarni.Algorithms.Negamax

type NegamaxAI
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

    override _.AICall d s acc pv = negamax d s acc pv
