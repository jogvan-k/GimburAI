module Kjarni.Test.Algorithms.PrincipalVariationSearchTest

open System

open NUnit.Framework

open Kjarni
open Kjarni.AI.MinimaxTypes
open Kjarni.Algorithms.Accumulator
open Kjarni.Algorithms.PVS
open Kjarni.Algorithms.TranspositionTable
open Kjarni.Algorithms.AISupportTypes
open KjarniTest.TestTypes

open FsUnit

[<TestFixture>]
type PrincipalVariationSearchTest() =
    let mutable hashIndex = 0

    let evalFun counter =
        let d, h = counter

        let playerTurn =
            if d % 2 = 0 then
                Player.Player1
            else
                Player.Player2

        hashIndex <- hashIndex + 1
        playerTurn, d, h, hashIndex

    let state = (complexTree evalFun 4 2).build ()
    let bestPath = [ 1; 0; 1; 0 ]

    [<Test>]
    member _.PrincipalPathIsBestPath() =
        let mutable nodesEvaluated = 0

        let evalFuncWithEvalCount =
            fun s ->
                nodesEvaluated <- nodesEvaluated + 1
                evaluatorFunc s

        let tTable =
            transpositionTable LoggingConfiguration.LogAll

        let acc =
            accumulator (evalFuncWithEvalCount, Unlimited, LoggingConfiguration.LogAll)

        let result =
            recPVS -Int32.MaxValue Int32.MaxValue 1 (Plies 10) state acc tTable bestPath

        fst result |> should equal 10
        snd result |> should equal bestPath
        nodesEvaluated |> should equal 7


    [<Test>]
    member _.PrincipalPathIsNotBestPath() =
        let mutable nodesEvaluated = 0

        let evalFuncWithEvalCount =
            fun s ->
                nodesEvaluated <- nodesEvaluated + 1
                evaluatorFunc s

        let tTable =
            transpositionTable LoggingConfiguration.LogAll

        let pv = [ 0; 0; 1; 0 ]

        let acc =
            accumulator (evalFuncWithEvalCount, Unlimited, LoggingConfiguration.LogAll)

        let result =
            recPVS -Int32.MaxValue Int32.MaxValue 1 (Plies 10) state acc tTable pv

        fst result |> should equal 10
        snd result |> should equal bestPath
        nodesEvaluated |> should equal 14
