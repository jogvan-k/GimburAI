namespace Kjarni.Test

open Kjarni
open Kjarni.AI.MinimaxTypes
open Kjarni.Algorithms.AISupportTypes
open Kjarni.Algorithms.Accumulator
open Kjarni.Algorithms.Negamax

open NUnit.Framework
open KjarniTest.TestTypes

open FsUnit

[<TestFixture>]
type NegamaxTest() =

    let basicTree =
        node_builder (
            p1,
            0,
            0,
            0,
            [ node_builder(p2, 1, -10, 1)
                .addChild (
                    node_builder(p1, 2, 20, 2)
                        .addChild (
                            node_builder(p2, 3, -40, 3)
                                .addChildren [ node_builder (p1, 4, 70, 4); node_builder (p1, 4, 80, 5) ]
                        )
                )
              node_builder(p2, 1, -20, 6)
                  .addChild (
                      node_builder(p1, 2, 30, 7)
                          .addChild (node_builder(p2, 3, -50, 8).addChild (node_builder (p1, 4, 75, 9)))
                  ) ]
        )

    let twoDepthsPerTurnTree =
        node_builder(
            p1,
            0,
            0,
            0,
            [ node_builder (
                p1,
                0,
                20,
                1,
                [ node_builder (p2, 1, 20, 2, [ node_builder(p2, 1, 0, 3).addChild (node_builder (p1, 2, -10, 4)) ])
                  node_builder(p2, 1, 25, 5)
                      .addChild (
                          node_builder(p2, 1, 20, 6)
                              .addChild (
                                  node_builder(p1, 2, -20, 7)
                                      .addChild (node_builder(p1, 2, 0, 8).addChild (node_builder (p2, 3, 30, 9)))
                              )
                      ) ]
              )
              node_builder (p1, 0, 10, 10) ]
        )
            .build ()

    [<Test>]
    member _.NoDepth_BasicTree() =
        let d = Plies 0

        let result =
            negamax
                d
                (basicTree.build ())
                (accumulator (evaluatorFunc, Unlimited, LoggingConfiguration.NoLogging))
                []

        fst result |> should equal 0
        snd result |> should be Empty

    [<TestCase(1, -10, [| 0 |])>]
    [<TestCase(2, 30, [| 1; 0 |])>]
    [<TestCase(3, -40, [| 0; 0; 0 |])>]
    [<TestCase(4, 75, [| 1; 0; 0; 0 |])>]
    member _.VariousDepth_BasicTree (depth: int) (expectedValue: int) (expectedPath: int []) =
        let d = Plies depth

        let result =
            negamax
                d
                (basicTree.build ())
                (accumulator (evaluatorFunc, Unlimited, LoggingConfiguration.NoLogging))
                []

        fst result |> should equal expectedValue

        snd result
        |> List.toArray
        |> should equal expectedPath

    [<TestCase(1, -20, [| 1 |])>]
    [<TestCase(2, 20, [| 0; 0 |])>]
    [<TestCase(3, -50, [| 1; 0; 0 |])>]
    [<TestCase(4, 75, [| 1; 0; 0; 0 |])>]
    member _.VariousDepth_InvertedBasicTree (depth: int) (expectedValue: int) (expectedPath: int []) =
        let invertedTree = invertTree basicTree
        let d = Plies depth

        let result =
            negamax
                d
                (invertedTree.build ())
                (accumulator (evaluatorFunc, Unlimited, LoggingConfiguration.NoLogging))
                []

        fst result |> should equal -expectedValue

        snd result
        |> List.toArray
        |> should equal expectedPath

    [<TestCase(1, 25, [| 0; 1 |])>]
    [<TestCase(2, 10, [| 1 |])>]
    [<TestCase(3, 30, [| 0; 1; 0; 0; 0; 0 |])>]
    member _.TurnDepthSearch (untilTurn: int) (expectedValue: int) (expectedPath: int []) =
        let d = Turns(untilTurn, 0)

        let result =
            negamax
                d
                twoDepthsPerTurnTree
                (accumulator (evaluatorFunc, Unlimited, LoggingConfiguration.NoLogging))
                []

        fst result |> should equal expectedValue

        snd result
        |> List.toArray
        |> should equal expectedPath
