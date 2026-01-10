module Kjarni.Test.HashMapTest

open NUnit.Framework
open Kjarni
open Kjarni.AI.MinimaxTypes
open Kjarni.AITypes
open KjarniTest.TestTypes

open FsUnit

[<TestFixture>]
type hashMapTest() =

    let tree =
        node_builder(p1, 0, 0, 1234)
            .addChildren(
                [ node_builder (p2, 1, 0, 4321, [ node_builder (p1, 2, 100, 1111) ])
                  node_builder (p2, 1, 0, 4321, [ node_builder (p1, 2, 100, 1111) ])
                  node_builder (p2, 1, 0, 4321, [ node_builder (p1, 2, 100, 1111) ]) ]
            )
            .build ()

    [<Test>]
    member _.SearchWithHashTableLookup() =
        let sut =
            NegamaxAI(evaluator, Turn(4, Unlimited))

        let path = (sut :> IGameAI).DetermineAction tree

        path |> should equal [| 0; 0 |]

        sut.LatestLogInfo.successfulHashMapLookups
        |> should equal 2
