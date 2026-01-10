module Kjarni.Algorithms.RandomMove

open System
open Kjarni

let randomMoveAI (rng: Random) (s: ICoreState) =
    let actions = s.Actions()
    rng.Next() % actions.Length
