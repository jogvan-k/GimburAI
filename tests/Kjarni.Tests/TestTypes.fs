module KjarniTest.TestTypes

open System
open Kjarni

let p1 = Player.Player1
let p2 = Player.Player2

type node(playerTurn, turnNumber, value, hash, parent: ICoreState option) =
    let mutable _children = list.Empty
    new(playerTurn, turnNumber, value, hash) = node (playerTurn, turnNumber, value, hash, None)

    member _.parent = parent
    member _.playerTurn = playerTurn
    member _.turnNumber = turnNumber
    member _.value = value

    member _.children
        with get () = _children
        and set value = _children <- value

    override _.GetHashCode() = hash
    override _.Equals other = hash = other.GetHashCode()

    interface ICoreState with
        member _.PlayerTurn = playerTurn
        member _.TurnNumber = turnNumber

        member this.Actions() =
            Array.map (fun n -> action (this, n) :> ICoreAction) (Array.ofList _children)

        member _.Scores() =
            let scores = Array.zeroCreate<float> 5
            let i = int playerTurn
            if i > 0 && i < 5 then
                scores.[i] <- float value
            scores

and action(origin, node) =
    interface ICoreAction with
        member _.Origin = origin :> ICoreState
        member _.DoCoreAction() = node :> ICoreState

    interface IComparable with
        member _.CompareTo _ = 0

    override _.Equals other = node.Equals other
    override _.GetHashCode() = node.GetHashCode()

type node_builder(playerTurn, turnNumber, value, hash, children: node_builder list) =
    new(playerTurn, turnNumber, value, hash, child: node_builder) = node_builder (playerTurn, turnNumber, value, hash, List.singleton child)
    new(playerTurn, turnNumber, value, hash) = node_builder (playerTurn, turnNumber, value, hash, list.Empty)
    member _.children = children
    member _.playerTurn = playerTurn
    member _.turnNumber = turnNumber
    member _.value = value
    member _.hash = hash

    member _.addChild(c: node_builder) =
        node_builder (playerTurn, turnNumber, value, hash, List.append children [ c ])

    member _.addChildren(c: node_builder list) =
        node_builder (playerTurn, turnNumber, value, hash, List.append children c)

    member this.build ?parent =
        let node =
            node (this.playerTurn, this.turnNumber, this.value, this.hash, parent)

        node.children <-
            children
            |> List.map (fun (c: node_builder) -> c.build node)

        node



type evaluator() =
    interface IEvaluator with
        member _.Evaluate s = (s :?> node).value

let evaluator = evaluator ()
let evaluatorFunc (s: ICoreState) = (s :?> node).value
// n: depth number
// b: new branches after each depth
// counter: tuple of given depth * given height
let rec recComplexTree evalFun counter n b =
    let player, turnNo, value, hash = evalFun counter
    let d, h = counter

    if d = n then
        node_builder (player, turnNo, value, hash)
    else
        let nodes =
            [ 0 .. 1 .. b - 1 ]
            |> List.map (fun i -> recComplexTree evalFun (d + 1, b * h + i) n b)

        node_builder (player, turnNo, value, hash, nodes)

let complexTree (evalFun: int * int -> Player * int * int * int) (n: int) (b: int) =
    let counter = 0, 0
    recComplexTree evalFun counter n b

let rec invertTree (t: node_builder) =
    let otherPlayer = if t.playerTurn = p1 then p2 else p1

    if t.children.Length = 0 then
        node_builder (otherPlayer, t.turnNumber, t.value, t.GetHashCode())
    else
        let nodes = t.children |> List.map invertTree
        node_builder (otherPlayer, t.turnNumber, t.value, t.GetHashCode(), nodes)
