# Kjarni MCTS Algorithm

Kjarni implements Monte Carlo Tree Search (MCTS) for multiplayer board games with
support for stochastic (chance) actions. This document describes the algorithm,
its data model, and the design decisions that extend classical MCTS to handle
multiple players and mixed deterministic/stochastic action spaces.

## Overview

The search loop repeats five phases until a budget (time or simulation count) is
exhausted:

```
while budget remains AND NOT all root actions are Terminal:
    1. Selection              — walk the tree from root to a frontier node
    2. Expansion              — create a new child node at the frontier
    3. Simulation             — random rollout from the new node to a terminal state
    4. Backpropagation        — propagate the rollout result up to the root
    5. Terminal propagation   — check if resolved subtrees can be collapsed
```

After the loop, `extractBestPath` reads the tree greedily to return the
recommended sequence of action indices.

## Data Model

### ICoreState (game interface)

Games plug into Kjarni by implementing `ICoreState`:

```fsharp
type ICoreState =
    abstract PlayerTurn : Player          // whose turn it is
    abstract NumberOfPlayers : int
    abstract TurnNumber : int
    abstract Actions : unit -> CoreAction[]
    abstract Scores : unit -> float[]     // per-player scores for depth-limited rollouts
```

`Actions()` returns an array of `CoreAction`, which is a discriminated union:

```fsharp
type CoreAction =
    | Deterministic of IDeterministicCoreAction   // player-chosen move
    | Stochastic of IStochasticCoreAction         // chance event (dice, card draw, etc.)
```

A deterministic action exposes a single successor state. A stochastic action
exposes a weighted set of outcomes:

```fsharp
type IDeterministicCoreAction =
    abstract State : unit -> ICoreState

type IStochasticCoreAction =
    abstract Outcomes : unit -> (int * ICoreState)[]   // (weight, resulting state)
```

### MCTSState (search tree node)

Each node in the search tree wraps an `ICoreState` and tracks visit statistics:

| Field      | Type                 | Description                              |
|------------|----------------------|------------------------------------------|
| State      | ICoreState           | The underlying game state                |
| Actions    | Action[]             | Per-action status (see below)            |
| Rollouts   | int (mutable)        | Total visit count                        |
| WinCounts  | float[] (mutable)    | Per-player cumulative wins               |

`Actions` is initialized by wrapping each `CoreAction` in `Unexplored`. As the
tree grows, entries are replaced:

```fsharp
type Action =
    | Unexplored of CoreAction              // not yet expanded
    | DeterministicAction of MCTSState      // expanded deterministic child
    | StochasticAction of StochasticOutcome[]  // expanded stochastic children
    | Terminal of float[]                    // leaf with known outcome
```

A `StochasticOutcome` pairs a probability weight with its own `MCTSState`:

```fsharp
type StochasticOutcome = { ProbabilityWeight: int; State: MCTSState }
```

## Phase 1 — Selection

Selection walks from the root toward the frontier. At each node it picks the
action with the highest evaluation score (UCB1-based) and follows it.

```
select(root):
    visitedStates = [root]
    current = root
    loop:
        if current has no actions → return Exhausted
        pick action with highest actionEvaluator score
        match action:
            Unexplored       → return Candidate(visitedStates, actionIndex)
            DeterministicAction(child) → push child, continue
            StochasticAction(outcomes) → sample an outcome by weight:
                if outcome unvisited → return StochasticCandidate(...)
                else push outcome state, continue
            Terminal(outcome) → return Exhausted(visitedStates, outcome)
```

The result is one of three cases:

- **Candidate** — an unexplored action was found; expansion is needed.
- **StochasticCandidate** — an already-expanded stochastic action has an
  unvisited outcome; no expansion needed, just simulate from that outcome.
- **Exhausted** — the path reached a terminal node; backpropagate immediately.

### Action Evaluation (UCB1)

```
actionEvaluator(state, action):
    match action:
        Unexplored → 10.0   (high constant ensures exploration)
        DeterministicAction(child) →
            winRate(child, actingPlayer) + explorationRate(state, child)
        StochasticAction(outcomes) →
            if totalRollouts = 0 → 10.0
            else sampledWinRate(outcomes, actingPlayer) + explorationRate(state, totalRollouts)
        Terminal(outcome) → outcome[actingPlayer]
```

The exploration term uses the standard UCB1 formula:

```
explorationRate(C, parentVisits, childVisits) = C * sqrt(ln(parentVisits) / childVisits)
```

where `C` defaults to `sqrt(2)` and is configurable via `MCTSConfig.ExplorationConstant`.

## Phase 2 — Expansion

When selection returns a `Candidate`, the unexplored action is expanded:

- **Deterministic**: call `action.State()` to get the successor, wrap it in a
  new `MCTSState`, and replace `Unexplored` with `DeterministicAction(newNode)`.
- **Stochastic**: call `action.Outcomes()` to get all weighted outcomes, wrap
  each in a `MCTSState` and a `StochasticOutcome`, and replace `Unexplored` with
  `StochasticAction(outcomes)`. One outcome is sampled by weight and returned
  as the node to simulate from.

When selection returns a `StochasticCandidate`, the stochastic action is already
expanded — the unvisited outcome state is used directly for simulation.

## Phase 3 — Simulation (Rollout)

From the expanded node, a random playout proceeds until a terminal state is
reached or the maximum rollout depth is exceeded:

```
simulate(state, depth):
    actions = state.Actions()
    if no actions → oneHotOutcome(state.PlayerTurn)
    if depth >= maxRolloutDepth → scoreBasedOutcome(state)
    pick a random action:
        Deterministic → recurse into its state
        Stochastic    → sample an outcome by weight, recurse
```

The rollout result is a float array of length `NumberOfPlayers`.

### Terminal outcome

When the rollout reaches a state with no actions, the current player is treated
as the winner and a one-hot outcome vector is returned (e.g. `[0, 1, 0, 0]` for
Player2 in a 4-player game).

### Depth-limited outcome

When `maxRolloutDepth` is reached, `Scores()` is called on the current state.
The player(s) with the highest score share a win equally. If all scores are zero
or negative, a draw (all zeros) is returned.

## Phase 4 — Backpropagation

The outcome vector is added to every node along the selection path (the
`visitedStates` list plus the expanded/simulated node):

```
backPropagate(visitedStates, outcome):
    for state in visitedStates:
        state.Rollouts += 1
        for each player j:
            state.WinCounts[j] += outcome[j]
```

This is a simple additive update — every node on the path gets the full outcome.

## Phase 5 — Terminal Propagation

Terminal propagation replaces expanded actions with `Terminal` when the subtree
below them is fully resolved — the outcome is known with certainty and no
further simulation is needed. This redirects the search budget to unresolved
parts of the tree and is especially valuable in endgame positions where many
branches lead to forced outcomes.

After backpropagation, the selection path is checked bottom-up (deepest visited
node first, working toward the root). An expanded action (`DeterministicAction`
or `StochasticAction`) is replaced by `Terminal(outcome)` when either of the
following conditions holds for the **resulting state**:

### Condition 1 — Guaranteed win

The resulting state has at least one action that is `Terminal` with a 100% win
probability for the **active player of the resulting state**. The active player
can choose a move that guarantees a win, so the outcome is known.

The outcome stored in the new `Terminal` is the best possible outcome for the
active player — the `Terminal` action that gives them the highest win value.

```
Parent (Player 1's turn)
  └─ action[2] = DeterministicAction(child)
       child (Player 2's turn)
         ├─ action[0] = Terminal([0.0, 1.0, 0.0])   ← P2 wins 100%
         ├─ action[1] = DeterministicAction(...)
         └─ action[2] = Terminal([1.0, 0.0, 0.0])   ← P1 wins 100%

→ Condition 1 met: action[0] gives P2 (active player) a 100% win
→ Parent's action[2] becomes Terminal([0.0, 1.0, 0.0])
  (best outcome for P2, the active player of child)
```

### Condition 2 — Fully resolved subtree

**All** actions of the resulting state are `Terminal`. Since every possible move
leads to a known outcome, the active player will choose the action that gives
them the highest win value.

```
Parent (Player 1's turn)
  └─ action[0] = DeterministicAction(child)
       child (Player 2's turn)
         ├─ action[0] = Terminal([0.5, 0.3, 0.2])
         └─ action[1] = Terminal([0.1, 0.6, 0.3])

→ Condition 2 met: all actions are Terminal
→ P2 picks action[1] (0.6 > 0.3, best for P2)
→ Parent's action[0] becomes Terminal([0.1, 0.6, 0.3])
```

### Terminal game states (no actions)

When expansion creates a child that has no actions (an empty `Actions` array),
the parent's action pointing to it is immediately replaced with
`Terminal(oneHotOutcome(child.PlayerTurn))`.

### Stochastic actions

A `StochasticAction` becomes `Terminal` under the same two conditions, but
applied across all its outcomes. Since the game (not the player) chooses the
stochastic outcome:

- **Condition 1**: Only met if **all** outcome states satisfy condition 1 or 2
  for the **same** player winning with 100%.

- **Condition 2**: All outcome states have all their actions resolved as
  `Terminal`. The stored outcome is the **weighted average** of the best outcomes
  across the stochastic branches.

```
Parent
  └─ action[1] = StochasticAction(outcomes)
       outcome[0] (weight=1, P1's turn)
         ├─ Terminal([1.0, 0.0])
         └─ Terminal([0.0, 1.0])
       outcome[1] (weight=2, P1's turn)
         └─ Terminal([1.0, 0.0])

→ All outcome states are fully resolved (condition 2)
→ outcome[0]: P1 picks [1.0, 0.0] (best for P1)
→ outcome[1]: P1 picks [1.0, 0.0] (best for P1)
→ Weighted average: (1*[1.0, 0.0] + 2*[1.0, 0.0]) / 3 = [1.0, 0.0]
→ Parent's action[1] becomes Terminal([1.0, 0.0])
```

Propagation stops as soon as a state does not satisfy either condition, since
its parent cannot become Terminal if the state itself has unresolved actions.

### Early search termination

When all root actions become `Terminal`, the search stops early — no further
simulation can change the outcome. The game itself continues to play out (to
produce training labels, etc.), but no additional MCTS searches are required for
the remaining moves.

`LogInfo` includes a `reachedTerminal` field that is `true` when the search
terminated early because the root was fully resolved.

## Best Path Extraction

After the search loop, `extractBestPath` greedily follows the highest-valued
action at each node using pure exploitation (no exploration term):

```
extractionEvaluator(player, action):
    Terminal(outcome)           → outcome[player]
    DeterministicAction(child)  → winRate(child, player)
    StochasticAction(outcomes)  → sampledWinRate(outcomes, player)
    Unexplored                  → 0.0
```

Extraction continues through deterministic actions and stops when it encounters
a stochastic action, a terminal, or a node with no actions. The result is a list
of action indices representing the recommended move sequence.

## Multiplayer Handling

Standard MCTS literature focuses on two-player zero-sum games. Kjarni
generalizes to N players with the following design:

### Per-player win tracking

Every `MCTSState` maintains a `WinCounts` array of length `NumberOfPlayers`.
Backpropagation adds the full outcome vector to every node on the path,
regardless of whose turn it is at that node. This means each node accumulates
independent win statistics for all players.

### Perspective-relative evaluation

During selection, `actionEvaluator` reads `winRate` from the perspective of the
**acting player** at the current node (`state.State.PlayerTurn`). A node where
it is Player 1's turn evaluates children by Player 1's win rate; a child where
it is Player 2's turn evaluates its children by Player 2's win rate.

This is the key mechanism: each player maximizes their own win rate during
selection, which naturally produces adversarial play without any negation or
min/max alternation. It also generalizes cleanly beyond two players — in a
3-player game, Player 1 simply maximizes Player 1's wins, while Player 2 and
Player 3 each do the same for themselves.

### Outcome vectors

Rollout results are not scalar (+1/-1) but float arrays. A terminal state
produces a one-hot vector for the winning player. A depth-limited rollout
produces a fractional vector based on scores, which allows ties and partial
credit.

### Extraction

Best-path extraction also uses the acting player's perspective at each node,
ensuring the recommended path accounts for opponent responses.

## Stochastic Actions

Many board games include chance events — dice rolls, card draws, tile reveals.
Kjarni models these as first-class stochastic actions alongside deterministic
player moves.

### Game-side contract

A game state can return a mix of `Deterministic` and `Stochastic` actions from
`Actions()`. This is flexible: a single state might offer several player choices
and one chance event (e.g. "roll the dice" alongside "play a development card").

A stochastic action defines its outcome space via `Outcomes()`, which returns an
array of `(weight, ICoreState)` pairs. Weights are integers representing
relative probability (e.g. `[(1, stateA); (5, stateB)]` means stateB is 5x more
likely). The engine normalizes these to probabilities internally.

### Tree structure

When a stochastic action is expanded, the tree creates a `StochasticAction` node
containing an array of `StochasticOutcome` values — one `MCTSState` per possible
outcome, each tagged with its probability weight. This is conceptually similar
to a "chance node" in expectimax trees.

```
        [Root]
       /      \
  det[0]    stoch[1]
    |        /     \
  child   out[0]  out[1]     ← each outcome is its own MCTSState
            |       |
           ...     ...
```

### Selection through stochastic nodes

When selection reaches an expanded `StochasticAction`, it does **not** evaluate
outcomes by UCB1. Instead, it **samples** an outcome according to the
probability weights using `rollStochasticAction`. This reflects the fact that the
game — not the player — chooses the outcome.

If the sampled outcome has zero rollouts, selection returns a
`StochasticCandidate` so the search loop can simulate from that outcome without
further expansion. If the outcome has been visited before, selection recurses
into it normally.

### Evaluation of stochastic actions

When `actionEvaluator` compares a `StochasticAction` against other actions at
the same node, it computes a **weighted win rate** across all visited outcomes:

```
sampledWinRate(outcomes, player):
    visited = outcomes where rollouts > 0
    sum(weight * winRate(outcome, player) for outcome in visited) / sum(weights of visited)
```

The exploration bonus uses the total rollout count across all outcomes as the
"child visit count" in the UCB1 formula. If no outcome has been visited yet, the
action scores 10.0 (same as unexplored).

### Rollout through stochastic actions

During random rollouts, stochastic actions are handled by sampling an outcome
by weight and continuing the playout from the sampled state. This ensures
rollout results naturally reflect the probability distribution.

### Best-path extraction

`extractBestPath` stops when it encounters a stochastic action. Since the game
controls the outcome, recommending a specific branch would be meaningless. The
extracted path therefore represents the sequence of *player decisions* up to the
next chance event.

## Configuration

`MCTSConfig` controls the search:

| Field                | Type       | Default        | Description                                        |
|----------------------|------------|----------------|----------------------------------------------------|
| SearchTime           | searchTime | Unlimited      | Wall-clock budget (Minutes, Seconds, MilliSeconds)  |
| MaxSimulations       | int        | Int32.MaxValue | Maximum number of rollouts                         |
| MaxRolloutDepth      | int        | 500            | Maximum depth per rollout before score fallback     |
| ExplorationConstant  | float      | sqrt(2)        | UCB1 exploration weight (C)                        |
| ActionRolloutLimit   | int        | Int32.MaxValue | Stop when any single action reaches this many rollouts |

The search stops when any budget is exhausted, or when all root actions have
been resolved to `Terminal` via terminal propagation.

## Logging

`LogInfo` tracks per-search statistics:

| Field            | Type     | Description                                                    |
|------------------|----------|----------------------------------------------------------------|
| simulations      | int      | Number of iterations performed                                 |
| elapsedTime      | TimeSpan | Wall-clock time spent in the search                            |
| reachedTerminal  | bool     | `true` if all root actions became `Terminal` during the search |
