# Kjarni MCTS Algorithm

Kjarni implements Monte Carlo Tree Search (MCTS) for multiplayer board games with
support for stochastic (chance) actions. This document describes the algorithm,
its data model, and the design decisions that extend classical MCTS to handle
multiple players and mixed deterministic/stochastic action spaces.

## Overview

The search loop repeats seven phases until a budget (time or simulation count) is
exhausted:

```
while budget remains AND NOT all root actions are Terminal:
    1. Selection              — walk the tree from root to a frontier node
    2. Expansion              — create a new child node at the frontier
    3. Prior request          — enqueue async NN evaluation for the expanded node
    4. Simulation             — random rollout from the new node to a terminal state
    5. Backpropagation        — propagate the rollout result up to the root
    6. Terminal propagation   — check if resolved subtrees can be collapsed
    7. Prior collection       — apply any completed NN priors to tree nodes
```

Steps 2 and 3 are skipped when selection returns `Exhausted` (terminal path) or
`Horizon` (expansion guard boundary — see
[Expansion Guard](#expansion-guard-horizon)).


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
    | HorizonAction of MCTSState            // expansion-blocked rollout leaf
    | Terminal of float[]                    // leaf with known outcome
```

A `HorizonAction` wraps an `MCTSState` that accumulates rollout statistics but
is never expanded further. It is created when the expansion guard blocks an
`Unexplored` action (see [Expansion Guard](#expansion-guard-horizon)).

A `StochasticOutcome` pairs a probability weight with its own `MCTSState`:

```fsharp
type StochasticOutcome = { ProbabilityWeight: int; State: MCTSState }
```

## Phase 1 — Selection

Selection walks from the root toward the frontier. At each node it picks the
action with the highest evaluation score (PUCT-based) and follows it.

```
select(root, expansionGuard):
    visitedStates = [root]
    current = root
    loop:
        if current has no actions → return Exhausted
        pick action with highest actionEvaluator score
        match action:
            Unexplored →
                if expansionGuard(current.State, action) →
                    replace action with HorizonAction(new MCTSState)
                    return Horizon(visitedStates, horizonState)
                else → return Candidate(visitedStates, actionIndex)
            DeterministicAction(child) → push child, continue
            StochasticAction(outcomes) → sample an outcome by weight:
                if outcome unvisited → return StochasticCandidate(...)
                else push outcome state, continue
            HorizonAction(child) → return Horizon(visitedStates, child)
            Terminal(outcome) → return Exhausted(visitedStates, outcome)
```

The result is one of four cases:

- **Candidate** — an unexplored action was found; expansion is needed.
- **StochasticCandidate** — an already-expanded stochastic action has an
  unvisited outcome; no expansion needed, just simulate from that outcome.
- **Horizon** — selection reached a `HorizonAction` node (an expansion-blocked
  boundary). No expansion occurs; a full rollout is performed from the horizon
  node's state and backpropagated. See
  [Expansion Guard](#expansion-guard-horizon).
- **Exhausted** — the path reached a terminal node; backpropagate immediately.


### Action Evaluation (PUCT)

```
actionEvaluator(state, action, index):
    P = state.Priors[index] if priors exist, else 1 / len(actions)
    match action:
        Unexplored →
            C_puct * P * sqrt(N_parent) / 1
        DeterministicAction(child) | HorizonAction(child) →
            winRate(child, actingPlayer) + C_puct * P * sqrt(N_parent) / (1 + child.Rollouts)
        StochasticAction(outcomes) →
            if totalRollouts = 0 →
                C_puct * P * sqrt(N_parent) / 1
            else
                sampledWinRate(outcomes, actingPlayer)
                + C_puct * P * sqrt(N_parent) / (1 + totalRollouts)
        Terminal(outcome) → outcome[actingPlayer]
```

`HorizonAction` is evaluated identically to `DeterministicAction` — the wrapped
`MCTSState` accumulates rollout statistics that drive the win rate and
exploration balance.

This is a variant of the PUCT (Predictor + UCB applied to Trees) formula. When
no neural network priors are available, `P` defaults to a uniform distribution
(`1 / number_of_actions`), which produces equivalent behavior to standard UCB1
exploration. When NN priors are applied to a node, `P` is overwritten with the
NN-informed policy, biasing selection toward actions the network considers
promising.

The exploration constant `C_puct` is configured via
`MCTSConfig.ExplorationConstant` (default: `sqrt(2)`).


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

When selection returns a `Horizon`, the `Unexplored` action has already been
replaced with a `HorizonAction` during selection. The `HorizonAction` wraps an
`MCTSState` that tracks statistics but is never expanded further. No additional
expansion step is needed. See [Expansion Guard](#expansion-guard-horizon) for
details.


## Phase 3 — Prior Request

After expansion, if a prior client is configured, the search loop enqueues an
asynchronous prior request for the newly expanded node. The request contains the
serialized result state of every action from the expanded node — the NN evaluates
each resulting position and returns one **prior score** per action. Depending on
how the model was trained, those scores have one of two meanings:

- **Win-rate prior** (legacy): each score is the predicted win probability of
  the corresponding result state for the acting player. Scores are independent
  per action and are not constrained to any particular sum.
- **Policy prior** (placement model only, going forward): the scores form an
  approximate policy distribution over the legal actions of the parent node and
  approximately sum to 1.

In both cases the prior client treats the returned values as raw scores and
normalises them before they are stored on the parent node (see
[Converting Prior Scores to Priors](#converting-prior-scores-to-priors)). The
call is non-blocking; the search continues immediately with simulation. See
[Asynchronous Neural Network Priors](#asynchronous-neural-network-priors) for
the full request lifecycle.

## Phase 4 — Simulation (Rollout)

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

## Phase 5 — Backpropagation

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

## Phase 6 — Terminal Propagation

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

## Phase 7 — Prior Collection

After terminal propagation, if a prior client is configured, the search loop
calls `collectPriors()` to drain any completed NN inference responses from the
mailbox. For each response the per-action scores are normalised into a prior
policy (sum-to-1 over the parent node's legal actions) and stored on the
corresponding `MCTSState` node. This step is target-agnostic: it produces a
valid policy whether the model was trained to emit win rates or policy logits.
This is non-blocking — if no responses are ready, the loop continues
immediately.

Prior collection runs after terminal propagation so that priors are not applied
to nodes that have just been collapsed into `Terminal` actions. See
[Asynchronous Neural Network Priors](#asynchronous-neural-network-priors) for
details.

## Best Path Extraction

After the search loop, `extractBestPath` greedily follows the highest-valued
action at each node using pure exploitation (no exploration term):

```
extractionEvaluator(player, action):
    Terminal(outcome)           → outcome[player]
    DeterministicAction(child)  → winRate(child, player)
    HorizonAction(child)       → winRate(child, player)
    StochasticAction(outcomes)  → sampledWinRate(outcomes, player)
    Unexplored                  → 0.0
```

Extraction continues through deterministic actions and stops when it encounters
a stochastic action, a horizon action, a terminal, or a node with no actions.
The result is a list of action indices representing the recommended move
sequence.

When the best action at the root is a `HorizonAction`, the game simulation
should stop — the search has reached the expansion boundary and continuing play
past it is not meaningful for label generation. The caller (e.g. `Gimbur.Cli`)
detects this by inspecting `mctsRoot.Actions[bestPath.Head]` after extraction.

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
"child visit count" in the PUCT formula. If no outcome has been visited yet, the
action is treated as unexplored.

### Rollout through stochastic actions

During random rollouts, stochastic actions are handled by sampling an outcome
by weight and continuing the playout from the sampled state. This ensures
rollout results naturally reflect the probability distribution.

### Best-path extraction

`extractBestPath` stops when it encounters a stochastic action. Since the game
controls the outcome, recommending a specific branch would be meaningless. The
extracted path therefore represents the sequence of *player decisions* up to the
next chance event.

## Asynchronous Neural Network Priors

The search uses neural network priors to bias action selection toward promising
moves. The NN evaluates the resulting state of each action to produce a win
probability, which is then converted into a prior policy over the parent node's
actions. Because NN inference is expensive relative to a single MCTS iteration,
prior requests are issued asynchronously — the search loop does not block while
waiting for results.

### Components

Three components cooperate:

1. **MCTS search loop** (Kjarni, F#) — fires prior requests on node expansion,
   collects responses after terminal propagation.
2. **Prior client** (Gimbur, C#) — manages HTTP communication with the inference
   server and a local response mailbox.
3. **Inference server** (gimbur-nn, Python) — receives prior requests, queues
   them by priority, runs batched GPU inference, and returns results.

```
MCTS Search Loop                   Prior Client                    Inference Server
────────────────                   ────────────                    ────────────────
expand(node)                              │                              │
  ├─ create child MCTSState(s)            │                              │
  ├─ requestPrior(                        │                              │
  │    parentNode,                        │                              │
  │    actionStates[], depth) ───────────►├─ POST /prior-enqueue ───────►├─ enqueue by depth
  │   (non-blocking, fire & forget)       │   { states[], depth }        │
  │                                       │                              ├─ dequeue batch
  ├─ simulate(child)                      │                              │   (lowest depth first)
  ├─ backPropagate(path, outcome)         │                              ├─ GPU inference
  ├─ propagateTerminals(path)             │                              │
  │                                       │                              │
  ├─ collectPriors() ◄────────────────────┤◄─────── response ────────────┤
  │   check mailbox for responses         │   { id, win_probs[] }        │
  │   if found: normalize win_probs       │                              │
  │   into prior policy, apply to node    │                              │
  │                                       │                              │
  └─ continue loop                        │                              │
                                          │                              │
search ends ──────────────────────────────┼─ POST /prior-flush ─────────►├─ clear queue
```

### Prior Request Lifecycle

#### 1. Request on expansion

When `expand` creates a new child `MCTSState`, the search loop calls
`requestPrior(parentNode, actionStates, depth)` where `depth` is the number of
edges from the root to the newly expanded node. This call is **non-blocking** —
it enqueues the request and returns immediately. The MCTS iteration continues
with rollout and backpropagation as normal.

The request contains a list of serialized game states — one per action from the
expanded node. Each state represents the result of applying that action. The
inference server does not know how to apply game actions; Kjarni is responsible
for materializing the resulting states and sending them.

For **deterministic actions**, the resulting state is `action.State()`. For
**stochastic actions**, each possible outcome is a separate state. A node with
3 deterministic actions and 1 stochastic action with 4 outcomes sends 7 states
total.

The request payload contains:
- A client-side ID to correlate the response back to the parent `MCTSState`.
- The list of serialized action-result states (compact strings, same format as
  training data).
- The acting player (1-based, for server-side player rotation).
- The priority (depth from root; lower = more important).

#### 2. Priority queuing on the server

The inference server maintains a **min-heap priority queue** ordered by depth.
When the server is ready to run inference, it dequeues a batch of the
highest-priority (lowest-depth) requests and processes them together in a single
GPU forward pass.

Positions near the root are visited many more times during the search than deep
positions. A prior applied to a depth-1 node influences every subsequent
selection pass; a prior applied to a depth-20 node affects very few future
iterations. Processing shallow positions first maximizes the impact of each
inference batch.

The queue has a bounded capacity. When the queue is full, new requests with
higher depth (lower priority) than the worst queued item are **dropped
silently** — the MCTS continues without priors for those nodes, falling back to
rollout-derived statistics. Requests with lower depth (higher priority) than the
worst queued item evict the lowest-priority entry.

#### 3. Response collection after terminal propagation

After each backpropagation and terminal propagation step, the search loop calls
`collectPriors()` which checks a **lock-free mailbox** (concurrent queue) for
completed prior responses. For each response:

1. Look up the corresponding parent `MCTSState` by its ID.
2. Convert the per-action win probabilities into a normalized prior policy.
3. Store the prior on the parent node.

If no responses are available, `collectPriors` returns immediately with no
blocking. Multiple responses may arrive between iterations; all are consumed in
a single pass.

#### 4. Flush on search completion

When the MCTS search finishes (budget exhausted or root resolved), the caller
invokes `flush()` on the server. This clears the priority queue and drops all
pending requests. Any in-flight responses that arrive after the flush are
discarded by the client (the node IDs no longer exist in the new tree).

This is necessary because the tree is advanced to a new root after each game
move. Pending requests reference nodes in the old tree that are no longer
relevant.

### Converting Prior Scores to Priors

The NN returns one **score** per action-result state. The semantics of the
score depend on which target the model was trained for (see
[Prior Targets](#prior-targets) below), but Kjarni's conversion path is the
same in both cases.

For **stochastic actions** with multiple outcomes, the action's score is the
probability-weighted average across its outcomes:

```
score(stochastic_action) = sum(weight_k * score(outcome_k)) / sum(weight_k)
```

This collapses multiple outcome evaluations into a single value per action.

The resulting per-action scores are then **normalised** to a probability
distribution over the parent node's legal actions:

```
P(action_i) = max(score(action_i), 0) / sum(max(score(action_j), 0) for all j)
```

If every score is non-positive (or the sum is otherwise zero), the prior falls
back to the uniform distribution `1 / number_of_actions`. The resulting prior
policy is stored on the parent `MCTSState` node and used by `actionEvaluator`
(see [Action Evaluation (PUCT)](#action-evaluation-puct)).

#### Prior Targets

The score returned by the inference server depends on how the model was
trained. The MCTS engine and prior client are agnostic to which target was
used — the per-action normalisation above is what guarantees a valid policy.

| Target              | Per-action score meaning                                                | Rough sum of raw scores | Used by         |
|---------------------|-------------------------------------------------------------------------|-------------------------|-----------------|
| `winrate` (legacy)  | Predicted win probability of the action-result state for the acting player. Each score is computed independently per action. | Unconstrained (typically not 1) | State model, legacy placement model |
| `policy`            | Approximate posterior probability that the action is the best move at the parent state — the model's policy head output, softmaxed across the parent's legal actions. | ~1 by construction      | New placement model |

Why both are supported:

- The **`winrate` target** is what the existing simulation export already
  contains as `winRate` per action. Models trained on this target produce
  normalised priors that are nearly uniform when sibling win rates are similar
  (e.g. `0.48, 0.49, 0.50, 0.51` → `~0.245, ~0.245, ~0.250, ~0.255`). The
  resulting policy provides only weak guidance to PUCT.
- The **`policy` target** is trained on MCTS visit-count distributions
  (`rollouts[i] / sum(rollouts)`) — the search's own preference. This produces
  a much more peaked distribution (e.g. `0.01, 0.04, 0.80, 0.15`) and therefore
  much stronger PUCT guidance. The placement model is the first to adopt this
  target; the state model continues to use `winrate` for now.

Both targets share the same wire format (a list of per-action floats); the
inference server tags each response with the target it was trained for, but the
normalisation step above is what makes the engine indifferent.

### Prior Data Format

#### Request (MCTS → server)

```json
{
    "requests": [
        {
            "id": "node-0x1a2b3c",
            "states": [
                "<compact_state_for_action_0>",
                "<compact_state_for_action_1>",
                "<compact_state_for_action_2>"
            ],
            "player": 2,
            "priority": 3
        }
    ]
}
```

- `id` — opaque string the server echoes back to correlate responses.
- `states` — serialized game states, one per action result. For deterministic
  actions this is the single successor state; for stochastic actions this is
  one state per outcome. The order matches the parent node's `Actions()` array,
  with stochastic actions expanded inline (outcomes in weight order).
- `player` — the acting player (1-based), for server-side player rotation.
- `priority` — depth from root (0 = root's children). Lower = serve first.

#### Response (server → MCTS)

```json
{
    "responses": [
        {
            "id": "node-0x1a2b3c",
            "win_probabilities": [0.62, 0.45, 0.38],
            "target": "winrate"
        }
    ]
}
```

- `id` — matches the request ID.
- `win_probabilities` — per-state prior score, in the same order as the
  request's `states` array. The field name is preserved for backward
  compatibility, but the values may be either win probabilities (`target =
  "winrate"`) or policy probabilities (`target = "policy"`). Kjarni maps these
  back to actions, computes weighted averages for stochastic actions, and
  normalises to produce the prior policy (see
  [Converting Prior Scores to Priors](#converting-prior-scores-to-priors)).
- `target` — optional, identifies how the model was trained. Used for logging
  and audits. The MCTS engine itself does not branch on this field.

### Server Endpoints

#### `POST /prior-enqueue`

Accepts a batch of prior requests. Each request is inserted into the priority
queue. Returns immediately with 202 Accepted (results are delivered
asynchronously via `/prior-collect`).

#### `POST /prior-collect`

Returns all completed inference results since the last call. The response body
contains an array of prior responses. If no results are ready, returns an empty
array immediately (non-blocking).

#### `POST /prior-flush`

Clears the priority queue and discards any pending results. Called when the game
advances to a new position and the old tree is abandoned.

### MCTSState Changes

`MCTSState` has an optional prior field:

```fsharp
type MCTSState(state: ICoreState) =
    // ... existing fields ...
    let mutable _priors: float[] option = None
    member _.Priors
        with get () = _priors
        and set value = _priors <- value
```

When `Priors` is `None`, `actionEvaluator` uses the uniform default
(`1 / number_of_actions`). When `Priors` is `Some(p)`, it uses `p[i]` as
`P(action_i)` in the PUCT formula.

### Graceful Degradation

The system degrades gracefully when the inference server is slow or unavailable:

- **Server unreachable**: Prior requests fail silently. MCTS runs with uniform
  priors and rollout-based evaluation — identical to search without an NN.
- **Server slow**: Priors arrive late (after many rollouts have already visited
  the node). The NN policy still improves selection for remaining iterations,
  but the benefit diminishes as the node accumulates its own statistics.
- **Degenerate scores**: If every per-action score is non-positive (or sums to
  zero after the stochastic-action collapse), the prior falls back to the
  uniform distribution rather than producing NaN or a one-hot artefact.
- **Queue full**: Low-priority (deep) requests are dropped. Shallow nodes that
  matter most still receive priors.
- **Stale responses**: After a flush, any late-arriving responses from the
  previous search are discarded (the node IDs no longer exist in the new tree).


## Expansion Guard (Horizon)

The expansion guard is an optional predicate (`MCTSConfig.ExpansionGuard`) that
prevents the MCTS tree from growing past a configurable boundary. When the
guard blocks an `Unexplored` action, the action is replaced with a
`HorizonAction` — a new `MCTSState` that accumulates rollout statistics but is
never expanded further. This focuses the search budget on the region of the
game tree where high-quality labels are needed.

The motivating use case is Settlers of Catan placement: expanding into the main
game (past the first `RollDiceAction`) is counterproductive because dice
variance dominates and dilutes placement-phase labels. The guard stops tree
growth at that boundary while rollouts still pass through the main game to
estimate placement quality.

### Guard predicate

```fsharp
ExpansionGuard: (ICoreState -> CoreAction -> bool) option
```

When `None` (the default), expansion is unrestricted. When `Some(guard)`, the
predicate is called during selection on each `Unexplored` action. Arguments are
the parent node’s `ICoreState` and the unexplored `CoreAction`. Returning
`true` blocks expansion.

### HorizonAction

On the first visit to a guarded action, selection materializes the successor
state (`action.State()` for deterministic, or the canonical successor for
stochastic), wraps it in a new `MCTSState`, and replaces the `Unexplored` entry
with `HorizonAction(horizonState)`. On subsequent visits the `HorizonAction` is
already present. In both cases selection returns `Horizon(visitedStates,
horizonState)`.

The `MCTSState` inside a `HorizonAction` is never recursed into by selection —
it exists solely to accumulate `Rollouts` and `WinCounts` from rollouts.
Because `HorizonAction` is not `Terminal`, the parent node can never become
fully resolved, so the search will not terminate early due to horizon nodes.

### Gimbur integration

In `Gimbur.Cli`, the guard targets the placement/main-game boundary:

```csharp
ExpansionGuard = (state, action) =>
    action is Stochastic sa && sa is RollDiceAction
```

## Configuration

`MCTSConfig` controls the search:

| Field                | Type                                   | Default        | Description                                        |
|----------------------|----------------------------------------|----------------|----------------------------------------------------|
| SearchTime           | searchTime                             | Unlimited      | Wall-clock budget (Minutes, Seconds, MilliSeconds)  |
| MaxSimulations       | int                                    | Int32.MaxValue | Maximum number of rollouts                         |
| MaxRolloutDepth      | int                                    | 500            | Maximum depth per rollout before score fallback     |
| ExplorationConstant  | float                                  | sqrt(2)        | UCB1 exploration weight (C)                        |
| ActionRolloutLimit   | int                                    | Int32.MaxValue | Stop when any single action reaches this many rollouts |
| PriorClient          | IPriorClient option                    | None           | Async prior client for NN-guided search (see below) |
| ExpansionGuard       | (ICoreState -> CoreAction -> bool) opt | None           | Predicate that blocks expansion of specific actions ([details](#expansion-guard-horizon)) |

The search stops when any budget is exhausted, or when all root actions have
been resolved to `Terminal` via terminal propagation.

When `PriorClient` is `None`, the search uses uniform priors and rollout-based
evaluation (the current default behavior). When set, the search fires prior
requests on node expansion and collects responses after terminal propagation.

When `ExpansionGuard` is `None`, all unexplored actions are expanded normally.
When set, the predicate is checked during selection before expanding any
`Unexplored` action. See [Expansion Guard](#expansion-guard-horizon).

## Logging

`LogInfo` tracks per-search statistics:

| Field               | Type     | Description                                                    |
|---------------------|----------|----------------------------------------------------------------|
| simulations         | int      | Number of iterations performed                                 |
| elapsedTime         | TimeSpan | Wall-clock time spent in the search                            |
| reachedTerminal     | bool     | `true` if all root actions became `Terminal` during the search |
| priorsRequested     | int      | Number of prior requests enqueued (one per expanded node)      |
| priorsApplied       | int      | Number of prior responses received and applied to nodes        |
| priorStatesEvaluated| int      | Total action-result states sent for NN evaluation              |
| horizonSkips        | int      | Number of selection passes that returned `Horizon`             |
