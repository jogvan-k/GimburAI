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
`Horizon` (result-state boundary; see [Leaf Boundary](#leaf-boundary-horizon)).


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

| Field       | Type                 | Description                              |
|-------------|----------------------|------------------------------------------|
| State       | ICoreState           | The underlying game state                |
| Actions     | Action[]             | Per-action tree status (see below)       |
| ActionStats | ActionStats[]        | Per-action edge visits and value sums    |
| Rollouts    | int (mutable)        | Total node visit count                   |
| WinCounts   | float[] (mutable)    | Per-player cumulative node values        |

`Actions` is initialized by wrapping each `CoreAction` in `Unexplored`. As the
tree grows, entries are replaced:

```fsharp
type Action =
    | Unexplored of CoreAction              // not yet expanded
    | DeterministicAction of MCTSState      // expanded deterministic child
    | StochasticAction of StochasticOutcome[]  // expanded stochastic children
    | HorizonAction of MCTSState            // result-state boundary rollout leaf
    | Terminal of float[]                    // leaf with known outcome
```

A `HorizonAction` wraps an `MCTSState` that accumulates rollout statistics but
is never expanded further. It is created when a deterministic action's exact
result state matches the configured leaf boundary.

A `StochasticOutcome` pairs a probability weight with its own `MCTSState`:

```fsharp
type StochasticOutcome = { ProbabilityWeight: int; State: MCTSState }
```

Each action index also has mutable edge statistics independent of its child
nodes: `CompletedVisits`, `PendingVisits`, and per-player `ValueSums`. This
separation is required because one stochastic action visit can evaluate or
traverse several outcome nodes. `PendingVisits` is currently zero in synchronous
rollout search, but is included in PUCT so asynchronous evaluation can reserve
an edge without changing the selection formula later.

## Phase 1 — Selection

Selection walks from the root toward the frontier. At each node it picks the
action with the highest evaluation score (PUCT-based) and follows it.

```
select(root):
    path = { States = [root]; Edges = [] }
    current = root
    loop:
        if current has no actions → return Exhausted
        pick action with highest actionEvaluator score
        match action:
            Unexplored → return Candidate(path, actionIndex)
            DeterministicAction(child) → push child and selected edge, continue
            StochasticAction(outcomes) → sample an outcome by weight:
                if outcome unvisited → return StochasticCandidate(...)
                else push outcome state, continue
            HorizonAction(child) → return Horizon(visitedStates, child)
            Terminal(outcome) → push selected edge, return Exhausted(path, outcome)
```

The result is one of four cases:

- **Candidate** — an unexplored action was found; expansion is needed.
- **StochasticCandidate** — an already-expanded stochastic action has an
  unvisited outcome; no expansion needed, just simulate from that outcome.
- **Horizon** — selection reached a `HorizonAction` node (a result-state
  boundary). No further expansion occurs; a full rollout is performed from the
  boundary state and backpropagated. See [Leaf Boundary](#leaf-boundary-horizon).
- **Exhausted** — the path reached a terminal node; backpropagate immediately.


### Action Evaluation (PUCT)

```
actionEvaluator(state, action, index):
    P = state.Priors[index] if priors exist, else 1 / len(actions)
    edge = state.ActionStats[index]
    match action:
        Unexplored | DeterministicAction | HorizonAction | StochasticAction →
            edge.ValueSums[actingPlayer] / edge.CompletedVisits (or 0 if unvisited)
            + C_puct * P * sqrt(N_parent)
              / (1 + edge.CompletedVisits + edge.PendingVisits)
        Terminal(outcome) → outcome[actingPlayer]
```

All non-terminal action kinds use their own edge Q and visit count. Child node
rollouts do not define the parent action's statistics.

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

- **Deterministic**: call `action.State()` to get the successor and wrap it in a
  new `MCTSState`. If `LeafBoundary` matches that successor, replace
  `Unexplored` with `HorizonAction(newNode)` and return the boundary for rollout;
  otherwise use `DeterministicAction(newNode)`.
- **Stochastic**: call `action.Outcomes()` to get all weighted outcomes, wrap
  each in a `MCTSState` and a `StochasticOutcome`, and replace `Unexplored` with
  `StochasticAction(outcomes)`. On this first expansion, run one existing random
  rollout from every outcome and probability-weight the per-player results.
  Commit that expectation as exactly one visit to the stochastic action edge
  and its ancestor nodes. Each outcome node receives its own rollout result.

When selection returns a `StochasticCandidate`, the stochastic action is already
expanded — the unvisited outcome state is used directly for simulation.

When deterministic expansion reaches the boundary, the exact result node is
stored as a `HorizonAction`, simulated, and backpropagated. Later selections of
that action return `Horizon` without recursing into it.


## Phase 3 — Prior Request

After expansion, if a prior client is configured, the search loop enqueues an
asynchronous prior request for the newly expanded node. Full-state value models
evaluate serialized action-result states. The placement client instead sends one
placement state to the state-only `placement_state_v3` model and receives a dense
full-vocabulary policy plus an optional value estimate.

- **Win-rate prior** (legacy): each score is the predicted win probability of
  the corresponding result state for the acting player. Scores are independent
  per action and are not constrained to any particular sum.
- **Placement policy prior**: the server softmaxes the model's raw dense policy
  logits. C# applies authoritative legality masking and normalization.

The prior client normalises legal scores before storing them on the parent node (see
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

The outcome vector is added to every node and selected action edge along the
selection path:

```
backPropagate(path, outcome):
    for state in path.States:
        state.Rollouts += 1
        for each player j:
            state.WinCounts[j] += outcome[j]
    for edge in path.Edges:
        edge.CompletedVisits += 1
        for each player j:
            edge.ValueSums[j] += outcome[j]
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
    any other action            → edge.ValueSums[player] / edge.CompletedVisits
                                  (or 0 if unvisited)
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

`actionEvaluator` uses the stochastic action edge's accumulated Q and visit
count, exactly like a deterministic edge. Initial expansion computes one exact
expectation over one rollout from every outcome. Later visits sample one outcome
by weight and backpropagate that sample once; they do not multiply it by the
outcome probability again. This avoids both counting all initial outcomes as
separate action visits and double probability weighting sampled revisits.

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
moves. Full-state models evaluate action-result states. Placement models evaluate
one parent placement state and return a dense canonical policy. Because NN inference is expensive relative to a single MCTS iteration,
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
  │    parentNode, state, depth) ────────►├─ POST /placement/prior-enqueue ► enqueue by depth
  │   (non-blocking, fire & forget)       │   { id, state, priority }    │
  │                                       │                              ├─ dequeue batch
  ├─ simulate(child)                      │                              │   (lowest depth first)
  ├─ backPropagate(path, outcome)         │                              ├─ GPU inference
  ├─ propagateTerminals(path)             │                              │
  │                                       │                              │
  ├─ collectPriors() ◄────────────────────┤◄─────── response ────────────┤
  │   check mailbox for responses         │   { id, priors[],            │
  │   if found: mask and normalize        │     value_estimate }         │
  │   legal priors, apply to node         │                              │
  │                                       │                              │
  └─ continue loop                        │                              │
                                          │                              │
search ends ──────────────────────────────┼─ discard local pending IDs   │
```

### Prior Request Lifecycle

#### 1. Request on expansion

When `expand` creates a new child `MCTSState`, the search loop calls
`requestPrior(parentNode, actionStates, depth)` where `depth` is the number of
edges from the root to the newly expanded node. This call is **non-blocking** —
it enqueues the request and returns immediately. The MCTS iteration continues
with rollout and backpropagation as normal.

For state-model priors, the request contains serialized result states. For
placement priors, it contains only the parent placement state. The placement
action vocabulary is output indexing, not request input.

For **deterministic actions**, the resulting state is `action.State()`. For
**stochastic actions**, each possible outcome is a separate state. A node with
3 deterministic actions and 1 stochastic action with 4 outcomes sends 7 states
total.

The queued placement request contains:
- A client-side ID to correlate the response back to the parent `MCTSState`.
- One compact placement state.
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
2. For placement, mask the dense policy to C#-legal composites and normalize it.
3. Store the prior on the parent node.

If no responses are available, `collectPriors` returns immediately with no
blocking. Multiple responses may arrive between iterations; all are consumed in
a single pass.

#### 4. Cleanup on search completion

When the MCTS search finishes, the client drops pending IDs and mailbox responses
for that search. It does not clear the shared server queue, which may also contain
requests from concurrent searches. Late responses become harmless orphans.

This is necessary because the tree is advanced to a new root after each game
move. Pending requests reference nodes in the old tree that are no longer
relevant.

### Converting Prior Scores to Priors

For state priors, the NN returns one score per action-result state. For placement,
the server returns a dense policy of width 60, 82, or 144. C# is authoritative
for legality: it rejects invalid dense vectors, masks illegal composites, and
normalizes the remaining scores.

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

#### Placement Composite Priors

At a settlement node, the settlement prior is the sum of dense probabilities for
all C#-legal roads paired with that settlement. At the subsequent road node, the
road prior is conditional within that settlement: each legal composite probability
is divided by the settlement marginal. Thus the MCTS two-step decision preserves
the model's composite settlement-road distribution.

### Prior Data Format

#### Request (MCTS → server)

```json
{
    "requests": [
        {
            "id": "node-0x1a2b3c",
            "state": "<compact_placement_state>",
            "priority": 3
        }
    ]
}
```

- `id` — opaque string the server echoes back to correlate responses.
- `state` — the serialized placement parent state.
- `priority` — depth from root (0 = root's children). Lower = serve first.

#### Response (server → MCTS)

```json
{
    "responses": [
        {
            "id": "node-0x1a2b3c",
            "priors": [0.01, 0.0, 0.03, 0.02],
            "value_estimate": 0.62
        }
    ]
}
```

- `id` — matches the request ID.
- `priors` — dense full-vocabulary placement policy probabilities. C# masks and
  normalizes them over currently legal composites. The abbreviated example has
  four entries; actual widths are 60, 82, or 144.
- `player_win_probabilities` — normalized per-player values from the value head.

### Server Endpoints

#### `POST /placement/prior-enqueue`

Accepts a batch of prior requests. Each request is inserted into the priority
queue. Returns immediately with 202 Accepted (results are delivered
asynchronously via `/placement/prior-collect`).

#### `POST /placement/prior-collect`

Returns all completed inference results since the last call. The response body
contains an array of prior responses. If no results are ready, returns an empty
array immediately (non-blocking).

#### `POST /placement/prior-flush`

Administratively clears the placement priority queue and pending results. Normal
per-search cleanup stays client-side so concurrent searches are not disrupted.

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


## Leaf Boundary (Horizon)

The leaf boundary is an optional result-state predicate (`MCTSConfig.LeafBoundary`)
that prevents the MCTS tree from growing past a configurable deterministic
boundary. After a deterministic action produces its successor, a matching state
is stored as a `HorizonAction`. The exact successor accumulates rollout
statistics but is never expanded further.

The motivating use case is Settlers of Catan placement: expanding into the main
game is counterproductive because dice variance dominates and dilutes
placement-phase labels. The boundary is the exact post-final-road state, before
the first dice roll, while rollouts still pass through the main game to estimate
placement quality.

### Boundary predicate

```fsharp
LeafBoundary: (ICoreState -> bool) option
```

When `None` (the default), expansion is unrestricted. For deterministic actions,
the predicate receives the newly produced result state. Returning `true` makes
that exact state the horizon. Stochastic actions always expand all weighted
outcomes normally; the predicate is not applied to an arbitrary outcome.

### HorizonAction

On first expansion of a matching deterministic action, expansion wraps its
result in a new `MCTSState` and replaces the `Unexplored` entry with
`HorizonAction(horizonState)`. That iteration rolls out from the horizon state.
On subsequent visits selection returns `Horizon(visitedStates, horizonState)`.

The `MCTSState` inside a `HorizonAction` is never recursed into by selection —
it exists solely to accumulate `Rollouts` and `WinCounts` from rollouts.
Because `HorizonAction` is not `Terminal`, the parent node can never become
fully resolved, so the search will not terminate early due to horizon nodes.

### Gimbur integration

In `Gimbur.Cli`, placement-only search targets the exact placement/main-game
boundary:

```csharp
LeafBoundary = state =>
    state is CatanState { TurnNumber: 1, Stage: TurnStage.PreRoll }
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
| LeafEvaluator        | ILeafEvaluator option                  | None           | Async per-player neural leaf values                 |
| LeafBoundary         | (ICoreState -> bool) option            | None           | Deterministic result-state horizon predicate ([details](#leaf-boundary-horizon)) |
| MaxPriorDepth        | int                                    | Int32.MaxValue | Deepest node that requests priors                    |
| MaxPendingEvaluations| int                                    | 32             | Maximum reserved neural leaf requests                |
| LeafEvaluationTimeoutMs | int                                 | 500            | Per-request timeout before rollout fallback          |
| DrainTimeoutMs       | int                                    | 1000           | Bounded post-deadline response drain                 |

The search stops when any budget is exhausted, or when all root actions have
been resolved to `Terminal` via terminal propagation.

When `PriorClient` is `None`, the search uses uniform priors and rollout-based
evaluation (the current default behavior). When set, the search fires prior
requests on node expansion and collects responses after terminal propagation.

When `LeafBoundary` is `None`, all actions are expanded normally. When set, it
is checked against each deterministic result state after that action executes.

## Asynchronous Leaf Evaluation

With `LeafEvaluator` configured, newly expanded deterministic leaves, horizon
leaves, and stochastic outcome sets are submitted to a shared non-blocking value
queue instead of immediately running random rollouts. Exact game terminals are
still evaluated by the rules engine. A stochastic action submits all nonterminal
outcomes as one request; the returned per-player vectors are probability-weighted
and commit exactly one action-edge visit.

Selection reserves every edge in an outstanding path through `PendingVisits`.
Reserved edges are excluded from selection, reducing duplicate leaf requests
without counting unfinished work as simulations. Completed responses first remove
the reservation and then backpropagate once. Invalid responses and requests that
time out while the search still has budget fall back to one random rollout. At the
search deadline, Kjarni stops enqueueing, drains responses for at most
`DrainTimeoutMs`, and cancels anything unresolved without adding visits.

If every selectable path is reserved, or `MaxPendingEvaluations` is reached, the
search calls `ILeafEvaluator.WaitForResults` rather than spinning. A shared HTTP
evaluator can therefore batch work from other concurrent games while each blocked
search sleeps on the same completion signal.

The HTTP evaluator also coalesces locally queued leaves for 2 ms, sending up to 64
requests in one `/state/leaf-enqueue` POST. Cancelled requests are omitted before
send, and transport failures or server-reported drops complete as invalid responses
so MCTS can use its normal fallback instead of waiting for a timeout.

Placement search reuses `ValueEstimates` already carried by an `IPriorClient`
response when that response is available immediately after expansion. This avoids
a duplicate placement-model request. It deliberately does not race a random
rollout against a late placement value; absent an immediately collected value,
placement retains the existing rollout path. Main-game values use the dedicated
`/state/leaf-enqueue` queue.

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
| leafEvaluationsSubmitted/applied | int | Neural leaf requests submitted and committed       |
| leafEvaluationTimeouts/Invalid | int | Timed-out and malformed responses                    |
| leafEvaluationsCancelled/fallbacks/orphans | int | Cleanup, rollout fallback, and stale response counts |
| leafEvaluationBatches/states | int | Response batches and state vectors evaluated          |
| leafEvaluationLatencyMs | int64 | Sum of client-observed request latency                  |
