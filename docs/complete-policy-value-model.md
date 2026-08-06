# Complete Policy-Value Model

## Scope

`catan_policy_value_v1` is the current-only full-game model. It predicts a
per-player value distribution and one fixed-width policy vector for every Catan
decision state, including initial placement and normal play. Older placement and
state checkpoints are incompatible and are not loaded or migrated.

Compound choices are represented as staged game states. Playing a development
card, buying a piece, upgrading a city, and initiating a bank trade are separate
actions from their subsequent spatial or resource choices. Once initiated, the
choice is mandatory; there is no cancellation/refund action.

## Staged Action Graph

- `PlayKnight` -> `ChooseRobberLocation` -> optional `ChooseRobberVictim`.
- `PlayRoadBuilding` -> `PlaceRoadBuildingFirst` -> `PlaceRoadCommitted` for the
  second free road when another legal road exists.
- `PlayYearOfPlenty` -> `ChooseYearOfPlentyFirst` ->
  `ChooseYearOfPlentySecond`.
- `PlayMonopoly` -> `ChooseMonopolyResource`.
- `BuyRoad` -> `PlaceRoadCommitted`.
- `BuySettlement` -> `PlaceSettlementCommitted`.
- `UpgradeCity` -> `PlaceCityCommitted`.
- `TradeWithBank` -> `ChooseBankTradeGive` -> `ChooseBankTradeReceive`.

Costs and development cards are consumed by the initiating action. Completion
actions do not charge again. Road Building reevaluates legal roads after the
first placement and skips the second only when no legal edge or road piece
remains. At most one development card may be initiated per turn.

## Policy Vector

Let `T` be tile count, `V` vertex count, `E` edge count, and `N` player count.
Resource order is `Wood, Brick, Sheep, Wheat, Ore`.

| Segment | Width | Offset | Meaning |
|---|---:|---:|---|
| Tiles | `T` | `0` | Robber tile placement. |
| Vertices | `V` | `T` | Settlement placement and city upgrade. |
| Edges | `E` | `T+V` | Initial, purchased, and Road Building roads. |
| Resources | `5` | `T+V+E` | Monopoly, Year of Plenty, and bank-trade choices. |
| Buy/trade | `5` | `T+V+E+5` | Buy road, buy settlement, upgrade city, buy development card, trade with bank. |
| Play dev card | `4` | `T+V+E+10` | Knight, Road Building, Monopoly, Year of Plenty. |
| Victim players | `N` | `T+V+E+14` | Robber victim in canonical player-slot order. |
| Controls | `2` | `T+V+E+14+N` | Roll dice, end turn. |

Policy width:

```text
T + V + E + 5 + 5 + 4 + N + 2
```

Current widths are mini 2P `79`, small 2P `101`, small 3P `102`, standard
3P `164`, and standard 4P `165`.

The same index may represent different concrete action classes only when the
turn stage makes the meaning unambiguous. For example, the vertex segment is
used for both settlement and city stages, and the resource segment is reused
for Monopoly, Year of Plenty, and bank-trade stages.

## Stage Masks

- Initial or committed settlement: legal vertex indices.
- Initial, purchased, or Road Building road: legal edge indices.
- City upgrade placement: owned upgradeable settlement vertices.
- Robber location: legal tile indices.
- Robber victim: legal canonical player slots.
- Monopoly, Year of Plenty, and bank trade: legal resource indices.
- `PreRoll`: Roll plus legal development-card initiations.
- `BuildTrade`: legal buy/trade initiations, legal development-card initiations,
  and End Turn.
- Terminal: empty mask.

C# rules are authoritative for legality. Serving returns the complete raw policy
distribution; C# masks and renormalizes legal entries before PUCT or direct play.
Stochastic outcomes such as dice totals, bought development-card identity, and
stolen resource are chance outcomes and never receive policy indices.

## Symmetry And Player Rotation

Board symmetry transforms tile, vertex, and edge segments through their topology
permutations. Resource, buy/trade, development-card, victim, and control segments
are geometrically invariant. Player canonicalization rotates ownership/value
slots and robber-victim indices so the acting player is canonical player 1.

## Model And Checkpoint Contract

Input is the full serialized game state. Output is:

```text
value:  [B, N]
policy: [B, policy_width]
```

Training uses normalized MCTS root visits as the policy target and the legal
action mask exported by C#. Value targets use exact resolved distributions when
available and otherwise the configured MCTS/terminal blend. Checkpoints use
architecture `catan_policy_value_v1` and a new checkpoint version; no previous
checkpoint architecture is accepted.
