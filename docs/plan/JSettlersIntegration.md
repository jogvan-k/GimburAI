# JSettlers2 Integration Plan

Integrate GimburAI as a third-party client bot in [JSettlers2](https://github.com/jdmonin/JSettlers2) for benchmarking against JSettlers' built-in bots.

## Overview

The integration consists of three components:

1. **`Gimbur.Server`** -- A new .NET Minimal API project exposing an HTTP endpoint that accepts a serialized game state, runs MCTS, and returns the chosen action.
2. **`jsettlers/`** -- A directory containing a Java bot client (extending JSettlers' third-party bot API), Gradle build scripts, and an init script that clones JSettlers2 into a `.gitignore`d path.
3. **Coordinate mapping** -- A bidirectional mapping between JSettlers and Gimbur board coordinate systems, embedded in the Java bot.

```
GimburAI/
├── src/
│   ├── Gimbur.Server/                  # NEW .NET project
│   │   ├── Gimbur.Server.csproj
│   │   ├── Program.cs                  # Minimal API: /choose-action, /health
│   │   └── ActionRequest.cs            # Request/response DTOs
│   └── ...existing projects...
├── jsettlers/                           # NEW integration directory
│   ├── init.sh                          # Clones JSettlers2 at a pinned tag
│   ├── .gitignore                       # Ignores jsettlers2/ clone, build/
│   ├── build.gradle                     # Builds the Gimbur bot JAR
│   ├── run-server.sh                    # Starts JSettlers + Gimbur.Server together
│   └── src/main/java/
│       └── gimbur/jsettlers/
│           ├── GimburClient.java        # extends SOCRobotClient
│           ├── GimburBrain.java         # extends SOCRobotBrain
│           ├── GimburServerClient.java  # HTTP client to Gimbur.Server
│           ├── StateTranslator.java     # SOCGame -> Gimbur serialized state
│           ├── ActionTranslator.java    # Gimbur action -> JSettlers messages
│           └── CoordinateMap.java       # Hex/vertex/edge mapping tables
└── docs/
    └── plan/
        └── JSettlersIntegration.md      # This file
```

## Background: Architecture Differences

| Aspect              | Gimbur                                     | JSettlers2                                          |
|---------------------|--------------------------------------------|-----------------------------------------------------|
| Language            | C# + F# (.NET 10)                          | Java 8+                                             |
| Game loop           | Bot owns the loop; calls `Actions()`, applies chosen action to immutable state clones | Server owns the loop; bots are network clients that receive messages and respond |
| State model         | Single `CatanState` object the bot fully controls | Distributed: server has full `SOCGame`, bot receives incremental updates via `SOCMessage` |
| Bot interface       | `ICoreState.Actions()` returns all legal moves; MCTS picks one | `SOCRobotBrain` is event-driven, reacting to messages in a queue |
| Coordinates         | Axial `HexCoord(Q, R)`, vertices/edges indexed by sorted spatial position | Diagonal-axis encoded hex coordinates (classic board) |
| Player indexing     | 1-based                                    | 0-based                                            |

The systems implement the same Catan rules, so the game semantics align. The work is purely a translation problem.

---

## Component 1: Gimbur.Server

### Project Setup

**Path:** `src/Gimbur.Server/Gimbur.Server.csproj`

```xml
<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="../Kjarni/Kjarni.fsproj" />
    <ProjectReference Include="../Gimbur/Gimbur.csproj" />
  </ItemGroup>
</Project>
```

- Uses `Microsoft.NET.Sdk.Web` for Kestrel/Minimal API (the existing CLI uses `Microsoft.NET.Sdk`; a separate project keeps concerns clean).
- Add to the solution: `dotnet sln add src/Gimbur.Server/Gimbur.Server.csproj`.

### Endpoint: `POST /choose-action`

**Request:**
```json
{
  "config": "standard",
  "playerCount": 4,
  "state": "<human-readable serialized state string>",
  "searchTimeMs": 1000,
  "maxRolloutDepth": 500
}
```

- `config`: one of `"standard"`, `"small"`, `"mini"` -- selects the `GameConfig` preset.
- `playerCount`: 2-4.
- `state`: the full human-readable serialization from `CatanStateSerializer.SerializeHumanReadable()` (10 pipe-delimited sections).
- `searchTimeMs`: MCTS time budget in milliseconds.
- `maxRolloutDepth`: max random rollout depth (default 500).

**Response:**
```json
{
  "typeTag": 0,
  "arg1": 23,
  "arg2": 0,
  "actionName": "PlaceSettlement",
  "visits": 4820,
  "winRate": 0.34,
  "allActions": [
    { "typeTag": 0, "arg1": 23, "arg2": 0, "visits": 4820, "winRate": 0.34 },
    { "typeTag": 0, "arg1": 17, "arg2": 0, "visits": 2100, "winRate": 0.28 }
  ]
}
```

- `typeTag`, `arg1`, `arg2`: the chosen action's identity tuple (see action table below).
- `actionName`: human-readable label for logging/debugging.
- `allActions`: all legal actions with their MCTS visit counts and win rates, for analysis.

**Endpoint: `GET /health`** -- returns 200 OK when the server is ready.

### Implementation Logic

```
1. Parse request, select GameConfig preset
2. Deserialize state: CatanState.DeserializeHumanReadable(config, playerCount, stateString)
3. Get legal actions: state.Actions()
4. If only 1 action -> return it immediately (forced move, no MCTS needed)
5. Create MCTSState root from the CatanState
6. Configure and run MonteCarloTreeSearch.RunSimulation()
7. Extract best path and per-action statistics
8. Return chosen action + diagnostics
```

Each HTTP call gets a fresh MCTS tree root. Tree reuse across calls would require session state, which adds complexity for marginal benchmarking benefit.

---

## Component 2: jsettlers/ Directory

### `jsettlers/init.sh`

Clones JSettlers2 at a pinned release tag into a gitignored subdirectory, then builds it:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JSETTLERS_DIR="$SCRIPT_DIR/jsettlers2"
JSETTLERS_TAG="release-2.6.10"

if [ -d "$JSETTLERS_DIR" ]; then
    echo "JSettlers2 already present at $JSETTLERS_DIR"
    exit 0
fi

git clone --depth 1 --branch "$JSETTLERS_TAG" \
    https://github.com/jdmonin/JSettlers2.git "$JSETTLERS_DIR"

echo "Building JSettlers2..."
(cd "$JSETTLERS_DIR" && gradle assemble)

echo "JSettlers2 ready."
```

### `jsettlers/.gitignore`

```
jsettlers2/
build/
.gradle/
```

### `jsettlers/build.gradle`

Compiles the Gimbur bot Java sources against the JSettlers2 build output and produces a JAR:

```groovy
plugins {
    id 'java'
}

java {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
}

repositories {
    mavenCentral()
}

dependencies {
    implementation fileTree(dir: 'jsettlers2/build/libs', include: ['*.jar'])
    implementation 'com.google.code.gson:gson:2.10.1'
}

jar {
    manifest {
        attributes 'Main-Class': 'gimbur.jsettlers.GimburClient'
    }
    from {
        configurations.runtimeClasspath.collect {
            it.isDirectory() ? it : zipTree(it)
        }
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    archiveBaseName = 'gimbur-jsettlers'
}
```

### `jsettlers/run-server.sh`

Orchestrates starting both servers:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GIMBUR_PORT="${GIMBUR_PORT:-5123}"
JSETTLERS_PORT="${JSETTLERS_PORT:-8880}"
GIMBUR_BOTS="${GIMBUR_BOTS:-2}"
SEARCH_TIME_MS="${SEARCH_TIME_MS:-1000}"

# 1. Build Gimbur.Server
echo "Building Gimbur.Server..."
dotnet build "$REPO_ROOT/src/Gimbur.Server/Gimbur.Server.csproj" -c Release -q

# 2. Build Java bot
echo "Building Gimbur JSettlers bot..."
(cd "$SCRIPT_DIR" && gradle build -q)

# 3. Start Gimbur.Server
echo "Starting Gimbur.Server on :$GIMBUR_PORT..."
dotnet run --project "$REPO_ROOT/src/Gimbur.Server/Gimbur.Server.csproj" \
    -c Release -- --urls "http://0.0.0.0:$GIMBUR_PORT" &
GIMBUR_PID=$!

until curl -sf "http://localhost:$GIMBUR_PORT/health" > /dev/null 2>&1; do
    sleep 0.2
done
echo "Gimbur.Server ready."

# 4. Start JSettlers with Gimbur bots
echo "Starting JSettlers on :$JSETTLERS_PORT with $GIMBUR_BOTS Gimbur bot(s)..."
java -cp "$SCRIPT_DIR/build/libs/gimbur-jsettlers.jar" \
    soc.server.SOCServer \
    -Djsettlers.bots.start3p="$GIMBUR_BOTS,gimbur.jsettlers.GimburClient" \
    -Djsettlers.bots.percent3p=100 \
    -Djsettlers.bots.cookie=gimbur \
    -Djsettlers.bots.timeout.turn=60 \
    -Djsettlers.allow.debug=Y \
    -Djsettlers.gameopt._EXT_BOT="http://localhost:$GIMBUR_PORT,$SEARCH_TIME_MS" \
    "$JSETTLERS_PORT" &
JSETTLERS_PID=$!

echo ""
echo "Ready. JSettlers :$JSETTLERS_PORT | Gimbur.Server :$GIMBUR_PORT"
echo "Connect a JSettlers client to localhost:$JSETTLERS_PORT to play."
echo "Press Ctrl+C to stop both servers."

trap "kill $GIMBUR_PID $JSETTLERS_PID 2>/dev/null; exit" INT TERM
wait
```

**Configuration via the `_EXT_BOT` game option:** JSettlers provides `_EXT_BOT` as a built-in mechanism for passing configuration strings to third-party bots. The Gimbur bot reads this at game join to get the server URL and search time.

---

## Component 3: Java Bot Classes

### `GimburClient.java`

Extends `SOCRobotClient`. Minimal -- its only job is to override `createBrain()`:

```java
package gimbur.jsettlers;

import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotParameters;
import soc.game.SOCGame;
import soc.server.genericServer.ServerConnectInfo;
import soc.util.CappedQueue;
import soc.message.SOCMessage;

public class GimburClient extends SOCRobotClient {

    public GimburClient(ServerConnectInfo sci, String nn, String pw) {
        super(sci, nn, pw);
        rbclass = GimburClient.class.getName();
    }

    @Override
    public SOCRobotBrain createBrain(
            SOCRobotParameters params, SOCGame game, CappedQueue<SOCMessage> mq) {
        return new GimburBrain(this, params, game, mq);
    }

    public static void main(String[] args) {
        // args: hostname port botname password cookie
        if (args.length < 5) {
            System.err.println("Usage: GimburClient hostname port botname password cookie");
            System.exit(1);
        }
        ServerConnectInfo sci = new ServerConnectInfo(args[0], Integer.parseInt(args[1]), args[4]);
        GimburClient client = new GimburClient(sci, args[2], args[3]);
        client.init();
    }
}
```

### `GimburBrain.java`

Extends `SOCRobotBrain`. Intercepts decision points, delegates to Gimbur.Server:

```java
package gimbur.jsettlers;

import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotParameters;
import soc.game.SOCGame;
import soc.util.CappedQueue;
import soc.message.SOCMessage;

public class GimburBrain extends SOCRobotBrain {

    private GimburServerClient gimburClient;
    private StateTranslator stateTranslator;
    private ActionTranslator actionTranslator;

    public GimburBrain(SOCRobotClient rc, SOCRobotParameters params,
                       SOCGame ga, CappedQueue<SOCMessage> mq) {
        super(rc, params, ga, mq);
    }

    @Override
    public void setOurPlayerData() {
        super.setOurPlayerData();

        // Parse _EXT_BOT: "http://localhost:5123,1000"
        String extBot = game.getGameOptionStringValue("_EXT_BOT");
        String[] parts = extBot.split(",", 2);
        String serverUrl = parts[0];
        int searchTimeMs = parts.length > 1 ? Integer.parseInt(parts[1]) : 1000;

        gimburClient = new GimburServerClient(serverUrl, searchTimeMs);
        stateTranslator = new StateTranslator(game);  // builds coord maps
        actionTranslator = new ActionTranslator(stateTranslator);
    }

    // -- Override each decision point to query Gimbur.Server --
    // See "Decision Point Mapping" section below for details.
}
```

**Decision point mapping -- which `SOCRobotBrain` methods to override:**

The brain's `run()` loop dispatches on `SOCGame` state constants. At each state where the bot must choose, the default brain calls into strategy objects and the decision maker. The Gimbur brain overrides these to query the server instead:

| Game State                    | Default Brain Method                 | Gimbur Override Strategy                                    |
|-------------------------------|--------------------------------------|-------------------------------------------------------------|
| `START1A` / `START2A`         | `openingBuildStrategy.planInitialSettlement()` | Override: query Gimbur, returns `PlaceSettlement` action   |
| `START1B` / `START2B`         | `openingBuildStrategy.planSecondSettlement()` (and road) | Override: query Gimbur, returns `PlaceRoad` action         |
| `ROLL_OR_CARD`                | `rollOrPlayKnightOrExpectDice()`     | Override: query Gimbur, returns `RollDice` or `PlayKnight` |
| `PLAY1`                       | `planAndDoActionForPLAY1()`          | Override: query Gimbur, returns build/trade/dev card/end turn |
| `PLACING_ROBBER`              | `robberStrategy.getBestRobberHex()`  | Override: query Gimbur, returns `ChooseRobberTile`         |
| `WAITING_FOR_ROB_CHOOSE_PLAYER` | `chooseRobberVictim()`            | Override: query Gimbur, returns `ChooseRobberVictim`       |
| `WAITING_FOR_DISCARDS`        | `discardStrategy.discard()`          | Override: query Gimbur (see note below)                    |
| `WAITING_FOR_MONOPOLY`        | `monopolyStrategy.getMonopolyChoice()` | Override: query Gimbur, returns `PlayMonopoly`           |
| `WAITING_FOR_DISCOVERY`       | (pick 2 resources)                   | Override: query Gimbur, returns `PlayYearOfPlenty`         |
| `PLACING_FREE_ROAD1/2`        | (place road)                         | Override: query Gimbur, returns `PlaceRoad`                |
| Trade offer received          | `considerOffer()`                    | Reject all offers (MCTS doesn't model negotiation)         |

**Note on discards:** Gimbur doesn't have an explicit "discard" action type -- in Gimbur's model, rolling a 7 triggers the robber flow directly. For the JSettlers integration, the discard decision can fall back to the default `SOCRobotBrain.discardStrategy` or use a simple heuristic (discard lowest-value resources). This is acceptable for benchmarking.

### `StateTranslator.java`

Converts `SOCGame` state into the Gimbur human-readable serialization format (10 pipe-delimited sections).

This is the most complex class. It has two responsibilities:

1. **Build coordinate mapping tables** (once, at game start)
2. **Serialize the current `SOCGame` state** (each time an action decision is needed)

#### Coordinate Mapping

Both systems place hexes on a grid. The standard 4-player board has 19 land hexes, 54 vertices, and 72 edges.

**Approach: static mapping tables for the standard board.**

Rather than deriving a general formula between coordinate systems, match positions spatially:

- Enumerate all JSettlers hex coordinates on the standard board (`SOCBoard4p` hex layout constants).
- Enumerate all Gimbur hex coordinates (axial radius-2, sorted by screen position).
- Match them by spatial position (row and column within the hex grid).
- Do the same for vertices (54) and edges (72).

The mapping tables are `int[]` arrays: `jsettlersHexToGimburTile[i]`, `gimburVertexToJSettlersNode[i]`, etc.

**Validation:** At game start, after building the mapping, compare the resource type and dice number at each mapped hex pair. If they don't match, the mapping is wrong -- fail loudly.

#### Serialization

For each decision, `StateTranslator.translate(SOCGame game)` produces the 10-section string:

| Section | Source in JSettlers | Gimbur Format |
|---------|---------------------|---------------|
| 1. Tiles (19) | `board.getHexTypeFromCoord()`, `board.getNumberOnHexFromCoord()` | `{resourceChar}{pipDigit}{sideChar}` per tile |
| 2. Ports (9) | `board.getPortTypeFromNodeCoord()` | `{portTypeChar}` per port |
| 3. Robber (1) | `board.getRobberHex()` | Crockford base-32 tile index |
| 4. Turn (3) | `game.getCurrentPlayerNumber()`, `game.getGameState()` | `{playerChar}{stageChar}{postDevChar}` |
| 5. Longest/Largest (2) | `game.getPlayerWithLongestRoad()`, `game.getPlayerWithLargestArmy()` | `{playerChar}{playerChar}` |
| 6. Vertices (54) | `board.settlementAtNode()` for each vertex | `{buildingChar}{playerChar}` per vertex |
| 7. Edges (72) | check roads at each edge coordinate | `{playerChar}` per edge |
| 8. Resources | `player.getResources()` for each player | Crockford base-32 counts, `/`-delimited |
| 9. Knights | `player.getNumKnights()` for each player | Crockford base-32, `/`-delimited |
| 10. Dev cards | `player.getInventory()` for each player | Crockford base-32 counts, `/`-delimited |

**Character encoding reference (from `StateToken.cs`):**

| Category   | Characters                                                        |
|------------|-------------------------------------------------------------------|
| Resource   | `d` (desert) `w` (wood) `b` (brick) `s` (sheep) `W` (wheat) `o` (ore) |
| Player     | `_` (none) `-` (P1) `+` (P2) `*` (P3) `^` (P4)                  |
| Building   | `.` (empty) `v` (settlement) `c` (city)                          |
| Turn stage | `a` `e` `f` `i` `r` `x` `y` `t`                                 |
| Port       | `g` (generic 3:1) + resource chars                                |
| Quantities | Crockford base-32: `0-9 A-H J K M N P-T V-Z`                    |

**Turn stage mapping:**

| JSettlers `SOCGame` State          | Gimbur `TurnStage`       | Char |
|------------------------------------|--------------------------|------|
| `START1A`                          | `PlaceFirstSettlement`   | `a`  |
| `START1B`                          | `PlaceFirstRoad`         | `e`  |
| `START2A`                          | `PlaceSecondSettlement`  | `f`  |
| `START2B`                          | `PlaceSecondRoad`        | `i`  |
| `ROLL_OR_CARD`                     | `PreRoll`                | `r`  |
| `PLACING_ROBBER`                   | `ChooseRobberLocation`   | `x`  |
| `PLAY1`                            | `BuildTrade`             | `t`  |
| `WAITING_FOR_MONOPOLY`             | `BuildTrade` (dev card)  | `y`  |
| `WAITING_FOR_DISCOVERY`            | `BuildTrade` (dev card)  | `y`  |

### `ActionTranslator.java`

Converts Gimbur `(TypeTag, Arg1, Arg2)` tuples into JSettlers client method calls.

Uses the inverse coordinate mapping from `StateTranslator` (Gimbur index -> JSettlers coordinate).

| TypeTag | Gimbur Action         | JSettlers Client Call                                                          |
|---------|-----------------------|--------------------------------------------------------------------------------|
| 0       | PlaceSettlement(v)    | `client.putPiece(game, new SOCSettlement(player, mapVertex(arg1), board))`     |
| 1       | PlaceRoad(e)          | `client.putPiece(game, new SOCRoad(player, mapEdge(arg1), board))`             |
| 2       | RollDice              | `client.rollDice(game)`                                                        |
| 3       | ChooseRobberTile(t)   | `client.moveRobber(game, mapHex(arg1))`                                        |
| 4       | BuildCity(v)          | `client.putPiece(game, new SOCCity(player, mapVertex(arg1), board))`           |
| 5       | BankTrade(give, recv) | `client.bankTrade(game, toResourceSet(arg1), toResourceSet(arg2))`             |
| 6       | BuyDevCard            | `client.buyDevCard(game)`                                                      |
| 7       | PlayKnight            | `client.playDevCard(game, SOCDevCardConstants.KNIGHT)`                         |
| 8       | PlayRoadBuilding      | `client.playDevCard(game, SOCDevCardConstants.ROADS)`                          |
| 9       | PlayMonopoly(res)     | `client.playDevCard(game, SOCDevCardConstants.MONO)` + pick resource           |
| 10      | PlayYearOfPlenty(r,r) | `client.playDevCard(game, SOCDevCardConstants.DISC)` + pick 2 resources        |
| 11      | EndTurn               | `client.endTurn(game)`                                                         |
| 12      | ChooseRobberVictim(p) | `client.choosePlayer(game, mapPlayer(arg1))`                                   |

**Resource type mapping:**

| Gimbur `ResourceType` (for Arg1/Arg2) | Value | JSettlers `SOCResourceConstants` |
|----------------------------------------|-------|----------------------------------|
| Desert                                 | 0     | (N/A)                            |
| Wood                                   | 1     | `WOOD = 5`                       |
| Brick                                  | 2     | `CLAY = 1`                       |
| Sheep                                  | 3     | `SHEEP = 3`                      |
| Wheat                                  | 4     | `WHEAT = 4`                      |
| Ore                                    | 5     | `ORE = 2`                        |

**Player mapping:** Gimbur 1-based -> JSettlers 0-based: `jsettlersPlayer = gimburPlayer - 1`.

### `GimburServerClient.java`

HTTP client that calls `POST /choose-action` on the Gimbur.Server.

Uses `java.net.HttpURLConnection` (available in Java 8) or `java.net.http.HttpClient` (Java 11+) depending on compatibility needs. Parses JSON responses using Gson.

```java
public class GimburServerClient {
    private final String baseUrl;
    private final int searchTimeMs;

    public GimburServerClient(String baseUrl, int searchTimeMs) { ... }

    public ActionResponse chooseAction(String serializedState, int playerCount) {
        // POST to baseUrl + "/choose-action"
        // Body: { config: "standard", playerCount, state, searchTimeMs }
        // Returns: { typeTag, arg1, arg2, actionName, visits, winRate }
    }
}
```

### `CoordinateMap.java`

Encapsulates the bidirectional coordinate mapping tables. Constructed once per game from the `SOCBoard` instance. Provides:

```java
public class CoordinateMap {
    public CoordinateMap(SOCBoard board) { ... }  // builds all tables

    public int jsettlersHexToGimburTile(int hexCoord);
    public int gimburTileToJSettlersHex(int tileIndex);
    public int jsettlersNodeToGimburVertex(int nodeCoord);
    public int gimburVertexToJSettlersNode(int vertexIndex);
    public int jsettlersEdgeToGimburEdge(int edgeCoord);
    public int gimburEdgeToJSettlersEdge(int edgeIndex);

    public void validate(SOCBoard board);  // compare resource types at each hex
}
```

---

## Implementation Phases

### Phase 1: Gimbur.Server (1-2 days)

1. Create `src/Gimbur.Server/Gimbur.Server.csproj` with `Microsoft.NET.Sdk.Web`.
2. Add to the solution: `dotnet sln add src/Gimbur.Server/Gimbur.Server.csproj`.
3. Implement `Program.cs` with Minimal API:
   - `GET /health` -- returns 200.
   - `POST /choose-action` -- deserializes state, runs MCTS, returns action.
4. Define request/response record types in `ActionRequest.cs`.
5. Test with `curl` using manually crafted serialized states.

### Phase 2: jsettlers/ Scaffolding (1 day)

1. Create `jsettlers/init.sh`, `jsettlers/.gitignore`.
2. Create `jsettlers/build.gradle` referencing the JSettlers2 build output.
3. Run `init.sh`, verify JSettlers2 builds.
4. Create skeleton `GimburClient.java` with `main()` -- verify it can connect to a JSettlers server as a third-party bot (even if it makes no decisions yet).

### Phase 3: Coordinate Mapping (2-3 days)

1. Document both coordinate systems side by side (JSettlers hex IDs from `SOCBoard4p`, Gimbur axial coords from `BoardTopology`).
2. Build the static mapping tables for the standard 4-player board in `CoordinateMap.java`.
3. Implement `validate()` -- at game start, compare hex resource types through the mapping.
4. Write unit tests for the mapping (a few known positions checked by hand).

### Phase 4: State Translation (2-3 days)

1. Implement `StateTranslator.translate()` section by section (tiles, ports, robber, turn, vertices, edges, resources, knights, dev cards).
2. Test by running a JSettlers game with debug logging, capturing the serialized output, and deserializing it back through `CatanState.DeserializeHumanReadable()` on the .NET side to verify round-trip correctness.

### Phase 5: Action Translation (1-2 days)

1. Implement `ActionTranslator.execute()` for all 13 action types.
2. Handle multi-step JSettlers sequences (e.g., `PlayMonopoly` requires `playDevCard` then `pickResources`).
3. Handle resource set construction for `BankTrade` (determining the trade ratio from ports).

### Phase 6: GimburBrain Integration (2-3 days)

1. Override each decision method in `GimburBrain` to call `stateTranslator.translate()` -> `gimburClient.chooseAction()` -> `actionTranslator.execute()`.
2. Handle the fallback for decisions Gimbur doesn't model (discards, trade offers).
3. Add timeout handling -- if Gimbur.Server is slow, fall back to `super` behavior.
4. Add logging at each decision point (chosen action, MCTS stats).

### Phase 7: run-server.sh and Integration Testing (3-5 days)

1. Create `jsettlers/run-server.sh`.
2. Run bot-only games: `jsettlers.bots.botgames.total=10` with a mix of Gimbur and JSettlers bots.
3. Debug coordinate mapping errors (these will manifest as "illegal placement" rejections from the server).
4. Debug state reconstruction errors (compare Gimbur's view of the game vs JSettlers' view at each decision point).
5. Run longer benchmarks (100+ games) and collect win rates.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Coordinate mapping errors cause illegal placements | High | `CoordinateMap.validate()` compares board layouts at game start; fail fast if mismatched |
| JSettlers game state diverges from reconstructed Gimbur state | High | Add a `/validate-state` debug endpoint that returns the parsed state for comparison; log both states at each decision |
| MCTS response time exceeds JSettlers bot timeout | Medium | Set `jsettlers.bots.timeout.turn=60`; set MCTS search time well below that; handle HTTP timeouts gracefully with fallback |
| Multi-step JSettlers message sequences (play dev card -> pick resource) don't map to single Gimbur actions | Medium | `ActionTranslator` handles the full sequence: sends the dev card play message, then waits for the server's prompt and sends the resource pick |
| Dev card tracking mismatch (Gimbur tracks exact deck composition; JSettlers reveals partial info) | Low | Bot has full knowledge of its own cards; opponent cards aren't needed for its own decisions |
| JSettlers 2.6.10 targets Java 8; `java.net.http.HttpClient` requires Java 11+ | Low | Use `java.net.HttpURLConnection` (available in Java 8) or bundle a lightweight HTTP client |

---

## Scope and Limitations

This integration covers **benchmarking on the standard 4-player classic board only**.

**Not in scope:**
- Sea boards, 6-player boards, or scenarios (would require additional coordinate mapping tables).
- Player-to-player trade negotiation (Gimbur's MCTS doesn't model negotiation; `considerOffer()` will reject all trade offers).
- Tree reuse across HTTP calls (each decision starts a fresh MCTS tree).
- The discard decision on rolling a 7 (falls back to default JSettlers heuristic).

---

## Server Communication Flow

```
 JSettlers Server (Java)          GimburBrain (Java)           Gimbur.Server (.NET)
        |                              |                              |
        |--- TURN (your turn) -------->|                              |
        |--- GAMESTATE (ROLL_OR_CARD)->|                              |
        |                              |                              |
        |                              |-- translate SOCGame state -->|
        |                              |   POST /choose-action        |
        |                              |                              |-- deserialize state
        |                              |                              |-- run MCTS
        |                              |                              |-- extract best action
        |                              |<---- { typeTag, arg1, arg2 } |
        |                              |                              |
        |<-- rollDice() ---------------|                              |
        |                              |                              |
        |--- DICERESULT (8) ---------->|                              |
        |--- GAMESTATE (PLAY1) ------->|                              |
        |                              |                              |
        |                              |-- translate SOCGame state -->|
        |                              |   POST /choose-action        |
        |                              |                              |-- run MCTS
        |                              |<---- { typeTag:0, arg1:23 }  |
        |                              |                              |
        |<-- putPiece(settlement@23) --|                              |
        |                              |                              |
       ...                            ...                            ...
```
