# jsettlers/ -- JSettlers2 Integration

This directory contains a Java bot client and benchmark CLI for running GimburAI bots against [JSettlers2](https://github.com/jdmonin/JSettlers2)'s built-in AI players.

The current implementation (`GimburBrain` v1) replicates JSettlers' strongest heuristic AI (`SMART_STRATEGY`) for baseline benchmarking. A future v2 will delegate decisions to the GimburAI MCTS engine over HTTP.

## Requirements

| Tool       | Version | Notes |
|------------|---------|-------|
| **Java**   | 17      | OpenJDK 17. JSettlers2 uses `mainClassName` (removed in Gradle 8+) and Java 17 class format. Other versions will not work. |
| **Gradle** | 7.x     | Tested with 7.5.1. Gradle 8+ is incompatible with JSettlers2's `build.gradle`. |


### Installing requirements

**Arch Linux:**
```bash
sudo pacman -S jdk17-openjdk git gradle
```

**macOS (Homebrew):**
```bash
brew install openjdk@17 gradle git
```

**Other systems:** Install Java 17, Gradle 7.x, and Git through your package
manager or from the official downloads:
- Java 17: https://adoptium.net/
- Gradle 7.5.1: https://gradle.org/releases/

Make sure `JAVA_HOME` points to your Java 17 installation. The init script
attempts to auto-detect it but you can always set it explicitly:
```bash
export JAVA_HOME=/path/to/java-17
```

## Setup

Run the init script from anywhere in the repository:
```bash
./jsettlers/init.sh
```

This will:
1. Verify Java 17 and Gradle 7.x are available
2. Clone JSettlers2 at the pinned commit into `jsettlers/JSettlers2/` (gitignored)
3. Build JSettlers2
4. Build the Gimbur bot JAR (`jsettlers/build/libs/gimbur-jsettlers.jar`)

## Running the Benchmark

```bash
java -jar jsettlers/build/libs/gimbur-jsettlers.jar [options]
```

Or equivalently:
```bash
java -cp jsettlers/build/libs/gimbur-jsettlers.jar gimbur.jsettlers.BenchmarkCli [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--games, -g N` | Number of games to run | 100 |
| `--gimbur-bots N` | Gimbur bots per game | 2 |
| `--jsettlers-bots N` | JSettlers built-in bots per game | 2 |
| `--port N` | JSettlers server port | 8880 |
| `--parallel N` | Max concurrent games | 4 |
| `--output, -o FILE` | Write JSON results to file | -- |
| `--verbose` | Print per-game details | off |
| `--help, -h` | Show help | -- |

The total number of bots (`--gimbur-bots` + `--jsettlers-bots`) must equal 4.

### Examples

```bash
# Quick smoke test (10 games)
java -jar jsettlers/build/libs/gimbur-jsettlers.jar --games 10

# Full benchmark with JSON export
java -jar jsettlers/build/libs/gimbur-jsettlers.jar \
    --games 100 \
    --gimbur-bots 2 \
    --jsettlers-bots 2 \
    --output results.json

# 3 Gimbur bots vs 1 JSettlers bot
java -jar jsettlers/build/libs/gimbur-jsettlers.jar \
    --games 50 \
    --gimbur-bots 3 \
    --jsettlers-bots 1
```

### Output

The benchmark prints a summary when all games complete:

```
=== Benchmark Results ===

Games completed: 100 / 100
Total time: 142.3 s
Avg time per game: 1.4 s
Avg rounds per game: 23.5

--- Win Rates ---
Gimbur (SMART):   48 / 100  (48.0%)
JSettlers (mix):  52 / 100  (52.0%)

--- Win Rate by Seat ---
Seat 0: 27 / 100 wins (27.0%)
Seat 1: 25 / 100 wins (25.0%)
Seat 2: 24 / 100 wins (24.0%)
Seat 3: 24 / 100 wins (24.0%)
```

With `--output`, a JSON file is written containing per-game details (winner,
VP, rounds, duration, per-seat player info).

## Rebuilding

After the initial `init.sh` setup, rebuild just the bot JAR:
```bash
gradle -p jsettlers assemble
```

If Gradle or Java are not on your `PATH` with the right versions, specify them:
```bash
JAVA_HOME=/usr/lib/jvm/java-17-openjdk /path/to/gradle-7.5.1/bin/gradle -p jsettlers assemble
```

## Directory Structure

```
jsettlers/
├── README.md                 # This file
├── AGENTS.md                 # AI agent context (for LLM assistants)
├── init.sh                   # Setup script (clone + build)
├── build.gradle              # Gradle build producing gimbur-jsettlers.jar
├── settings.gradle           # Gradle project settings
├── .gitignore                # Ignores JSettlers2/, .tools/, build/, .gradle/
├── JSettlers2/               # JSettlers2 clone (gitignored)
├── build/                    # Build output (gitignored)
│   └── libs/
│       └── gimbur-jsettlers.jar
└── src/main/java/gimbur/jsettlers/
    ├── BenchmarkCli.java     # Benchmark runner CLI
    ├── GimburClient.java     # Bot client (extends SOCRobotClient)
    └── GimburBrain.java      # Bot brain v1 (SMART_STRATEGY heuristic)
```

## JSettlers2 Pinned Version

- **Repository:** https://github.com/jdmonin/JSettlers2
- **Commit:** `60e4d7145261bf023be91c8507d9155e7cda7edc`

## Bot Architecture

**Phase 1 (current):** `GimburBrain` extends `SOCRobotBrain` with
`SMART_STRATEGY` parameters -- identical to JSettlers' strongest built-in
heuristic AI. Expected win rate is ~50% against JSettlers' SMART bots.

**Phase 2 (planned):** `GimburBrain` will delegate decisions to `Gimbur.Server`
(the MCTS engine) over HTTP. See `docs/plan/JSettlersIntegration.md` for the
full architecture.
