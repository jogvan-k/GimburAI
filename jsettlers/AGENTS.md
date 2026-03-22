# jsettlers/ — JSettlers2 Integration

## Overview

Java bot client and benchmark CLI for running GimburAI bots against JSettlers2's built-in AI.

Phase 1 (current): `GimburBrain` v1 replicates JSettlers' strongest heuristic AI (`SMART_STRATEGY`) for baseline benchmarking.  
Phase 2 (future): `GimburBrain` v2 delegates decisions to `Gimbur.Server` (MCTS engine over HTTP).

## Prerequisites

- **Java 17** (OpenJDK) — required for building JSettlers2 with Gradle 7.5.1
- **Git** — for cloning JSettlers2
- Internet access (first run only, to clone JSettlers2 and download Gradle wrapper deps)

On Arch Linux:
```bash
sudo pacman -S jdk17-openjdk git
```

## Setup

Run the init script from the repository root:
```bash
./jsettlers/init.sh
```

This will:
1. Clone JSettlers2 at the pinned commit into `jsettlers/JSettlers2/` (gitignored)
2. Download Gradle 7.5.1 to `jsettlers/.tools/gradle-7.5.1/` (gitignored)
3. Build JSettlers2 with Java 17
4. Build the Gimbur bot JAR

## Directory Structure

```
jsettlers/
├── .gitignore              # Ignores JSettlers2/, .tools/, build/, .gradle/
├── AGENTS.md               # This file
├── init.sh                 # Clone + build script
├── build.gradle            # Gradle build for the Gimbur bot JAR
├── settings.gradle         # Gradle settings
└── src/main/java/gimbur/jsettlers/
    ├── GimburClient.java   # Bot client (extends SOCRobotClient)
    ├── GimburBrain.java    # Bot brain v1 (extends SOCRobotBrain, SMART_STRATEGY)
    └── BenchmarkCli.java   # Benchmark runner CLI
```

## Build

After `init.sh` has run, rebuild the bot JAR:
```bash
JAVA_HOME=/usr/lib/jvm/java-17-openjdk jsettlers/.tools/gradle-7.5.1/bin/gradle -p jsettlers assemble
```

## Running the Benchmark

```bash
java -cp jsettlers/build/libs/gimbur-jsettlers.jar gimbur.jsettlers.BenchmarkCli \
    --games 100 \
    --gimbur-bots 2 \
    --jsettlers-bots 2
```

Options:
- `--games N` — Number of games to run (default: 100)
- `--gimbur-bots N` — Number of Gimbur bots per game (default: 2)
- `--jsettlers-bots N` — Number of JSettlers built-in bots per game (default: 2)
- `--port N` — JSettlers server port (default: 8880)
- `--output FILE` — Write JSON results to file
- `--verbose` — Verbose logging

## JSettlers2 Pinned Version

- Repository: https://github.com/jdmonin/JSettlers2
- Commit: `60e4d7145261bf023be91c8507d9155e7cda7edc`
- Gradle: 7.5.1 (downloaded locally, not system Gradle)
- Java: 17 (build uses `mainClassName` removed in Gradle 8+; Java 25 produces incompatible class files)

## Bot Architecture

### Phase 1 (v1): Heuristic baseline
- `GimburBrain` extends `SOCRobotBrain` with `SMART_STRATEGY` parameters
- Uses the same decision-making as JSettlers' smarter built-in bots
- Purpose: establish a baseline win rate (should be ~50/50 against SMART bots)

### Phase 2 (v2): MCTS engine
- `GimburBrain` delegates decisions to `Gimbur.Server` via HTTP
- State/action translation between JSettlers and Gimbur coordinate systems
- See `docs/plan/JSettlersIntegration.md` for full architecture

## Test Commands

```bash
# Build everything
./jsettlers/init.sh

# Rebuild bot JAR only
JAVA_HOME=/usr/lib/jvm/java-17-openjdk jsettlers/.tools/gradle-7.5.1/bin/gradle -p jsettlers assemble

# Run quick benchmark (10 games)
java -cp jsettlers/build/libs/gimbur-jsettlers.jar gimbur.jsettlers.BenchmarkCli --games 10

# Run benchmark with JSON output
java -cp jsettlers/build/libs/gimbur-jsettlers.jar gimbur.jsettlers.BenchmarkCli --games 100 --output results.json
```
