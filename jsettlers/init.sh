#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JSETTLERS_DIR="$SCRIPT_DIR/JSettlers2"
JSETTLERS_COMMIT="60e4d7145261bf023be91c8507d9155e7cda7edc"

# ---- Check prerequisites ----

errors=0

# Check Java
if [ -n "$JAVA_HOME" ]; then
    JAVA_CMD="$JAVA_HOME/bin/java"
    JAVAC_CMD="$JAVA_HOME/bin/javac"
else
    JAVA_CMD="$(command -v java 2>/dev/null || true)"
    JAVAC_CMD="$(command -v javac 2>/dev/null || true)"
fi

if [ -z "$JAVA_CMD" ] || [ ! -x "$JAVA_CMD" ]; then
    echo "ERROR: Java not found."
    echo "  Install Java 17 (e.g. 'sudo pacman -S jdk17-openjdk') and ensure"
    echo "  it is on your PATH, or set JAVA_HOME to your Java 17 installation."
    errors=1
else
    JAVA_VERSION=$("$JAVA_CMD" -version 2>&1 | head -1 | sed 's/.*"\([0-9]*\)\..*/\1/')
    if [ "$JAVA_VERSION" != "17" ]; then
        echo "ERROR: Java 17 required, but found Java $JAVA_VERSION ($JAVA_CMD)."
        echo "  Install Java 17 and set JAVA_HOME to point to it."
        echo "  Example: export JAVA_HOME=/usr/lib/jvm/java-17-openjdk"
        errors=1
    else
        echo "Found Java 17: $JAVA_CMD"
        export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$JAVA_CMD")")")}"
    fi
fi

# Check Gradle
GRADLE_CMD="$(command -v gradle 2>/dev/null || true)"
if [ -z "$GRADLE_CMD" ]; then
    echo "ERROR: Gradle not found on PATH."
    echo "  Install Gradle 7.x (e.g. 'sudo pacman -S gradle') or download it"
    echo "  from https://gradle.org/releases/ and add it to your PATH."
    errors=1
else
    GRADLE_VERSION=$(gradle --version 2>/dev/null | grep '^Gradle ' | awk '{print $2}')
    GRADLE_MAJOR=$(echo "$GRADLE_VERSION" | cut -d. -f1)
    if [ "$GRADLE_MAJOR" != "7" ]; then
        echo "WARNING: Gradle 7.x recommended, but found Gradle $GRADLE_VERSION."
        echo "  JSettlers2 uses 'mainClassName' which was removed in Gradle 8+."
        echo "  If the build fails, install Gradle 7.5.1 from https://gradle.org/releases/"
        # Don't set errors=1; let the user try and see if it works
    else
        echo "Found Gradle $GRADLE_VERSION: $GRADLE_CMD"
    fi
fi

# Check Git
if ! command -v git &>/dev/null; then
    echo "ERROR: Git not found on PATH."
    echo "  Install git (e.g. 'sudo pacman -S git')."
    errors=1
else
    echo "Found Git: $(command -v git)"
fi

if [ "$errors" -ne 0 ]; then
    echo ""
    echo "Please install the missing prerequisites and try again."
    exit 1
fi

echo ""

# ---- Clone JSettlers2 ----

if [ -d "$JSETTLERS_DIR" ]; then
    echo "JSettlers2 already present at $JSETTLERS_DIR"
else
    echo "Cloning JSettlers2..."
    git clone https://github.com/jdmonin/JSettlers2.git "$JSETTLERS_DIR"
    git -C "$JSETTLERS_DIR" checkout "$JSETTLERS_COMMIT"
    echo "JSettlers2 cloned at commit $JSETTLERS_COMMIT"
fi

# ---- Build JSettlers2 ----

echo "Building JSettlers2..."
JAVA_HOME="$JAVA_HOME" gradle -p "$JSETTLERS_DIR" assemble
echo "JSettlers2 built successfully."

# ---- Build Gimbur bot ----

echo "Building Gimbur JSettlers bot..."
JAVA_HOME="$JAVA_HOME" gradle -p "$SCRIPT_DIR" assemble
echo "Gimbur bot built successfully."

echo ""
echo "Setup complete. Run the benchmark with:"
echo "  java -jar jsettlers/build/libs/gimbur-jsettlers.jar --games 10"
