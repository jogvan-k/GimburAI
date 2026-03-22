package gimbur.jsettlers;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import soc.robot.SOCRobotDM;
import soc.server.SOCGameListAtServer;
import soc.server.SOCServer;
import soc.util.SOCRobotParameters;

/**
 * Benchmark CLI that runs bot-only games on a JSettlers2 server
 * with Gimbur bots (using SMART_STRATEGY) vs JSettlers built-in bots.
 *
 * <p>Starts the JSettlers server in-process with the configured number of
 * bot-only games. Gimbur bots are registered as third-party bots and
 * launched in-process by the server. Results are collected via
 * {@link GameResult} and printed when all games complete.
 *
 * <p>Usage:
 * <pre>
 *   java gimbur.jsettlers.BenchmarkCli --games 100 --gimbur-bots 2 --jsettlers-bots 2
 * </pre>
 */
public class BenchmarkCli {

    // ---- Shared result collection (written by GimburClient/GimburBrain, read by this CLI) ----

    /**
     * Thread-safe map of per-game results, keyed by game name to deduplicate.
     * Multiple GimburClient instances may try to record the same game;
     * putIfAbsent ensures each game is recorded exactly once.
     */
    static final ConcurrentHashMap<String, GameResult> resultsByGame = new ConcurrentHashMap<>();

    /** Record holding per-game result data. */
    static class GameResult {
        final String gameName;
        final String winnerName;
        final boolean winnerIsGimbur;
        final int winnerVP;
        final int rounds;
        final int durationSeconds;
        final String[] playerNames;  // indexed by seat (0..3)
        final int[] playerVPs;       // indexed by seat (0..3)
        final boolean[] playerIsGimbur;  // indexed by seat (0..3)

        GameResult(String gameName, String winnerName, boolean winnerIsGimbur,
                   int winnerVP, int rounds, int durationSeconds,
                   String[] playerNames, int[] playerVPs, boolean[] playerIsGimbur) {
            this.gameName = gameName;
            this.winnerName = winnerName;
            this.winnerIsGimbur = winnerIsGimbur;
            this.winnerVP = winnerVP;
            this.rounds = rounds;
            this.durationSeconds = durationSeconds;
            this.playerNames = playerNames;
            this.playerVPs = playerVPs;
            this.playerIsGimbur = playerIsGimbur;
        }
    }

    // ---- CLI options ----

    private int games = 100;
    private int gimburBots = 2;
    private int jsettlersBots = 2;
    private int port = 8880;
    private int parallel = 4;
    private String outputFile = null;
    private boolean verbose = false;

    public static void main(String[] args) {
        BenchmarkCli cli = new BenchmarkCli();
        cli.parseArgs(args);
        cli.run();
    }

    private void parseArgs(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--games":
                case "-g":
                    games = Integer.parseInt(args[++i]);
                    break;
                case "--gimbur-bots":
                    gimburBots = Integer.parseInt(args[++i]);
                    break;
                case "--jsettlers-bots":
                    jsettlersBots = Integer.parseInt(args[++i]);
                    break;
                case "--port":
                    port = Integer.parseInt(args[++i]);
                    break;
                case "--parallel":
                    parallel = Integer.parseInt(args[++i]);
                    break;
                case "--output":
                case "-o":
                    outputFile = args[++i];
                    break;
                case "--verbose":
                    verbose = true;
                    break;
                case "--help":
                case "-h":
                    printUsage();
                    System.exit(0);
                    break;
                default:
                    System.err.println("Unknown option: " + args[i]);
                    printUsage();
                    System.exit(1);
            }
        }
    }

    private void printUsage() {
        System.out.println("Usage: BenchmarkCli [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --games, -g N          Number of games to run (default: 100)");
        System.out.println("  --gimbur-bots N        Gimbur bots per game (default: 2)");
        System.out.println("  --jsettlers-bots N     JSettlers built-in bots per game (default: 2)");
        System.out.println("  --port N               Server port (default: 8880)");
        System.out.println("  --parallel N           Max concurrent games (default: 4)");
        System.out.println("  --output, -o FILE      Write JSON results to file");
        System.out.println("  --verbose              Verbose logging");
        System.out.println("  --help, -h             Show this help");
        System.out.println();
        System.out.println("Gimbur bots use SMART_STRATEGY (JSettlers' strongest heuristic).");
        System.out.println("JSettlers bots also use SMART_STRATEGY for a fair comparison.");
        System.out.println("Total players per game = gimbur-bots + jsettlers-bots (must be 4).");
    }

    private void run() {
        int totalPlayers = gimburBots + jsettlersBots;
        if (totalPlayers != 4) {
            System.err.println("ERROR: gimbur-bots + jsettlers-bots must equal 4 (got " + totalPlayers + ")");
            System.exit(1);
        }
        if (games < 1) {
            System.err.println("ERROR: --games must be >= 1");
            System.exit(1);
        }

        long startTime = System.currentTimeMillis();

        System.out.println("=== GimburAI JSettlers Benchmark ===");
        System.out.println("Games: " + games);
        System.out.println("Gimbur bots: " + gimburBots + " (SMART_STRATEGY)");
        System.out.println("JSettlers bots: " + jsettlersBots + " (SMART_STRATEGY)");
        System.out.println("Parallel: " + parallel);
        System.out.println("Port: " + port);
        System.out.println();

        // Force all built-in JSettlers bots to use SMART_STRATEGY and disable
        // player-to-player trading.
        //
        // SMART_STRATEGY: By default, "droid N" bots get FAST_STRATEGY (weaker
        // heuristic). Overwriting ensures all bots use the same strategy.
        //
        // tradeFlag=0: Disables player-to-player trade offers and responses for
        // ALL bots. When tradeFlag=0, the brain sets doneTrading=true at the
        // start of each turn and never calls makeOffer(). Since no bot ever
        // makes an offer, no bot ever has to wait for the 5-second trade
        // response timeout (TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY). Bank/port
        // trading is NOT affected — it's not gated by tradeFlag.
        SOCRobotParameters noTradeSmartParams = new SOCRobotParameters(
                120,    // maxGameLength
                35,     // maxETA
                0.13f,  // etaBonusFactor
                1.0f,   // adversarialFactor
                1.0f,   // leaderAdversarialFactor
                3.0f,   // devCardMultiplier
                1.0f,   // threatMultiplier
                SOCRobotDM.SMART_STRATEGY,  // strategyType
                0       // tradeFlag: disabled
        );
        SOCServer.ROBOT_PARAMS_DEFAULT = noTradeSmartParams;
        SOCServer.ROBOT_PARAMS_SMARTER = noTradeSmartParams;

        // Reduce game expiry timeout for benchmark runs.
        //
        // ROOT CAUSE OF STALLS: JSettlers2 has NO timeout for bot join
        // requests — if a bot fails to complete the join-and-sit-down
        // sequence for a new game, that game sits in READY state until
        // the game expiry timer fires. The default expiry is 120 minutes,
        // which explains the observed ~2-hour stalls.
        //
        // Fix: Reduce the expiry to 2 minutes and check every 1 minute.
        // Stuck games are cleaned up quickly, freeing parallel slots for
        // new games. Normal games complete in ~10-13 seconds, so a
        // 2-minute expiry never fires for healthy games.
        SOCGameListAtServer.GAME_TIME_EXPIRE_MINUTES = 2;
        SOCServer.GAME_TIME_EXPIRE_WARN_MINUTES = 1;
        SOCServer.GAME_TIME_EXPIRE_CHECK_MINUTES = 1;

        // Configure server properties.
        // NOTE: We intentionally do NOT set botgames.shutdown=Y. That flag
        // causes the server to call System.exit(0) inside destroyGame()
        // as soon as the last game finishes — before brain threads have
        // processed their SOCGameStats messages. Instead we manage the
        // lifecycle ourselves: poll for results, then shut down cleanly.
        Properties props = new Properties();
        props.setProperty("jsettlers.startrobots", String.valueOf(jsettlersBots + 5));
        props.setProperty("jsettlers.bots.botgames.total", String.valueOf(games));
        props.setProperty("jsettlers.bots.botgames.parallel", String.valueOf(parallel));
        props.setProperty("jsettlers.bots.fast_pause_percent", "0");
        props.setProperty("jsettlers.allow.debug", "Y");
        props.setProperty("jsettlers.bots.timeout.turn", "120");

        if (gimburBots > 0) {
            props.setProperty("jsettlers.bots.start3p",
                    gimburBots + "," + GimburClient.class.getName());
            // percent3p controls what fraction of seats go to 3P bots
            int pct = (gimburBots * 100) / totalPlayers;
            props.setProperty("jsettlers.bots.percent3p", String.valueOf(pct));
            props.setProperty("jsettlers.bots.cookie", "gimbur");
            props.setProperty("jsettlers.bots.botgames.wait_sec", "2");
        }

        // Start server in-process
        System.out.println("Starting JSettlers server on port " + port + "...");
        SOCServer server = null;
        try {
            server = new SOCServer(port, props);
            server.setPriority(5);
            server.start();  // non-blocking; server runs in its own thread

            System.out.println("Server started. Running " + games + " bot-only games...");
            System.out.println();

        } catch (Exception e) {
            System.err.println("ERROR: Failed to start server: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // Poll until all game results have been captured by GimburBrain.handleGAMESTATS.
        // The server stays alive (no botgames.shutdown), so brain threads have
        // unlimited time to process their queued SOCGameStats messages.
        //
        // Timeout: With the 2-minute game expiry, stuck games are cleaned up
        // quickly. Normal games take ~10-13s. Allow 30s per game as baseline
        // plus 5 minutes of slack for occasional stalls and startup overhead.
        long timeoutMs = (long) games * 30_000L + 300_000L;
        long deadline = System.currentTimeMillis() + timeoutMs;
        int lastReportedCount = 0;
        long lastProgressTime = System.currentTimeMillis();
        while (resultsByGame.size() < games && System.currentTimeMillis() < deadline) {
            try { Thread.sleep(500); } catch (InterruptedException e) { break; }

            int currentCount = resultsByGame.size();
            long now = System.currentTimeMillis();

            // Report progress every 30 seconds or when new results arrive in bulk
            if (currentCount > lastReportedCount && currentCount - lastReportedCount >= 5) {
                double elapsed = (now - startTime) / 1000.0;
                System.out.printf("  Progress: %d / %d games (%.0fs elapsed)%n",
                        currentCount, games, elapsed);
                lastReportedCount = currentCount;
                lastProgressTime = now;
            } else if (now - lastProgressTime > 30_000 && currentCount < games) {
                double elapsed = (now - startTime) / 1000.0;
                long remaining = (deadline - now) / 1000;
                System.out.printf("  Waiting: %d / %d games (%.0fs elapsed, %ds until timeout)%n",
                        currentCount, games, elapsed, remaining);
                lastProgressTime = now;
            }
        }

        // Print results
        printSummary(startTime);
        if (outputFile != null) {
            exportResults(startTime);
        }

        // Shut down the server cleanly
        server.stopServer();
        System.exit(0);
    }

    private void printSummary(long startTime) {
        long elapsed = System.currentTimeMillis() - startTime;
        List<GameResult> results = new ArrayList<>(resultsByGame.values());
        int totalGames = results.size();

        System.out.println();
        System.out.println("=== Benchmark Results ===");
        System.out.println();

        if (totalGames == 0) {
            System.out.println("No games completed.");
            return;
        }

        // Count wins
        int gimburWins = 0;
        int jsettlersWins = 0;
        int totalRounds = 0;

        for (GameResult r : results) {
            if (r.winnerIsGimbur) {
                gimburWins++;
            } else {
                jsettlersWins++;
            }
            totalRounds += r.rounds;
        }

        double gimburWinRate = (double) gimburWins / totalGames * 100;
        double jsettlersWinRate = (double) jsettlersWins / totalGames * 100;
        double avgRounds = (double) totalRounds / totalGames;

        System.out.printf("Games completed: %d / %d%n", totalGames, games);
        System.out.printf("Total time: %.1f s%n", elapsed / 1000.0);
        System.out.printf("Avg time per game: %.1f s%n", elapsed / 1000.0 / totalGames);
        System.out.printf("Avg rounds per game: %.1f%n", avgRounds);
        System.out.println();
        System.out.println("--- Win Rates ---");
        System.out.printf("Gimbur (SMART):     %d / %d  (%.1f%%)%n",
                gimburWins, totalGames, gimburWinRate);
        System.out.printf("JSettlers (SMART):  %d / %d  (%.1f%%)%n",
                jsettlersWins, totalGames, jsettlersWinRate);

        // Per-seat analysis
        int[] seatWins = new int[4];
        int[] seatGames = new int[4];
        int[] seatGimburWins = new int[4];
        int[] seatGimburGames = new int[4];

        for (GameResult r : results) {
            for (int s = 0; s < 4; s++) {
                seatGames[s]++;
                if (r.playerIsGimbur[s]) {
                    seatGimburGames[s]++;
                }
            }
            // Find winning seat
            for (int s = 0; s < 4; s++) {
                if (r.playerNames[s] != null && r.playerNames[s].equals(r.winnerName)) {
                    seatWins[s]++;
                    if (r.winnerIsGimbur) {
                        seatGimburWins[s]++;
                    }
                    break;
                }
            }
        }

        System.out.println();
        System.out.println("--- Win Rate by Seat ---");
        for (int s = 0; s < 4; s++) {
            double rate = seatGames[s] > 0 ? (double) seatWins[s] / seatGames[s] * 100 : 0;
            System.out.printf("Seat %d: %d / %d wins (%.1f%%)%n", s, seatWins[s], seatGames[s], rate);
        }

        if (verbose) {
            System.out.println();
            System.out.println("--- Per-Game Details ---");
            for (int i = 0; i < results.size(); i++) {
                GameResult r = results.get(i);
                System.out.printf("Game %d: winner=%s (gimbur=%s) vp=%d rounds=%d time=%ds%n",
                        i + 1, r.winnerName, r.winnerIsGimbur, r.winnerVP,
                        r.rounds, r.durationSeconds);
            }
        }
    }

    private void exportResults(long startTime) {
        long elapsed = System.currentTimeMillis() - startTime;
        List<GameResult> results = new ArrayList<>(resultsByGame.values());
        int totalGames = results.size();

        if (totalGames == 0) return;

        int gimburWins = 0;
        for (GameResult r : results) {
            if (r.winnerIsGimbur) gimburWins++;
        }

        // Build JSON manually (avoiding Gson dependency for the CLI output)
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"totalGames\": ").append(totalGames).append(",\n");
        sb.append("  \"targetGames\": ").append(games).append(",\n");
        sb.append("  \"elapsedMs\": ").append(elapsed).append(",\n");
        sb.append("  \"gimburBots\": ").append(gimburBots).append(",\n");
        sb.append("  \"jsettlersBots\": ").append(jsettlersBots).append(",\n");
        sb.append("  \"gimburWins\": ").append(gimburWins).append(",\n");
        sb.append("  \"jsettlersWins\": ").append(totalGames - gimburWins).append(",\n");
        sb.append("  \"gimburWinRate\": ").append(String.format("%.4f", (double) gimburWins / totalGames)).append(",\n");
        sb.append("  \"games\": [\n");

        for (int i = 0; i < results.size(); i++) {
            GameResult r = results.get(i);
            sb.append("    {\n");
            sb.append("      \"game\": ").append(i + 1).append(",\n");
            sb.append("      \"winner\": \"").append(escapeJson(r.winnerName)).append("\",\n");
            sb.append("      \"winnerIsGimbur\": ").append(r.winnerIsGimbur).append(",\n");
            sb.append("      \"winnerVP\": ").append(r.winnerVP).append(",\n");
            sb.append("      \"rounds\": ").append(r.rounds).append(",\n");
            sb.append("      \"durationSeconds\": ").append(r.durationSeconds).append(",\n");
            sb.append("      \"players\": [\n");
            for (int s = 0; s < 4; s++) {
                sb.append("        {\"seat\": ").append(s);
                sb.append(", \"name\": \"").append(escapeJson(r.playerNames[s] != null ? r.playerNames[s] : ""));
                sb.append("\", \"vp\": ").append(r.playerVPs[s]);
                sb.append(", \"isGimbur\": ").append(r.playerIsGimbur[s]).append("}");
                if (s < 3) sb.append(",");
                sb.append("\n");
            }
            sb.append("      ]\n");
            sb.append("    }");
            if (i < results.size() - 1) sb.append(",");
            sb.append("\n");
        }

        sb.append("  ]\n");
        sb.append("}\n");

        try (FileWriter fw = new FileWriter(outputFile)) {
            fw.write(sb.toString());
            System.out.println("Results written to " + outputFile);
        } catch (IOException e) {
            System.err.println("ERROR: Failed to write results: " + e.getMessage());
        }
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
