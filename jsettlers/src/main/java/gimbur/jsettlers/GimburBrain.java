package gimbur.jsettlers;

import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.message.SOCGameStats;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotDM;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * Gimbur bot brain for JSettlers2.
 *
 * <p><b>Phase 1 (v1):</b> Forces SMART_STRATEGY parameters regardless of what
 * the server assigns. All actual decision-making is delegated to the parent
 * {@link SOCRobotBrain} — this brain behaves identically to JSettlers'
 * strongest built-in heuristic AI. The purpose is to establish a baseline
 * win rate for benchmarking.
 *
 * <p><b>Phase 2 (v2):</b> Will override decision methods to delegate to
 * Gimbur.Server (MCTS engine over HTTP).
 */
public class GimburBrain extends SOCRobotBrain {

    /**
     * SMART_STRATEGY robot parameters with trading disabled.
     * Same as {@code SOCServer.ROBOT_PARAMS_SMARTER} but with tradeFlag = 0
     * to disable player-to-player trade offers (bank/port trading still works).
     */
    private static final SOCRobotParameters SMART_PARAMS =
            new SOCRobotParameters(
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

    /**
     * Create a GimburBrain that always uses SMART_STRATEGY.
     *
     * @param rc      the robot client
     * @param params  robot parameters from the server (will be overridden with SMART_PARAMS)
     * @param ga      the game
     * @param mq      inbound message queue
     */
    public GimburBrain(SOCRobotClient rc, SOCRobotParameters params,
                       SOCGame ga, CappedQueue<SOCMessage> mq) {
        super(rc, SMART_PARAMS, ga, mq);
    }

    /**
     * Initialize player data and strategy fields.
     * Logs that GimburBrain v1 is active.
     */
    @Override
    public void setOurPlayerData() {
        super.setOurPlayerData();

        System.out.println("GimburBrain v1 (SMART_STRATEGY) active for "
                + client.getNickname() + " in game " + game.getName());
    }

    /**
     * Capture game results when the server sends final scores.
     *
     * <p>This is the reliable hook for result capture. The alternative
     * ({@code handleDELETEGAME}) suffers from two race conditions:
     * <ol>
     *   <li>The server sends {@code SOCRobotDismiss} before {@code SOCDeleteGame}.
     *       The brain thread may process the dismiss and remove entries from the
     *       client's maps before the reader thread reaches {@code handleDELETEGAME},
     *       causing it to find {@code null} and skip the result.</li>
     *   <li>For the very last game, {@code System.exit(0)} is called inside
     *       {@code destroyGame()} before the {@code SOCDeleteGame} broadcast.</li>
     * </ol>
     *
     * <p>{@code handleGAMESTATS} fires before any dismiss/cleanup, is explicitly
     * designed for third-party bot overrides, and has the true final scores
     * (including revealed VP cards).
     *
     * @param message game stats; only {@link SOCGameStats#TYPE_PLAYERS} at game end
     */
    @Override
    protected void handleGAMESTATS(SOCGameStats message) {
        super.handleGAMESTATS(message);

        if (message.getStatType() != SOCGameStats.TYPE_PLAYERS)
            return;

        // Only capture once per game (the first Gimbur brain to see it wins the putIfAbsent)
        if (BenchmarkCli.resultsByGame.containsKey(game.getName()))
            return;

        long[] scores = message.getScores();
        boolean[] robotSeats = message.getRobotSeats();

        // Find winner: highest score
        int winnerPN = -1;
        long highScore = -1;
        for (int pn = 0; pn < game.maxPlayers; pn++) {
            if (game.isSeatVacant(pn))
                continue;
            if (scores[pn] > highScore) {
                highScore = scores[pn];
                winnerPN = pn;
            }
        }

        if (winnerPN < 0)
            return;

        SOCPlayer winner = game.getPlayer(winnerPN);
        String[] playerNames = new String[4];
        int[] playerVPs = new int[4];
        boolean[] playerIsGimbur = new boolean[4];

        for (int pn = 0; pn < game.maxPlayers; pn++) {
            if (game.isSeatVacant(pn))
                continue;
            SOCPlayer p = game.getPlayer(pn);
            playerNames[pn] = p.getName();
            playerVPs[pn] = (int) scores[pn];
            // Gimbur bots are named "extrabot N" by the server
            playerIsGimbur[pn] = p.getName() != null && p.getName().startsWith("extrabot ");
        }

        boolean winnerIsGimbur = winner.getName() != null
                && winner.getName().startsWith("extrabot ");

        BenchmarkCli.GameResult result = new BenchmarkCli.GameResult(
                game.getName(),
                winner.getName(),
                winnerIsGimbur,
                (int) highScore,
                game.getRoundCount(),
                game.getDurationSeconds(),
                playerNames,
                playerVPs,
                playerIsGimbur
        );

        BenchmarkCli.resultsByGame.putIfAbsent(game.getName(), result);

        // Print progress
        int n = BenchmarkCli.resultsByGame.size();
        if (n % 10 == 0 || n <= 5) {
            int gimburWins = 0;
            for (BenchmarkCli.GameResult r : BenchmarkCli.resultsByGame.values()) {
                if (r.winnerIsGimbur) gimburWins++;
            }
            System.out.printf("[%d games] Gimbur wins: %d (%.1f%%)%n",
                    n, gimburWins, (double) gimburWins / n * 100);
        }
    }

    /**
     * No-op override: skip all artificial delays for maximum benchmark speed.
     * The server property {@code fast_pause_percent=0} achieves the same for
     * built-in bots, but overriding here makes GimburBrain unconditionally fast
     * regardless of server configuration.
     */
    @Override
    public void pause(int msec) {
        // intentionally empty
    }

    /**
     * Reject all trade offers. With tradeFlag=0 this method is normally
     * never called (the brain ignores offers entirely), but we keep the
     * override as a safety net in case the trade flag is changed.
     *
     * @param offer  the trade offer to consider
     * @return {@link SOCRobotNegotiator#REJECT_OFFER} always
     */
    @Override
    public int considerOffer(soc.game.SOCTradeOffer offer) {
        return soc.robot.SOCRobotNegotiator.REJECT_OFFER;
    }
}
