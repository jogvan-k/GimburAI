package gimbur.jsettlers;

import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.message.SOCMessage;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * Third-party bot client for JSettlers2 that uses a GimburBrain.
 *
 * <p>Phase 1 (v1): GimburBrain uses SMART_STRATEGY — identical to JSettlers'
 * strongest built-in heuristic. This establishes a baseline.
 *
 * <p>Phase 2 (v2): GimburBrain will delegate decisions to Gimbur.Server
 * (MCTS engine over HTTP).
 *
 * <p>Game results are captured by {@link GimburBrain#handleGAMESTATS} rather
 * than here, to avoid race conditions with {@code SOCRobotDismiss} cleanup.
 */
public class GimburClient extends SOCRobotClient {

    /** Robot class name reported to the server. */
    private static final String RBCLASSNAME = GimburClient.class.getName();

    /**
     * Create a GimburClient.
     *
     * @param sci  server connection info (host, port, cookie)
     * @param nn   bot nickname
     * @param pw   bot password
     */
    public GimburClient(final ServerConnectInfo sci, final String nn, final String pw)
            throws IllegalArgumentException {
        super(sci, nn, pw);
        rbclass = RBCLASSNAME;
    }

    /**
     * Factory method: create a GimburBrain for each game this client joins.
     */
    @Override
    public SOCRobotBrain createBrain(
            final SOCRobotParameters params,
            final SOCGame ga,
            final CappedQueue<SOCMessage> mq) {
        return new GimburBrain(this, params, ga, mq);
    }

    /**
     * Entry point for running the bot client standalone.
     * Args: hostname port botname password cookie
     */
    public static void main(String[] args) {
        if (args.length < 5) {
            System.err.println("Usage: GimburClient hostname port botname password cookie");
            System.exit(1);
        }

        ServerConnectInfo sci = new ServerConnectInfo(
                args[0], Integer.parseInt(args[1]), args[4]);
        GimburClient client = new GimburClient(sci, args[2], args[3]);
        client.init();
    }
}
