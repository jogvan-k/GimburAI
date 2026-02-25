using System.Text;
using Gimbur.Rules;

namespace Gimbur;

/// <summary>
/// Handles serialization and deserialization of <see cref="CatanState"/>
/// in both human-readable (section-delimited) and compact (fixed-length) forms.
/// </summary>
internal static class CatanStateSerializer
{
    public static string SerializeHumanReadable(CatanState state)
    {
        var sb = new StringBuilder(256);

        // Section 1: Tiles — resource/pip pairs concatenated (no separators)
        for (var ti = 0; ti < state.Board.Topology.TileCount; ti++)
        {
            sb.Append(CrockfordBase32.Encode((int)state.Board.TileResource(ti)));
            sb.Append(TilePip.Encode(state.Board.TileNumber(ti)));
        }

        // Section 2: Robber
        sb.Append('|');
        sb.Append(CrockfordBase32.Encode(state.Board.RobberTile));

        // Section 3: Current Turn (player + stage, concatenated)
        sb.Append('|');
        sb.Append(CrockfordBase32.Encode(state.CurrentPlayer));
        sb.Append(CrockfordBase32.Encode((int)state.Stage));

        // Section 4: Longest Road / Largest Army (concatenated)
        sb.Append('|');
        sb.Append(CrockfordBase32.Encode(state.LongestRoadOwner));
        sb.Append(CrockfordBase32.Encode(state.LargestArmyOwner));

        // Section 5: Vertices (concatenated)
        sb.Append('|');
        for (var vi = 0; vi < state.Board.Topology.VertexCount; vi++)
        {
            sb.Append(CrockfordBase32.Encode(state.Board.VertexOccupancy[vi].ToToken()));
        }

        // Section 6: Edges (concatenated)
        sb.Append('|');
        for (var ei = 0; ei < state.Board.Topology.EdgeCount; ei++)
        {
            sb.Append(CrockfordBase32.Encode(state.Board.EdgeOccupancy[ei].ToToken()));
        }

        // Section 7: Ports (concatenated)
        sb.Append('|');
        for (var pi = 0; pi < state.Board.Topology.PortCount; pi++)
        {
            sb.Append(CrockfordBase32.Encode((int)state.Board.PortType(pi)));
        }

        // Section 8: Per-Player Resources (5 chars per player, '/' between players)
        sb.Append('|');
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            if (player > 1)
            {
                sb.Append('/');
            }

            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Wood)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Brick)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Sheep)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Wheat)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Ore)]));
        }

        // Section 9: Per-Player Knights Played ('/' between players)
        sb.Append('|');
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            if (player > 1)
            {
                sb.Append('/');
            }

            sb.Append(CrockfordBase32.Encode(state._knightsPlayed[player]));
        }

        // Section 10: Per-Player Dev Cards (5 chars per player, '/' between players)
        sb.Append('|');
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            if (player > 1)
            {
                sb.Append('/');
            }

            for (var card = 0; card < CatanState.DevCardCount; card++)
            {
                sb.Append(CrockfordBase32.Encode(state._devCards[player, card]));
            }
        }

        return sb.ToString();
    }

    public static CatanState DeserializeHumanReadable(
        GameConfig config,
        int playerCount,
        string serialized)
    {
        if (string.IsNullOrWhiteSpace(serialized))
        {
            throw new ArgumentException("Serialized state cannot be empty.", nameof(serialized));
        }

        // Format: tiles|robber|currentTurn|longestArmy|vertices|edges|ports|resources|knights|devCards
        var sections = serialized.Split('|');
        if (sections.Length != 10)
        {
            throw new InvalidOperationException(
                $"Serialized state has {sections.Length} sections, expected 10.");
        }

        var topology = config.Map.Topology;

        // Section 1: Tiles — resource/pip pairs concatenated (2*T chars)
        var tileSection = sections[0];
        if (tileSection.Length != topology.TileCount * 2)
        {
            throw new InvalidOperationException(
                $"Tile section has {tileSection.Length} chars, expected {topology.TileCount * 2}.");
        }

        var tileResources = new ResourceType[topology.TileCount];
        var tileNumbers = new int[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tileResources[ti] = (ResourceType)CrockfordBase32.Decode(tileSection[ti * 2]);
            tileNumbers[ti] = TilePip.Decode(tileSection[(ti * 2) + 1]);
        }

        // Section 2: Robber (single char)
        var robberTile = CrockfordBase32.Decode(sections[1][0]);

        // Section 3: Current Turn (2 chars: player + stage)
        var currentPlayer = CrockfordBase32.Decode(sections[2][0]);
        var stage = (TurnStage)CrockfordBase32.Decode(sections[2][1]);

        // Section 4: Longest Road / Largest Army (2 chars)
        var longestRoadOwner = CrockfordBase32.Decode(sections[3][0]);
        var largestArmyOwner = CrockfordBase32.Decode(sections[3][1]);

        // Section 5: Vertices (concatenated single chars)
        if (sections[4].Length != topology.VertexCount)
        {
            throw new InvalidOperationException(
                $"Vertex section has {sections[4].Length} chars, expected {topology.VertexCount}.");
        }

        var vertices = new VertexOccupancy[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            vertices[vi] = VertexOccupancy.FromToken(CrockfordBase32.Decode(sections[4][vi]));
        }

        // Section 6: Edges (concatenated single chars)
        if (sections[5].Length != topology.EdgeCount)
        {
            throw new InvalidOperationException(
                $"Edge section has {sections[5].Length} chars, expected {topology.EdgeCount}.");
        }

        var edges = new EdgeOccupancy[topology.EdgeCount];
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            edges[ei] = EdgeOccupancy.FromToken(CrockfordBase32.Decode(sections[5][ei]));
        }

        // Section 7: Ports (concatenated single chars)
        if (sections[6].Length != topology.PortCount)
        {
            throw new InvalidOperationException(
                $"Port section has {sections[6].Length} chars, expected {topology.PortCount}.");
        }

        var ports = new PortType[topology.PortCount];
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            ports[pi] = (PortType)CrockfordBase32.Decode(sections[6][pi]);
        }

        // Section 8: Per-Player Resources (5 chars per player, '/' between players)
        var resourceGroups = sections[7].Split('/');
        if (resourceGroups.Length != playerCount)
        {
            throw new InvalidOperationException(
                $"Resource section has {resourceGroups.Length} player groups, expected {playerCount}.");
        }

        var resources = new int[playerCount + 1, CatanState.ResourceCount];
        for (var player = 1; player <= playerCount; player++)
        {
            var group = resourceGroups[player - 1];
            resources[player, CatanState.ResourceToIndex(ResourceType.Wood)] = CrockfordBase32.Decode(group[0]);
            resources[player, CatanState.ResourceToIndex(ResourceType.Brick)] = CrockfordBase32.Decode(group[1]);
            resources[player, CatanState.ResourceToIndex(ResourceType.Sheep)] = CrockfordBase32.Decode(group[2]);
            resources[player, CatanState.ResourceToIndex(ResourceType.Wheat)] = CrockfordBase32.Decode(group[3]);
            resources[player, CatanState.ResourceToIndex(ResourceType.Ore)] = CrockfordBase32.Decode(group[4]);
        }

        // Section 9: Per-Player Knights Played ('/' between players)
        var knightGroups = sections[8].Split('/');
        if (knightGroups.Length != playerCount)
        {
            throw new InvalidOperationException(
                $"Knights section has {knightGroups.Length} player groups, expected {playerCount}.");
        }

        var knightsPlayed = new int[playerCount + 1];
        for (var player = 1; player <= playerCount; player++)
        {
            knightsPlayed[player] = CrockfordBase32.Decode(knightGroups[player - 1][0]);
        }

        // Section 10: Per-Player Dev Cards (5 chars per player, '/' between players)
        var devCardGroups = sections[9].Split('/');
        if (devCardGroups.Length != playerCount)
        {
            throw new InvalidOperationException(
                $"Dev card section has {devCardGroups.Length} player groups, expected {playerCount}.");
        }

        var devCards = new int[playerCount + 1, CatanState.DevCardCount];
        for (var player = 1; player <= playerCount; player++)
        {
            var group = devCardGroups[player - 1];
            for (var card = 0; card < CatanState.DevCardCount; card++)
            {
                devCards[player, card] = CrockfordBase32.Decode(group[card]);
            }
        }

        var setup = new BoardSetup(topology, [.. tileResources], [.. tileNumbers], [.. ports], robberTile);
        var board = new Board(setup, config);
        Array.Copy(vertices, board.VertexOccupancy, vertices.Length);
        Array.Copy(edges, board.EdgeOccupancy, edges.Length);
        board.RobberTile = robberTile;

        var pendingSettlement = InferPendingSettlementVertex(board, currentPlayer, stage);
        var turnNumber = stage is TurnStage.PreRoll or TurnStage.ChooseRobberLocation or TurnStage.ChooseRobberVictim or TurnStage.BuildTrade ? 1 : 0;

        var deck = new int[CatanState.DevCardCount];
        foreach (var pair in config.DevCardCounts)
        {
            deck[(int)pair.Key] = pair.Value;
        }

        for (var player = 1; player <= playerCount; player++)
        {
            for (var card = 0; card < CatanState.DevCardCount; card++)
            {
                deck[card] -= devCards[player, card];
            }
        }

        for (var card = 0; card < CatanState.DevCardCount; card++)
        {
            if (deck[card] < 0)
            {
                throw new InvalidOperationException("Serialized dev card counts exceed deck size.");
            }
        }

        var state = new CatanState(
            config,
            board,
            playerCount,
            currentPlayer,
            stage,
            turnNumber,
            pendingSettlement,
            longestRoadOwner,
            largestArmyOwner,
            winnerPlayer: 0,
            resources,
            knightsPlayed,
            devCards,
            deck,
            new int[CatanState.DevCardCount],
            new int[playerCount + 1]);

        state.RefreshVictory();
        return state;
    }

    /// <summary>
    /// Produces the compact form: strips all '/' and '|' separators from the
    /// human-readable form, yielding a fixed-length Crockford base-32 string.
    /// </summary>
    public static string SerializeCompact(CatanState state)
    {
        var humanReadable = SerializeHumanReadable(state);
        var sb = new StringBuilder(humanReadable.Length);
        foreach (var c in humanReadable)
        {
            if (c is not '/' and not '|')
            {
                sb.Append(c);
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Parses the compact form by re-inserting separators at the known fixed
    /// positions and delegating to <see cref="DeserializeHumanReadable"/>.
    /// </summary>
    public static CatanState DeserializeCompact(
        GameConfig config,
        int playerCount,
        string compact)
    {
        if (string.IsNullOrWhiteSpace(compact))
        {
            throw new ArgumentException("Compact state cannot be empty.", nameof(compact));
        }

        var topology = config.Map.Topology;
        var pos = 0;

        // Section 1: Tiles — 2*TileCount chars, concatenated directly (no separators)
        var tileLen = topology.TileCount * 2;
        var sb = new StringBuilder(compact.Length + 32);
        for (var i = 0; i < tileLen; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 2: Robber — 1 char
        sb.Append('|');
        sb.Append(compact[pos++]);

        // Section 3: Current Turn — 2 chars
        sb.Append('|');
        sb.Append(compact[pos++]);
        sb.Append(compact[pos++]);

        // Section 4: Longest Road / Largest Army — 2 chars
        sb.Append('|');
        sb.Append(compact[pos++]);
        sb.Append(compact[pos++]);

        // Section 5: Vertices — VertexCount chars
        sb.Append('|');
        for (var i = 0; i < topology.VertexCount; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 6: Edges — EdgeCount chars
        sb.Append('|');
        for (var i = 0; i < topology.EdgeCount; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 7: Ports — PortCount chars
        sb.Append('|');
        for (var i = 0; i < topology.PortCount; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 8: Per-Player Resources — 5 chars per player, '/' between players
        sb.Append('|');
        for (var player = 0; player < playerCount; player++)
        {
            if (player > 0)
            {
                sb.Append('/');
            }

            for (var i = 0; i < CatanState.ResourceCount; i++)
            {
                sb.Append(compact[pos++]);
            }
        }

        // Section 9: Per-Player Knights — 1 char per player, '/' between players
        sb.Append('|');
        for (var player = 0; player < playerCount; player++)
        {
            if (player > 0)
            {
                sb.Append('/');
            }

            sb.Append(compact[pos++]);
        }

        // Section 10: Per-Player Dev Cards — 5 chars per player, '/' between players
        sb.Append('|');
        for (var player = 0; player < playerCount; player++)
        {
            if (player > 0)
            {
                sb.Append('/');
            }

            for (var i = 0; i < CatanState.DevCardCount; i++)
            {
                sb.Append(compact[pos++]);
            }
        }

        return DeserializeHumanReadable(config, playerCount, sb.ToString());
    }

    /// <summary>
    /// Serializes the board-invariant portion: tiles (section 1) and ports (section 7),
    /// separated by '|'. This is stable across turns within a single game.
    /// Format: "{tile_chars}|{port_chars}"
    /// </summary>
    public static string SerializeBoard(CatanState state)
    {
        var topology = state.Board.Topology;
        var sb = new StringBuilder((topology.TileCount * 2) + 1 + topology.PortCount);

        // Tiles — resource/pip pairs concatenated
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            sb.Append(CrockfordBase32.Encode((int)state.Board.TileResource(ti)));
            sb.Append(TilePip.Encode(state.Board.TileNumber(ti)));
        }

        // Ports
        sb.Append('|');
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            sb.Append(CrockfordBase32.Encode((int)state.Board.PortType(pi)));
        }

        return sb.ToString();
    }

    /// <summary>
    /// Serializes the turn-specific state (sections 2–6, 8–10) in compact form
    /// (no separators). This excludes tiles and ports which are board-invariant.
    /// Layout: [robber:1][player:1][stage:1][longestRoad:1][largestArmy:1]
    ///         [vertices:V][edges:E][resources:5*N][knights:N][devCards:5*N]
    /// </summary>
    public static string SerializeStateOnly(CatanState state)
    {
        var topology = state.Board.Topology;
        var playerCount = state.PlayerCount;
        var capacity = 1 + 2 + 2 + topology.VertexCount + topology.EdgeCount
                       + (5 * playerCount) + playerCount + (5 * playerCount);
        var sb = new StringBuilder(capacity);

        // Section 2: Robber
        sb.Append(CrockfordBase32.Encode(state.Board.RobberTile));

        // Section 3: Current Turn (player + stage)
        sb.Append(CrockfordBase32.Encode(state.CurrentPlayer));
        sb.Append(CrockfordBase32.Encode((int)state.Stage));

        // Section 4: Longest Road / Largest Army
        sb.Append(CrockfordBase32.Encode(state.LongestRoadOwner));
        sb.Append(CrockfordBase32.Encode(state.LargestArmyOwner));

        // Section 5: Vertices
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            sb.Append(CrockfordBase32.Encode(state.Board.VertexOccupancy[vi].ToToken()));
        }

        // Section 6: Edges
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            sb.Append(CrockfordBase32.Encode(state.Board.EdgeOccupancy[ei].ToToken()));
        }

        // Section 8: Per-Player Resources (concatenated, no '/' separators)
        for (var player = 1; player <= playerCount; player++)
        {
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Wood)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Brick)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Sheep)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Wheat)]));
            sb.Append(CrockfordBase32.Encode(state._resources[player, CatanState.ResourceToIndex(ResourceType.Ore)]));
        }

        // Section 9: Per-Player Knights Played (concatenated)
        for (var player = 1; player <= playerCount; player++)
        {
            sb.Append(CrockfordBase32.Encode(state._knightsPlayed[player]));
        }

        // Section 10: Per-Player Dev Cards (concatenated)
        for (var player = 1; player <= playerCount; player++)
        {
            for (var card = 0; card < CatanState.DevCardCount; card++)
            {
                sb.Append(CrockfordBase32.Encode(state._devCards[player, card]));
            }
        }

        return sb.ToString();
    }

    private static int? InferPendingSettlementVertex(Board board, int currentPlayer, TurnStage stage)
    {
        if (stage is not (TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad))
        {
            return null;
        }

        var candidates = new List<int>();
        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            var occ = board.VertexOccupancy[vi];
            if (occ.Building != BuildingType.Settlement || occ.Player != currentPlayer)
            {
                continue;
            }

            var hasOwnRoad = false;
            foreach (var edge in board.Topology.VertexEdges[vi])
            {
                if (board.EdgeOccupancy[edge].Player == currentPlayer)
                {
                    hasOwnRoad = true;
                    break;
                }
            }

            if (!hasOwnRoad)
            {
                candidates.Add(vi);
            }
        }

        return candidates.Count == 1 ? candidates[0] : null;
    }
}
