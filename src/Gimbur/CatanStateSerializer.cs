using System.Text;
using Gimbur.Rules;

namespace Gimbur;

/// <summary>
/// Handles serialization and deserialization of <see cref="CatanState"/>
/// in both human-readable (section-delimited) and compact (fixed-length) forms.
/// Uses semantically disjoint character alphabets as defined in
/// <c>docs/state-serialization.md</c>.
/// </summary>
internal static class CatanStateSerializer
{
    public static string SerializeHumanReadable(CatanState state)
    {
        var sb = new StringBuilder(320);

        // Section 1: Tiles — 3 tokens each: resource + pips + side (no separators)
        for (var ti = 0; ti < state.Board.Topology.TileCount; ti++)
        {
            sb.Append(StateToken.EncodeResource(state.Board.TileResource(ti)));
            sb.Append(StateToken.EncodeTilePips(state.Board.TileNumber(ti)));
            sb.Append(StateToken.EncodeTileSide(state.Board.TileNumber(ti)));
        }

        // Section 2: Ports
        sb.Append('|');
        for (var pi = 0; pi < state.Board.Topology.PortCount; pi++)
        {
            sb.Append(StateToken.EncodePort(state.Board.PortType(pi)));
        }

        // Section 3: Robber
        sb.Append('|');
        sb.Append(CrockfordBase32.Encode(state.Board.RobberTile));

        // Section 4: Current Turn (player + stage, concatenated)
        sb.Append('|');
        sb.Append(StateToken.EncodePlayer(state.CurrentPlayer));
        sb.Append(StateToken.EncodeTurnStage(state.Stage));

        // Section 5: Longest Road / Largest Army (concatenated)
        sb.Append('|');
        sb.Append(StateToken.EncodePlayer(state.LongestRoadOwner));
        sb.Append(StateToken.EncodePlayer(state.LargestArmyOwner));

        // Section 6: Vertices — 2 tokens each: building + player
        sb.Append('|');
        for (var vi = 0; vi < state.Board.Topology.VertexCount; vi++)
        {
            var occ = state.Board.VertexOccupancy[vi];
            sb.Append(StateToken.EncodeBuilding(occ.Building));
            sb.Append(StateToken.EncodePlayer(occ.Player));
        }

        // Section 7: Edges — 1 player ID token each
        sb.Append('|');
        for (var ei = 0; ei < state.Board.Topology.EdgeCount; ei++)
        {
            sb.Append(StateToken.EncodePlayer(state.Board.EdgeOccupancy[ei].Player));
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

        // Format: tiles|ports|robber|currentTurn|longestArmy|vertices|edges|resources|knights|devCards
        var sections = serialized.Split('|');
        if (sections.Length != 10)
        {
            throw new InvalidOperationException(
                $"Serialized state has {sections.Length} sections, expected 10.");
        }

        var topology = config.Map.Topology;

        // Section 1: Tiles — 3 tokens each: resource + pips + side (3*T chars)
        var tileSection = sections[0];
        if (tileSection.Length != topology.TileCount * 3)
        {
            throw new InvalidOperationException(
                $"Tile section has {tileSection.Length} chars, expected {topology.TileCount * 3}.");
        }

        var tileResources = new ResourceType[topology.TileCount];
        var tileNumbers = new int[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tileResources[ti] = StateToken.DecodeResource(tileSection[ti * 3]);
            tileNumbers[ti] = StateToken.DecodeTileNumber(tileSection[(ti * 3) + 1], tileSection[(ti * 3) + 2]);
        }

        // Section 2: Ports
        var portSection = sections[1];
        if (portSection.Length != topology.PortCount)
        {
            throw new InvalidOperationException(
                $"Port section has {portSection.Length} chars, expected {topology.PortCount}.");
        }

        var ports = new PortType[topology.PortCount];
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            ports[pi] = StateToken.DecodePort(portSection[pi]);
        }

        // Section 3: Robber (single char)
        var robberTile = CrockfordBase32.Decode(sections[2][0]);

        // Section 4: Current Turn (2 chars: player + stage)
        var currentPlayer = StateToken.DecodePlayer(sections[3][0]);
        var stage = StateToken.DecodeTurnStage(sections[3][1]);

        // Section 5: Longest Road / Largest Army (2 chars)
        var longestRoadOwner = StateToken.DecodePlayer(sections[4][0]);
        var largestArmyOwner = StateToken.DecodePlayer(sections[4][1]);

        // Section 6: Vertices — 2 chars each (building + player)
        if (sections[5].Length != topology.VertexCount * 2)
        {
            throw new InvalidOperationException(
                $"Vertex section has {sections[5].Length} chars, expected {topology.VertexCount * 2}.");
        }

        var vertices = new VertexOccupancy[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var building = StateToken.DecodeBuilding(sections[5][vi * 2]);
            var player = StateToken.DecodePlayer(sections[5][(vi * 2) + 1]);
            vertices[vi] = new VertexOccupancy(building, player);
        }

        // Section 7: Edges — 1 char each (player ID)
        if (sections[6].Length != topology.EdgeCount)
        {
            throw new InvalidOperationException(
                $"Edge section has {sections[6].Length} chars, expected {topology.EdgeCount}.");
        }

        var edges = new EdgeOccupancy[topology.EdgeCount];
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            edges[ei] = new EdgeOccupancy(StateToken.DecodePlayer(sections[6][ei]));
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
    /// human-readable form, yielding a fixed-length token string.
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

        // Section 1: Tiles — 3*TileCount chars
        var tileLen = topology.TileCount * 3;
        var sb = new StringBuilder(compact.Length + 32);
        for (var i = 0; i < tileLen; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 2: Ports — PortCount chars
        sb.Append('|');
        for (var i = 0; i < topology.PortCount; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 3: Robber — 1 char
        sb.Append('|');
        sb.Append(compact[pos++]);

        // Section 4: Current Turn — 2 chars
        sb.Append('|');
        sb.Append(compact[pos++]);
        sb.Append(compact[pos++]);

        // Section 5: Longest Road / Largest Army — 2 chars
        sb.Append('|');
        sb.Append(compact[pos++]);
        sb.Append(compact[pos++]);

        // Section 6: Vertices — 2*VertexCount chars
        sb.Append('|');
        for (var i = 0; i < topology.VertexCount * 2; i++)
        {
            sb.Append(compact[pos++]);
        }

        // Section 7: Edges — EdgeCount chars
        sb.Append('|');
        for (var i = 0; i < topology.EdgeCount; i++)
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
    /// Serializes the board-invariant portion: tiles (section 1) and ports (section 2),
    /// separated by '|'. This is stable across turns within a single game.
    /// Format: "{tile_chars}|{port_chars}"
    /// </summary>
    public static string SerializeBoard(CatanState state)
    {
        var topology = state.Board.Topology;
        var sb = new StringBuilder((topology.TileCount * 3) + 1 + topology.PortCount);

        // Tiles — resource + pips + side per tile
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            sb.Append(StateToken.EncodeResource(state.Board.TileResource(ti)));
            sb.Append(StateToken.EncodeTilePips(state.Board.TileNumber(ti)));
            sb.Append(StateToken.EncodeTileSide(state.Board.TileNumber(ti)));
        }

        // Ports
        sb.Append('|');
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            sb.Append(StateToken.EncodePort(state.Board.PortType(pi)));
        }

        return sb.ToString();
    }

    /// <summary>
    /// Serializes the turn-specific state (sections 3–7, 8–10) in human-readable
    /// form. This excludes tiles (section 1) and ports (section 2) which are
    /// board-invariant. Sections are separated by '|' and per-player groups
    /// within sections 8–10 are separated by '/'.
    /// Layout: robber|playerStage|longestLargest|vertices|edges|resources|knights|devCards
    /// </summary>
    public static string SerializeStateOnly(CatanState state)
    {
        var topology = state.Board.Topology;
        var playerCount = state.PlayerCount;
        // Capacity: tokens + 7 section '|' separators + player '/' separators
        var capacity = 1 + 2 + 2 + (topology.VertexCount * 2) + topology.EdgeCount
                       + (5 * playerCount) + playerCount + (5 * playerCount)
                       + 7 + (3 * (playerCount - 1));
        var sb = new StringBuilder(capacity);

        // Section 3: Robber
        sb.Append(CrockfordBase32.Encode(state.Board.RobberTile));

        // Section 4: Current Turn (player + stage)
        sb.Append('|');
        sb.Append(StateToken.EncodePlayer(state.CurrentPlayer));
        sb.Append(StateToken.EncodeTurnStage(state.Stage));

        // Section 5: Longest Road / Largest Army
        sb.Append('|');
        sb.Append(StateToken.EncodePlayer(state.LongestRoadOwner));
        sb.Append(StateToken.EncodePlayer(state.LargestArmyOwner));

        // Section 6: Vertices — 2 tokens each
        sb.Append('|');
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var occ = state.Board.VertexOccupancy[vi];
            sb.Append(StateToken.EncodeBuilding(occ.Building));
            sb.Append(StateToken.EncodePlayer(occ.Player));
        }

        // Section 7: Edges — 1 player ID token each
        sb.Append('|');
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            sb.Append(StateToken.EncodePlayer(state.Board.EdgeOccupancy[ei].Player));
        }

        // Section 8: Per-Player Resources (5 chars per player, '/' between players)
        sb.Append('|');
        for (var player = 1; player <= playerCount; player++)
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
        for (var player = 1; player <= playerCount; player++)
        {
            if (player > 1)
            {
                sb.Append('/');
            }

            sb.Append(CrockfordBase32.Encode(state._knightsPlayed[player]));
        }

        // Section 10: Per-Player Dev Cards (5 chars per player, '/' between players)
        sb.Append('|');
        for (var player = 1; player <= playerCount; player++)
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
