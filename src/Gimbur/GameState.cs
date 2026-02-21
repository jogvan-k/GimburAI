using System.Text;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur;

public enum CatanActionType : byte
{
    PlaceSettlement = 0,
    PlaceRoad = 1,
    RollDice = 2,
    ChooseRobberTile = 3,
    BuildCity = 4,
    BankTrade = 5,
    BuyDevCard = 6,
    PlayKnight = 7,
    PlayRoadBuilding = 8,
    PlayMonopoly = 9,
    PlayYearOfPlenty = 10,
    EndTurn = 11,
}

public sealed class CatanAction : ICoreAction
{
    public CatanAction(CatanState origin, CatanActionType actionType, int arg1 = 0, int arg2 = 0)
    {
        OriginState = origin;
        ActionType = actionType;
        Arg1 = arg1;
        Arg2 = arg2;
    }

    public CatanState OriginState { get; }

    public CatanActionType ActionType { get; }

    public int Arg1 { get; }

    public int Arg2 { get; }

    public int TargetIndex => Arg1;

    public ICoreState Origin => OriginState;

    public ICoreState DoCoreAction() => OriginState.Apply(this);

    public int CompareTo(object? obj)
    {
        if (obj is not CatanAction other)
        {
            throw new ArgumentException("Cannot compare CatanAction with a different type", nameof(obj));
        }

        var typeCompare = ActionType.CompareTo(other.ActionType);
        if (typeCompare != 0)
        {
            return typeCompare;
        }

        var arg1Compare = Arg1.CompareTo(other.Arg1);
        if (arg1Compare != 0)
        {
            return arg1Compare;
        }

        return Arg2.CompareTo(other.Arg2);
    }

    public override bool Equals(object? obj)
    {
        return obj is CatanAction other
            && ActionType == other.ActionType
            && Arg1 == other.Arg1
            && Arg2 == other.Arg2;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine((int)ActionType, Arg1, Arg2);
    }
}

public sealed class CatanState : ICoreState
{
    private const int ResourceCount = 5;
    private const int DevCardCount = 5;

    private readonly int[,] _resources;
    private readonly int[] _knightsPlayed;
    private readonly int[,] _devCards;
    private readonly int[] _devDeckRemaining;
    private readonly int[] _newDevCardsThisTurn;
    private readonly int[] _pendingRoadBuildingPlacements;

    public CatanState()
        : this(GameConfig.Standard, playerCount: 3, new Random())
    {
    }

    public CatanState(GameConfig config, int playerCount, Random rng)
    {
        if (playerCount < config.MinPlayers || playerCount > config.MaxPlayers)
        {
            throw new ArgumentOutOfRangeException(
                nameof(playerCount),
                playerCount,
                $"Player count must be between {config.MinPlayers} and {config.MaxPlayers}.");
        }

        Config = config;
        PlayerCount = playerCount;
        Board = new Board(BoardSetup.Generate(config.Map, rng), config);
        CurrentPlayer = 1;
        Stage = TurnStage.PlaceFirstSettlement;
        TurnNumber = 0;
        PendingSettlementVertex = null;
        LongestRoadOwner = 0;
        LargestArmyOwner = 0;
        WinnerPlayer = 0;
        _resources = new int[playerCount + 1, ResourceCount];
        _knightsPlayed = new int[playerCount + 1];
        _devCards = new int[playerCount + 1, DevCardCount];
        _devDeckRemaining = new int[DevCardCount];
        _newDevCardsThisTurn = new int[DevCardCount];
        _pendingRoadBuildingPlacements = new int[playerCount + 1];

        foreach (var pair in config.DevCardCounts)
        {
            _devDeckRemaining[(int)pair.Key] = pair.Value;
        }
    }

    private CatanState(
        GameConfig config,
        Board board,
        int playerCount,
        int currentPlayer,
        TurnStage stage,
        int turnNumber,
        int? pendingSettlementVertex,
        int longestRoadOwner,
        int largestArmyOwner,
        int winnerPlayer,
        int[,] resources,
        int[] knightsPlayed,
        int[,] devCards,
        int[] devDeckRemaining,
        int[] newDevCardsThisTurn,
        int[] pendingRoadBuildingPlacements)
    {
        Config = config;
        Board = board;
        PlayerCount = playerCount;
        CurrentPlayer = currentPlayer;
        Stage = stage;
        TurnNumber = turnNumber;
        PendingSettlementVertex = pendingSettlementVertex;
        LongestRoadOwner = longestRoadOwner;
        LargestArmyOwner = largestArmyOwner;
        WinnerPlayer = winnerPlayer;
        _resources = resources;
        _knightsPlayed = knightsPlayed;
        _devCards = devCards;
        _devDeckRemaining = devDeckRemaining;
        _newDevCardsThisTurn = newDevCardsThisTurn;
        _pendingRoadBuildingPlacements = pendingRoadBuildingPlacements;
    }

    public GameConfig Config { get; }

    public Board Board { get; }

    public int PlayerCount { get; }

    public int CurrentPlayer { get; private set; }

    public TurnStage Stage { get; private set; }

    public int TurnNumber { get; private set; }

    public int? PendingSettlementVertex { get; private set; }

    public int LongestRoadOwner { get; private set; }

    public int LargestArmyOwner { get; private set; }

    public int WinnerPlayer { get; private set; }

    public Player PlayerTurn => CurrentPlayer switch
    {
        1 => Player.Player1,
        2 => Player.Player2,
        _ => Player.None,
    };

    public int ResourceCountFor(int player, ResourceType resource)
    {
        EnsurePlayer(player);
        EnsureCollectableResource(resource);
        return _resources[player, ResourceToIndex(resource)];
    }

    public int TotalResourceCards(int player)
    {
        EnsurePlayer(player);
        var total = 0;
        for (var i = 0; i < ResourceCount; i++)
        {
            total += _resources[player, i];
        }

        return total;
    }

    public int KnightsPlayedFor(int player)
    {
        EnsurePlayer(player);
        return _knightsPlayed[player];
    }

    public int DevCardsInHand(int player, DevCardType devCard)
    {
        EnsurePlayer(player);
        return _devCards[player, (int)devCard];
    }

    public int DevCardsRemaining(DevCardType devCard)
    {
        return _devDeckRemaining[(int)devCard];
    }

    public int PendingRoadBuildingPlacementsFor(int player)
    {
        EnsurePlayer(player);
        return _pendingRoadBuildingPlacements[player];
    }

    public int VictoryPointsFor(int player)
    {
        EnsurePlayer(player);
        var settlements = Board.SettlementCount(player);
        var cities = Board.CityCount(player);
        var devVp = _devCards[player, (int)DevCardType.VictoryPoint];
        var roadBonus = LongestRoadOwner == player ? 2 : 0;
        var armyBonus = LargestArmyOwner == player ? 2 : 0;
        return settlements + (cities * 2) + devVp + roadBonus + armyBonus;
    }

    public ICoreAction[] Actions()
    {
        if (WinnerPlayer != 0)
        {
            return [];
        }

        var actions = new List<CatanAction>();
        switch (Stage)
        {
            case TurnStage.PlaceFirstSettlement:
            case TurnStage.PlaceSecondSettlement:
                foreach (var vertexIndex in LegalSettlementVertices(initialPlacement: true))
                {
                    actions.Add(new CatanAction(this, CatanActionType.PlaceSettlement, vertexIndex));
                }

                break;

            case TurnStage.PlaceFirstRoad:
            case TurnStage.PlaceSecondRoad:
                foreach (var edgeIndex in LegalInitialRoadEdges())
                {
                    actions.Add(new CatanAction(this, CatanActionType.PlaceRoad, edgeIndex));
                }

                break;

            case TurnStage.PreRoll:
                for (var roll = 2; roll <= 12; roll++)
                {
                    actions.Add(new CatanAction(this, CatanActionType.RollDice, roll));
                }

                break;

            case TurnStage.ChooseRobberLocation:
                foreach (var tileIndex in LegalRobberTiles())
                {
                    actions.Add(new CatanAction(this, CatanActionType.ChooseRobberTile, tileIndex));
                }

                break;

            case TurnStage.BuildTrade:
                if (_pendingRoadBuildingPlacements[CurrentPlayer] > 0)
                {
                    foreach (var edgeIndex in LegalBuildRoadEdges(requireCost: false))
                    {
                        actions.Add(new CatanAction(this, CatanActionType.PlaceRoad, edgeIndex));
                    }

                    break;
                }

                foreach (var edgeIndex in LegalBuildRoadEdges(requireCost: true))
                {
                    actions.Add(new CatanAction(this, CatanActionType.PlaceRoad, edgeIndex));
                }

                foreach (var vertexIndex in LegalSettlementVertices(initialPlacement: false))
                {
                    actions.Add(new CatanAction(this, CatanActionType.PlaceSettlement, vertexIndex));
                }

                foreach (var vertexIndex in LegalCityVertices())
                {
                    actions.Add(new CatanAction(this, CatanActionType.BuildCity, vertexIndex));
                }

                foreach (var trade in LegalBankTrades())
                {
                    actions.Add(new CatanAction(this, CatanActionType.BankTrade, (int)trade.Give, (int)trade.Receive));
                }

                foreach (var devCardType in LegalDevCardPurchases())
                {
                    actions.Add(new CatanAction(this, CatanActionType.BuyDevCard, (int)devCardType));
                }

                foreach (var action in LegalDevCardPlays())
                {
                    actions.Add(action);
                }

                actions.Add(new CatanAction(this, CatanActionType.EndTurn));
                break;
        }

        actions.Sort();
        return [.. actions];
    }

    public CatanState Apply(CatanAction action)
    {
        if (!ReferenceEquals(action.OriginState, this))
        {
            throw new InvalidOperationException("Action origin does not match the current state.");
        }

        return action.ActionType switch
        {
            CatanActionType.PlaceSettlement => ApplySettlement(action.Arg1),
            CatanActionType.PlaceRoad => ApplyRoad(action.Arg1),
            CatanActionType.RollDice => ApplyRollDice(action.Arg1),
            CatanActionType.ChooseRobberTile => ApplyChooseRobberTile(action.Arg1),
            CatanActionType.BuildCity => ApplyBuildCity(action.Arg1),
            CatanActionType.BankTrade => ApplyBankTrade((ResourceType)action.Arg1, (ResourceType)action.Arg2),
            CatanActionType.BuyDevCard => ApplyBuyDevCard((DevCardType)action.Arg1),
            CatanActionType.PlayKnight => ApplyPlayKnight(),
            CatanActionType.PlayRoadBuilding => ApplyPlayRoadBuilding(),
            CatanActionType.PlayMonopoly => ApplyPlayMonopoly((ResourceType)action.Arg1),
            CatanActionType.PlayYearOfPlenty => ApplyPlayYearOfPlenty((ResourceType)action.Arg1, (ResourceType)action.Arg2),
            CatanActionType.EndTurn => ApplyEndTurn(),
            _ => throw new InvalidOperationException($"Unsupported action type: {action.ActionType}"),
        };
    }

    public string SerializeHumanReadable()
    {
        var tokens = new List<int>(
            Board.Topology.TileCount * 2
            + 1
            + 2
            + 2
            + Board.Topology.VertexCount
            + Board.Topology.EdgeCount
            + Board.Topology.PortCount
            + (11 * PlayerCount));

        for (var ti = 0; ti < Board.Topology.TileCount; ti++)
        {
            tokens.Add((int)Board.TileResource(ti));
            tokens.Add(Board.TileNumber(ti));
        }

        tokens.Add(Board.RobberTile);
        tokens.Add(CurrentPlayer);
        tokens.Add((int)Stage);
        tokens.Add(LongestRoadOwner);
        tokens.Add(LargestArmyOwner);

        for (var vi = 0; vi < Board.Topology.VertexCount; vi++)
        {
            tokens.Add(Board.VertexOccupancy[vi].ToToken());
        }

        for (var ei = 0; ei < Board.Topology.EdgeCount; ei++)
        {
            tokens.Add(Board.EdgeOccupancy[ei].ToToken());
        }

        for (var pi = 0; pi < Board.Topology.PortCount; pi++)
        {
            tokens.Add((int)Board.PortType(pi));
        }

        for (var player = 1; player <= PlayerCount; player++)
        {
            tokens.Add(_resources[player, ResourceToIndex(ResourceType.Wood)]);
            tokens.Add(_resources[player, ResourceToIndex(ResourceType.Brick)]);
            tokens.Add(_resources[player, ResourceToIndex(ResourceType.Sheep)]);
            tokens.Add(_resources[player, ResourceToIndex(ResourceType.Wheat)]);
            tokens.Add(_resources[player, ResourceToIndex(ResourceType.Ore)]);
        }

        for (var player = 1; player <= PlayerCount; player++)
        {
            tokens.Add(_knightsPlayed[player]);
        }

        for (var player = 1; player <= PlayerCount; player++)
        {
            for (var card = 0; card < DevCardCount; card++)
            {
                tokens.Add(_devCards[player, card]);
            }
        }

        var sb = new StringBuilder(tokens.Count * 3);
        for (var i = 0; i < tokens.Count; i++)
        {
            if (i > 0)
            {
                sb.Append('|');
            }

            sb.Append(tokens[i].ToString("D2"));
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

        var tokens = serialized.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var topology = config.Map.Topology;
        var expectedCount =
            (topology.TileCount * 2)
            + 1
            + 2
            + 2
            + topology.VertexCount
            + topology.EdgeCount
            + topology.PortCount
            + (11 * playerCount);

        if (tokens.Length != expectedCount)
        {
            throw new InvalidOperationException(
                $"Serialized state has {tokens.Length} tokens, expected {expectedCount}.");
        }

        var index = 0;

        int NextToken()
        {
            if (!int.TryParse(tokens[index], out var token))
            {
                throw new InvalidOperationException($"Invalid token '{tokens[index]}' at position {index}.");
            }

            index++;
            return token;
        }

        var tileResources = new ResourceType[topology.TileCount];
        var tileNumbers = new int[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tileResources[ti] = (ResourceType)NextToken();
            tileNumbers[ti] = NextToken();
        }

        var robberTile = NextToken();
        var currentPlayer = NextToken();
        var stage = (TurnStage)NextToken();
        var longestRoadOwner = NextToken();
        var largestArmyOwner = NextToken();

        var vertices = new VertexOccupancy[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            vertices[vi] = VertexOccupancy.FromToken(NextToken());
        }

        var edges = new EdgeOccupancy[topology.EdgeCount];
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            edges[ei] = EdgeOccupancy.FromToken(NextToken());
        }

        var ports = new PortType[topology.PortCount];
        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            ports[pi] = (PortType)NextToken();
        }

        var resources = new int[playerCount + 1, ResourceCount];
        for (var player = 1; player <= playerCount; player++)
        {
            resources[player, ResourceToIndex(ResourceType.Wood)] = NextToken();
            resources[player, ResourceToIndex(ResourceType.Brick)] = NextToken();
            resources[player, ResourceToIndex(ResourceType.Sheep)] = NextToken();
            resources[player, ResourceToIndex(ResourceType.Wheat)] = NextToken();
            resources[player, ResourceToIndex(ResourceType.Ore)] = NextToken();
        }

        var knightsPlayed = new int[playerCount + 1];
        for (var player = 1; player <= playerCount; player++)
        {
            knightsPlayed[player] = NextToken();
        }

        var devCards = new int[playerCount + 1, DevCardCount];
        for (var player = 1; player <= playerCount; player++)
        {
            for (var card = 0; card < DevCardCount; card++)
            {
                devCards[player, card] = NextToken();
            }
        }

        var setup = new BoardSetup(topology, [.. tileResources], [.. tileNumbers], [.. ports], robberTile);
        var board = new Board(setup, config);
        Array.Copy(vertices, board.VertexOccupancy, vertices.Length);
        Array.Copy(edges, board.EdgeOccupancy, edges.Length);
        board.RobberTile = robberTile;

        var pendingSettlement = InferPendingSettlementVertex(board, currentPlayer, stage);
        var turnNumber = stage is TurnStage.PreRoll or TurnStage.ChooseRobberLocation or TurnStage.BuildTrade ? 1 : 0;

        var deck = new int[DevCardCount];
        foreach (var pair in config.DevCardCounts)
        {
            deck[(int)pair.Key] = pair.Value;
        }

        for (var player = 1; player <= playerCount; player++)
        {
            for (var card = 0; card < DevCardCount; card++)
            {
                deck[card] -= devCards[player, card];
            }
        }

        for (var card = 0; card < DevCardCount; card++)
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
            new int[DevCardCount],
            new int[playerCount + 1]);

        state.RefreshVictory();
        return state;
    }

    public override string ToString() => SerializeHumanReadable();

    public override bool Equals(object? obj)
    {
        if (obj is not CatanState other)
        {
            return false;
        }

        if (!string.Equals(SerializeHumanReadable(), other.SerializeHumanReadable(), StringComparison.Ordinal)
            || TurnNumber != other.TurnNumber
            || WinnerPlayer != other.WinnerPlayer)
        {
            return false;
        }

        for (var card = 0; card < DevCardCount; card++)
        {
            if (_newDevCardsThisTurn[card] != other._newDevCardsThisTurn[card])
            {
                return false;
            }
        }

        for (var player = 1; player <= PlayerCount; player++)
        {
            if (_pendingRoadBuildingPlacements[player] != other._pendingRoadBuildingPlacements[player])
            {
                return false;
            }
        }

        return true;
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(PlayerCount);
        hash.Add(CurrentPlayer);
        hash.Add((int)Stage);
        hash.Add(TurnNumber);
        hash.Add(LongestRoadOwner);
        hash.Add(LargestArmyOwner);
        hash.Add(WinnerPlayer);
        hash.Add(Board.RobberTile);

        for (var ti = 0; ti < Board.Topology.TileCount; ti++)
        {
            hash.Add((int)Board.TileResource(ti));
            hash.Add(Board.TileNumber(ti));
        }

        for (var vi = 0; vi < Board.Topology.VertexCount; vi++)
        {
            hash.Add(Board.VertexOccupancy[vi].ToToken());
        }

        for (var ei = 0; ei < Board.Topology.EdgeCount; ei++)
        {
            hash.Add(Board.EdgeOccupancy[ei].ToToken());
        }

        for (var pi = 0; pi < Board.Topology.PortCount; pi++)
        {
            hash.Add((int)Board.PortType(pi));
        }

        for (var player = 1; player <= PlayerCount; player++)
        {
            for (var i = 0; i < ResourceCount; i++)
            {
                hash.Add(_resources[player, i]);
            }

            hash.Add(_knightsPlayed[player]);

            for (var card = 0; card < DevCardCount; card++)
            {
                hash.Add(_devCards[player, card]);
            }
        }

        for (var card = 0; card < DevCardCount; card++)
        {
            hash.Add(_devDeckRemaining[card]);
            hash.Add(_newDevCardsThisTurn[card]);
        }

        for (var player = 1; player <= PlayerCount; player++)
        {
            hash.Add(_pendingRoadBuildingPlacements[player]);
        }

        return hash.ToHashCode();
    }

    private CatanState ApplySettlement(int vertexIndex)
    {
        if (Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
        {
            return ApplyInitialSettlement(vertexIndex);
        }

        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException($"Settlement placement is not allowed during stage {Stage}.");
        }

        if (!LegalSettlementVertices(initialPlacement: false).Contains(vertexIndex))
        {
            throw new InvalidOperationException(
                $"Vertex {vertexIndex} is not a legal settlement location for player {CurrentPlayer}.");
        }

        var next = Clone();
        next.PayCost(next.Config.SettlementCost);
        next.Board.VertexOccupancy[vertexIndex] = new VertexOccupancy(BuildingType.Settlement, CurrentPlayer);
        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyRoad(int edgeIndex)
    {
        if (Stage is TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad)
        {
            return ApplyInitialRoad(edgeIndex);
        }

        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException($"Road placement is not allowed during stage {Stage}.");
        }

        var legalRoads = _pendingRoadBuildingPlacements[CurrentPlayer] > 0
            ? LegalBuildRoadEdges(requireCost: false)
            : LegalBuildRoadEdges(requireCost: true);
        if (!legalRoads.Contains(edgeIndex))
        {
            throw new InvalidOperationException(
                $"Edge {edgeIndex} is not a legal road location for player {CurrentPlayer}.");
        }

        var next = Clone();
        if (next._pendingRoadBuildingPlacements[CurrentPlayer] > 0)
        {
            next._pendingRoadBuildingPlacements[CurrentPlayer]--;
        }
        else
        {
            next.PayCost(next.Config.RoadCost);
        }

        next.Board.EdgeOccupancy[edgeIndex] = new EdgeOccupancy(CurrentPlayer);
        next.UpdateLongestRoadOwner();
        if (next._pendingRoadBuildingPlacements[CurrentPlayer] > 0 && next.LegalBuildRoadEdges(requireCost: false).Count == 0)
        {
            next._pendingRoadBuildingPlacements[CurrentPlayer] = 0;
        }

        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyInitialSettlement(int vertexIndex)
    {
        if (!Board.CanPlaceSettlement(vertexIndex, CurrentPlayer))
        {
            throw new InvalidOperationException(
                $"Vertex {vertexIndex} is not a legal settlement location for player {CurrentPlayer}.");
        }

        var next = Clone();
        next.Board.VertexOccupancy[vertexIndex] = new VertexOccupancy(BuildingType.Settlement, CurrentPlayer);
        if (Stage == TurnStage.PlaceSecondSettlement)
        {
            next.GrantSecondPlacementResources(vertexIndex);
        }

        next.PendingSettlementVertex = vertexIndex;
        next.Stage = Stage == TurnStage.PlaceFirstSettlement
            ? TurnStage.PlaceFirstRoad
            : TurnStage.PlaceSecondRoad;
        return next;
    }

    private CatanState ApplyInitialRoad(int edgeIndex)
    {
        var legal = LegalInitialRoadEdges();
        if (!legal.Contains(edgeIndex))
        {
            throw new InvalidOperationException(
                $"Edge {edgeIndex} is not a legal road location for player {CurrentPlayer}.");
        }

        var next = Clone();
        next.Board.EdgeOccupancy[edgeIndex] = new EdgeOccupancy(CurrentPlayer);
        next.PendingSettlementVertex = null;
        next.UpdateLongestRoadOwner();

        if (Stage == TurnStage.PlaceFirstRoad)
        {
            if (CurrentPlayer < PlayerCount)
            {
                next.CurrentPlayer = CurrentPlayer + 1;
                next.Stage = TurnStage.PlaceFirstSettlement;
            }
            else if (Config.InitialPlacementRounds >= 2)
            {
                next.Stage = TurnStage.PlaceSecondSettlement;
            }
            else
            {
                next.CurrentPlayer = 1;
                next.Stage = TurnStage.PreRoll;
                next.TurnNumber = 1;
            }
        }
        else
        {
            if (CurrentPlayer > 1)
            {
                next.CurrentPlayer = CurrentPlayer - 1;
                next.Stage = TurnStage.PlaceSecondSettlement;
            }
            else
            {
                next.CurrentPlayer = 1;
                next.Stage = TurnStage.PreRoll;
                next.TurnNumber = 1;
            }
        }

        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyRollDice(int roll)
    {
        if (Stage != TurnStage.PreRoll)
        {
            throw new InvalidOperationException("Dice can only be rolled during pre-roll stage.");
        }

        if (roll < 2 || roll > 12)
        {
            throw new ArgumentOutOfRangeException(nameof(roll), roll, "Roll must be in [2, 12].");
        }

        var next = Clone();
        if (roll == 7)
        {
            next.ApplyDiscardOnSeven();
            next.Stage = TurnStage.ChooseRobberLocation;
        }
        else
        {
            next.ProduceResources(roll);
            next.Stage = TurnStage.BuildTrade;
        }

        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyChooseRobberTile(int tileIndex)
    {
        if (Stage != TurnStage.ChooseRobberLocation)
        {
            throw new InvalidOperationException("Robber placement is not currently allowed.");
        }

        if (!LegalRobberTiles().Contains(tileIndex))
        {
            throw new InvalidOperationException($"Tile {tileIndex} is not a legal robber destination.");
        }

        var next = Clone();
        next.Board.RobberTile = tileIndex;
        next.TryStealFromRobberTile(tileIndex);
        next.Stage = TurnStage.BuildTrade;
        return next;
    }

    private CatanState ApplyBuildCity(int vertexIndex)
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("City upgrade is only allowed during build/trade stage.");
        }

        if (!LegalCityVertices().Contains(vertexIndex))
        {
            throw new InvalidOperationException($"Vertex {vertexIndex} is not a legal city upgrade.");
        }

        var next = Clone();
        next.PayCost(next.Config.CityCost);
        next.Board.VertexOccupancy[vertexIndex] = new VertexOccupancy(BuildingType.City, CurrentPlayer);
        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyBankTrade(ResourceType give, ResourceType receive)
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Trading is only allowed during build/trade stage.");
        }

        EnsureCollectableResource(give);
        EnsureCollectableResource(receive);
        if (give == receive)
        {
            throw new InvalidOperationException("Give and receive resources must differ.");
        }

        var ratio = Board.TradeRatio(CurrentPlayer, give);
        if (_resources[CurrentPlayer, ResourceToIndex(give)] < ratio)
        {
            throw new InvalidOperationException("Insufficient resources to perform trade.");
        }

        var next = Clone();
        next._resources[CurrentPlayer, ResourceToIndex(give)] -= ratio;
        next._resources[CurrentPlayer, ResourceToIndex(receive)] += 1;
        return next;
    }

    private CatanState ApplyBuyDevCard(DevCardType cardType)
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Buying development cards is only allowed during build/trade stage.");
        }

        if (_devDeckRemaining[(int)cardType] <= 0)
        {
            throw new InvalidOperationException($"No {cardType} cards remaining in deck.");
        }

        var next = Clone();
        next.PayCost(next.Config.DevCardCost);
        next._devDeckRemaining[(int)cardType]--;
        next._devCards[CurrentPlayer, (int)cardType]++;
        next._newDevCardsThisTurn[(int)cardType]++;
        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyPlayKnight()
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Playing a knight is only allowed during build/trade stage.");
        }

        if (_devCards[CurrentPlayer, (int)DevCardType.Knight] <= 0)
        {
            throw new InvalidOperationException("Player has no knight card to play.");
        }
        if (GetPlayableDevCardCount(DevCardType.Knight) <= 0)
        {
            throw new InvalidOperationException("Player cannot play a knight bought this turn.");
        }

        var next = Clone();
        next._devCards[CurrentPlayer, (int)DevCardType.Knight]--;
        next._knightsPlayed[CurrentPlayer]++;
        next.UpdateLargestArmyOwner();
        next.Stage = TurnStage.ChooseRobberLocation;
        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyPlayRoadBuilding()
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Playing road building is only allowed during build/trade stage.");
        }

        if (_devCards[CurrentPlayer, (int)DevCardType.RoadBuilding] <= 0)
        {
            throw new InvalidOperationException("Player has no road building card to play.");
        }
        if (GetPlayableDevCardCount(DevCardType.RoadBuilding) <= 0)
        {
            throw new InvalidOperationException("Player cannot play road building bought this turn.");
        }

        var next = Clone();
        next._devCards[CurrentPlayer, (int)DevCardType.RoadBuilding]--;
        next._pendingRoadBuildingPlacements[CurrentPlayer] = Math.Min(2, next.LegalBuildRoadEdges(requireCost: false).Count);
        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyPlayMonopoly(ResourceType resource)
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Playing monopoly is only allowed during build/trade stage.");
        }

        EnsureCollectableResource(resource);
        if (_devCards[CurrentPlayer, (int)DevCardType.Monopoly] <= 0)
        {
            throw new InvalidOperationException("Player has no monopoly card to play.");
        }
        if (GetPlayableDevCardCount(DevCardType.Monopoly) <= 0)
        {
            throw new InvalidOperationException("Player cannot play monopoly bought this turn.");
        }

        var next = Clone();
        next._devCards[CurrentPlayer, (int)DevCardType.Monopoly]--;
        var resourceIndex = ResourceToIndex(resource);

        for (var player = 1; player <= PlayerCount; player++)
        {
            if (player == CurrentPlayer)
            {
                continue;
            }

            var amount = next._resources[player, resourceIndex];
            if (amount <= 0)
            {
                continue;
            }

            next._resources[player, resourceIndex] = 0;
            next._resources[CurrentPlayer, resourceIndex] += amount;
        }

        return next;
    }

    private CatanState ApplyPlayYearOfPlenty(ResourceType first, ResourceType second)
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Playing year of plenty is only allowed during build/trade stage.");
        }

        EnsureCollectableResource(first);
        EnsureCollectableResource(second);
        if (_devCards[CurrentPlayer, (int)DevCardType.YearOfPlenty] <= 0)
        {
            throw new InvalidOperationException("Player has no year of plenty card to play.");
        }
        if (GetPlayableDevCardCount(DevCardType.YearOfPlenty) <= 0)
        {
            throw new InvalidOperationException("Player cannot play year of plenty bought this turn.");
        }

        var next = Clone();
        next._devCards[CurrentPlayer, (int)DevCardType.YearOfPlenty]--;
        next._resources[CurrentPlayer, ResourceToIndex(first)]++;
        next._resources[CurrentPlayer, ResourceToIndex(second)]++;
        return next;
    }

    private CatanState ApplyEndTurn()
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("End turn is only available during build/trade stage.");
        }

        var next = Clone();
        next.CurrentPlayer = (CurrentPlayer % PlayerCount) + 1;
        if (next.CurrentPlayer == 1)
        {
            next.TurnNumber++;
        }

        next.Stage = TurnStage.PreRoll;
        next.PendingSettlementVertex = null;
        Array.Clear(next._newDevCardsThisTurn, 0, DevCardCount);
        next._pendingRoadBuildingPlacements[CurrentPlayer] = 0;
        return next;
    }

    private IReadOnlyList<int> LegalSettlementVertices(bool initialPlacement)
    {
        var legal = new List<int>();
        for (var vi = 0; vi < Board.Topology.VertexCount; vi++)
        {
            if (!Board.CanPlaceSettlement(vi, CurrentPlayer))
            {
                continue;
            }

            if (initialPlacement)
            {
                legal.Add(vi);
                continue;
            }

            if (!CanBuildSettlementConnected(vi))
            {
                continue;
            }

            if (!CanAfford(Config.SettlementCost))
            {
                continue;
            }

            legal.Add(vi);
        }

        return legal;
    }

    private IReadOnlyList<int> LegalInitialRoadEdges()
    {
        if (PendingSettlementVertex is null)
        {
            return [];
        }

        var legal = new List<int>();
        foreach (var edgeIndex in Board.Topology.VertexEdges[PendingSettlementVertex.Value])
        {
            if (Board.EdgeOccupancy[edgeIndex].IsEmpty && Board.RoadCount(CurrentPlayer) < Board.Config.MaxRoads)
            {
                legal.Add(edgeIndex);
            }
        }

        return legal;
    }

    private IReadOnlyList<int> LegalBuildRoadEdges(bool requireCost)
    {
        if (requireCost && !CanAfford(Config.RoadCost))
        {
            return [];
        }

        var legal = new List<int>();
        for (var ei = 0; ei < Board.Topology.EdgeCount; ei++)
        {
            if (Board.CanPlaceRoad(ei, CurrentPlayer))
            {
                legal.Add(ei);
            }
        }

        return legal;
    }

    private IReadOnlyList<int> LegalCityVertices()
    {
        if (!CanAfford(Config.CityCost))
        {
            return [];
        }

        var legal = new List<int>();
        for (var vi = 0; vi < Board.Topology.VertexCount; vi++)
        {
            if (Board.CanUpgradeToCity(vi, CurrentPlayer))
            {
                legal.Add(vi);
            }
        }

        return legal;
    }

    private IReadOnlyList<int> LegalRobberTiles()
    {
        var legal = new List<int>();
        for (var tile = 0; tile < Board.Topology.TileCount; tile++)
        {
            if (tile != Board.RobberTile)
            {
                legal.Add(tile);
            }
        }

        return legal;
    }

    private IReadOnlyList<(ResourceType Give, ResourceType Receive)> LegalBankTrades()
    {
        var legal = new List<(ResourceType Give, ResourceType Receive)>();
        foreach (var give in CollectableResources())
        {
            var ratio = Board.TradeRatio(CurrentPlayer, give);
            if (_resources[CurrentPlayer, ResourceToIndex(give)] < ratio)
            {
                continue;
            }

            foreach (var receive in CollectableResources())
            {
                if (receive != give)
                {
                    legal.Add((give, receive));
                }
            }
        }

        return legal;
    }

    private IReadOnlyList<DevCardType> LegalDevCardPurchases()
    {
        if (!CanAfford(Config.DevCardCost))
        {
            return [];
        }

        var legal = new List<DevCardType>();
        for (var i = 0; i < DevCardCount; i++)
        {
            if (_devDeckRemaining[i] > 0)
            {
                legal.Add((DevCardType)i);
            }
        }

        return legal;
    }

    private IReadOnlyList<CatanAction> LegalDevCardPlays()
    {
        var legal = new List<CatanAction>();

        if (GetPlayableDevCardCount(DevCardType.Knight) > 0)
        {
            legal.Add(new CatanAction(this, CatanActionType.PlayKnight));
        }

        if (GetPlayableDevCardCount(DevCardType.RoadBuilding) > 0 && Board.RoadCount(CurrentPlayer) < Config.MaxRoads)
        {
            if (LegalBuildRoadEdges(requireCost: false).Count > 0)
            {
                legal.Add(new CatanAction(this, CatanActionType.PlayRoadBuilding));
            }
        }

        if (GetPlayableDevCardCount(DevCardType.Monopoly) > 0)
        {
            foreach (var resource in CollectableResources())
            {
                legal.Add(new CatanAction(this, CatanActionType.PlayMonopoly, (int)resource));
            }
        }

        if (GetPlayableDevCardCount(DevCardType.YearOfPlenty) > 0)
        {
            var resources = CollectableResources().ToArray();
            for (var i = 0; i < resources.Length; i++)
            {
                for (var j = i; j < resources.Length; j++)
                {
                    legal.Add(new CatanAction(this, CatanActionType.PlayYearOfPlenty, (int)resources[i], (int)resources[j]));
                }
            }
        }

        return legal;
    }

    private void ProduceResources(int roll)
    {
        foreach (var tileIndex in Board.TilesForRoll(roll))
        {
            var resource = Board.TileResource(tileIndex);
            if (resource == ResourceType.Desert)
            {
                continue;
            }

            var resourceIndex = ResourceToIndex(resource);
            foreach (var vertexIndex in Board.Topology.TileVertices[tileIndex])
            {
                var occ = Board.VertexOccupancy[vertexIndex];
                if (occ.IsEmpty)
                {
                    continue;
                }

                var amount = occ.Building == BuildingType.City ? 2 : 1;
                _resources[occ.Player, resourceIndex] += amount;
            }
        }
    }

    private void ApplyDiscardOnSeven()
    {
        foreach (var player in Enumerable.Range(1, PlayerCount))
        {
            var total = TotalResourceCards(player);
            if (total <= Config.DiscardThreshold)
            {
                continue;
            }

            var toDiscard = total / 2;
            for (var i = 0; i < toDiscard; i++)
            {
                var resourceIndex = PickDiscardResourceIndex(player);
                if (resourceIndex < 0)
                {
                    break;
                }

                _resources[player, resourceIndex]--;
            }
        }
    }

    private int PickDiscardResourceIndex(int player)
    {
        var bestIndex = -1;
        var bestCount = -1;
        for (var i = 0; i < ResourceCount; i++)
        {
            var count = _resources[player, i];
            if (count > bestCount)
            {
                bestCount = count;
                bestIndex = i;
            }
        }

        return bestCount > 0 ? bestIndex : -1;
    }

    private void TryStealFromRobberTile(int tileIndex)
    {
        var candidates = new HashSet<int>();
        foreach (var vertex in Board.Topology.TileVertices[tileIndex])
        {
            var occ = Board.VertexOccupancy[vertex];
            if (occ.IsEmpty || occ.Player == CurrentPlayer)
            {
                continue;
            }

            if (TotalResourceCards(occ.Player) > 0)
            {
                candidates.Add(occ.Player);
            }
        }

        if (candidates.Count == 0)
        {
            return;
        }

        var victim = candidates.OrderBy(p => p).First();
        for (var i = 0; i < ResourceCount; i++)
        {
            if (_resources[victim, i] <= 0)
            {
                continue;
            }

            _resources[victim, i]--;
            _resources[CurrentPlayer, i]++;
            break;
        }
    }

    private bool CanBuildSettlementConnected(int vertexIndex)
    {
        foreach (var edgeIndex in Board.Topology.VertexEdges[vertexIndex])
        {
            if (Board.EdgeOccupancy[edgeIndex].Player == CurrentPlayer)
            {
                return true;
            }
        }

        return false;
    }

    private bool CanAfford(IReadOnlyDictionary<ResourceType, int> cost)
    {
        foreach (var pair in cost)
        {
            if (_resources[CurrentPlayer, ResourceToIndex(pair.Key)] < pair.Value)
            {
                return false;
            }
        }

        return true;
    }

    private void PayCost(IReadOnlyDictionary<ResourceType, int> cost)
    {
        if (!CanAfford(cost))
        {
            throw new InvalidOperationException("Current player cannot afford cost.");
        }

        foreach (var pair in cost)
        {
            _resources[CurrentPlayer, ResourceToIndex(pair.Key)] -= pair.Value;
        }
    }

    private void GrantSecondPlacementResources(int settlementVertex)
    {
        foreach (var tileIndex in Board.Topology.VertexTiles[settlementVertex])
        {
            var resource = Board.TileResource(tileIndex);
            if (resource == ResourceType.Desert)
            {
                continue;
            }

            _resources[CurrentPlayer, ResourceToIndex(resource)]++;
        }
    }

    private void UpdateLongestRoadOwner()
    {
        var lengths = new int[PlayerCount + 1];
        for (var player = 1; player <= PlayerCount; player++)
        {
            lengths[player] = LongestRoadLength(player);
        }

        var maxLength = lengths.Max();
        if (maxLength < Config.LongestRoadMinimum)
        {
            LongestRoadOwner = 0;
            return;
        }

        var winners = Enumerable.Range(1, PlayerCount)
            .Where(player => lengths[player] == maxLength)
            .ToArray();

        if (winners.Length == 1)
        {
            LongestRoadOwner = winners[0];
            return;
        }

        if (LongestRoadOwner != 0 && winners.Contains(LongestRoadOwner))
        {
            return;
        }

        LongestRoadOwner = 0;
    }

    private int LongestRoadLength(int player)
    {
        var ownedEdges = new HashSet<int>();
        for (var ei = 0; ei < Board.Topology.EdgeCount; ei++)
        {
            if (Board.EdgeOccupancy[ei].Player == player)
            {
                ownedEdges.Add(ei);
            }
        }

        var best = 0;
        foreach (var edge in ownedEdges)
        {
            var (a, b) = Board.Topology.Edges[edge];
            var used = new HashSet<int> { edge };
            best = Math.Max(best, 1 + DfsRoadLength(player, b, used));
            best = Math.Max(best, 1 + DfsRoadLength(player, a, used));
        }

        return best;
    }

    private int DfsRoadLength(int player, int vertex, HashSet<int> usedEdges)
    {
        var occ = Board.VertexOccupancy[vertex];
        if (!occ.IsEmpty && occ.Player != player)
        {
            return 0;
        }

        var best = 0;
        foreach (var edge in Board.Topology.VertexEdges[vertex])
        {
            if (Board.EdgeOccupancy[edge].Player != player || usedEdges.Contains(edge))
            {
                continue;
            }

            usedEdges.Add(edge);
            var endpoints = Board.Topology.Edges[edge];
            var nextVertex = endpoints.VertexA == vertex ? endpoints.VertexB : endpoints.VertexA;
            best = Math.Max(best, 1 + DfsRoadLength(player, nextVertex, usedEdges));
            usedEdges.Remove(edge);
        }

        return best;
    }

    private void UpdateLargestArmyOwner()
    {
        var maxKnights = _knightsPlayed.Max();
        if (maxKnights < Config.LargestArmyMinimum)
        {
            LargestArmyOwner = 0;
            return;
        }

        var winners = Enumerable.Range(1, PlayerCount)
            .Where(player => _knightsPlayed[player] == maxKnights)
            .ToArray();

        if (winners.Length == 1)
        {
            LargestArmyOwner = winners[0];
            return;
        }

        if (LargestArmyOwner != 0 && winners.Contains(LargestArmyOwner))
        {
            return;
        }

        LargestArmyOwner = 0;
    }

    private void RefreshVictory()
    {
        for (var player = 1; player <= PlayerCount; player++)
        {
            if (VictoryPointsFor(player) >= Config.VictoryPointsToWin)
            {
                WinnerPlayer = player;
                return;
            }
        }

        WinnerPlayer = 0;
    }

    private static IEnumerable<ResourceType> CollectableResources()
    {
        yield return ResourceType.Wood;
        yield return ResourceType.Brick;
        yield return ResourceType.Sheep;
        yield return ResourceType.Wheat;
        yield return ResourceType.Ore;
    }

    private CatanState Clone()
    {
        return new CatanState(
            Config,
            Board.Clone(),
            PlayerCount,
            CurrentPlayer,
            Stage,
            TurnNumber,
            PendingSettlementVertex,
            LongestRoadOwner,
            LargestArmyOwner,
            WinnerPlayer,
            (int[,])_resources.Clone(),
            (int[])_knightsPlayed.Clone(),
            (int[,])_devCards.Clone(),
            (int[])_devDeckRemaining.Clone(),
            (int[])_newDevCardsThisTurn.Clone(),
            (int[])_pendingRoadBuildingPlacements.Clone());
    }

    private int GetPlayableDevCardCount(DevCardType cardType)
    {
        var typeIndex = (int)cardType;
        var total = _devCards[CurrentPlayer, typeIndex];
        var newThisTurn = _newDevCardsThisTurn[typeIndex];
        var playable = total - newThisTurn;
        return playable > 0 ? playable : 0;
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

    private static int ResourceToIndex(ResourceType resource) => resource switch
    {
        ResourceType.Wood => 0,
        ResourceType.Brick => 1,
        ResourceType.Sheep => 2,
        ResourceType.Wheat => 3,
        ResourceType.Ore => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(resource), resource, "Resource must be one of wood/brick/sheep/wheat/ore."),
    };

    private static void EnsureCollectableResource(ResourceType resource)
    {
        if (resource == ResourceType.Desert)
        {
            throw new ArgumentOutOfRangeException(nameof(resource), resource, "Desert is not a collectable resource.");
        }
    }

    private void EnsurePlayer(int player)
    {
        if (player < 1 || player > PlayerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(player), player, $"Player must be between 1 and {PlayerCount}.");
        }
    }
}
