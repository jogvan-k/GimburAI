using Gimbur.Rules;
using Kjarni;

namespace Gimbur;

public sealed class CatanState : ICoreState
{
    internal const int ResourceCount = 5;
    internal const int DevCardCount = 5;
    /// Dev card types serialized in the "new this turn" section.
    internal static readonly DevCardType[] NewDevCardTypes =
        [DevCardType.Knight, DevCardType.RoadBuilding, DevCardType.Monopoly, DevCardType.YearOfPlenty, DevCardType.VictoryPoint];
    internal const int NewDevCardSerializedCount = 5;

    internal readonly int[,] _resources;
    internal readonly int[] _knightsPlayed;
    internal readonly int[,] _devCards;
    internal readonly int[] _devDeckRemaining;
    internal readonly int[] _newDevCardsThisTurn;
    internal readonly int[] _pendingRoadBuildingPlacements;
    internal TurnStage? _postDevCardStage;
    /// <summary>
    /// Per-vertex placement round: 0 = empty, 1 = placed during first round,
    /// 2 = placed during second round. Used by placement phase serialization.
    /// </summary>
    internal readonly byte[] _vertexPlacementRound;
    private static readonly object RngLock = new();
    private static readonly Random SharedRng = new();
    private static readonly Dictionary<int, int> DiceRollCounts = new()
    {
        [2] = 1,
        [3] = 2,
        [4] = 3,
        [5] = 4,
        [6] = 5,
        [7] = 6,
        [8] = 5,
        [9] = 4,
        [10] = 3,
        [11] = 2,
        [12] = 1,
    };

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
        _postDevCardStage = null;
        _vertexPlacementRound = new byte[config.Map.Topology.VertexCount];
        foreach (var pair in config.DevCardCounts)
        {
            _devDeckRemaining[(int)pair.Key] = pair.Value;
        }
    }

    internal CatanState(
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
        int[] pendingRoadBuildingPlacements,
        TurnStage? postDevCardStage,
        byte[] vertexPlacementRound)
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
        _postDevCardStage = postDevCardStage;
        _vertexPlacementRound = vertexPlacementRound;
    }

    public GameConfig Config { get; }

    public Board Board { get; }

    public int PlayerCount { get; }

    public int NumberOfPlayers => PlayerCount;

    public int CurrentPlayer { get; private set; }

    public TurnStage Stage { get; private set; }

    public int TurnNumber { get; private set; }

    public int? PendingSettlementVertex { get; private set; }

    public int LongestRoadOwner { get; private set; }

    public int LargestArmyOwner { get; private set; }

    public int WinnerPlayer { get; private set; }

    public int LastDiceRoll { get; private set; }

    public Player PlayerTurn => CurrentPlayer switch
    {
        1 => Player.Player1,
        2 => Player.Player2,
        3 => Player.Player3,
        4 => Player.Player4,
        _ => Player.Player1,
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

    internal IReadOnlyList<(CatanState State, int Weight)> RollDiceOutcomes()
    {
        var outcomes = new List<(CatanState State, int Weight)>(11);
        foreach (var pair in DiceRollCounts)
        {
            outcomes.Add((ApplyRollDiceOutcome(pair.Key), pair.Value));
        }

        return outcomes;
    }

    internal IReadOnlyList<(CatanState State, int Weight)> BuyDevCardOutcomes()
    {
        var total = 0;
        for (var i = 0; i < DevCardCount; i++)
        {
            total += _devDeckRemaining[i];
        }

        if (total <= 0)
        {
            return [];
        }

        var outcomes = new List<(CatanState State, int Weight)>(DevCardCount);
        for (var i = 0; i < DevCardCount; i++)
        {
            var count = _devDeckRemaining[i];
            if (count <= 0)
            {
                continue;
            }

            outcomes.Add((ApplyBuyDevCardType((DevCardType)i), count));
        }

        return outcomes;
    }

    internal IReadOnlyList<(CatanState State, int Weight)> ChooseRobberTileOutcomes(int tileIndex)
    {
        if (Stage != TurnStage.ChooseRobberLocation)
        {
            throw new InvalidOperationException("Robber placement is not currently allowed.");
        }

        if (!LegalRobberTiles().Contains(tileIndex))
        {
            throw new InvalidOperationException($"Tile {tileIndex} is not a legal robber destination.");
        }

        var victims = RobberVictims(tileIndex);
        if (victims.Count == 0)
        {
            return [ (ApplyChooseRobberTileNoSteal(tileIndex), 1) ];
        }

        if (victims.Count > 1)
        {
            return [ (ApplyChooseRobberTileAwaitVictim(tileIndex), 1) ];
        }

        var victim = victims[0];
        return ChooseRobberTileVictimStealOutcomes(tileIndex, victim);
    }

    internal IReadOnlyList<(CatanState State, int Weight)> ChooseRobberVictimOutcomes(int victimPlayer)
    {
        if (Stage != TurnStage.ChooseRobberVictim)
        {
            throw new InvalidOperationException("Robber victim choice is not currently allowed.");
        }

        if (!LegalRobberVictims().Contains(victimPlayer))
        {
            throw new InvalidOperationException($"Player {victimPlayer} is not a legal robber victim.");
        }

        return ChooseRobberTileVictimStealOutcomes(Board.RobberTile, victimPlayer);
    }

    private IReadOnlyList<(CatanState State, int Weight)> ChooseRobberTileVictimStealOutcomes(int tileIndex, int victim)
    {
        var outcomes = new List<(CatanState State, int Weight)>();
        var victimTotal = TotalResourceCards(victim);
        if (victimTotal <= 0)
        {
            outcomes.Add((ApplyChooseRobberTileNoSteal(tileIndex), 1));
            return outcomes;
        }

        for (var i = 0; i < ResourceCount; i++)
        {
            var count = _resources[victim, i];
            if (count <= 0)
            {
                continue;
            }

            outcomes.Add((ApplyChooseRobberTileSteal(tileIndex, victim, i), count));
        }

        return outcomes;
    }

    internal IReadOnlyList<(CatanState State, int Weight)> PlayKnightOutcomes()
    {
        return [ (ApplyPlayKnight(), 1) ];
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

    public double[] Scores()
    {
        var scores = new double[PlayerCount]; // Indexed by Player enum value (0-based)
        for (var p = 1; p <= PlayerCount; p++)
        {
            scores[p - 1] = VictoryPointsFor(p);
        }
        return scores;
    }

    public CoreAction[] Actions()
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
                    actions.Add(new PlaceSettlementAction(this, vertexIndex));
                }

                break;

            case TurnStage.PlaceFirstRoad:
            case TurnStage.PlaceSecondRoad:
                foreach (var edgeIndex in LegalInitialRoadEdges())
                {
                    actions.Add(new PlaceRoadAction(this, edgeIndex));
                }

                break;

            case TurnStage.PreRoll:
                actions.Add(new RollDiceAction(this));
                foreach (var action in LegalDevCardPlays())
                {
                    actions.Add(action);
                }

                break;

            case TurnStage.ChooseRobberLocation:
                foreach (var tileIndex in LegalRobberTiles())
                {
                    actions.Add(new ChooseRobberTileAction(this, tileIndex));
                }

                break;

            case TurnStage.ChooseRobberVictim:
                foreach (var victim in LegalRobberVictims())
                {
                    actions.Add(new ChooseRobberVictimAction(this, victim));
                }
                break;

            case TurnStage.BuildTrade:
                if (_pendingRoadBuildingPlacements[CurrentPlayer] > 0)
                {
                    foreach (var edgeIndex in LegalBuildRoadEdges(requireCost: false))
                    {
                        actions.Add(new PlaceRoadAction(this, edgeIndex));
                    }

                    break;
                }

                foreach (var edgeIndex in LegalBuildRoadEdges(requireCost: true))
                {
                    actions.Add(new PlaceRoadAction(this, edgeIndex));
                }

                foreach (var vertexIndex in LegalSettlementVertices(initialPlacement: false))
                {
                    actions.Add(new PlaceSettlementAction(this, vertexIndex));
                }

                foreach (var vertexIndex in LegalCityVertices())
                {
                    actions.Add(new BuildCityAction(this, vertexIndex));
                }

                foreach (var trade in LegalBankTrades())
                {
                    actions.Add(new BankTradeAction(this, trade.Give, trade.Receive));
                }

                if (LegalDevCardPurchases().Count > 0)
                {
                    actions.Add(new BuyDevCardAction(this));
                }

                foreach (var action in LegalDevCardPlays())
                {
                    actions.Add(action);
                }

                actions.Add(new EndTurnAction(this));
                break;
        }

        actions.Sort();
        return [.. actions.Select(a => a switch
        {
            CatanDeterministicAction d => CoreAction.NewDeterministic(d),
            CatanStochasticAction s => CoreAction.NewStochastic(s),
            _ => throw new InvalidOperationException($"Unsupported action type: {a.GetType().Name}")
        })];
    }

    internal CatanState Apply(CatanAction action)
    {
        if (!ReferenceEquals(action.OriginState, this))
        {
            throw new InvalidOperationException("Action origin does not match the current state.");
        }

        return action switch
        {
            PlaceSettlementAction a => ApplySettlement(a.VertexIndex),
            PlaceRoadAction a => ApplyRoad(a.EdgeIndex),
            RollDiceAction => ApplyRollDice(),
            ChooseRobberTileAction a => ApplyChooseRobberTile(a.TileIndex),
            ChooseRobberVictimAction a => ApplyChooseRobberVictim(a.VictimPlayer),
            BuildCityAction a => ApplyBuildCity(a.VertexIndex),
            BankTradeAction a => ApplyBankTrade(a.Give, a.Receive),
            BuyDevCardAction => ApplyBuyDevCard(),
            PlayKnightAction => ApplyPlayKnight(),
            PlayRoadBuildingAction => ApplyPlayRoadBuilding(),
            PlayMonopolyAction a => ApplyPlayMonopoly(a.Resource),
            PlayYearOfPlentyAction a => ApplyPlayYearOfPlenty(a.First, a.Second),
            EndTurnAction => ApplyEndTurn(),
            _ => throw new InvalidOperationException($"Unsupported action type: {action.GetType().Name}"),
        };
    }

    public string SerializeHumanReadable() => CatanStateSerializer.SerializeHumanReadable(this);

    /// <summary>
    /// Serializes the board-invariant portion: tiles and ports, separated by '|'.
    /// </summary>
    public string SerializeBoard() => CatanStateSerializer.SerializeBoard(this);

    /// <summary>
    /// Serializes the turn-specific state (sections 2–6, 8–10) in human-readable
    /// form, excluding tiles and ports which are board-invariant.
    /// </summary>
    public string SerializeStateOnly() => CatanStateSerializer.SerializeStateOnly(this);

    public static CatanState DeserializeHumanReadable(
        GameConfig config,
        int playerCount,
        string serialized) => CatanStateSerializer.DeserializeHumanReadable(config, playerCount, serialized);

    /// <summary>
    /// Produces the compact form: strips all '/' and '|' separators from the
    /// human-readable form, yielding a fixed-length token string.
    /// </summary>
    public string SerializeCompact() => CatanStateSerializer.SerializeCompact(this);

    /// <summary>
    /// Parses the compact form by re-inserting separators at the known fixed
    /// positions and delegating to <see cref="DeserializeHumanReadable"/>.
    /// </summary>
    public static CatanState DeserializeCompact(
        GameConfig config,
        int playerCount,
        string compact) => CatanStateSerializer.DeserializeCompact(config, playerCount, compact);

    /// <summary>
    /// Serializes the placement phase state in human-readable form.
    /// Format: tiles|ports|stage|placementVertices|edges
    /// </summary>
    public string SerializePlacementPhase() => CatanStateSerializer.SerializePlacementPhase(this);

    /// <summary>
    /// Produces the compact form of the placement phase state: strips all pipe separators.
    /// </summary>
    public string SerializePlacementPhaseCompact() => CatanStateSerializer.SerializePlacementPhaseCompact(this);

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

        if (_postDevCardStage != other._postDevCardStage)
        {
            return false;
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
            hash.Add((int)Board.VertexOccupancy[vi].Building);
            hash.Add(Board.VertexOccupancy[vi].Player);
        }

        for (var ei = 0; ei < Board.Topology.EdgeCount; ei++)
        {
            hash.Add(Board.EdgeOccupancy[ei].Player);
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

        hash.Add(_postDevCardStage.HasValue ? (int)_postDevCardStage.Value : -1);

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

        // When all road building placements are done and a return stage is
        // pending (e.g. road building played before dice roll), transition back.
        if (next._pendingRoadBuildingPlacements[CurrentPlayer] == 0 && next._postDevCardStage.HasValue)
        {
            next.Stage = next._postDevCardStage.Value;
            next._postDevCardStage = null;
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
        next._vertexPlacementRound[vertexIndex] = (byte)(Stage == TurnStage.PlaceFirstSettlement ? 1 : 2);
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

    private CatanState ApplyRollDice()
    {
        if (Stage != TurnStage.PreRoll)
        {
            throw new InvalidOperationException("Dice can only be rolled during pre-roll stage.");
        }

        var roll = RollTwoDice();
        return ApplyRollDiceOutcome(roll);
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

        var victims = RobberVictims(tileIndex);
        if (victims.Count == 0)
        {
            return ApplyChooseRobberTileNoSteal(tileIndex);
        }

        if (victims.Count > 1)
        {
            return ApplyChooseRobberTileAwaitVictim(tileIndex);
        }

        var victim = victims[0];
        var victimCardCount = TotalResourceCards(victim);
        if (victimCardCount <= 0)
        {
            return ApplyChooseRobberTileNoSteal(tileIndex);
        }

        int pick;
        lock (RngLock)
        {
            pick = SharedRng.Next(victimCardCount);
        }

        var running = pick;
        for (var i = 0; i < ResourceCount; i++)
        {
            running -= _resources[victim, i];
            if (running < 0)
            {
                return ApplyChooseRobberTileSteal(tileIndex, victim, i);
            }
        }

        return ApplyChooseRobberTileNoSteal(tileIndex);
    }

    private CatanState ApplyChooseRobberVictim(int victimPlayer)
    {
        if (Stage != TurnStage.ChooseRobberVictim)
        {
            throw new InvalidOperationException("Robber victim choice is not currently allowed.");
        }

        if (!LegalRobberVictims().Contains(victimPlayer))
        {
            throw new InvalidOperationException($"Player {victimPlayer} is not a legal robber victim.");
        }

        var victimCardCount = TotalResourceCards(victimPlayer);
        if (victimCardCount <= 0)
        {
            return ApplyChooseRobberTileNoSteal(Board.RobberTile);
        }

        int pick;
        lock (RngLock)
        {
            pick = SharedRng.Next(victimCardCount);
        }

        var running = pick;
        for (var i = 0; i < ResourceCount; i++)
        {
            running -= _resources[victimPlayer, i];
            if (running < 0)
            {
                return ApplyChooseRobberTileSteal(Board.RobberTile, victimPlayer, i);
            }
        }

        return ApplyChooseRobberTileNoSteal(Board.RobberTile);
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

    private CatanState ApplyBuyDevCard()
    {
        if (Stage != TurnStage.BuildTrade)
        {
            throw new InvalidOperationException("Buying development cards is only allowed during build/trade stage.");
        }

        return ApplyBuyDevCardType(DrawRandomDevCard() ?? throw new InvalidOperationException("No development cards remaining in deck."));
    }

    private CatanState ApplyPlayKnight()
    {
        if (Stage is not (TurnStage.BuildTrade or TurnStage.PreRoll))
        {
            throw new InvalidOperationException("Playing a knight is only allowed during pre-roll or build/trade stage.");
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
        next._postDevCardStage = Stage;
        next.Stage = TurnStage.ChooseRobberLocation;
        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyPlayRoadBuilding()
    {
        if (Stage is not (TurnStage.BuildTrade or TurnStage.PreRoll))
        {
            throw new InvalidOperationException("Playing road building is only allowed during pre-roll or build/trade stage.");
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
        if (Stage == TurnStage.PreRoll)
        {
            next._postDevCardStage = TurnStage.PreRoll;
            next.Stage = TurnStage.BuildTrade;
        }

        next.RefreshVictory();
        return next;
    }

    private CatanState ApplyPlayMonopoly(ResourceType resource)
    {
        if (Stage is not (TurnStage.BuildTrade or TurnStage.PreRoll))
        {
            throw new InvalidOperationException("Playing monopoly is only allowed during pre-roll or build/trade stage.");
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
        if (Stage is not (TurnStage.BuildTrade or TurnStage.PreRoll))
        {
            throw new InvalidOperationException("Playing year of plenty is only allowed during pre-roll or build/trade stage.");
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

    private IReadOnlyList<int> LegalRobberVictims()
    {
        if (Stage != TurnStage.ChooseRobberVictim)
        {
            return [];
        }

        return RobberVictims(Board.RobberTile);
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
            legal.Add(new PlayKnightAction(this));
        }

        if (GetPlayableDevCardCount(DevCardType.RoadBuilding) > 0 && Board.RoadCount(CurrentPlayer) < Config.MaxRoads)
        {
            if (LegalBuildRoadEdges(requireCost: false).Count > 0)
            {
                legal.Add(new PlayRoadBuildingAction(this));
            }
        }

        if (GetPlayableDevCardCount(DevCardType.Monopoly) > 0)
        {
            foreach (var resource in CollectableResources())
            {
                legal.Add(new PlayMonopolyAction(this, resource));
            }
        }

        if (GetPlayableDevCardCount(DevCardType.YearOfPlenty) > 0)
        {
            var resources = CollectableResources().ToArray();
            for (var i = 0; i < resources.Length; i++)
            {
                for (var j = i; j < resources.Length; j++)
                {
                    legal.Add(new PlayYearOfPlentyAction(this, resources[i], resources[j]));
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

    private List<int> RobberVictims(int tileIndex)
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

        return [.. candidates.OrderBy(p => p)];
    }

    private CatanState ApplyChooseRobberTileNoSteal(int tileIndex)
    {
        var next = Clone();
        next.Board.RobberTile = tileIndex;
        next.Stage = next._postDevCardStage ?? TurnStage.BuildTrade;
        next._postDevCardStage = null;
        return next;
    }

    private CatanState ApplyChooseRobberTileAwaitVictim(int tileIndex)
    {
        var next = Clone();
        next.Board.RobberTile = tileIndex;
        next.Stage = TurnStage.ChooseRobberVictim;
        return next;
    }

    private CatanState ApplyChooseRobberTileSteal(int tileIndex, int victim, int resourceIndex)
    {
        var next = ApplyChooseRobberTileNoSteal(tileIndex);
        next._resources[victim, resourceIndex]--;
        next._resources[CurrentPlayer, resourceIndex]++;
        return next;
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

    internal void RefreshVictory()
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

    private static int RollTwoDice()
    {
        lock (RngLock)
        {
            return SharedRng.Next(1, 7) + SharedRng.Next(1, 7);
        }
    }

    private DevCardType? DrawRandomDevCard()
    {
        var totalRemaining = 0;
        for (var i = 0; i < DevCardCount; i++)
        {
            totalRemaining += _devDeckRemaining[i];
        }

        if (totalRemaining <= 0)
        {
            return null;
        }

        int pick;
        lock (RngLock)
        {
            pick = SharedRng.Next(totalRemaining);
        }

        var cumulative = 0;
        for (var i = 0; i < DevCardCount; i++)
        {
            cumulative += _devDeckRemaining[i];
            if (pick < cumulative)
            {
                return (DevCardType)i;
            }
        }

        return (DevCardType)(DevCardCount - 1);
    }

    private CatanState ApplyRollDiceOutcome(int roll)
    {
        var next = Clone();
        next.LastDiceRoll = roll;
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

    private CatanState ApplyBuyDevCardType(DevCardType cardType)
    {
        if (!CanAfford(Config.DevCardCost))
        {
            throw new InvalidOperationException("Current player cannot afford development card.");
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
            (int[])_pendingRoadBuildingPlacements.Clone(),
            _postDevCardStage,
            (byte[])_vertexPlacementRound.Clone());
    }

    private int GetPlayableDevCardCount(DevCardType cardType)
    {
        var typeIndex = (int)cardType;
        var total = _devCards[CurrentPlayer, typeIndex];
        var newThisTurn = _newDevCardsThisTurn[typeIndex];
        var playable = total - newThisTurn;
        return playable > 0 ? playable : 0;
    }

    internal static int ResourceToIndex(ResourceType resource) => resource switch
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
