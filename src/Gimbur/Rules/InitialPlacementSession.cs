namespace Gimbur.Rules;

/// <summary>
/// Tracks and applies the initial placement phase (settlement+road pairs).
/// Supports first round (player order ascending) and optional second round
/// (player order descending), based on <see cref="GameConfig.InitialPlacementRounds"/>.
/// </summary>
public sealed class InitialPlacementSession
{
    private readonly int[] _pairOrder;
    private int _pairIndex;
    private int? _pendingSettlementVertex;

    public Board Board { get; }
    public int PlayerCount { get; }
    public int CurrentPlayer => IsComplete ? 0 : _pairOrder[_pairIndex];
    public bool IsComplete { get; private set; }
    public int? PendingSettlementVertex => _pendingSettlementVertex;
    public TurnStage Stage { get; private set; }

    private InitialPlacementSession(Board board, int playerCount)
    {
        Board = board;
        PlayerCount = playerCount;

        if (playerCount < board.Config.MinPlayers || playerCount > board.Config.MaxPlayers)
            throw new ArgumentOutOfRangeException(
                nameof(playerCount),
                playerCount,
                $"Player count must be between {board.Config.MinPlayers} and {board.Config.MaxPlayers}.");

        var order = new List<int>(playerCount * board.Config.InitialPlacementRounds);
        for (var player = 1; player <= playerCount; player++)
            order.Add(player);

        if (board.Config.InitialPlacementRounds >= 2)
        {
            for (var player = playerCount; player >= 1; player--)
                order.Add(player);
        }

        _pairOrder = [.. order];
        _pairIndex = 0;
        _pendingSettlementVertex = null;
        IsComplete = _pairOrder.Length == 0;
        Stage = IsComplete ? TurnStage.PreRoll : TurnStage.PlaceFirstSettlement;
    }

    /// <summary>
    /// Creates a new randomized board and initial placement session.
    /// </summary>
    public static InitialPlacementSession Create(GameConfig config, int playerCount, Random rng)
    {
        var setup = BoardSetup.Generate(config.Map, rng);
        var board = new Board(setup, config);
        return new InitialPlacementSession(board, playerCount);
    }

    /// <summary>
    /// Returns legal settlement vertices for the current player in settlement stages.
    /// </summary>
    public IReadOnlyList<int> LegalSettlementVertices()
    {
        EnsureNotComplete();
        EnsureSettlementStage();

        var result = new List<int>();
        for (var vi = 0; vi < Board.Topology.VertexCount; vi++)
        {
            if (Board.CanPlaceSettlement(vi, CurrentPlayer))
                result.Add(vi);
        }
        return result;
    }

    /// <summary>
    /// Returns legal road edges for the current player in road stages.
    /// During initial placement, roads must connect to the settlement placed in the same pair.
    /// </summary>
    public IReadOnlyList<int> LegalRoadEdges()
    {
        EnsureNotComplete();
        EnsureRoadStage();

        if (_pendingSettlementVertex is null)
            throw new InvalidOperationException("No pending settlement for road placement.");

        var result = new List<int>();
        foreach (var edgeIndex in Board.Topology.VertexEdges[_pendingSettlementVertex.Value])
        {
            if (Board.EdgeOccupancy[edgeIndex].IsEmpty && Board.RoadCount(CurrentPlayer) < Board.Config.MaxRoads)
                result.Add(edgeIndex);
        }
        return result;
    }

    /// <summary>
    /// Places a settlement for the current player at a legal vertex.
    /// </summary>
    public void PlaceSettlement(int vertexIndex)
    {
        EnsureNotComplete();
        EnsureSettlementStage();

        if (!Board.CanPlaceSettlement(vertexIndex, CurrentPlayer))
            throw new InvalidOperationException(
                $"Vertex {vertexIndex} is not a legal settlement location for player {CurrentPlayer}.");

        Board.VertexOccupancy[vertexIndex] = new VertexOccupancy(BuildingType.Settlement, CurrentPlayer);
        _pendingSettlementVertex = vertexIndex;
        Stage = IsFirstRoundPair(_pairIndex) ? TurnStage.PlaceFirstRoad : TurnStage.PlaceSecondRoad;
    }

    /// <summary>
    /// Places a road for the current player at a legal edge and advances turn order.
    /// </summary>
    public void PlaceRoad(int edgeIndex)
    {
        EnsureNotComplete();
        EnsureRoadStage();

        var legalRoads = LegalRoadEdges();
        if (!legalRoads.Contains(edgeIndex))
            throw new InvalidOperationException(
                $"Edge {edgeIndex} is not a legal road location for player {CurrentPlayer}.");

        Board.EdgeOccupancy[edgeIndex] = new EdgeOccupancy(CurrentPlayer);
        _pendingSettlementVertex = null;

        _pairIndex++;
        if (_pairIndex >= _pairOrder.Length)
        {
            IsComplete = true;
            Stage = TurnStage.PreRoll;
            return;
        }

        Stage = IsFirstRoundPair(_pairIndex)
            ? TurnStage.PlaceFirstSettlement
            : TurnStage.PlaceSecondSettlement;
    }

    private bool IsFirstRoundPair(int pairIndex) => pairIndex < PlayerCount;

    private void EnsureNotComplete()
    {
        if (IsComplete)
            throw new InvalidOperationException("Initial placement is already complete.");
    }

    private void EnsureSettlementStage()
    {
        if (Stage is not (TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement))
            throw new InvalidOperationException($"Current stage is {Stage}; settlement placement is not allowed.");
    }

    private void EnsureRoadStage()
    {
        if (Stage is not (TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad))
            throw new InvalidOperationException($"Current stage is {Stage}; road placement is not allowed.");
    }
}
