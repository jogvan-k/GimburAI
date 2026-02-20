namespace Gimbur.Rules;

/// <summary>
/// Mutable board state for a Catan game. Combines the fixed topology and randomized
/// setup with the mutable in-game state (vertex/edge occupancy, robber position).
/// </summary>
public sealed class Board
{
    /// <summary>The underlying board topology.</summary>
    public BoardTopology Topology { get; }

    /// <summary>The initial setup (tile resources, number tokens, port types).</summary>
    public BoardSetup Setup { get; }

    // ── Mutable state ───────────────────────────────────────────────

    /// <summary>Occupancy of each vertex, indexed by vertex index.</summary>
    public VertexOccupancy[] VertexOccupancy { get; }

    /// <summary>Occupancy of each edge, indexed by edge index.</summary>
    public EdgeOccupancy[] EdgeOccupancy { get; }

    /// <summary>Current tile index of the robber.</summary>
    public int RobberTile { get; set; }

    // ── Convenience accessors ───────────────────────────────────────

    /// <summary>Resource type of a tile.</summary>
    public ResourceType TileResource(int tileIndex) => Setup.TileResources[tileIndex];

    /// <summary>Number token of a tile (0 for desert).</summary>
    public int TileNumber(int tileIndex) => Setup.TileNumbers[tileIndex];

    /// <summary>Port type at a port position.</summary>
    public PortType PortType(int portIndex) => Setup.PortTypes[portIndex];

    /// <summary>The two vertex indices connected by a port.</summary>
    public (int VertexA, int VertexB) PortVertices(int portIndex) => Topology.Ports[portIndex];

    // ── Constructor ─────────────────────────────────────────────────

    public Board(BoardSetup setup)
    {
        Topology = setup.Topology;
        Setup = setup;
        VertexOccupancy = new VertexOccupancy[Topology.VertexCount];
        EdgeOccupancy = new EdgeOccupancy[Topology.EdgeCount];
        RobberTile = setup.InitialRobberTile;

        // Initialize all vertices and edges to empty.
        Array.Fill(VertexOccupancy, Rules.VertexOccupancy.Empty);
        Array.Fill(EdgeOccupancy, Rules.EdgeOccupancy.Empty);
    }

    /// <summary>
    /// Creates a deep copy of this board (for MCTS state branching).
    /// </summary>
    public Board Clone()
    {
        var clone = new Board(Setup);
        Array.Copy(VertexOccupancy, clone.VertexOccupancy, VertexOccupancy.Length);
        Array.Copy(EdgeOccupancy, clone.EdgeOccupancy, EdgeOccupancy.Length);
        clone.RobberTile = RobberTile;
        return clone;
    }

    // ── Query methods ───────────────────────────────────────────────

    /// <summary>
    /// Returns the best trade ratio a player has for a given resource,
    /// considering port access. Default is 4:1; generic port gives 3:1;
    /// matching resource port gives 2:1.
    /// </summary>
    public int TradeRatio(int player, ResourceType resource)
    {
        var ratio = 4;
        for (var pi = 0; pi < Topology.PortCount; pi++)
        {
            var (va, vb) = Topology.Ports[pi];
            var ownsPort = VertexOccupancy[va].Player == player
                        || VertexOccupancy[vb].Player == player;
            if (!ownsPort) continue;

            var portType = Setup.PortTypes[pi];
            if (portType == Rules.PortType.Generic)
            {
                ratio = Math.Min(ratio, 3);
            }
            else
            {
                // Resource-specific port: PortType value - 1 maps to ResourceType.
                // PortType.Wood(2) -> ResourceType.Wood(1), etc.
                var portResource = (ResourceType)((int)portType - 1);
                if (portResource == resource)
                    ratio = Math.Min(ratio, 2);
            }
        }
        return ratio;
    }

    /// <summary>
    /// Returns tile indices that produce resources for a given dice roll.
    /// </summary>
    public IEnumerable<int> TilesForRoll(int roll)
    {
        for (var ti = 0; ti < Topology.TileCount; ti++)
        {
            if (Setup.TileNumbers[ti] == roll && ti != RobberTile)
                yield return ti;
        }
    }

    /// <summary>
    /// Returns whether a vertex is available for settlement placement.
    /// A vertex is available if it is empty and no adjacent vertex has a building.
    /// </summary>
    public bool CanPlaceSettlement(int vertexIndex)
    {
        if (!VertexOccupancy[vertexIndex].IsEmpty)
            return false;

        // Distance rule: no building on any adjacent vertex.
        foreach (var neighbor in Topology.VertexNeighbors[vertexIndex])
        {
            if (!VertexOccupancy[neighbor].IsEmpty)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Returns whether a vertex can be upgraded to a city by the given player.
    /// </summary>
    public bool CanUpgradeToCity(int vertexIndex, int player)
    {
        var occ = VertexOccupancy[vertexIndex];
        return occ.Building == BuildingType.Settlement && occ.Player == player;
    }

    /// <summary>
    /// Returns whether an edge is available for road placement by the given player.
    /// An edge must be empty and adjacent to the player's existing road or building.
    /// </summary>
    public bool CanPlaceRoad(int edgeIndex, int player)
    {
        if (!EdgeOccupancy[edgeIndex].IsEmpty)
            return false;

        // Must connect to player's existing network.
        var (va, vb) = Topology.Edges[edgeIndex];
        return IsConnectedToPlayer(va, player) || IsConnectedToPlayer(vb, player);
    }

    private bool IsConnectedToPlayer(int vertexIndex, int player)
    {
        // Player has a building here.
        if (VertexOccupancy[vertexIndex].Player == player)
            return true;

        // Player has a road on an adjacent edge (and no opponent building blocking).
        if (!VertexOccupancy[vertexIndex].IsEmpty && VertexOccupancy[vertexIndex].Player != player)
            return false; // Opponent building blocks connection.

        foreach (var ei in Topology.VertexEdges[vertexIndex])
        {
            if (EdgeOccupancy[ei].Player == player)
                return true;
        }
        return false;
    }
}
