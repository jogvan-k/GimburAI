using Gimbur.Rules;

namespace Gimbur;

/// <summary>Current-only fixed complete-policy vocabulary for Catan actions.</summary>
public sealed class CatanPolicySerializer
{
    public const int ResourceCount = 5;
    public const int BuyTradeCount = 5;
    public const int PlayDevCardCount = 4;
    public const int ControlCount = 2;

    private readonly BoardTopology _topology;
    private readonly int _playerCount;

    public CatanPolicySerializer(BoardTopology topology, int playerCount)
    {
        _topology = topology;
        _playerCount = playerCount;
    }

    public int TilesOffset => 0;
    public int VerticesOffset => _topology.TileCount;
    public int EdgesOffset => VerticesOffset + _topology.VertexCount;
    public int ResourcesOffset => EdgesOffset + _topology.EdgeCount;
    public int BuyTradeOffset => ResourcesOffset + ResourceCount;
    public int PlayDevCardOffset => BuyTradeOffset + BuyTradeCount;
    public int VictimsOffset => PlayDevCardOffset + PlayDevCardCount;
    public int ControlsOffset => VictimsOffset + _playerCount;
    public int PolicySize => ControlsOffset + ControlCount;

    public int IndexOf(CatanState state, CatanAction action) => action switch
    {
        ChooseRobberTileAction robber => TilesOffset + robber.TileIndex,
        PlaceSettlementAction settlement => VerticesOffset + settlement.VertexIndex,
        PlaceCityAction city => VerticesOffset + city.VertexIndex,
        PlaceRoadAction road => EdgesOffset + road.EdgeIndex,
        ChooseBankTradeGiveAction choice => ResourceIndex(choice.Resource),
        ChooseBankTradeReceiveAction choice => ResourceIndex(choice.Resource),
        ChooseMonopolyResourceAction choice => ResourceIndex(choice.Resource),
        ChooseYearOfPlentyResourceAction choice => ResourceIndex(choice.Resource),
        BuyRoadAction => BuyTradeOffset,
        BuySettlementAction => BuyTradeOffset + 1,
        UpgradeCityAction => BuyTradeOffset + 2,
        BuyDevCardAction => BuyTradeOffset + 3,
        TradeWithBankAction => BuyTradeOffset + 4,
        PlayKnightAction => PlayDevCardOffset,
        PlayRoadBuildingAction => PlayDevCardOffset + 1,
        PlayMonopolyAction => PlayDevCardOffset + 2,
        PlayYearOfPlentyAction => PlayDevCardOffset + 3,
        ChooseRobberVictimAction victim => VictimsOffset + CanonicalPlayerSlot(
            victim.VictimPlayer, state.CurrentPlayer, state.PlayerCount),
        RollDiceAction => ControlsOffset,
        EndTurnAction => ControlsOffset + 1,
        _ => throw new InvalidOperationException(
            $"Action {action.GetType().Name} has no complete-policy index."),
    };

    public int TransformIndex(int policyIndex, SymmetryPermutation permutation)
    {
        if (policyIndex < VerticesOffset)
            return TilesOffset + permutation.Tiles[policyIndex - TilesOffset];
        if (policyIndex < EdgesOffset)
            return VerticesOffset + permutation.Vertices[policyIndex - VerticesOffset];
        if (policyIndex < ResourcesOffset)
            return EdgesOffset + permutation.Edges[policyIndex - EdgesOffset];
        return policyIndex;
    }

    public double[] MaskAndNormalize(
        IReadOnlyList<double> policy,
        IReadOnlyList<int> legalIndices)
    {
        var result = new double[legalIndices.Count];
        if (result.Length == 0)
            return result;
        if (policy.Count == PolicySize && policy.All(value => double.IsFinite(value) && value >= 0))
        {
            for (var i = 0; i < result.Length; i++)
                result[i] = policy[legalIndices[i]];
        }
        var total = result.Sum();
        if (total > 0 && double.IsFinite(total))
        {
            for (var i = 0; i < result.Length; i++)
                result[i] /= total;
        }
        else
        {
            Array.Fill(result, 1.0 / result.Length);
        }
        return result;
    }

    private int ResourceIndex(ResourceType resource)
    {
        if (resource is < ResourceType.Wood or > ResourceType.Ore)
            throw new ArgumentOutOfRangeException(nameof(resource));
        return ResourcesOffset + (int)resource - (int)ResourceType.Wood;
    }

    private static int CanonicalPlayerSlot(int player, int actingPlayer, int playerCount) =>
        (player - actingPlayer + playerCount) % playerCount;
}
