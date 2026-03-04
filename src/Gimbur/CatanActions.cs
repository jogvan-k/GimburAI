using Gimbur.Rules;
using Kjarni;

namespace Gimbur;

public abstract class CatanAction : IComparable
{
    protected CatanAction(CatanState origin)
    {
        OriginState = origin;
    }

    public CatanState OriginState { get; }

    /// <summary>
    /// Stable numeric tag identifying the concrete action type.
    /// Used for deterministic ordering, equality, and hashing.
    /// </summary>
    public abstract byte TypeTag { get; }

    public virtual int Arg1 => 0;

    public virtual int Arg2 => 0;

    public int TargetIndex => Arg1;

    public abstract ICoreState DoCoreAction();

    public int CompareTo(object? obj)
    {
        if (obj is not CatanAction other)
        {
            throw new ArgumentException("Cannot compare CatanAction with a different type", nameof(obj));
        }

        var typeCompare = TypeTag.CompareTo(other.TypeTag);
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
            && TypeTag == other.TypeTag
            && Arg1 == other.Arg1
            && Arg2 == other.Arg2;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(TypeTag, Arg1, Arg2);
    }
}

public abstract class CatanDeterministicAction : CatanAction, IDeterministicCoreAction
{
    protected CatanDeterministicAction(CatanState origin)
        : base(origin)
    {
    }

    public ICoreState State() => DoCoreAction();
}

public abstract class CatanStochasticAction : CatanAction, IStochasticCoreAction
{
    protected CatanStochasticAction(CatanState origin)
        : base(origin)
    {
    }

    public abstract Tuple<int, ICoreState>[] Outcomes();
}

public sealed class PlaceSettlementAction(CatanState origin, int vertexIndex) : CatanDeterministicAction(origin)
{
    public const byte Tag = 0;
    public int VertexIndex { get; } = vertexIndex;
    public override int Arg1 => VertexIndex;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class PlaceRoadAction(CatanState origin, int edgeIndex) : CatanDeterministicAction(origin)
{
    public const byte Tag = 1;
    public int EdgeIndex { get; } = edgeIndex;
    public override int Arg1 => EdgeIndex;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class RollDiceAction(CatanState origin) : CatanStochasticAction(origin)
{
    public const byte Tag = 2;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
    public override Tuple<int, ICoreState>[] Outcomes() =>
        [.. OriginState.RollDiceOutcomes().Select(x => Tuple.Create(x.Weight, (ICoreState)x.State))];
}

public sealed class ChooseRobberTileAction(CatanState origin, int tileIndex) : CatanStochasticAction(origin)
{
    public const byte Tag = 3;
    public int TileIndex { get; } = tileIndex;
    public override int Arg1 => TileIndex;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
    public override Tuple<int, ICoreState>[] Outcomes() =>
        [.. OriginState.ChooseRobberTileOutcomes(TileIndex).Select(x => Tuple.Create(x.Weight, (ICoreState)x.State))];
}

public sealed class ChooseRobberVictimAction(CatanState origin, int victimPlayer) : CatanStochasticAction(origin)
{
    public const byte Tag = 12;
    public int VictimPlayer { get; } = victimPlayer;
    public override int Arg1 => VictimPlayer;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
    public override Tuple<int, ICoreState>[] Outcomes() =>
        [.. OriginState.ChooseRobberVictimOutcomes(VictimPlayer).Select(x => Tuple.Create(x.Weight, (ICoreState)x.State))];
}

public sealed class BuildCityAction(CatanState origin, int vertexIndex) : CatanDeterministicAction(origin)
{
    public const byte Tag = 4;
    public int VertexIndex { get; } = vertexIndex;
    public override int Arg1 => VertexIndex;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class BankTradeAction(CatanState origin, ResourceType give, ResourceType receive) : CatanDeterministicAction(origin)
{
    public const byte Tag = 5;
    public ResourceType Give { get; } = give;
    public ResourceType Receive { get; } = receive;
    public override int Arg1 => (int)Give;
    public override int Arg2 => (int)Receive;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class BuyDevCardAction(CatanState origin) : CatanStochasticAction(origin)
{
    public const byte Tag = 6;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
    public override Tuple<int, ICoreState>[] Outcomes() =>
        [.. OriginState.BuyDevCardOutcomes().Select(x => Tuple.Create(x.Weight, (ICoreState)x.State))];
}

public sealed class PlayKnightAction(CatanState origin) : CatanStochasticAction(origin)
{
    public const byte Tag = 7;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
    public override Tuple<int, ICoreState>[] Outcomes() =>
        [.. OriginState.PlayKnightOutcomes().Select(x => Tuple.Create(x.Weight, (ICoreState)x.State))];
}

public sealed class PlayRoadBuildingAction(CatanState origin) : CatanDeterministicAction(origin)
{
    public const byte Tag = 8;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class PlayMonopolyAction(CatanState origin, ResourceType resource) : CatanDeterministicAction(origin)
{
    public const byte Tag = 9;
    public ResourceType Resource { get; } = resource;
    public override int Arg1 => (int)Resource;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class PlayYearOfPlentyAction(CatanState origin, ResourceType first, ResourceType second) : CatanDeterministicAction(origin)
{
    public const byte Tag = 10;
    public ResourceType First { get; } = first;
    public ResourceType Second { get; } = second;
    public override int Arg1 => (int)First;
    public override int Arg2 => (int)Second;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}

public sealed class EndTurnAction(CatanState origin) : CatanDeterministicAction(origin)
{
    public const byte Tag = 11;
    public override byte TypeTag => Tag;
    public override ICoreState DoCoreAction() => OriginState.Apply(this);
}
