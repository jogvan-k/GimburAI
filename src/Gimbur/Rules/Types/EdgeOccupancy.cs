namespace Gimbur.Rules;

/// <summary>
/// Occupancy state of an edge (path between two vertices).
/// Encodes the owning player (0 = empty, 1–4 = road by that player).
/// </summary>
public readonly record struct EdgeOccupancy(int Player)
{
    public static readonly EdgeOccupancy Empty = new(0);

    public bool IsEmpty => Player == 0;
}
