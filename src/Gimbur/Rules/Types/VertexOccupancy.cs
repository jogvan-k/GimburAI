namespace Gimbur.Rules;

/// <summary>
/// Occupancy state of a vertex (intersection).
/// Encodes both the building type and the owning player.
/// </summary>
public readonly record struct VertexOccupancy(BuildingType Building, int Player)
{
    public static readonly VertexOccupancy Empty = new(BuildingType.None, 0);

    public bool IsEmpty => Building == BuildingType.None;
}
