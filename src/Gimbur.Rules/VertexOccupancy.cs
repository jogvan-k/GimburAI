namespace Gimbur.Rules;

/// <summary>
/// Building type placed on a vertex.
/// </summary>
public enum BuildingType : byte
{
    None = 0,
    Settlement = 1,
    City = 2,
}

/// <summary>
/// Occupancy state of a vertex (intersection).
/// Encodes both the building type and the owning player.
/// </summary>
/// <remarks>
/// Serialization encoding: 0 = empty, 1-4 = settlement by player 1-4,
/// 5-8 = city by player 1-4 (playerId + 4).
/// </remarks>
public readonly record struct VertexOccupancy(BuildingType Building, int Player)
{
    public static readonly VertexOccupancy Empty = new(BuildingType.None, 0);

    public bool IsEmpty => Building == BuildingType.None;

    /// <summary>
    /// Converts to the serialization token value (0-8).
    /// </summary>
    public int ToToken() => Building switch
    {
        BuildingType.None => 0,
        BuildingType.Settlement => Player,
        BuildingType.City => Player + 4,
        _ => throw new InvalidOperationException($"Unknown building type: {Building}"),
    };

    /// <summary>
    /// Parses from the serialization token value (0-8).
    /// </summary>
    public static VertexOccupancy FromToken(int token) => token switch
    {
        0 => Empty,
        >= 1 and <= 4 => new VertexOccupancy(BuildingType.Settlement, token),
        >= 5 and <= 8 => new VertexOccupancy(BuildingType.City, token - 4),
        _ => throw new ArgumentOutOfRangeException(nameof(token), token, "Must be 0-8"),
    };
}
