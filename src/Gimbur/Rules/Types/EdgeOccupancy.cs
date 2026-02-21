namespace Gimbur.Rules;

/// <summary>
/// Occupancy state of an edge (path between two vertices).
/// Encodes the owning player (0 = empty, 1-4 = road by that player).
/// </summary>
public readonly record struct EdgeOccupancy(int Player)
{
    public static readonly EdgeOccupancy Empty = new(0);

    public bool IsEmpty => Player == 0;

    /// <summary>
    /// Converts to the serialization token value (0-4).
    /// </summary>
    public int ToToken() => Player;

    /// <summary>
    /// Parses from the serialization token value (0-4).
    /// </summary>
    public static EdgeOccupancy FromToken(int token) => token switch
    {
        >= 0 and <= 4 => new EdgeOccupancy(token),
        _ => throw new ArgumentOutOfRangeException(nameof(token), token, "Must be 0-4"),
    };
}
