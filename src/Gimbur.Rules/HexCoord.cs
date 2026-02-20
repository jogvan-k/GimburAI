namespace Gimbur.Rules;

/// <summary>
/// Axial hex coordinate (q, r).
/// </summary>
public readonly record struct HexCoord(int Q, int R) : IComparable<HexCoord>
{
    /// <summary>
    /// Axial neighbor directions, clockwise from east:
    /// E(+1,0), SE(+1,-1), SW(0,-1), W(-1,0), NW(-1,+1), NE(0,+1).
    /// </summary>
    public static readonly HexCoord[] Directions =
    [
        new(1, 0), new(1, -1), new(0, -1),
        new(-1, 0), new(-1, 1), new(0, 1),
    ];

    public static HexCoord operator +(HexCoord a, HexCoord b)
        => new(a.Q + b.Q, a.R + b.R);

    public int CompareTo(HexCoord other)
    {
        var cmp = Q.CompareTo(other.Q);
        return cmp != 0 ? cmp : R.CompareTo(other.R);
    }
}
