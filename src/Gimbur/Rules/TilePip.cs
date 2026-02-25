namespace Gimbur.Rules;

/// <summary>
/// Encodes and decodes tile number tokens as lowercase letters (a–k)
/// that represent pip count and side (below/above 7).
/// <para>
/// The alphabet is intentionally disjoint from Crockford base-32
/// (uppercase + digits) so that a tokenizer can distinguish tile
/// likelihood tokens from all other field types.
/// </para>
/// <list type="table">
///   <listheader><term>Char</term><description>Pips / Side / Number</description></listheader>
///   <item><term>a</term><description>0 pips — desert (number 0)</description></item>
///   <item><term>b</term><description>1 pip, low  (number 2)</description></item>
///   <item><term>c</term><description>1 pip, high (number 12)</description></item>
///   <item><term>d</term><description>2 pips, low  (number 3)</description></item>
///   <item><term>e</term><description>2 pips, high (number 11)</description></item>
///   <item><term>f</term><description>3 pips, low  (number 4)</description></item>
///   <item><term>g</term><description>3 pips, high (number 10)</description></item>
///   <item><term>h</term><description>4 pips, low  (number 5)</description></item>
///   <item><term>i</term><description>4 pips, high (number 9)</description></item>
///   <item><term>j</term><description>5 pips, low  (number 6)</description></item>
///   <item><term>k</term><description>5 pips, high (number 8)</description></item>
/// </list>
/// </summary>
public static class TilePip
{
    /// <summary>
    /// Encodes a tile number (0, 2–6, 8–12) as a single lowercase letter.
    /// </summary>
    public static char Encode(int tileNumber) => tileNumber switch
    {
        0 => 'a',
        2 => 'b',
        12 => 'c',
        3 => 'd',
        11 => 'e',
        4 => 'f',
        5 => 'h',
        10 => 'g',
        6 => 'j',
        9 => 'i',
        8 => 'k',
        _ => throw new ArgumentOutOfRangeException(
            nameof(tileNumber), tileNumber,
            "Must be 0 or 2–6 or 8–12."),
    };

    /// <summary>
    /// Decodes a lowercase letter (a–k) back to the original tile number.
    /// </summary>
    public static int Decode(char c) => c switch
    {
        'a' => 0,
        'b' => 2,
        'c' => 12,
        'd' => 3,
        'e' => 11,
        'f' => 4,
        'g' => 10,
        'h' => 5,
        'i' => 9,
        'j' => 6,
        'k' => 8,
        _ => throw new ArgumentOutOfRangeException(
            nameof(c), c,
            "Must be a lowercase letter a–k."),
    };
}
