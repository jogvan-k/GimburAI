namespace Gimbur.Rules;

/// <summary>
/// Encodes and decodes single-digit Crockford base-32 values (0–31).
/// Alphabet: 0123456789ABCDEFGHJKMNPQRSTVWXYZ
/// </summary>
public static class CrockfordBase32
{
    private const string Alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

    /// <summary>
    /// Encodes a value (0–31) as a single Crockford base-32 character.
    /// </summary>
    public static char Encode(int value)
    {
        if (value is < 0 or > 31)
        {
            throw new ArgumentOutOfRangeException(nameof(value), value, "Must be 0–31.");
        }

        return Alphabet[value];
    }

    /// <summary>
    /// Decodes a single Crockford base-32 character to its integer value (0–31).
    /// Case-insensitive. Accepts common confusables (I/L → 1, O → 0).
    /// </summary>
    public static int Decode(char c)
    {
        // Normalize to uppercase.
        c = char.ToUpperInvariant(c);

        // Handle Crockford confusables.
        if (c is 'I' or 'L')
        {
            return 1;
        }

        if (c == 'O')
        {
            return 0;
        }

        var index = Alphabet.IndexOf(c);
        if (index < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(c), c, "Not a valid Crockford base-32 character.");
        }

        return index;
    }
}
