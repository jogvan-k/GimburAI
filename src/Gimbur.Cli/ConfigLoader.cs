using System.Text.Json;

namespace Gimbur.Cli;

/// <summary>
/// Loads a JSON configuration file with optional <c>//</c> comment support.
/// Keys are expected in camelCase matching the CLI option names without the
/// leading dashes.
/// </summary>
internal static class ConfigLoader
{
    /// <summary>
    /// Reads a JSON file, strips <c>//</c> line comments, and returns the
    /// parsed <see cref="JsonElement"/>.
    /// </summary>
    public static JsonElement Load(FileInfo file)
    {
        var text = File.ReadAllText(file.FullName);
        var stripped = StripJsonComments(text);
        return JsonDocument.Parse(stripped).RootElement;
    }

    /// <summary>
    /// Returns the value of a string property, or <c>null</c> if the key
    /// is missing or the value is not a string.
    /// </summary>
    public static string? GetString(JsonElement root, string key)
    {
        return root.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.String
            ? el.GetString()
            : null;
    }

    /// <summary>
    /// Returns the value of an integer property, or <c>null</c> if the key
    /// is missing or the value is not a number.
    /// </summary>
    public static int? GetInt(JsonElement root, string key)
    {
        return root.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.Number
            ? el.GetInt32()
            : null;
    }

    /// <summary>
    /// Returns the value of an unsigned integer property, or <c>null</c> if
    /// the key is missing or the value is not a number.
    /// </summary>
    public static uint? GetUInt(JsonElement root, string key)
    {
        return root.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.Number
            ? el.GetUInt32()
            : null;
    }

    public static double? GetDouble(JsonElement root, string key)
    {
        return root.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.Number
            ? el.GetDouble()
            : null;
    }

    /// <summary>
    /// Returns the value of a boolean property, or <c>null</c> if the key
    /// is missing or the value is not a boolean.
    /// </summary>
    public static bool? GetBool(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var el)) return null;
        return el.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };
    }

    /// <summary>
    /// Returns the value of a string-array property, or <c>null</c> if the
    /// key is missing or the value is not an array.
    /// </summary>
    public static string[]? GetStringArray(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var el) || el.ValueKind != JsonValueKind.Array)
            return null;

        var result = new string[el.GetArrayLength()];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = el[i].GetString() ?? string.Empty;
        }

        return result;
    }

    /// <summary>
    /// Removes <c>// ...</c> comments from JSON text while preserving
    /// strings that contain <c>//</c>.
    /// </summary>
    private static string StripJsonComments(string text)
    {
        var sb = new System.Text.StringBuilder(text.Length);
        foreach (var rawLine in text.Split('\n'))
        {
            var line = rawLine;
            var inString = false;
            var escape = false;

            for (var i = 0; i < line.Length; i++)
            {
                var ch = line[i];
                if (escape) { escape = false; continue; }
                if (ch == '\\') { escape = true; continue; }
                if (ch == '"') { inString = !inString; continue; }
                if (ch == '/' && !inString && i + 1 < line.Length && line[i + 1] == '/')
                {
                    line = line[..i].TrimEnd();
                    break;
                }
            }

            sb.AppendLine(line);
        }

        return sb.ToString();
    }
}
