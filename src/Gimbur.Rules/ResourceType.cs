namespace Gimbur.Rules;

/// <summary>
/// Resource type for tiles and resource-specific ports.
/// Values match the serialization spec (docs/state-serialization.md).
/// </summary>
public enum ResourceType : byte
{
    Desert = 0,
    Wood = 1,
    Brick = 2,
    Sheep = 3,
    Wheat = 4,
    Ore = 5,
}
