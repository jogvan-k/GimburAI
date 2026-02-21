namespace Gimbur.Rules;

/// <summary>
/// Port type (harbor). Generic 3:1 or resource-specific 2:1.
/// Values match the serialization spec (docs/state-serialization.md).
/// </summary>
public enum PortType : byte
{
    /// <summary>3:1 generic trade ratio.</summary>
    Generic = 1,

    /// <summary>2:1 wood trade ratio.</summary>
    Wood = 2,

    /// <summary>2:1 brick trade ratio.</summary>
    Brick = 3,

    /// <summary>2:1 sheep trade ratio.</summary>
    Sheep = 4,

    /// <summary>2:1 wheat trade ratio.</summary>
    Wheat = 5,

    /// <summary>2:1 ore trade ratio.</summary>
    Ore = 6,
}
