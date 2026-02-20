namespace Gimbur.Rules;

/// <summary>
/// Turn stage within a player's turn.
/// Values match the serialization spec (docs/state-serialization.md).
/// </summary>
public enum TurnStage : byte
{
    /// <summary>Initial placement: place 1st settlement.</summary>
    PlaceFirstSettlement = 0,

    /// <summary>Initial placement: place 1st road.</summary>
    PlaceFirstRoad = 1,

    /// <summary>Initial placement: place 2nd settlement.</summary>
    PlaceSecondSettlement = 2,

    /// <summary>Initial placement: place 2nd road.</summary>
    PlaceSecondRoad = 3,

    /// <summary>Pre-roll phase (may play dev card before rolling).</summary>
    PreRoll = 4,

    /// <summary>Choose robber location (after rolling 7 or playing knight).</summary>
    ChooseRobberLocation = 5,

    /// <summary>Build and trade phase.</summary>
    BuildTrade = 6,
}
