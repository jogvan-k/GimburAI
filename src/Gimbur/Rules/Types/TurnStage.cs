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

    /// <summary>Choose robber victim (only when multiple adjacent opponents can be robbed).</summary>
    ChooseRobberVictim = 6,

    /// <summary>Build and trade phase.</summary>
    BuildTrade = 7,

    /// <summary>Place the first free Road Building road.</summary>
    PlaceRoadBuildingFirst = 8,

    /// <summary>Place a paid or second free committed road.</summary>
    PlaceRoadCommitted = 9,

    /// <summary>Place a paid committed settlement.</summary>
    PlaceSettlementCommitted = 10,

    /// <summary>Place a committed city upgrade.</summary>
    PlaceCityCommitted = 11,

    /// <summary>Choose the resource given to the bank.</summary>
    ChooseBankTradeGive = 12,

    /// <summary>Choose the resource received from the bank.</summary>
    ChooseBankTradeReceive = 13,

    /// <summary>Choose the resource monopolized from opponents.</summary>
    ChooseMonopolyResource = 14,

    /// <summary>Choose the first Year of Plenty resource.</summary>
    ChooseYearOfPlentyFirst = 15,

    /// <summary>Choose the second Year of Plenty resource.</summary>
    ChooseYearOfPlentySecond = 16,
}
