using System.Collections.Immutable;

namespace Gimbur.Rules;

/// <summary>
/// Immutable game configuration defining all tunable parameters for a Catan game variant.
/// Covers building supply limits, victory conditions, card pools, costs, thresholds,
/// initial placement rules, and map configuration.
/// </summary>
public sealed class GameConfig
{
    // ── Map ─────────────────────────────────────────────────────────

    /// <summary>Map configuration (tile/number/port distributions and topology).</summary>
    public MapConfig Map { get; }

    // ── Players ─────────────────────────────────────────────────────

    /// <summary>Minimum number of players.</summary>
    public int MinPlayers { get; }

    /// <summary>Maximum number of players.</summary>
    public int MaxPlayers { get; }

    // ── Building supply per player ──────────────────────────────────

    /// <summary>Maximum settlements per player.</summary>
    public int MaxSettlements { get; }

    /// <summary>Maximum cities per player.</summary>
    public int MaxCities { get; }

    /// <summary>Maximum roads per player.</summary>
    public int MaxRoads { get; }

    // ── Victory conditions ──────────────────────────────────────────

    /// <summary>Victory points required to win.</summary>
    public int VictoryPointsToWin { get; }

    // ── Thresholds ──────────────────────────────────────────────────

    /// <summary>Minimum contiguous road length to claim longest road.</summary>
    public int LongestRoadMinimum { get; }

    /// <summary>Minimum knights played to claim largest army.</summary>
    public int LargestArmyMinimum { get; }

    /// <summary>
    /// Hand size threshold for discard on rolling 7. Players with more than
    /// this many resource cards must discard half (rounded down).
    /// </summary>
    public int DiscardThreshold { get; }

    // ── Resource bank ───────────────────────────────────────────────

    /// <summary>Number of cards of each resource type in the bank at game start.</summary>
    public int ResourceCardsPerType { get; }

    // ── Development card pool ───────────────────────────────────────

    /// <summary>Number of each development card type in the deck.</summary>
    public ImmutableDictionary<DevCardType, int> DevCardCounts { get; }

    /// <summary>Total number of development cards in the deck.</summary>
    public int TotalDevCards { get; }

    // ── Building costs ──────────────────────────────────────────────

    /// <summary>Cost to build a road: (Wood, Brick, Sheep, Wheat, Ore).</summary>
    public ImmutableDictionary<ResourceType, int> RoadCost { get; }

    /// <summary>Cost to build a settlement: (Wood, Brick, Sheep, Wheat, Ore).</summary>
    public ImmutableDictionary<ResourceType, int> SettlementCost { get; }

    /// <summary>Cost to upgrade to a city: (Wood, Brick, Sheep, Wheat, Ore).</summary>
    public ImmutableDictionary<ResourceType, int> CityCost { get; }

    /// <summary>Cost to buy a development card: (Wood, Brick, Sheep, Wheat, Ore).</summary>
    public ImmutableDictionary<ResourceType, int> DevCardCost { get; }

    // ── Initial placement ───────────────────────────────────────────

    /// <summary>
    /// Number of settlement+road placement rounds during setup.
    /// Standard Catan has 2 rounds (place first settlement/road, then second).
    /// Mini variant has 1 round and collects resources from that settlement.
    /// </summary>
    public int InitialPlacementRounds { get; }

    // ── Constructor ─────────────────────────────────────────────────

    public GameConfig(
        MapConfig map,
        int minPlayers,
        int maxPlayers,
        int maxSettlements,
        int maxCities,
        int maxRoads,
        int victoryPointsToWin,
        int longestRoadMinimum,
        int largestArmyMinimum,
        int discardThreshold,
        int resourceCardsPerType,
        ImmutableDictionary<DevCardType, int> devCardCounts,
        ImmutableDictionary<ResourceType, int> roadCost,
        ImmutableDictionary<ResourceType, int> settlementCost,
        ImmutableDictionary<ResourceType, int> cityCost,
        ImmutableDictionary<ResourceType, int> devCardCost,
        int initialPlacementRounds)
    {
        Map = map;
        MinPlayers = minPlayers;
        MaxPlayers = maxPlayers;
        MaxSettlements = maxSettlements;
        MaxCities = maxCities;
        MaxRoads = maxRoads;
        VictoryPointsToWin = victoryPointsToWin;
        LongestRoadMinimum = longestRoadMinimum;
        LargestArmyMinimum = largestArmyMinimum;
        DiscardThreshold = discardThreshold;
        ResourceCardsPerType = resourceCardsPerType;
        DevCardCounts = devCardCounts;
        TotalDevCards = devCardCounts.Values.Sum();
        RoadCost = roadCost;
        SettlementCost = settlementCost;
        CityCost = cityCost;
        DevCardCost = devCardCost;
        InitialPlacementRounds = initialPlacementRounds;
    }

    // ── Standard costs (shared) ─────────────────────────────────────

    private static ImmutableDictionary<ResourceType, int> MakeCost(
        int wood = 0, int brick = 0, int sheep = 0, int wheat = 0, int ore = 0)
    {
        var builder = ImmutableDictionary.CreateBuilder<ResourceType, int>();
        if (wood > 0) builder[ResourceType.Wood] = wood;
        if (brick > 0) builder[ResourceType.Brick] = brick;
        if (sheep > 0) builder[ResourceType.Sheep] = sheep;
        if (wheat > 0) builder[ResourceType.Wheat] = wheat;
        if (ore > 0) builder[ResourceType.Ore] = ore;
        return builder.ToImmutable();
    }

    // Road: 1 wood + 1 brick
    private static readonly ImmutableDictionary<ResourceType, int> StandardRoadCost =
        MakeCost(wood: 1, brick: 1);

    // Settlement: 1 wood + 1 brick + 1 sheep + 1 wheat
    private static readonly ImmutableDictionary<ResourceType, int> StandardSettlementCost =
        MakeCost(wood: 1, brick: 1, sheep: 1, wheat: 1);

    // City: 2 wheat + 3 ore
    private static readonly ImmutableDictionary<ResourceType, int> StandardCityCost =
        MakeCost(wheat: 2, ore: 3);

    // Dev card: 1 sheep + 1 wheat + 1 ore
    private static readonly ImmutableDictionary<ResourceType, int> StandardDevCardCost =
        MakeCost(sheep: 1, wheat: 1, ore: 1);

    // ── Precomputed configs ─────────────────────────────────────────

    /// <summary>
    /// Standard 3-4 player Catan game configuration.
    /// 19-tile board, 10 VP to win, 2 initial placement rounds.
    /// </summary>
    public static GameConfig Standard { get; } = new(
        map: MapConfig.Standard,
        minPlayers: 3,
        maxPlayers: 4,
        maxSettlements: 5,
        maxCities: 4,
        maxRoads: 15,
        victoryPointsToWin: 10,
        longestRoadMinimum: 5,
        largestArmyMinimum: 3,
        discardThreshold: 7,
        resourceCardsPerType: 19,
        devCardCounts: ImmutableDictionary.CreateRange(new[]
        {
            KeyValuePair.Create(DevCardType.Knight, 14),
            KeyValuePair.Create(DevCardType.VictoryPoint, 5),
            KeyValuePair.Create(DevCardType.RoadBuilding, 2),
            KeyValuePair.Create(DevCardType.Monopoly, 2),
            KeyValuePair.Create(DevCardType.YearOfPlenty, 2),
        }),
        roadCost: StandardRoadCost,
        settlementCost: StandardSettlementCost,
        cityCost: StandardCityCost,
        devCardCost: StandardDevCardCost,
        initialPlacementRounds: 2);

    /// <summary>
    /// Mini 2-player Catan game configuration.
    /// 7-tile board, 5 VP to win, 1 initial placement round. Resources are collected
    /// from that single setup settlement.
    /// Scaled supply limits and dev card pool.
    /// </summary>
    public static GameConfig Mini { get; } = new(
        map: MapConfig.Mini,
        minPlayers: 2,
        maxPlayers: 2,
        maxSettlements: 4,
        maxCities: 3,
        maxRoads: 10,
        victoryPointsToWin: 5,
        longestRoadMinimum: 4,
        largestArmyMinimum: 2,
        discardThreshold: 5,
        resourceCardsPerType: 10,
        devCardCounts: ImmutableDictionary.CreateRange(new[]
        {
            KeyValuePair.Create(DevCardType.Knight, 7),
            KeyValuePair.Create(DevCardType.VictoryPoint, 2),
            KeyValuePair.Create(DevCardType.RoadBuilding, 1),
            KeyValuePair.Create(DevCardType.Monopoly, 1),
            KeyValuePair.Create(DevCardType.YearOfPlenty, 1),
        }),
        roadCost: StandardRoadCost,
        settlementCost: StandardSettlementCost,
        cityCost: StandardCityCost,
        devCardCost: StandardDevCardCost,
        initialPlacementRounds: 1);

    /// <summary>
    /// Small 2-3 player Catan game configuration.
    /// 10-tile oval board, 7 VP to win, 2 initial placement rounds.
    /// Intermediate supply limits and dev card pool between Mini and Standard.
    /// </summary>
    public static GameConfig Small { get; } = new(
        map: MapConfig.Small,
        minPlayers: 2,
        maxPlayers: 3,
        maxSettlements: 5,
        maxCities: 3,
        maxRoads: 12,
        victoryPointsToWin: 7,
        longestRoadMinimum: 5,
        largestArmyMinimum: 3,
        discardThreshold: 6,
        resourceCardsPerType: 14,
        devCardCounts: ImmutableDictionary.CreateRange(new[]
        {
            KeyValuePair.Create(DevCardType.Knight, 10),
            KeyValuePair.Create(DevCardType.VictoryPoint, 3),
            KeyValuePair.Create(DevCardType.RoadBuilding, 1),
            KeyValuePair.Create(DevCardType.Monopoly, 1),
            KeyValuePair.Create(DevCardType.YearOfPlenty, 1),
        }),
        roadCost: StandardRoadCost,
        settlementCost: StandardSettlementCost,
        cityCost: StandardCityCost,
        devCardCost: StandardDevCardCost,
        initialPlacementRounds: 2);
}
