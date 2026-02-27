using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class TypeTests
{
    // ── VertexOccupancy ─────────────────────────────────────────────

    [Test]
    public void VertexOccupancy_Empty_IsEmpty()
    {
        Assert.That(VertexOccupancy.Empty.IsEmpty, Is.True);
        Assert.That(VertexOccupancy.Empty.Building, Is.EqualTo(BuildingType.None));
        Assert.That(VertexOccupancy.Empty.Player, Is.EqualTo(0));
    }

    [TestCase(1)]
    [TestCase(2)]
    [TestCase(3)]
    [TestCase(4)]
    public void VertexOccupancy_Settlement_IsNotEmpty(int player)
    {
        var occ = new VertexOccupancy(BuildingType.Settlement, player);
        Assert.That(occ.IsEmpty, Is.False);
        Assert.That(occ.Building, Is.EqualTo(BuildingType.Settlement));
        Assert.That(occ.Player, Is.EqualTo(player));
    }

    [TestCase(1)]
    [TestCase(2)]
    [TestCase(3)]
    [TestCase(4)]
    public void VertexOccupancy_City_IsNotEmpty(int player)
    {
        var occ = new VertexOccupancy(BuildingType.City, player);
        Assert.That(occ.IsEmpty, Is.False);
        Assert.That(occ.Building, Is.EqualTo(BuildingType.City));
        Assert.That(occ.Player, Is.EqualTo(player));
    }

    // ── EdgeOccupancy ───────────────────────────────────────────────

    [Test]
    public void EdgeOccupancy_Empty_IsEmpty()
    {
        Assert.That(EdgeOccupancy.Empty.IsEmpty, Is.True);
        Assert.That(EdgeOccupancy.Empty.Player, Is.EqualTo(0));
    }

    [TestCase(1)]
    [TestCase(2)]
    [TestCase(3)]
    [TestCase(4)]
    public void EdgeOccupancy_WithPlayer_IsNotEmpty(int player)
    {
        var occ = new EdgeOccupancy(player);
        Assert.That(occ.IsEmpty, Is.False);
        Assert.That(occ.Player, Is.EqualTo(player));
    }

    // ── StateToken: Resource ────────────────────────────────────────

    [TestCase(ResourceType.Desert, 'd')]
    [TestCase(ResourceType.Wood, 'w')]
    [TestCase(ResourceType.Brick, 'b')]
    [TestCase(ResourceType.Sheep, 's')]
    [TestCase(ResourceType.Wheat, 'W')]
    [TestCase(ResourceType.Ore, 'o')]
    public void StateToken_Resource_RoundTrips(ResourceType resource, char expected)
    {
        var encoded = StateToken.EncodeResource(resource);
        Assert.That(encoded, Is.EqualTo(expected));
        Assert.That(StateToken.DecodeResource(encoded), Is.EqualTo(resource));
    }

    // ── StateToken: Port ────────────────────────────────────────────

    [TestCase(PortType.Generic, 'g')]
    [TestCase(PortType.Wood, 'w')]
    [TestCase(PortType.Brick, 'b')]
    [TestCase(PortType.Sheep, 's')]
    [TestCase(PortType.Wheat, 'W')]
    [TestCase(PortType.Ore, 'o')]
    public void StateToken_Port_RoundTrips(PortType port, char expected)
    {
        var encoded = StateToken.EncodePort(port);
        Assert.That(encoded, Is.EqualTo(expected));
        Assert.That(StateToken.DecodePort(encoded), Is.EqualTo(port));
    }

    // ── StateToken: Tile Number ─────────────────────────────────────

    [TestCase(0, '0', 'n')]
    [TestCase(2, '1', 'l')]
    [TestCase(12, '1', 'h')]
    [TestCase(3, '2', 'l')]
    [TestCase(11, '2', 'h')]
    [TestCase(4, '3', 'l')]
    [TestCase(10, '3', 'h')]
    [TestCase(5, '4', 'l')]
    [TestCase(9, '4', 'h')]
    [TestCase(6, '5', 'l')]
    [TestCase(8, '5', 'h')]
    public void StateToken_TileNumber_RoundTrips(int tileNumber, char expectedPips, char expectedSide)
    {
        var pips = StateToken.EncodeTilePips(tileNumber);
        var side = StateToken.EncodeTileSide(tileNumber);
        Assert.That(pips, Is.EqualTo(expectedPips));
        Assert.That(side, Is.EqualTo(expectedSide));
        Assert.That(StateToken.DecodeTileNumber(pips, side), Is.EqualTo(tileNumber));
    }

    // ── StateToken: Player ID ───────────────────────────────────────

    [TestCase(0, '_')]
    [TestCase(1, '-')]
    [TestCase(2, '+')]
    [TestCase(3, '*')]
    [TestCase(4, '^')]
    public void StateToken_Player_RoundTrips(int player, char expected)
    {
        var encoded = StateToken.EncodePlayer(player);
        Assert.That(encoded, Is.EqualTo(expected));
        Assert.That(StateToken.DecodePlayer(encoded), Is.EqualTo(player));
    }

    // ── StateToken: Building Type ───────────────────────────────────

    [TestCase(BuildingType.None, '.')]
    [TestCase(BuildingType.Settlement, 'v')]
    [TestCase(BuildingType.City, 'c')]
    public void StateToken_Building_RoundTrips(BuildingType building, char expected)
    {
        var encoded = StateToken.EncodeBuilding(building);
        Assert.That(encoded, Is.EqualTo(expected));
        Assert.That(StateToken.DecodeBuilding(encoded), Is.EqualTo(building));
    }

    // ── StateToken: Turn Stage ──────────────────────────────────────

    [TestCase(TurnStage.PlaceFirstSettlement, 'a')]
    [TestCase(TurnStage.PlaceFirstRoad, 'e')]
    [TestCase(TurnStage.PlaceSecondSettlement, 'f')]
    [TestCase(TurnStage.PlaceSecondRoad, 'i')]
    [TestCase(TurnStage.PreRoll, 'r')]
    [TestCase(TurnStage.ChooseRobberLocation, 'x')]
    [TestCase(TurnStage.ChooseRobberVictim, 'y')]
    [TestCase(TurnStage.BuildTrade, 't')]
    public void StateToken_TurnStage_RoundTrips(TurnStage stage, char expected)
    {
        var encoded = StateToken.EncodeTurnStage(stage);
        Assert.That(encoded, Is.EqualTo(expected));
        Assert.That(StateToken.DecodeTurnStage(encoded), Is.EqualTo(stage));
    }

    // ── HexCoord ────────────────────────────────────────────────────

    [Test]
    public void HexCoord_Addition()
    {
        var a = new HexCoord(1, 2);
        var b = new HexCoord(-1, 3);
        Assert.That(a + b, Is.EqualTo(new HexCoord(0, 5)));
    }

    [Test]
    public void HexCoord_Directions_Has6()
    {
        Assert.That(HexCoord.Directions.Length, Is.EqualTo(6));
    }

    [Test]
    public void HexCoord_CompareTo_SortsByQThenR()
    {
        var coords = new[]
        {
            new HexCoord(1, 0),
            new HexCoord(-1, 1),
            new HexCoord(0, 0),
        };
        var sorted = coords.OrderBy(c => c).ToArray();

        Assert.That(sorted[0], Is.EqualTo(new HexCoord(-1, 1)));
        Assert.That(sorted[1], Is.EqualTo(new HexCoord(0, 0)));
        Assert.That(sorted[2], Is.EqualTo(new HexCoord(1, 0)));
    }

    // ── VertexKey ───────────────────────────────────────────────────

    [Test]
    public void VertexKey_Create_SortsCoordinates()
    {
        var c0 = new HexCoord(1, 0);
        var c1 = new HexCoord(-1, 0);
        var c2 = new HexCoord(0, 0);

        var key = VertexKey.Create(c0, c1, c2);

        // Should be sorted: (-1,0), (0,0), (1,0)
        Assert.That(key.A, Is.EqualTo(new HexCoord(-1, 0)));
        Assert.That(key.B, Is.EqualTo(new HexCoord(0, 0)));
        Assert.That(key.C, Is.EqualTo(new HexCoord(1, 0)));
    }

    [Test]
    public void VertexKey_Create_OrderDoesNotMatter()
    {
        var c0 = new HexCoord(1, 0);
        var c1 = new HexCoord(-1, 0);
        var c2 = new HexCoord(0, 0);

        var key1 = VertexKey.Create(c0, c1, c2);
        var key2 = VertexKey.Create(c2, c0, c1);
        var key3 = VertexKey.Create(c1, c2, c0);

        Assert.That(key1, Is.EqualTo(key2));
        Assert.That(key2, Is.EqualTo(key3));
    }
}
