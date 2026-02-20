using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class TypeTests
{
    // ── VertexOccupancy ─────────────────────────────────────────────

    [Test]
    public void VertexOccupancy_Empty_ToToken_Returns0()
    {
        Assert.That(VertexOccupancy.Empty.ToToken(), Is.EqualTo(0));
    }

    [TestCase(1, 1)]
    [TestCase(2, 2)]
    [TestCase(3, 3)]
    [TestCase(4, 4)]
    public void VertexOccupancy_Settlement_ToToken(int player, int expected)
    {
        var occ = new VertexOccupancy(BuildingType.Settlement, player);
        Assert.That(occ.ToToken(), Is.EqualTo(expected));
    }

    [TestCase(1, 5)]
    [TestCase(2, 6)]
    [TestCase(3, 7)]
    [TestCase(4, 8)]
    public void VertexOccupancy_City_ToToken(int player, int expected)
    {
        var occ = new VertexOccupancy(BuildingType.City, player);
        Assert.That(occ.ToToken(), Is.EqualTo(expected));
    }

    [TestCase(0)]
    [TestCase(1)]
    [TestCase(4)]
    [TestCase(5)]
    [TestCase(8)]
    public void VertexOccupancy_FromToken_RoundTrips(int token)
    {
        var occ = VertexOccupancy.FromToken(token);
        Assert.That(occ.ToToken(), Is.EqualTo(token));
    }

    [Test]
    public void VertexOccupancy_FromToken_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => VertexOccupancy.FromToken(9));
        Assert.Throws<ArgumentOutOfRangeException>(() => VertexOccupancy.FromToken(-1));
    }

    // ── EdgeOccupancy ───────────────────────────────────────────────

    [Test]
    public void EdgeOccupancy_Empty_ToToken_Returns0()
    {
        Assert.That(EdgeOccupancy.Empty.ToToken(), Is.EqualTo(0));
    }

    [TestCase(0)]
    [TestCase(1)]
    [TestCase(4)]
    public void EdgeOccupancy_FromToken_RoundTrips(int token)
    {
        var occ = EdgeOccupancy.FromToken(token);
        Assert.That(occ.ToToken(), Is.EqualTo(token));
    }

    [Test]
    public void EdgeOccupancy_FromToken_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeOccupancy.FromToken(5));
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeOccupancy.FromToken(-1));
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
