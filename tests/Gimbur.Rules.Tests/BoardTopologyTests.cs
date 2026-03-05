using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
public class BoardTopologyTests
{
    // ── Standard map (radius 2) ─────────────────────────────────────

    [Test]
    public void Standard_HasCorrectCounts()
    {
        var t = BoardTopology.Standard;
        Assert.Multiple(() =>
        {
            Assert.That(t.TileCount, Is.EqualTo(19));
            Assert.That(t.VertexCount, Is.EqualTo(54));
            Assert.That(t.EdgeCount, Is.EqualTo(72));
            Assert.That(t.PortCount, Is.EqualTo(9));
            Assert.That(t.CoastalEdges.Length, Is.EqualTo(30));
        });
    }

    [Test]
    public void Standard_TileCoordinatesMatchDocs()
    {
        var t = BoardTopology.Standard;
        // First tile (top-left): (-2, +2)
        Assert.That(t.Tiles[0], Is.EqualTo(new HexCoord(-2, 2)));
        // Center tile: (0, 0)
        Assert.That(t.Tiles[9], Is.EqualTo(new HexCoord(0, 0)));
        // Last tile (bottom-right): (+2, -2)
        Assert.That(t.Tiles[18], Is.EqualTo(new HexCoord(2, -2)));
    }

    [Test]
    public void Standard_PortVertexPairsMatchDocs()
    {
        var t = BoardTopology.Standard;
        // From docs: P0:(4,1), P1:(2,6), P2:(15,20), P3:(37,42),
        // P4:(50,53), P5:(52,48), P6:(43,38), P7:(27,21), P8:(11,7)
        var expected = new (int, int)[]
        {
            (4, 1), (2, 6), (15, 20), (37, 42), (50, 53),
            (52, 48), (43, 38), (27, 21), (11, 7),
        };

        Assert.That(t.Ports.Length, Is.EqualTo(9));
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.That(t.Ports[i], Is.EqualTo(expected[i]),
                $"Port P{i} mismatch");
        }
    }

    [Test]
    public void Standard_EachTileHas6Vertices()
    {
        var t = BoardTopology.Standard;
        for (var ti = 0; ti < t.TileCount; ti++)
        {
            Assert.That(t.TileVertices[ti].Length, Is.EqualTo(6),
                $"Tile {ti} should have 6 vertices");
        }
    }

    [Test]
    public void Standard_EachTileHas6Edges()
    {
        var t = BoardTopology.Standard;
        for (var ti = 0; ti < t.TileCount; ti++)
        {
            Assert.That(t.TileEdges[ti].Length, Is.EqualTo(6),
                $"Tile {ti} should have 6 edges");
        }
    }

    [Test]
    public void Standard_VertexDegrees()
    {
        var t = BoardTopology.Standard;
        var degree2 = 0;
        var degree3 = 0;
        for (var vi = 0; vi < t.VertexCount; vi++)
        {
            var deg = t.VertexEdges[vi].Length;
            Assert.That(deg, Is.EqualTo(2).Or.EqualTo(3),
                $"Vertex {vi} has unexpected degree {deg}");
            if (deg == 2) degree2++;
            else degree3++;
        }
        Assert.Multiple(() =>
        {
            Assert.That(degree2, Is.EqualTo(18), "Boundary vertices (degree 2)");
            Assert.That(degree3, Is.EqualTo(36), "Interior vertices (degree 3)");
        });
    }

    [Test]
    public void Standard_EdgeEndpointsAreOrdered()
    {
        var t = BoardTopology.Standard;
        for (var ei = 0; ei < t.EdgeCount; ei++)
        {
            var (a, b) = t.Edges[ei];
            Assert.That(a, Is.LessThan(b),
                $"Edge {ei} endpoints not ordered: ({a}, {b})");
        }
    }

    [Test]
    public void Standard_CoastalEdgesHaveExactlyOneOnBoardTile()
    {
        var t = BoardTopology.Standard;
        foreach (var ei in t.CoastalEdges)
        {
            Assert.That(t.EdgeTiles[ei].Length, Is.EqualTo(1),
                $"Coastal edge {ei} should border exactly 1 tile");
        }
    }

    [Test]
    public void Standard_InteriorEdgesHaveTwoOnBoardTiles()
    {
        var t = BoardTopology.Standard;
        var coastalSet = t.CoastalEdges.ToHashSet();
        for (var ei = 0; ei < t.EdgeCount; ei++)
        {
            if (!coastalSet.Contains(ei))
            {
                Assert.That(t.EdgeTiles[ei].Length, Is.EqualTo(2),
                    $"Interior edge {ei} should border exactly 2 tiles");
            }
        }
    }

    [Test]
    public void Standard_CenterTileHas6Neighbors()
    {
        var t = BoardTopology.Standard;
        // Tile 9 is (0,0), the center tile.
        Assert.That(t.TileNeighbors[9].Length, Is.EqualTo(6));
    }

    [Test]
    public void Standard_CornerTileHas3Neighbors()
    {
        var t = BoardTopology.Standard;
        // Tile 0 is (-2, +2), a corner tile.
        Assert.That(t.TileNeighbors[0].Length, Is.EqualTo(3));
    }

    // ── Mini map (radius 1) ─────────────────────────────────────────

    [Test]
    public void Mini_HasCorrectCounts()
    {
        var t = BoardTopology.Mini;
        Assert.Multiple(() =>
        {
            Assert.That(t.TileCount, Is.EqualTo(7));
            Assert.That(t.VertexCount, Is.EqualTo(24));
            Assert.That(t.EdgeCount, Is.EqualTo(30));
            Assert.That(t.PortCount, Is.EqualTo(6));
            Assert.That(t.CoastalEdges.Length, Is.EqualTo(18));
        });
    }

    [Test]
    public void Mini_PortVertexPairsMatchDocs()
    {
        var t = BoardTopology.Mini;
        // From docs: P0:(3,1), P1:(7,11), P2:(18,21), P3:(20,22), P4:(16,12), P5:(5,2)
        var expected = new (int, int)[]
        {
            (3, 1), (7, 11), (18, 21), (20, 22), (16, 12), (5, 2),
        };

        Assert.That(t.Ports.Length, Is.EqualTo(6));
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.That(t.Ports[i], Is.EqualTo(expected[i]),
                $"Port P{i} mismatch");
        }
    }

    [Test]
    public void Mini_VertexDegrees()
    {
        var t = BoardTopology.Mini;
        var degree2 = 0;
        var degree3 = 0;
        for (var vi = 0; vi < t.VertexCount; vi++)
        {
            var deg = t.VertexEdges[vi].Length;
            if (deg == 2) degree2++;
            else degree3++;
        }
        Assert.Multiple(() =>
        {
            Assert.That(degree2, Is.EqualTo(12), "Boundary vertices (degree 2)");
            Assert.That(degree3, Is.EqualTo(12), "Interior vertices (degree 3)");
        });
    }

    // ── Small map (10 tiles, non-circular) ─────────────────────────

    [Test]
    public void Small_HasCorrectCounts()
    {
        var t = BoardTopology.Small;
        Assert.Multiple(() =>
        {
            Assert.That(t.TileCount, Is.EqualTo(10));
            Assert.That(t.VertexCount, Is.EqualTo(32));
            Assert.That(t.EdgeCount, Is.EqualTo(41));
            Assert.That(t.PortCount, Is.EqualTo(6));
            Assert.That(t.CoastalEdges.Length, Is.EqualTo(22));
        });
    }

    [Test]
    public void Small_TileCoordinatesMatchExpected()
    {
        var t = BoardTopology.Small;
        // Sorted by screen position: top row, middle row, bottom row.
        Assert.Multiple(() =>
        {
            Assert.That(t.Tiles[0], Is.EqualTo(new HexCoord(-1, 1)));
            Assert.That(t.Tiles[1], Is.EqualTo(new HexCoord(0, 1)));
            Assert.That(t.Tiles[2], Is.EqualTo(new HexCoord(1, 1)));
            Assert.That(t.Tiles[3], Is.EqualTo(new HexCoord(-1, 0)));
            Assert.That(t.Tiles[4], Is.EqualTo(new HexCoord(0, 0)));
            Assert.That(t.Tiles[5], Is.EqualTo(new HexCoord(1, 0)));
            Assert.That(t.Tiles[6], Is.EqualTo(new HexCoord(2, 0)));
            Assert.That(t.Tiles[7], Is.EqualTo(new HexCoord(0, -1)));
            Assert.That(t.Tiles[8], Is.EqualTo(new HexCoord(1, -1)));
            Assert.That(t.Tiles[9], Is.EqualTo(new HexCoord(2, -1)));
        });
    }

    [Test]
    public void Small_PortVertexPairsMatchExpected()
    {
        var t = BoardTopology.Small;
        var expected = new (int, int)[]
        {
            (4, 1), (2, 6), (20, 24), (27, 30), (29, 25), (11, 7),
        };

        Assert.That(t.Ports.Length, Is.EqualTo(6));
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.That(t.Ports[i], Is.EqualTo(expected[i]),
                $"Port P{i} mismatch");
        }
    }

    [Test]
    public void Small_EachTileHas6Vertices()
    {
        var t = BoardTopology.Small;
        for (var ti = 0; ti < t.TileCount; ti++)
        {
            Assert.That(t.TileVertices[ti].Length, Is.EqualTo(6),
                $"Tile {ti} should have 6 vertices");
        }
    }

    [Test]
    public void Small_EachTileHas6Edges()
    {
        var t = BoardTopology.Small;
        for (var ti = 0; ti < t.TileCount; ti++)
        {
            Assert.That(t.TileEdges[ti].Length, Is.EqualTo(6),
                $"Tile {ti} should have 6 edges");
        }
    }

    [Test]
    public void Small_VertexDegrees()
    {
        var t = BoardTopology.Small;
        var degree2 = 0;
        var degree3 = 0;
        for (var vi = 0; vi < t.VertexCount; vi++)
        {
            var deg = t.VertexEdges[vi].Length;
            Assert.That(deg, Is.EqualTo(2).Or.EqualTo(3),
                $"Vertex {vi} has unexpected degree {deg}");
            if (deg == 2) degree2++;
            else degree3++;
        }
        Assert.Multiple(() =>
        {
            Assert.That(degree2, Is.EqualTo(14), "Boundary vertices (degree 2)");
            Assert.That(degree3, Is.EqualTo(18), "Interior vertices (degree 3)");
        });
    }

    [Test]
    public void Small_EdgeEndpointsAreOrdered()
    {
        var t = BoardTopology.Small;
        for (var ei = 0; ei < t.EdgeCount; ei++)
        {
            var (a, b) = t.Edges[ei];
            Assert.That(a, Is.LessThan(b),
                $"Edge {ei} endpoints not ordered: ({a}, {b})");
        }
    }

    [Test]
    public void Small_CoastalEdgesHaveExactlyOneOnBoardTile()
    {
        var t = BoardTopology.Small;
        foreach (var ei in t.CoastalEdges)
        {
            Assert.That(t.EdgeTiles[ei].Length, Is.EqualTo(1),
                $"Coastal edge {ei} should border exactly 1 tile");
        }
    }

    [Test]
    public void Small_InteriorEdgesHaveTwoOnBoardTiles()
    {
        var t = BoardTopology.Small;
        var coastalSet = t.CoastalEdges.ToHashSet();
        for (var ei = 0; ei < t.EdgeCount; ei++)
        {
            if (!coastalSet.Contains(ei))
            {
                Assert.That(t.EdgeTiles[ei].Length, Is.EqualTo(2),
                    $"Interior edge {ei} should border exactly 2 tiles");
            }
        }
    }

    [Test]
    public void Small_CenterTilesHave6Neighbors()
    {
        var t = BoardTopology.Small;
        // Tile 4 is (0,0) and tile 5 is (1,0) -- the two central tiles.
        Assert.That(t.TileNeighbors[4].Length, Is.EqualTo(6),
            "Center tile (0,0) should have 6 neighbors");
        Assert.That(t.TileNeighbors[5].Length, Is.EqualTo(6),
            "Center tile (1,0) should have 6 neighbors");
    }

    [Test]
    public void Small_CornerTilesHave3Neighbors()
    {
        var t = BoardTopology.Small;
        // Tile 0 is (-1,1) -- a corner tile.
        Assert.That(t.TileNeighbors[0].Length, Is.EqualTo(3),
            "Corner tile (-1,1) should have 3 neighbors");
        // Tile 2 is (1,1) -- another corner tile.
        Assert.That(t.TileNeighbors[2].Length, Is.EqualTo(3),
            "Corner tile (1,1) should have 3 neighbors");
    }

    // ── Adjacency symmetry ──────────────────────────────────────────

    [TestCase(1)]
    [TestCase(2)]
    public void TileNeighbors_AreSymmetric(int radius)
    {
        var t = BoardTopology.FromRadius(radius);
        for (var ti = 0; ti < t.TileCount; ti++)
        {
            foreach (var neighbor in t.TileNeighbors[ti])
            {
                Assert.That(t.TileNeighbors[neighbor], Does.Contain(ti),
                    $"Tile {ti} -> {neighbor} but not {neighbor} -> {ti}");
            }
        }
    }

    [TestCase(1)]
    [TestCase(2)]
    public void VertexNeighbors_AreSymmetric(int radius)
    {
        var t = BoardTopology.FromRadius(radius);
        for (var vi = 0; vi < t.VertexCount; vi++)
        {
            foreach (var neighbor in t.VertexNeighbors[vi])
            {
                Assert.That(t.VertexNeighbors[neighbor], Does.Contain(vi),
                    $"Vertex {vi} -> {neighbor} but not {neighbor} -> {vi}");
            }
        }
    }

    [TestCase(1)]
    [TestCase(2)]
    public void EdgeEndpoints_AppearInVertexEdges(int radius)
    {
        var t = BoardTopology.FromRadius(radius);
        for (var ei = 0; ei < t.EdgeCount; ei++)
        {
            var (a, b) = t.Edges[ei];
            Assert.That(t.VertexEdges[a], Does.Contain(ei),
                $"Edge {ei}: vertex {a} should list this edge");
            Assert.That(t.VertexEdges[b], Does.Contain(ei),
                $"Edge {ei}: vertex {b} should list this edge");
        }
    }

    // ── Euler's formula ─────────────────────────────────────────────

    [TestCase(1)]
    [TestCase(2)]
    public void EulersFormula_VMinusEPlusFEquals2(int radius)
    {
        // For a planar graph: V - E + F = 2
        // F = tiles + 1 (outer face)
        var t = BoardTopology.FromRadius(radius);
        var v = t.VertexCount;
        var e = t.EdgeCount;
        var f = t.TileCount + 1;
        Assert.That(v - e + f, Is.EqualTo(2),
            $"Euler's formula: {v} - {e} + {f} = {v - e + f}, expected 2");
    }

    // ── Small-specific adjacency and Euler tests ────────────────────

    [Test]
    public void Small_TileNeighbors_AreSymmetric()
    {
        var t = BoardTopology.Small;
        for (var ti = 0; ti < t.TileCount; ti++)
        {
            foreach (var neighbor in t.TileNeighbors[ti])
            {
                Assert.That(t.TileNeighbors[neighbor], Does.Contain(ti),
                    $"Tile {ti} -> {neighbor} but not {neighbor} -> {ti}");
            }
        }
    }

    [Test]
    public void Small_VertexNeighbors_AreSymmetric()
    {
        var t = BoardTopology.Small;
        for (var vi = 0; vi < t.VertexCount; vi++)
        {
            foreach (var neighbor in t.VertexNeighbors[vi])
            {
                Assert.That(t.VertexNeighbors[neighbor], Does.Contain(vi),
                    $"Vertex {vi} -> {neighbor} but not {neighbor} -> {vi}");
            }
        }
    }

    [Test]
    public void Small_EdgeEndpoints_AppearInVertexEdges()
    {
        var t = BoardTopology.Small;
        for (var ei = 0; ei < t.EdgeCount; ei++)
        {
            var (a, b) = t.Edges[ei];
            Assert.That(t.VertexEdges[a], Does.Contain(ei),
                $"Edge {ei}: vertex {a} should list this edge");
            Assert.That(t.VertexEdges[b], Does.Contain(ei),
                $"Edge {ei}: vertex {b} should list this edge");
        }
    }

    [Test]
    public void Small_EulersFormula_VMinusEPlusFEquals2()
    {
        var t = BoardTopology.Small;
        var v = t.VertexCount;
        var e = t.EdgeCount;
        var f = t.TileCount + 1;
        Assert.That(v - e + f, Is.EqualTo(2),
            $"Euler's formula: {v} - {e} + {f} = {v - e + f}, expected 2");
    }
}
