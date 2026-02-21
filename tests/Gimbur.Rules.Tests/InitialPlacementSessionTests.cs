using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

public class InitialPlacementSessionTests
{
    [Test]
    public void Standard_UsesSnakeOrder()
    {
        var session = InitialPlacementSession.Create(GameConfig.Standard, 3, new Random(123));

        var playersAtSettlementTurns = new List<int>();
        while (!session.IsComplete)
        {
            if (session.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
            {
                playersAtSettlementTurns.Add(session.CurrentPlayer);
                var v = session.LegalSettlementVertices().First();
                session.PlaceSettlement(v);
            }
            else
            {
                var e = session.LegalRoadEdges().First();
                session.PlaceRoad(e);
            }
        }

        Assert.That(playersAtSettlementTurns, Is.EqualTo(new[] { 1, 2, 3, 3, 2, 1 }));
    }

    [Test]
    public void RoadMustTouchPendingSettlement()
    {
        var session = InitialPlacementSession.Create(GameConfig.Standard, 3, new Random(123));

        var vertex = session.LegalSettlementVertices().First();
        session.PlaceSettlement(vertex);

        var legalRoads = session.LegalRoadEdges().ToHashSet();
        Assert.That(legalRoads.Count, Is.GreaterThan(0));

        for (var ei = 0; ei < session.Board.Topology.EdgeCount; ei++)
        {
            if (!legalRoads.Contains(ei))
            {
                Assert.Throws<InvalidOperationException>(() => session.PlaceRoad(ei));
                return;
            }
        }

        Assert.Fail("Expected to find at least one illegal edge.");
    }

    [Test]
    public void StandardCompletesWithTwoPlacementsPerPlayer()
    {
        var session = InitialPlacementSession.Create(GameConfig.Standard, 4, new Random(321));

        while (!session.IsComplete)
        {
            if (session.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
            {
                session.PlaceSettlement(session.LegalSettlementVertices().First());
            }
            else
            {
                session.PlaceRoad(session.LegalRoadEdges().First());
            }
        }

        for (var player = 1; player <= 4; player++)
        {
            Assert.That(session.Board.SettlementCount(player), Is.EqualTo(2));
            Assert.That(session.Board.RoadCount(player), Is.EqualTo(2));
        }

        Assert.That(session.Stage, Is.EqualTo(TurnStage.PreRoll));
    }

    [Test]
    public void MiniCompletesWithOnePlacementPerPlayer()
    {
        var session = InitialPlacementSession.Create(GameConfig.Mini, 2, new Random(55));

        while (!session.IsComplete)
        {
            if (session.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
            {
                session.PlaceSettlement(session.LegalSettlementVertices().First());
            }
            else
            {
                session.PlaceRoad(session.LegalRoadEdges().First());
            }
        }

        for (var player = 1; player <= 2; player++)
        {
            Assert.That(session.Board.SettlementCount(player), Is.EqualTo(1));
            Assert.That(session.Board.RoadCount(player), Is.EqualTo(1));
        }
    }
}
