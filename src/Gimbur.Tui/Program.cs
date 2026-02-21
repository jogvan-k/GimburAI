using Gimbur.Rules;
using Gimbur;

namespace Gimbur.Tui;

internal static class Program
{
    private const double Sqrt3 = 1.7320508075688772;

    private const string Reset = "\u001b[0m";
    private const string FgWhite = "\u001b[37m";
    private const string FgBrightWhite = "\u001b[97m";
    private const string FgBrightBlack = "\u001b[90m";
    private const string FgRed = "\u001b[31m";
    private const string FgBrightRed = "\u001b[91m";
    private const string FgYellow = "\u001b[33m";
    private const string FgBrightGreen = "\u001b[92m";
    private const string FgGreen = "\u001b[32m";
    private const string FgSilver = "\u001b[38;5;250m";
    private const string FgBeige = "\u001b[38;5;223m";
    private const string FgCyan = "\u001b[36m";
    private static int _lastFrameLineCount;
    private static readonly Random UiRng = new();
    private static string _statusMessage = "";

    private static void Main()
    {
        Console.WriteLine("Gimbur TUI");
        Console.WriteLine();

        var mapChoice = PromptMapTopology();
        var config = mapChoice == MapChoice.Mini ? GameConfig.Mini : GameConfig.Standard;
        var players = PromptPlayerCount(config.MinPlayers, config.MaxPlayers);

        var rng = new Random();
        var state = new CatanState(config, players, rng);

        RunGameLoop(state);
    }

    private static void RunGameLoop(CatanState state)
    {
        Console.Clear();
        _lastFrameLineCount = 0;
        Console.CursorVisible = false;

        while (true)
        {
            var actions = state.Actions().OfType<CatanAction>().ToArray();
            if (actions.Length == 0)
            {
                break;
            }

            if (state.Stage is TurnStage.PlaceFirstSettlement or TurnStage.PlaceSecondSettlement)
            {
                state = ExecuteSettlementPlacement(state);
                continue;
            }

            if (state.Stage is TurnStage.PlaceFirstRoad or TurnStage.PlaceSecondRoad)
            {
                state = ExecuteRoadPlacement(state);
                continue;
            }

            if (state.Stage == TurnStage.BuildTrade && state.PendingRoadBuildingPlacementsFor(state.CurrentPlayer) > 0)
            {
                state = ExecuteRoadPlacement(state);
                continue;
            }

            if (state.Stage == TurnStage.ChooseRobberLocation)
            {
                state = ExecuteRobberPlacement(state);
                continue;
            }

            state = ExecuteActionMenu(state, actions);
        }

        DrawFrame(() =>
        {
            RenderBoard(state.Board);
            Console.WriteLine();
            Console.WriteLine("Game finished.");
            Console.WriteLine($"Winner: Player {state.WinnerPlayer}");
            Console.WriteLine();
            for (var player = 1; player <= state.PlayerCount; player++)
            {
                Console.WriteLine(
                    $"Player {player}: VP={state.VictoryPointsFor(player)} settlements={state.Board.SettlementCount(player)} cities={state.Board.CityCount(player)} roads={state.Board.RoadCount(player)}");
            }
            Console.WriteLine();
            Console.WriteLine("Legend: tile text stack = resource name, number, robber(*); o empty vertex, s settlement, c city, |/\\/- road");
            Console.WriteLine("Ports: 3:1 generic plus Wood/Brick/Sheep/Wheat/Ore resource ports");
        });

        Console.CursorVisible = true;
    }

    private static CatanState ExecuteSettlementPlacement(CatanState state)
    {
        var legal = state.Actions()
            .OfType<CatanAction>()
            .Where(a => a.ActionType == CatanActionType.PlaceSettlement)
            .Select(a => a.TargetIndex)
            .ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal settlement placements available.");
            return state;
        }

        var selected = SelectLocation(
            title: "Select settlement location",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            resourceSummary: BuildPlayerSummary(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).VertexPoints,
            neighborProvider: vertex => state.Board.Topology.VertexNeighbors[vertex],
            mode: LocationSelectionMode.Vertex);

        if (selected is int vertex)
        {
            var action = new CatanAction(state, CatanActionType.PlaceSettlement, vertex);
            return (CatanState)action.DoCoreAction();
        }

        return state;
    }

    private static CatanState ExecuteRobberPlacement(CatanState state)
    {
        var legal = state.Actions()
            .OfType<CatanAction>()
            .Where(a => a.ActionType == CatanActionType.ChooseRobberTile)
            .Select(a => a.Arg1)
            .ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal robber placements available.");
            return state;
        }

        var selected = SelectLocation(
            title: "Select robber destination tile",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            resourceSummary: BuildPlayerSummary(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).TilePoints,
            neighborProvider: tile => state.Board.Topology.TileNeighbors[tile],
            mode: LocationSelectionMode.Tile);

        if (selected is int tile)
        {
            var action = new CatanAction(state, CatanActionType.ChooseRobberTile, tile);
            return (CatanState)action.DoCoreAction();
        }

        return state;
    }

    private static CatanState ExecuteActionMenu(CatanState state, IReadOnlyList<CatanAction> actions)
    {
        var context = ActionMenuContext.Root;
        var menuEntries = BuildMenuEntries(state, actions, context);
        var selectedIndex = 0;

        while (true)
        {
            DrawFrame(() =>
            {
                RenderBoard(state.Board);
                WriteFixedLine();
                WriteFixedLine($"Turn {state.TurnNumber} - Player {state.CurrentPlayer}");
                WriteFixedLine($"Stage: {StageLabel(state.Stage),-28}");
                WriteFixedLine($"Last: {_statusMessage,-32}");
                WriteFixedLine(BuildPlayerSummary(state));
                WriteFixedLine($"Longest Road: {(state.LongestRoadOwner == 0 ? "none" : $"Player {state.LongestRoadOwner}")}, Largest Army: {(state.LargestArmyOwner == 0 ? "none" : $"Player {state.LargestArmyOwner}")}");
                WriteFixedLine();
                WriteFixedLine($"Legal actions - {ContextLabel(context)} (Up/Down + Enter):");

                for (var i = 0; i < menuEntries.Count; i++)
                {
                    var prefix = i == selectedIndex ? ">" : " ";
                    WriteFixedLine($"{prefix} {menuEntries[i].Label}");
                }
            });

            var key = Console.ReadKey(intercept: true).Key;
            if (key == ConsoleKey.UpArrow)
            {
                selectedIndex = selectedIndex > 0 ? selectedIndex - 1 : menuEntries.Count - 1;
                continue;
            }

            if (key == ConsoleKey.DownArrow)
            {
                selectedIndex = selectedIndex < menuEntries.Count - 1 ? selectedIndex + 1 : 0;
                continue;
            }

            if (key != ConsoleKey.Enter)
            {
                continue;
            }

            var selected = menuEntries[selectedIndex];
            if (selected.Mode == ActionSelectionMode.OpenTradeMenu)
            {
                context = ActionMenuContext.Trade;
                menuEntries = BuildMenuEntries(state, actions, context);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.OpenYearOfPlentyMenu)
            {
                context = ActionMenuContext.YearOfPlenty;
                menuEntries = BuildMenuEntries(state, actions, context);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.Back)
            {
                context = ActionMenuContext.Root;
                menuEntries = BuildMenuEntries(state, actions, context);
                selectedIndex = 0;
                continue;
            }

            if (selected.Mode == ActionSelectionMode.Direct)
            {
                return (CatanState)selected.Action!.DoCoreAction();
            }

            if (selected.Mode == ActionSelectionMode.RollDice)
            {
                return ExecuteRandomRoll(state);
            }

            if (selected.Mode == ActionSelectionMode.BuyDevCardRandom)
            {
                return ExecuteRandomDevCardPurchase(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceSettlement)
            {
                return ExecuteSettlementPlacement(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceRoad)
            {
                return ExecuteRoadPlacement(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceCity)
            {
                return ExecuteCityPlacement(state);
            }

            if (selected.Mode == ActionSelectionMode.PlaceRobber)
            {
                return ExecuteRobberPlacement(state);
            }
        }
    }

    private static List<MenuEntry> BuildMenuEntries(
        CatanState state,
        IReadOnlyList<CatanAction> actions,
        ActionMenuContext context)
    {
        var entries = new List<MenuEntry>();
        if (context == ActionMenuContext.Trade)
        {
            foreach (var action in actions.Where(a => a.ActionType == CatanActionType.BankTrade))
            {
                entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
            }

            entries.Add(new MenuEntry("Back", ActionSelectionMode.Back, null));
            return entries;
        }

        if (context == ActionMenuContext.YearOfPlenty)
        {
            foreach (var action in actions.Where(a => a.ActionType == CatanActionType.PlayYearOfPlenty))
            {
                entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
            }

            entries.Add(new MenuEntry("Back", ActionSelectionMode.Back, null));
            return entries;
        }

        if (actions.Any(a => a.ActionType == CatanActionType.RollDice))
        {
            entries.Add(new MenuEntry("Roll dice", ActionSelectionMode.RollDice, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.PlaceSettlement))
        {
            entries.Add(new MenuEntry("Place settlement", ActionSelectionMode.PlaceSettlement, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.PlaceRoad))
        {
            entries.Add(new MenuEntry("Place road", ActionSelectionMode.PlaceRoad, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.BuildCity))
        {
            entries.Add(new MenuEntry("Place city", ActionSelectionMode.PlaceCity, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.ChooseRobberTile))
        {
            entries.Add(new MenuEntry("Place robber", ActionSelectionMode.PlaceRobber, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.BuyDevCard))
        {
            entries.Add(new MenuEntry("Buy dev card", ActionSelectionMode.BuyDevCardRandom, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.BankTrade))
        {
            entries.Add(new MenuEntry("Trade", ActionSelectionMode.OpenTradeMenu, null));
        }

        if (actions.Any(a => a.ActionType == CatanActionType.PlayYearOfPlenty))
        {
            entries.Add(new MenuEntry("Year of Plenty", ActionSelectionMode.OpenYearOfPlentyMenu, null));
        }

        foreach (var action in actions)
        {
            if (action.ActionType is
                CatanActionType.RollDice or
                CatanActionType.BuyDevCard or
                CatanActionType.BankTrade or
                CatanActionType.PlayYearOfPlenty or
                CatanActionType.PlaceSettlement or
                CatanActionType.PlaceRoad or
                CatanActionType.BuildCity or
                CatanActionType.ChooseRobberTile)
            {
                continue;
            }

            entries.Add(new MenuEntry(DescribeAction(state, action), ActionSelectionMode.Direct, action));
        }

        return entries;
    }

    private static string ContextLabel(ActionMenuContext context) =>
        context switch
        {
            ActionMenuContext.Root => "Root",
            ActionMenuContext.Trade => "Trade",
            ActionMenuContext.YearOfPlenty => "Year Of Plenty",
            _ => "Root",
        };

    private static CatanState ExecuteRandomRoll(CatanState state)
    {
        var roll = UiRng.Next(1, 7) + UiRng.Next(1, 7);
        var action = state.Actions()
            .OfType<CatanAction>()
            .FirstOrDefault(a => a.ActionType == CatanActionType.RollDice && a.Arg1 == roll);

        if (action is null)
        {
            action = state.Actions()
                .OfType<CatanAction>()
                .First(a => a.ActionType == CatanActionType.RollDice);
            roll = action.Arg1;
        }

        var next = (CatanState)action.DoCoreAction();
        _statusMessage = $"Rolled {roll}";
        return next;
    }

    private static CatanState ExecuteRandomDevCardPurchase(CatanState state)
    {
        var options = state.Actions()
            .OfType<CatanAction>()
            .Where(a => a.ActionType == CatanActionType.BuyDevCard)
            .ToArray();

        if (options.Length == 0)
        {
            return state;
        }

        var weighted = options
            .Select(a => new
            {
                Action = a,
                Weight = state.DevCardsRemaining((DevCardType)a.Arg1),
            })
            .Where(x => x.Weight > 0)
            .ToArray();

        if (weighted.Length == 0)
        {
            return state;
        }

        var totalWeight = weighted.Sum(x => x.Weight);
        var pick = UiRng.Next(totalWeight);
        var cumulative = 0;
        CatanAction selected = weighted[0].Action;
        foreach (var option in weighted)
        {
            cumulative += option.Weight;
            if (pick < cumulative)
            {
                selected = option.Action;
                break;
            }
        }

        var next = (CatanState)selected.DoCoreAction();
        _statusMessage = $"Bought dev card: {(DevCardType)selected.Arg1}";
        return next;
    }

    private static string DescribeAction(CatanState state, CatanAction action)
    {
        return action.ActionType switch
        {
            CatanActionType.RollDice => $"Roll dice = {action.Arg1}",
            CatanActionType.BuildCity => $"Build city at vertex {action.Arg1}",
            CatanActionType.BankTrade => $"Trade {state.Board.TradeRatio(state.CurrentPlayer, (ResourceType)action.Arg1)} {(ResourceType)action.Arg1} -> {(ResourceType)action.Arg2}",
            CatanActionType.BuyDevCard => $"Buy dev card ({(DevCardType)action.Arg1})",
            CatanActionType.PlayKnight => "Play knight",
            CatanActionType.PlayRoadBuilding => "Play road building",
            CatanActionType.PlayMonopoly => $"Play monopoly on {(ResourceType)action.Arg1}",
            CatanActionType.PlayYearOfPlenty => $"Play year of plenty: {(ResourceType)action.Arg1} + {(ResourceType)action.Arg2}",
            CatanActionType.EndTurn => "End turn",
            _ => action.ActionType.ToString(),
        };
    }

    private static CatanState ExecuteRoadPlacement(CatanState state)
    {
        var legal = state.Actions()
            .OfType<CatanAction>()
            .Where(a => a.ActionType == CatanActionType.PlaceRoad)
            .Select(a => a.TargetIndex)
            .ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal road placements available.");
            return state;
        }

        var selected = SelectLocation(
            title: "Select road location",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            resourceSummary: BuildPlayerSummary(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).EdgePoints,
            neighborProvider: BuildEdgeNeighborProvider(state.Board.Topology),
            mode: LocationSelectionMode.Edge);

        if (selected is int edge)
        {
            var action = new CatanAction(state, CatanActionType.PlaceRoad, edge);
            return (CatanState)action.DoCoreAction();
        }

        return state;
    }

    private static CatanState ExecuteCityPlacement(CatanState state)
    {
        var legal = state.Actions()
            .OfType<CatanAction>()
            .Where(a => a.ActionType == CatanActionType.BuildCity)
            .Select(a => a.Arg1)
            .ToArray();

        if (legal.Length == 0)
        {
            Pause("No legal city placements available.");
            return state;
        }

        var selected = SelectLocation(
            title: "Select city location",
            stageLabel: StageLabel(state.Stage),
            currentPlayer: state.CurrentPlayer,
            board: state.Board,
            resourceSummary: BuildPlayerSummary(state),
            legalCandidates: legal,
            pointProvider: BuildRenderLayout(state.Board.Topology).VertexPoints,
            neighborProvider: vertex => state.Board.Topology.VertexNeighbors[vertex],
            mode: LocationSelectionMode.Vertex);

        if (selected is int vertex)
        {
            var action = new CatanAction(state, CatanActionType.BuildCity, vertex);
            return (CatanState)action.DoCoreAction();
        }

        return state;
    }

    private static int? SelectLocation(
        string title,
        string stageLabel,
        int currentPlayer,
        Board board,
        string resourceSummary,
        int[] legalCandidates,
        IReadOnlyList<(int X, int Y)> pointProvider,
        Func<int, IEnumerable<int>> neighborProvider,
        LocationSelectionMode mode)
    {
        var legalSet = legalCandidates.ToHashSet();
        var current = legalCandidates[0];

        while (true)
        {
            DrawFrame(() =>
            {
                if (mode == LocationSelectionMode.Vertex)
                {
                    RenderBoard(
                        board,
                        highlightedVertices: legalCandidates,
                        selectedVertex: current);
                }
                else if (mode == LocationSelectionMode.Edge)
                {
                    RenderBoard(
                        board,
                        highlightedEdges: legalCandidates,
                        selectedEdge: current);
                }
                else
                {
                    RenderBoard(
                        board,
                        highlightedTiles: legalCandidates,
                        selectedTile: current);
                }

                WriteFixedLine();
                WriteFixedLine($"Player {currentPlayer}");
                WriteFixedLine($"Stage: {stageLabel,-28}");
                WriteFixedLine(resourceSummary);
                WriteFixedLine($"{title}: arrows to move, Enter to confirm, Esc to cancel");
            });

            var key = Console.ReadKey(intercept: true).Key;
            if (key == ConsoleKey.Enter)
            {
                return current;
            }

            if (key == ConsoleKey.Escape)
            {
                return null;
            }

            current = key switch
            {
                ConsoleKey.LeftArrow => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, -1, 0),
                ConsoleKey.RightArrow => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, 1, 0),
                ConsoleKey.UpArrow => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, 0, -1),
                ConsoleKey.DownArrow => MoveSelection(current, legalCandidates, legalSet, pointProvider, neighborProvider, 0, 1),
                _ => current,
            };
        }
    }

    private static int MoveSelection(
        int current,
        IReadOnlyList<int> legalCandidates,
        HashSet<int> legalSet,
        IReadOnlyList<(int X, int Y)> pointProvider,
        Func<int, IEnumerable<int>> neighborProvider,
        int dirX,
        int dirY)
    {
        var currentPoint = pointProvider[current];

        var neighborCandidates = neighborProvider(current).Where(legalSet.Contains).ToArray();
        var bestNeighbor = PickDirectionalBest(currentPoint, neighborCandidates, pointProvider, dirX, dirY);
        if (bestNeighbor.HasValue)
        {
            return bestNeighbor.Value;
        }

        var fallback = PickDirectionalBest(currentPoint, legalCandidates, pointProvider, dirX, dirY);
        return fallback ?? current;
    }

    private static int? PickDirectionalBest(
        (int X, int Y) currentPoint,
        IEnumerable<int> candidates,
        IReadOnlyList<(int X, int Y)> pointProvider,
        int dirX,
        int dirY)
    {
        var best = -1;
        var bestScore = double.MaxValue;
        var found = false;

        foreach (var candidate in candidates)
        {
            var point = pointProvider[candidate];
            var vx = point.X - currentPoint.X;
            var vy = point.Y - currentPoint.Y;

            var projection = (vx * dirX) + (vy * dirY);
            if (projection <= 0)
                continue;

            var perpendicular = Math.Abs((vx * dirY) - (vy * dirX));
            var score = (perpendicular * 1000.0) + projection;
            if (score < bestScore)
            {
                bestScore = score;
                best = candidate;
                found = true;
            }
        }

        return found ? best : null;
    }

    private static Func<int, IEnumerable<int>> BuildEdgeNeighborProvider(BoardTopology topology)
    {
        var neighbors = new List<int>[topology.EdgeCount];
        for (var i = 0; i < topology.EdgeCount; i++)
        {
            neighbors[i] = new List<int>();
        }

        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var edges = topology.VertexEdges[vi];
            for (var i = 0; i < edges.Length; i++)
            {
                for (var j = i + 1; j < edges.Length; j++)
                {
                    var a = edges[i];
                    var b = edges[j];
                    neighbors[a].Add(b);
                    neighbors[b].Add(a);
                }
            }
        }

        var frozen = neighbors.Select(n => n.Distinct().ToArray()).ToArray();
        return edge => frozen[edge];
    }

    private static string StageLabel(TurnStage stage) =>
        stage switch
        {
            TurnStage.PlaceFirstSettlement => "Place first settlement",
            TurnStage.PlaceFirstRoad => "Place first road",
            TurnStage.PlaceSecondSettlement => "Place second settlement",
            TurnStage.PlaceSecondRoad => "Place second road",
            TurnStage.PreRoll => "Pre-roll",
            TurnStage.ChooseRobberLocation => "Choose robber location",
            TurnStage.BuildTrade => "Build/trade",
            _ => stage.ToString(),
        };

    private static string BuildPlayerSummary(CatanState state)
    {
        var parts = new List<string>();
        for (var player = 1; player <= state.PlayerCount; player++)
        {
            parts.Add(
                $"{PlayerColor(player)}P{player}{Reset} VP:{state.VictoryPointsFor(player)} K:{state.KnightsPlayedFor(player)} W:{state.ResourceCountFor(player, ResourceType.Wood)} B:{state.ResourceCountFor(player, ResourceType.Brick)} S:{state.ResourceCountFor(player, ResourceType.Sheep)} Wh:{state.ResourceCountFor(player, ResourceType.Wheat)} O:{state.ResourceCountFor(player, ResourceType.Ore)}");
        }

        return string.Join(" | ", parts);
    }

    private static MapChoice PromptMapTopology()
    {
        while (true)
        {
            Console.Write("Select map topology ([m]ini/[s]tandard): ");
            var input = Console.ReadLine()?.Trim().ToLowerInvariant();

            if (input is "m" or "mini")
            {
                return MapChoice.Mini;
            }

            if (input is "s" or "standard")
            {
                return MapChoice.Standard;
            }

            Console.WriteLine("Please enter 'mini' (or 'm') or 'standard' (or 's').");
        }
    }

    private static int PromptPlayerCount(int minPlayers, int maxPlayers)
    {
        while (true)
        {
            if (minPlayers == maxPlayers)
            {
                Console.Write($"How many players should be included? ({minPlayers}): ");
            }
            else
            {
                Console.Write($"How many players should be included? ({minPlayers}-{maxPlayers}): ");
            }

            var input = Console.ReadLine()?.Trim();

            if (!int.TryParse(input, out var players))
            {
                Console.WriteLine("Please enter a valid whole number.");
                continue;
            }

            if (players < minPlayers || players > maxPlayers)
            {
                if (minPlayers == maxPlayers)
                {
                    Console.WriteLine($"Only {minPlayers} players are supported for this map.");
                }
                else
                {
                    Console.WriteLine($"Player count must be between {minPlayers} and {maxPlayers}.");
                }
                continue;
            }

            return players;
        }
    }

    private static void RenderBoard(
        Board board,
        IEnumerable<int>? highlightedVertices = null,
        int? selectedVertex = null,
        IEnumerable<int>? highlightedEdges = null,
        int? selectedEdge = null,
        IEnumerable<int>? highlightedTiles = null,
        int? selectedTile = null)
    {
        var layout = BuildRenderLayout(board.Topology);
        var highlightedVertexSet = highlightedVertices is null ? null : highlightedVertices.ToHashSet();
        var highlightedEdgeSet = highlightedEdges is null ? null : highlightedEdges.ToHashSet();
        var highlightedTileSet = highlightedTiles is null ? null : highlightedTiles.ToHashSet();

        var canvas = new Canvas(layout.Width, layout.Height);

        for (var ei = 0; ei < board.Topology.EdgeCount; ei++)
        {
            var edge = board.Topology.Edges[ei];
            DrawEdge(
                canvas,
                layout.VertexPoints[edge.VertexA],
                layout.VertexPoints[edge.VertexB],
                board.EdgeOccupancy[ei].Player,
                highlighted: highlightedEdgeSet?.Contains(ei) ?? false,
                selected: selectedEdge == ei);
        }

        for (var vi = 0; vi < board.Topology.VertexCount; vi++)
        {
            DrawVertex(
                canvas,
                layout.VertexPoints[vi],
                board.VertexOccupancy[vi],
                highlighted: highlightedVertexSet?.Contains(vi) ?? false,
                selected: selectedVertex == vi);
        }

        for (var ti = 0; ti < board.Topology.TileCount; ti++)
        {
            DrawTile(
                canvas,
                layout.TilePoints[ti],
                board,
                ti,
                highlighted: highlightedTileSet?.Contains(ti) ?? false,
                selected: selectedTile == ti);
        }

        for (var pi = 0; pi < board.Topology.PortCount; pi++)
        {
            var (va, vb) = board.Topology.Ports[pi];
            var port = board.PortType(pi);
            var portColor = PortColor(port);
            var mid = (
                X: (layout.VertexPixels[va].X + layout.VertexPixels[vb].X) / 2.0,
                Y: (layout.VertexPixels[va].Y + layout.VertexPixels[vb].Y) / 2.0);
            var outward = (
                X: mid.X - layout.BoardCenter.X,
                Y: mid.Y - layout.BoardCenter.Y);
            var length = Math.Sqrt((outward.X * outward.X) + (outward.Y * outward.Y));
            if (length < 0.0001)
            {
                length = 1.0;
                outward = (1.0, 0.0);
            }

            var portPos = (
                X: mid.X + (outward.X / length) * 1.15,
                Y: mid.Y + (outward.Y / length) * 1.15);
            var point = ToCanvasPoint(portPos, layout.MinX, layout.MinY, layout.ScaleX, layout.ScaleY, layout.MarginX, layout.MarginY);
            var label = PortLabel(port);
            var labelStartX = point.X - (label.Length / 2);
            var labelEndX = labelStartX + label.Length - 1;
            var leftAnchor = (X: labelStartX - 1, Y: point.Y);
            var rightAnchor = (X: labelEndX + 1, Y: point.Y);
            var aAnchor = layout.VertexPoints[va].X <= point.X ? leftAnchor : rightAnchor;
            var bAnchor = layout.VertexPoints[vb].X <= point.X ? leftAnchor : rightAnchor;

            DrawConnector(canvas, aAnchor, layout.VertexPoints[va], portColor);
            DrawConnector(canvas, bAnchor, layout.VertexPoints[vb], portColor);
            DrawString(canvas, labelStartX, point.Y, label, portColor);
        }

        canvas.Print();
    }

    private static BoardRenderLayout BuildRenderLayout(BoardTopology topology)
    {
        var tilePixels = new (double X, double Y)[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tilePixels[ti] = AxialToPixel(topology.Tiles[ti]);
        }

        var vertexPixels = new (double X, double Y)[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            var key = topology.Vertices[vi];
            var a = AxialToPixel(key.A);
            var b = AxialToPixel(key.B);
            var c = AxialToPixel(key.C);
            vertexPixels[vi] = ((a.X + b.X + c.X) / 3.0, (a.Y + b.Y + c.Y) / 3.0);
        }

        var minX = Math.Min(tilePixels.Min(p => p.X), vertexPixels.Min(p => p.X));
        var maxX = Math.Max(tilePixels.Max(p => p.X), vertexPixels.Max(p => p.X));
        var minY = Math.Min(tilePixels.Min(p => p.Y), vertexPixels.Min(p => p.Y));
        var maxY = Math.Max(tilePixels.Max(p => p.Y), vertexPixels.Max(p => p.Y));

        const int marginX = 8;
        const int marginY = 4;
        const double scaleX = 8.0;
        const double scaleY = 4.0;

        var boardCenter = (
            X: tilePixels.Average(p => p.X),
            Y: tilePixels.Average(p => p.Y));

        var vertexPoints = new (int X, int Y)[topology.VertexCount];
        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            vertexPoints[vi] = ToCanvasPoint(vertexPixels[vi], minX, minY, scaleX, scaleY, marginX, marginY);
        }

        var tilePoints = new (int X, int Y)[topology.TileCount];
        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            tilePoints[ti] = ToCanvasPoint(tilePixels[ti], minX, minY, scaleX, scaleY, marginX, marginY);
        }

        var edgePoints = new (int X, int Y)[topology.EdgeCount];
        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            var (va, vb) = topology.Edges[ei];
            edgePoints[ei] = ((vertexPoints[va].X + vertexPoints[vb].X) / 2, (vertexPoints[va].Y + vertexPoints[vb].Y) / 2);
        }

        var width = (int)Math.Ceiling((maxX - minX) * scaleX) + marginX * 2 + 20;
        var height = (int)Math.Ceiling((maxY - minY) * scaleY) + marginY * 2 + 10;

        return new BoardRenderLayout(
            tilePixels,
            vertexPixels,
            tilePoints,
            vertexPoints,
            edgePoints,
            minX,
            minY,
            scaleX,
            scaleY,
            marginX,
            marginY,
            width,
            height,
            boardCenter);
    }

    private static (double X, double Y) AxialToPixel(HexCoord c)
    {
        var x = Sqrt3 * (c.Q + c.R / 2.0);
        var y = -1.5 * c.R;
        return (x, y);
    }

    private static (int X, int Y) ToCanvasPoint(
        (double X, double Y) point,
        double minX,
        double minY,
        double scaleX,
        double scaleY,
        int marginX,
        int marginY)
    {
        var x = (int)Math.Round((point.X - minX) * scaleX) + marginX;
        var y = (int)Math.Round((point.Y - minY) * scaleY) + marginY;
        return (x, y);
    }

    private static void DrawEdge(
        Canvas canvas,
        (int X, int Y) p0,
        (int X, int Y) p1,
        int player,
        bool highlighted,
        bool selected)
    {
        var dx = p1.X - p0.X;
        var dy = p1.Y - p0.Y;
        var steps = Math.Max(Math.Abs(dx), Math.Abs(dy));
        if (steps <= 1)
        {
            return;
        }

        char edgeChar;
        if (Math.Abs(dy) <= 1)
        {
            edgeChar = '-';
        }
        else if (Math.Abs(dy) > Math.Abs(dx) * 2)
        {
            edgeChar = '|';
        }
        else if ((dx > 0 && dy > 0) || (dx < 0 && dy < 0))
        {
            edgeChar = '\\';
        }
        else
        {
            edgeChar = '/';
        }

        var color = player > 0
            ? PlayerColor(player)
            : selected
                ? FgBrightRed
                : highlighted
                    ? FgBrightWhite
                    : FgBrightBlack;

        for (var i = 1; i < steps; i++)
        {
            var x = p0.X + (dx * i) / steps;
            var y = p0.Y + (dy * i) / steps;
            canvas.Set(x, y, edgeChar, color);
        }
    }

    private static void DrawVertex(
        Canvas canvas,
        (int X, int Y) point,
        VertexOccupancy occupancy,
        bool highlighted,
        bool selected)
    {
        if (occupancy.IsEmpty)
        {
            var charToDraw = selected ? 'O' : 'o';
            var color = selected ? FgBrightRed : highlighted ? FgBrightWhite : FgBrightBlack;
            canvas.Set(point.X, point.Y, charToDraw, color);
            return;
        }

        var marker = occupancy.Building == BuildingType.City ? 'c' : 's';
        if (selected)
        {
            canvas.Set(point.X, point.Y, char.ToUpperInvariant(marker), FgBrightRed);
            return;
        }

        if (highlighted)
        {
            canvas.Set(point.X, point.Y, marker, FgBrightWhite);
            return;
        }

        canvas.Set(point.X, point.Y, marker, PlayerColor(occupancy.Player));
    }

    private static void DrawTile(
        Canvas canvas,
        (int X, int Y) center,
        Board board,
        int tileIndex,
        bool highlighted,
        bool selected)
    {
        var resource = board.TileResource(tileIndex);
        var resourceText = resource switch
        {
            ResourceType.Desert => "Desert",
            ResourceType.Wood => "Wood",
            ResourceType.Brick => "Brick",
            ResourceType.Sheep => "Sheep",
            ResourceType.Wheat => "Wheat",
            ResourceType.Ore => "Ore",
            _ => "Unknown",
        };

        var number = board.TileNumber(tileIndex);
        var numberText = number == 0 ? "" : number.ToString();
        var numberColor = number is 6 or 8 ? FgBrightRed : FgBrightWhite;

        DrawString(
            canvas,
            center.X - (resourceText.Length / 2),
            center.Y - 1,
            resourceText,
            ResourceStyle(resource));

        if (numberText.Length > 0)
        {
            DrawString(canvas, center.X - (numberText.Length / 2), center.Y, numberText, numberColor);
        }

        if (board.RobberTile == tileIndex)
        {
            DrawString(canvas, center.X, center.Y + 1, "*", FgRed);
        }

        if (selected || highlighted)
        {
            var marker = selected ? "@" : "+";
            var markerColor = selected ? FgBrightRed : FgCyan;
            DrawString(canvas, center.X, center.Y + 2, marker, markerColor);
        }
    }

    private static void DrawString(Canvas canvas, int x, int y, string text, string color)
    {
        for (var i = 0; i < text.Length; i++)
        {
            canvas.Set(x + i, y, text[i], color);
        }
    }

    private static void DrawConnector(Canvas canvas, (int X, int Y) p0, (int X, int Y) p1, string color)
    {
        var x0 = p0.X;
        var y0 = p0.Y;
        var x1 = p1.X;
        var y1 = p1.Y;
        var dx = Math.Abs(x1 - x0);
        var sx = x0 < x1 ? 1 : -1;
        var dy = -Math.Abs(y1 - y0);
        var sy = y0 < y1 ? 1 : -1;
        var err = dx + dy;

        while (!(x0 == x1 && y0 == y1))
        {
            var e2 = err * 2;
            var nextX = x0;
            var nextY = y0;
            if (e2 >= dy)
            {
                err += dy;
                nextX += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                nextY += sy;
            }

            if (!(nextX == x1 && nextY == y1))
            {
                var cx = nextX - x0;
                var cy = nextY - y0;
                var ch = PickLineChar(cx, cy);
                canvas.SetIfEmpty(nextX, nextY, ch, color);
            }

            x0 = nextX;
            y0 = nextY;
        }
    }

    private static char PickLineChar(int dx, int dy)
    {
        if (dx == 0)
        {
            return '|';
        }

        if (dy == 0)
        {
            return '-';
        }

        return (dx > 0 && dy > 0) || (dx < 0 && dy < 0) ? '\\' : '/';
    }

    private static string PortLabel(PortType port) =>
        port switch
        {
            PortType.Generic => "3:1",
            PortType.Wood => "Wood",
            PortType.Brick => "Brick",
            PortType.Sheep => "Sheep",
            PortType.Wheat => "Wheat",
            PortType.Ore => "Ore",
            _ => "??",
        };

    private static string PortColor(PortType port) =>
        port switch
        {
            PortType.Generic => FgBrightWhite,
            PortType.Wood => FgGreen,
            PortType.Brick => FgRed,
            PortType.Sheep => FgBrightGreen,
            PortType.Wheat => FgYellow,
            PortType.Ore => FgSilver,
            _ => FgWhite,
        };

    private static string PlayerColor(int player) =>
        player switch
        {
            1 => FgRed,
            2 => FgBrightGreen,
            3 => FgYellow,
            4 => FgBrightWhite,
            _ => FgWhite,
        };

    private static string ResourceStyle(ResourceType resource) =>
        resource switch
        {
            ResourceType.Wood => FgGreen,
            ResourceType.Brick => FgRed,
            ResourceType.Sheep => FgBrightGreen,
            ResourceType.Wheat => FgYellow,
            ResourceType.Ore => FgSilver,
            ResourceType.Desert => FgBeige,
            _ => Reset,
        };

    private static void Pause(string message)
    {
        DrawFrame(() =>
        {
            Console.WriteLine();
            Console.WriteLine(message);
            Console.WriteLine("Press any key to continue...");
        });
        Console.ReadKey(intercept: true);
    }

    private static void DrawFrame(Action draw)
    {
        Console.SetCursorPosition(0, 0);
        draw();

        var usedLines = Console.CursorTop + (Console.CursorLeft > 0 ? 1 : 0);
        var clearWidth = Math.Max(1, Console.WindowWidth - 1);
        for (var i = usedLines; i < _lastFrameLineCount; i++)
        {
            Console.SetCursorPosition(0, i);
            Console.Write(new string(' ', clearWidth));
        }

        Console.SetCursorPosition(0, usedLines);
        _lastFrameLineCount = usedLines;
    }

    private static void WriteFixedLine(string text = "")
    {
        var width = Math.Max(1, Console.WindowWidth - 1);
        if (text.Length > width)
        {
            text = text[..width];
        }

        Console.Write(text);
        Console.WriteLine(new string(' ', Math.Max(0, width - text.Length)));
    }
}

internal enum ActionSelectionMode
{
    Direct,
    RollDice,
    BuyDevCardRandom,
    OpenTradeMenu,
    OpenYearOfPlentyMenu,
    Back,
    PlaceSettlement,
    PlaceRoad,
    PlaceCity,
    PlaceRobber,
}

internal enum ActionMenuContext
{
    Root,
    Trade,
    YearOfPlenty,
}

internal enum LocationSelectionMode
{
    Vertex,
    Edge,
    Tile,
}

internal sealed record MenuEntry(string Label, ActionSelectionMode Mode, CatanAction? Action);

internal sealed record BoardRenderLayout(
    (double X, double Y)[] TilePixels,
    (double X, double Y)[] VertexPixels,
    (int X, int Y)[] TilePoints,
    (int X, int Y)[] VertexPoints,
    (int X, int Y)[] EdgePoints,
    double MinX,
    double MinY,
    double ScaleX,
    double ScaleY,
    int MarginX,
    int MarginY,
    int Width,
    int Height,
    (double X, double Y) BoardCenter);

internal sealed class Canvas
{
    private readonly char[,] _chars;
    private readonly string?[,] _colors;

    public int Width { get; }
    public int Height { get; }

    public Canvas(int width, int height)
    {
        Width = width;
        Height = height;
        _chars = new char[height, width];
        _colors = new string?[height, width];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                _chars[y, x] = ' ';
                _colors[y, x] = null;
            }
        }
    }

    public void Set(int x, int y, char ch, string color)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height)
        {
            return;
        }

        _chars[y, x] = ch;
        _colors[y, x] = color;
    }

    public void SetIfEmpty(int x, int y, char ch, string color)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height)
        {
            return;
        }

        if (_chars[y, x] != ' ')
        {
            return;
        }

        _chars[y, x] = ch;
        _colors[y, x] = color;
    }

    public void Print()
    {
        for (var y = 0; y < Height; y++)
        {
            var lineHasContent = false;
            for (var x = 0; x < Width; x++)
            {
                if (_chars[y, x] != ' ')
                {
                    lineHasContent = true;
                    break;
                }
            }

            if (!lineHasContent)
            {
                continue;
            }

            string? activeColor = null;
            var trailingSpaceStart = Width;
            for (var x = Width - 1; x >= 0; x--)
            {
                if (_chars[y, x] != ' ')
                {
                    trailingSpaceStart = x + 1;
                    break;
                }
            }

            for (var x = 0; x < trailingSpaceStart; x++)
            {
                var color = _colors[y, x];
                if (!string.Equals(color, activeColor, StringComparison.Ordinal))
                {
                    Console.Write(color ?? "\u001b[0m");
                    activeColor = color;
                }

                Console.Write(_chars[y, x]);
            }

            if (activeColor is not null)
            {
                Console.Write("\u001b[0m");
            }
            Console.WriteLine();
        }
    }
}

internal enum MapChoice
{
    Mini,
    Standard,
}
