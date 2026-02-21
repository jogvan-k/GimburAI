using Gimbur.Rules;

namespace Gimbur.Tui;

internal static class Program
{
    private const double Sqrt3 = 1.7320508075688772;

    private const string Reset = "\u001b[0m";
    private const string FgBlack = "\u001b[30m";
    private const string FgWhite = "\u001b[37m";
    private const string FgBrightWhite = "\u001b[97m";
    private const string FgBrightBlack = "\u001b[90m";
    private const string FgRed = "\u001b[31m";
    private const string FgBrightRed = "\u001b[91m";
    private const string FgYellow = "\u001b[33m";
    private const string FgBrightGreen = "\u001b[92m";
    private const string FgBrightYellow = "\u001b[93m";
    private const string FgGreen = "\u001b[32m";
    private const string FgCyan = "\u001b[36m";
    private const string FgSilver = "\u001b[38;5;250m";
    private const string FgBeige = "\u001b[38;5;223m";
    private static void Main()
    {
        Console.WriteLine("Gimbur TUI");
        Console.WriteLine();

        var mapChoice = PromptMapTopology();
        var config = mapChoice == MapChoice.Mini ? GameConfig.Mini : GameConfig.Standard;
        var players = PromptPlayerCount(config.MinPlayers, config.MaxPlayers);

        var rng = new Random();
        var setup = BoardSetup.Generate(config.Map, rng);
        var board = new Board(setup, config);

        Console.WriteLine();
        Console.WriteLine($"Initialized {mapChoice.ToString().ToLowerInvariant()} map for {players} players.");
        Console.WriteLine($"Tiles: {board.Topology.TileCount}, Vertices: {board.Topology.VertexCount}, Edges: {board.Topology.EdgeCount}");
        Console.WriteLine();
        RenderBoard(board);
        Console.WriteLine();
        Console.WriteLine("Legend: [RsNN*] tile, o empty vertex, s settlement, c city, . empty edge, -/\\ road");
        Console.WriteLine("Ports: 3:1 generic plus Wood/Brick/Sheep/Wheat/Ore resource ports");
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

    private static void RenderBoard(Board board)
    {
        var topology = board.Topology;
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

        var width = (int)Math.Ceiling((maxX - minX) * scaleX) + marginX * 2 + 20;
        var height = (int)Math.Ceiling((maxY - minY) * scaleY) + marginY * 2 + 10;
        var canvas = new Canvas(width, height);

        for (var ei = 0; ei < topology.EdgeCount; ei++)
        {
            var edge = topology.Edges[ei];
            var p0 = vertexPoints[edge.VertexA];
            var p1 = vertexPoints[edge.VertexB];
            DrawEdge(canvas, p0, p1, board.EdgeOccupancy[ei].Player);
        }

        for (var vi = 0; vi < topology.VertexCount; vi++)
        {
            DrawVertex(canvas, vertexPoints[vi], board.VertexOccupancy[vi]);
        }

        for (var ti = 0; ti < topology.TileCount; ti++)
        {
            DrawTile(canvas, tilePoints[ti], board, ti);
        }

        for (var pi = 0; pi < topology.PortCount; pi++)
        {
            var (va, vb) = topology.Ports[pi];
            var port = board.PortType(pi);
            var portColor = PortColor(port);
            var mid = (
                X: (vertexPixels[va].X + vertexPixels[vb].X) / 2.0,
                Y: (vertexPixels[va].Y + vertexPixels[vb].Y) / 2.0);
            var outward = (
                X: mid.X - boardCenter.X,
                Y: mid.Y - boardCenter.Y);
            var length = Math.Sqrt((outward.X * outward.X) + (outward.Y * outward.Y));
            if (length < 0.0001)
            {
                length = 1.0;
                outward = (1.0, 0.0);
            }

            var portPos = (
                X: mid.X + (outward.X / length) * 1.15,
                Y: mid.Y + (outward.Y / length) * 1.15);
            var point = ToCanvasPoint(portPos, minX, minY, scaleX, scaleY, marginX, marginY);
            var label = PortLabel(port);
            var labelStartX = point.X - (label.Length / 2);
            var labelEndX = labelStartX + label.Length - 1;
            var leftAnchor = (X: labelStartX - 1, Y: point.Y);
            var rightAnchor = (X: labelEndX + 1, Y: point.Y);
            var aAnchor = vertexPoints[va].X <= point.X ? leftAnchor : rightAnchor;
            var bAnchor = vertexPoints[vb].X <= point.X ? leftAnchor : rightAnchor;

            DrawConnector(canvas, aAnchor, vertexPoints[va], portColor);
            DrawConnector(canvas, bAnchor, vertexPoints[vb], portColor);
            DrawString(canvas, labelStartX, point.Y, label, portColor);
        }

        canvas.Print();
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

    private static void DrawEdge(Canvas canvas, (int X, int Y) p0, (int X, int Y) p1, int player)
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

        var color = player == 0 ? FgBrightBlack : PlayerColor(player);

        for (var i = 1; i < steps; i++)
        {
            var x = p0.X + (dx * i) / steps;
            var y = p0.Y + (dy * i) / steps;
            canvas.Set(x, y, edgeChar, color);
        }
    }

    private static void DrawVertex(Canvas canvas, (int X, int Y) point, VertexOccupancy occupancy)
    {
        if (occupancy.IsEmpty)
        {
            canvas.Set(point.X, point.Y, 'o', FgBrightBlack);
            return;
        }

        var marker = occupancy.Building == BuildingType.City ? 'c' : 's';
        canvas.Set(point.X, point.Y, marker, PlayerColor(occupancy.Player));
    }

    private static void DrawTile(Canvas canvas, (int X, int Y) center, Board board, int tileIndex)
    {
        var startX = center.X - 3;
        canvas.Set(startX, center.Y, '[', FgWhite);

        var resource = board.TileResource(tileIndex);
        var resourceCode = resource switch
        {
            ResourceType.Desert => "Ds",
            ResourceType.Wood => "Wd",
            ResourceType.Brick => "Br",
            ResourceType.Sheep => "Sh",
            ResourceType.Wheat => "Wh",
            ResourceType.Ore => "Or",
            _ => "??",
        };

        DrawString(canvas, startX + 1, center.Y, resourceCode, ResourceStyle(resource));

        var number = board.TileNumber(tileIndex);
        var numberText = number == 0 ? "--" : number.ToString("00");
        var numberColor = number is 6 or 8 ? FgBrightRed : FgBrightWhite;
        DrawString(canvas, startX + 3, center.Y, numberText, numberColor);

        var robberMarker = board.RobberTile == tileIndex ? "*" : " ";
        DrawString(canvas, startX + 5, center.Y, robberMarker, board.RobberTile == tileIndex ? FgRed : FgWhite);
        canvas.Set(startX + 6, center.Y, ']', FgWhite);
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
            2 => FgCyan,
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
}

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
                    if (color is null)
                    {
                        Console.Write(ProgramReset());
                    }
                    else
                    {
                        Console.Write(color);
                    }
                    activeColor = color;
                }

                Console.Write(_chars[y, x]);
            }

            if (activeColor is not null)
            {
                Console.Write(ProgramReset());
            }
            Console.WriteLine();
        }
    }

    private static string ProgramReset() => "\u001b[0m";
}

internal enum MapChoice
{
    Mini,
    Standard,
}
