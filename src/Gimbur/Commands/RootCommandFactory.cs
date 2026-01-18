using System.CommandLine;

namespace Gimbur.Commands;

internal static class RootCommandFactory
{
    private const string RootDescription = "Gimbur – Settlers of Catan simulator,";

    internal static RootCommand Create()
    {
        var rootCommand = new RootCommand(RootDescription);

        var configOption = new Option<FileInfo?>("--config", "-c")
        {
            Description = "Path to a configuration file",
            Recursive = true,
        };
        var searchTimeOption = new Option<TimeSpan>("--search-time", "-s")
        {
            Description = "Time budget per move",
            Recursive = true,
            DefaultValueFactory = _ => TimeSpan.FromSeconds(2)
        };
        var maxSimulationsOption = new Option<uint?>("--max-simulations", "-m")
        {
            Description = "Maximum simulations per turn ",
            Recursive = true,
        };
        var seedOption = new Option<int?>("--seed")
        {
            Description = "Random seed to ensure reproducibility",
            Recursive = true,
        };
        var noOfPlayersOption = new Option<int>("--players", "-p")
        {
            Description = "Player count for the simulation",
            Recursive = true,
        };
        var mapConfigOption = new Option<string?>("--map-config")
        {
            Description = "Map layout identifier",
            Recursive = true,
        };
        var verbosityOption = new Option<string>("--verbosity", "-v")
        {
            Description = "Logging verbosity for simulation output",
            Recursive = true,
        };

        noOfPlayersOption.Validators.Add(result =>
        {
            if (result.Tokens.Count == 0)
            {
                return; // Allow empty value
            }

            var value = result.GetValue(noOfPlayersOption);
            if (!(1 <= value && value <= 4))
            {
                result.AddError($"Argument '{value}' must be between 1 and 4");
            }
        });
        // Add -q and --verbose as a separate options for verbosity.
        Option<bool> quietOption = new("-q")
        {
            Description = "Set verbosity to quiet (shorthand for --verbosity quiet)",
            Recursive = true
        };
        Option<bool> verboseOption = new("--verbose")
        {
            Description = "Set verbosity to verbose (shorthand for --verbosity verbose)",
            Recursive = true
        };
        // Handle both short and long forms.
        verbosityOption.Validators.Add(result =>
        {
            if (result.Tokens.Count == 0)
            {
                return; // Allow default value.
            }

            string value = result.Tokens.Single().Value.ToLowerInvariant();
            string[] validValues = new[] { "quiet", "q", "minimal", "m", "normal", "n", "detailed", "d", "diagnostic", "diag" };

            if (!validValues.Contains(value))
            {
                result.AddError($"Argument '{value}' not recognized. Must be one of: 'q[uiet]', 'm[inimal]', 'n[ormal]', 'd[etailed]', 'diag[nostic]'");
            }
        });

        var globals = new
        {
            Config = configOption,
            SearchTime = searchTimeOption,
            MaxSimulations = maxSimulationsOption,
            Seed = seedOption,
            NoOfPlayers = noOfPlayersOption,
            MapConfiguration = mapConfigOption,
            Verbosity = verbosityOption,
            Quiet = quietOption,
            Verbose = verboseOption,

        };

        rootCommand.Options.Add(globals.Config);
        rootCommand.Options.Add(globals.SearchTime);
        rootCommand.Options.Add(globals.MaxSimulations);
        rootCommand.Options.Add(globals.Seed);
        rootCommand.Options.Add(globals.NoOfPlayers);
        rootCommand.Options.Add(globals.MapConfiguration);
        rootCommand.Options.Add(globals.Verbosity);

        rootCommand.Subcommands.Add(CreateSimulateCommand(globals));
        rootCommand.Subcommands.Add(CreatePlayCommand(globals));

        rootCommand.SetAction(parserResults => Console.WriteLine("Gimbur CLI placeholder – use --help to explore commands."));

        return rootCommand;
    }

    private static Command CreateSimulateCommand(dynamic globals)
    {
        var noOfGamesOption = new Option<uint>("--games", "-g")
        {
            Description = "Number of games to simulate",
            DefaultValueFactory = _ => 1
        };

        var exportOption = new Option<FileInfo?>("--export")
        {
            Description = "Optional path to export game transcripts",
        };

        var command = new Command("simulate", "Run Settlers of Catan AI self-play simulations.")
        {
          noOfGamesOption,
          exportOption,
        };

        command.SetAction(parseResult =>
        {
            FileInfo? config = parseResult.GetValue(globals.Config);
            uint noOfGames = parseResult.GetValue(noOfGamesOption);
            TimeSpan searchTime = parseResult.GetValue(globals.SearchTime);
            uint? maxSimulations = parseResult.GetValue(globals.MaxSimulations);
            int seed = parseResult.GetValue(globals.Seed) ?? new Random().Next();
            int noOfPlayers = parseResult.GetValue(globals.NoOfPlayers);
            string? mapConfig = parseResult.GetValue(globals.MapConfiguration);
            string? verbosity = ParseVerbosity(parseResult, globals);
            FileInfo? export = parseResult.GetValue(exportOption);


            Console.WriteLine($"Config: {config?.FullName ?? "(null)"}");
            Console.WriteLine($"NoOfGames: {noOfGames}");
            Console.WriteLine($"SearchTime: {searchTime}");
            Console.WriteLine($"MaxSimulations: {maxSimulations}");
            Console.WriteLine($"Seed: {seed}");
            Console.WriteLine($"NoOfPlayers: {noOfPlayers}");
            Console.WriteLine($"MapConfig: {mapConfig ?? "(null)"}");
            Console.WriteLine($"Export: {export?.FullName ?? "(null)"}");
            Console.WriteLine($"Verbosity: {verbosity}");
            Console.WriteLine("TODO: implement simulation runner");
        });

        return command;
    }

    private static Command CreatePlayCommand(dynamic globals)
    {
        var humanPositionOption = new Option<int?>("--player-position", "-p")
        {
            Description = "Board position (seat) for the human player",
        };

        var aiOption = new Option<string[]>("--ai", "-a")
        {
            Description = "AI identifiers for automated players",
            AllowMultipleArgumentsPerToken = true,
            DefaultValueFactory = _ => ["R", "R", "R"]
        };

        var command = new Command("play", "Play a Settlers of Catan match with human and AI players.")
        {
            humanPositionOption,
            aiOption,
        };

        command.SetAction(parseResult =>
        {
            FileInfo? config = parseResult.GetValue(globals.Config);
            int? humanPosition = parseResult.GetValue(humanPositionOption);
            string[] ai = parseResult.GetValue(aiOption)!;
            TimeSpan searchTime = parseResult.GetValue(globals.SearchTime);
            uint? maxSimulations = parseResult.GetValue(globals.MaxSimulations);
            string? mapConfig = parseResult.GetValue(globals.MapConfiguration);
            string? verbosity = ParseVerbosity(parseResult, globals);

            Console.WriteLine($"Config: {config?.FullName ?? "(null)"}");
            Console.WriteLine($"verbosity: {verbosity}");
            Console.WriteLine($"HumanPosition: {humanPosition?.ToString() ?? "(null)"}");
            Console.WriteLine($"AI: {string.Join(",", ai)}");
            Console.WriteLine($"SearchTime: {searchTime}");
            Console.WriteLine($"MaxSimulations: {maxSimulations}");
            Console.WriteLine($"MapConfiguration: {mapConfig}");
            Console.WriteLine("TODO: implement interactive play mode");
        });

        return command;
    }

    private static string? ParseVerbosity(ParseResult parseResult, dynamic globals)
    {
        string? verbosity;
        if (parseResult.GetValue(globals.Quiet))
        {
            verbosity = "quiet";
        }
        else if (parseResult.GetValue(globals.Verbose))
        {
            verbosity = "verbose";
        }
        else
        {
            verbosity = parseResult.GetValue(globals.Verbosity) ?? "diagnostic";
        }

        return verbosity;
    }

}
