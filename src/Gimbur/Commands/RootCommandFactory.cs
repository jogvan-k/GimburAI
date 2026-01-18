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
            Description = "Path to a configuration file"
        };

        var verboseOption = new Option<bool>("--verbose")
        {
            Description = "Increase logging verbosity"
        };

        rootCommand.Options.Add(configOption);
        rootCommand.Options.Add(verboseOption);

        rootCommand.Subcommands.Add(CreateSimulateCommand());
        rootCommand.Subcommands.Add(CreatePlayCommand());

        rootCommand.SetAction(parserResults => Console.WriteLine("Gimbur CLI placeholder – use --help to explore commands."));

        return rootCommand;
    }

    private static Command CreateSimulateCommand()
    {
        var noOfGamesOption = new Option<uint>("--games", "-g")
        {
            Description = "Number of games to simulate",
            DefaultValueFactory = _ => 1
        };
        var searchTimeOption = new Option<TimeSpan>("--search-time", "-s")
        {
            Description = "Time budget per move",
            DefaultValueFactory = _ => TimeSpan.FromSeconds(2)
        };
        var maxSimulationsOption = new Option<uint?>("--max-simulations", "-m")
        {
            Description = "Maximum simulations per turn (0 = unlimited)",
        };
        var seedOption = new Option<int?>("--seed")
        {
            Description = "Random seed to ensure reproducibility",
        };
        var noOfPlayersOption = new Option<int>("--players", "-p")
        {
            Description = "Player count for the simulation",
        };
        var mapConfigOption = new Option<string?>("--map-config")
        {
            Description = "Map layout identifier",
        };
        var verbosityOption = new Option<string>("--verbosity", "-v")
        {
            Description = "Logging verbosity for simulation output",
        };

        noOfPlayersOption.Validators.Add(result =>
        {
            if (result.Tokens.Count == 0)
            {
              return; // ALlow empty value
            }

            var value = result.GetValue(noOfPlayersOption);
            if (!(1 <= value && value <= 4))
            {
              result.AddError($"Argument '{value}' must be between 1 and 4");
            }
        });
        // Add -q as a separate option for quiet verbosity.
        Option<bool> quietOption = new("-q")
        {
            Description = "Set verbosity to quiet (shorthand for --verbosity quiet)",
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

        var exportOption = new Option<FileInfo?>("--export")
        {
            Description = "Optional path to export game transcripts",
        };

        var command = new Command("simulate", "Run Settlers of Catan AI self-play simulations.")
        {
            noOfGamesOption,
            searchTimeOption,
            maxSimulationsOption,
            seedOption,
            noOfPlayersOption,
            mapConfigOption,
            verbosityOption,
            exportOption
        };

        command.SetAction(parseResult =>
            {
                uint noOfGames = parseResult.GetValue(noOfGamesOption);
                TimeSpan searchTime = parseResult.GetValue(searchTimeOption);
                uint? maxSimulations = parseResult.GetValue(maxSimulationsOption);
                int seed = parseResult.GetValue(seedOption) ?? new Random().Next();
                int noOfPlayers = parseResult.GetValue(noOfPlayersOption);
                string? mapConfig = parseResult.GetValue(mapConfigOption);
                string? verbosity;
                FileInfo? export = parseResult.GetValue(exportOption);

                // Check if -q was specified.
                if (parseResult.GetValue(quietOption))
                {
                    verbosity = "quiet";
                }
                else
                {
                    verbosity = parseResult.GetValue(verbosityOption) ?? "diagnostic";
                }
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

    private static Command CreatePlayCommand()
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

        var searchTimeOption = new Option<TimeSpan>("--search-time", "-s")
        {
            Description = "Time budget per move for AI players",
            DefaultValueFactory = _ => TimeSpan.FromSeconds(2),
        };

        var maxSimulationsOption = new Option<int>("--max-simulations", "-m")
        {
            Description = "Maximum simulations per AI turn (0 = unlimited)",
            DefaultValueFactory = _ => 0,
        };

        var command = new Command("play", "Play a Settlers of Catan match with human and AI players.")
        {
            humanPositionOption,
            aiOption,
            searchTimeOption,
            maxSimulationsOption,
        };

        command.SetAction(parserResults => Console.WriteLine("TODO: implement interactive play mode"));

        return command;
    }
}
