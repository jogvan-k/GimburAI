using System.CommandLine;
using Gimbur.Cli;

namespace Gimbur.Commands;

internal static class RootCommandFactory
{
    private const string RootDescription = "Gimbur – Settlers of Catan simulator";

    internal static RootCommand Create()
    {
        var rootCommand = new RootCommand(RootDescription);

        var configOption = new Option<FileInfo?>("--config", "-c")
        {
            Description = "Path to a configuration file",
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
            Description = "Map layout identifier (mini, small, or standard)",
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
            Seed = seedOption,
            NoOfPlayers = noOfPlayersOption,
            MapConfiguration = mapConfigOption,
            Verbosity = verbosityOption,
            Quiet = quietOption,
            Verbose = verboseOption,
        };

        rootCommand.Options.Add(globals.Config);
        rootCommand.Options.Add(globals.Seed);
        rootCommand.Options.Add(globals.NoOfPlayers);
        rootCommand.Options.Add(globals.MapConfiguration);
        rootCommand.Options.Add(globals.Verbosity);

        rootCommand.Subcommands.Add(CreateSimulateCommand(globals));

        rootCommand.SetAction(parserResults => Console.WriteLine("Gimbur CLI – use --help to explore commands."));

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
            Description = "Path to export training data as JSONL (one JSON object per game)",
        };

        var searchTimeOption = new Option<int>("--search-time")
        {
            Description = "MCTS search time limit in milliseconds per decision",
            DefaultValueFactory = _ => 1000
        };

        var maxSimulationsOption = new Option<int>("--max-simulations")
        {
            Description = "Maximum MCTS simulations per decision (default: unlimited, time-limited)",
            DefaultValueFactory = _ => int.MaxValue
        };

        var maxRolloutDepthOption = new Option<int>("--max-rollout-depth")
        {
            Description = "Maximum rollout depth for MCTS simulations (default: 500)",
            DefaultValueFactory = _ => 500
        };

        var noSymmetriesOption = new Option<bool>("--no-symmetries")
        {
            Description = "Disable board symmetry permutations in exported training data",
        };

        var command = new Command("simulate", "Run Settlers of Catan AI self-play simulations.")
        {
          noOfGamesOption,
          exportOption,
          searchTimeOption,
          maxSimulationsOption,
          maxRolloutDepthOption,
          noSymmetriesOption,
        };

        command.SetAction(parseResult =>
        {
            uint noOfGames = parseResult.GetValue(noOfGamesOption);
            int seed = parseResult.GetValue(globals.Seed) ?? new Random().Next();
            int noOfPlayers = parseResult.GetValue(globals.NoOfPlayers);
            string? mapConfig = parseResult.GetValue(globals.MapConfiguration);
            string? verbosity = ParseVerbosity(parseResult, globals);
            FileInfo? export = parseResult.GetValue(exportOption);
            int searchTimeMs = parseResult.GetValue(searchTimeOption);
            int maxSimulations = parseResult.GetValue(maxSimulationsOption);
            int maxRolloutDepth = parseResult.GetValue(maxRolloutDepthOption);
            bool noSymmetries = parseResult.GetValue(noSymmetriesOption);

            var options = new SimulationOptions
            {
                NumberOfGames = noOfGames,
                Seed = seed,
                NumberOfPlayers = noOfPlayers,
                MapConfig = mapConfig,
                ExportPath = export,
                Verbosity = verbosity ?? "normal",
                SearchTimeMs = searchTimeMs,
                MaxSimulations = maxSimulations,
                MaxRolloutDepth = maxRolloutDepth,
                Symmetries = !noSymmetries,
            };

            var runner = new SimulationRunner(options);
            runner.Run();
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
            verbosity = parseResult.GetValue(globals.Verbosity) ?? "normal";
        }

        return verbosity;
    }

}
