using System.CommandLine;
using System.CommandLine.Parsing;
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
            MapConfiguration = mapConfigOption,
            Verbosity = verbosityOption,
            Quiet = quietOption,
            Verbose = verboseOption,
        };

        rootCommand.Options.Add(globals.Config);
        rootCommand.Options.Add(globals.Seed);
        rootCommand.Options.Add(globals.MapConfiguration);
        rootCommand.Options.Add(globals.Verbosity);

        rootCommand.Subcommands.Add(CreateSimulateCommand(globals));
        rootCommand.Subcommands.Add(CreateBenchmarkCommand(globals));

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

        var noOfPlayersOption = new Option<int>("--players", "-p")
        {
            Description = "Player count for the simulation",
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

        var exportOption = new Option<FileInfo?>("--export")
        {
            Description = "Path for training data export (file for jsonl, directory for json)",
        };

        var exportFormatOption = new Option<ExportFormat>("--export-format")
        {
            Description = "Export format: none, jsonl (single file), or json (one file per game)",
            DefaultValueFactory = _ => ExportFormat.Jsonl,
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

        var actionRolloutLimitOption = new Option<int>("--action-rollout-limit")
        {
            Description = "Stop MCTS search when any action reaches this many rollouts (default: unlimited)",
            DefaultValueFactory = _ => int.MaxValue
        };

        var noSymmetriesOption = new Option<bool>("--no-symmetries")
        {
            Description = "Disable board symmetry permutations in exported training data",
        };

        var priorOption = new Option<bool>("--prior")
        {
            Description = "Enable async NN prior evaluation during MCTS search (requires running inference server)",
        };

        var nnUrlOption = new Option<string>("--nn-url")
        {
            Description = "Base URL of the NN inference server (e.g. http://localhost:8000)",
            DefaultValueFactory = _ => "http://localhost:8000",
        };

        var placementOnlyOption = new Option<bool>("--placement-only")
        {
            Description = "Stop MCTS expansion at the placement/main-game boundary (RollDiceAction). " +
                          "The game loop ends when the best action is a HorizonAction.",
        };

        var command = new Command("simulate", "Run Settlers of Catan AI self-play simulations.")
        {
          noOfGamesOption,
          noOfPlayersOption,
          exportOption,
          exportFormatOption,
          searchTimeOption,
          maxSimulationsOption,
          maxRolloutDepthOption,
          actionRolloutLimitOption,
          noSymmetriesOption,
          priorOption,
          nnUrlOption,
          placementOnlyOption,
        };

        command.SetAction(parseResult =>
        {
            uint noOfGames = parseResult.GetValue(noOfGamesOption);
            int seed = parseResult.GetValue(globals.Seed) ?? new Random().Next();
            int noOfPlayers = parseResult.GetValue(noOfPlayersOption);
            string? mapConfig = parseResult.GetValue(globals.MapConfiguration);
            string? verbosity = ParseVerbosity(parseResult, globals);
            FileInfo? export = parseResult.GetValue(exportOption);
            ExportFormat exportFormat = parseResult.GetValue(exportFormatOption);
            int searchTimeMs = parseResult.GetValue(searchTimeOption);
            int maxSimulations = parseResult.GetValue(maxSimulationsOption);
            int maxRolloutDepth = parseResult.GetValue(maxRolloutDepthOption);
            int actionRolloutLimit = parseResult.GetValue(actionRolloutLimitOption);
            bool noSymmetries = parseResult.GetValue(noSymmetriesOption);
            bool prior = parseResult.GetValue(priorOption);
            string nnUrl = parseResult.GetValue(nnUrlOption)!;
            bool placementOnly = parseResult.GetValue(placementOnlyOption);

            // Auto-enable prior when --nn-url is explicitly provided.
            if (!prior && parseResult.Tokens.Any(t => t.Value == "--nn-url"))
                prior = true;;

            var options = new SimulationOptions
            {
                NumberOfGames = noOfGames,
                Seed = seed,
                NumberOfPlayers = noOfPlayers,
                MapConfig = mapConfig,
                ExportPath = export,
                ExportFormat = exportFormat,
                Verbosity = verbosity ?? "normal",
                SearchTimeMs = searchTimeMs,
                MaxSimulations = maxSimulations,
                MaxRolloutDepth = maxRolloutDepth,
                ActionRolloutLimit = actionRolloutLimit,
                Symmetries = !noSymmetries,
                Prior = prior,
                NnUrl = nnUrl,
                PlacementOnly = placementOnly,
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

    private static Command CreateBenchmarkCommand(dynamic globals)
    {
        var noOfGamesOption = new Option<uint>("--games", "-g")
        {
            Description = "Number of games to run",
            DefaultValueFactory = _ => (uint)100,
        };

        var playersOption = new Option<string[]>("--ai")
        {
            Description = "AI for each player seat (e.g. --ai random greedy mcts nn). " +
                          "Available: random, greedy, mcts, nn",
            AllowMultipleArgumentsPerToken = true,
        };
        playersOption.DefaultValueFactory = _ => new[] { "random", "greedy" };

        var outputOption = new Option<FileInfo?>("--output", "-o")
        {
            Description = "Path to write JSON results file",
        };

        var searchTimeOption = new Option<int>("--search-time")
        {
            Description = "MCTS search time limit in milliseconds per decision",
            DefaultValueFactory = _ => 1000,
        };

        var maxSimulationsOption = new Option<int>("--max-simulations")
        {
            Description = "Maximum MCTS simulations per decision (default: unlimited, time-limited)",
            DefaultValueFactory = _ => int.MaxValue,
        };

        var maxRolloutDepthOption = new Option<int>("--max-rollout-depth")
        {
            Description = "Maximum rollout depth for MCTS simulations",
            DefaultValueFactory = _ => 500,
        };

        var nnUrlOption = new Option<string>("--nn-url")
        {
            Description = "Base URL of the NN inference server (e.g. http://localhost:8000)",
            DefaultValueFactory = _ => "http://localhost:8000",
        };

        var command = new Command("benchmark", "Run AI-vs-AI games and compute win rates.")
        {
            noOfGamesOption,
            playersOption,
            outputOption,
            searchTimeOption,
            maxSimulationsOption,
            maxRolloutDepthOption,
            nnUrlOption,
        };

        command.SetAction(parseResult =>
        {
            uint noOfGames = parseResult.GetValue(noOfGamesOption);
            int seed = parseResult.GetValue(globals.Seed) ?? new Random().Next();
            string? mapConfig = parseResult.GetValue(globals.MapConfiguration);
            string? verbosity = ParseVerbosity(parseResult, globals);
            FileInfo? output = parseResult.GetValue(outputOption);
            string[] aiNames = parseResult.GetValue(playersOption)!;
            int searchTimeMs = parseResult.GetValue(searchTimeOption);
            int maxSimulations = parseResult.GetValue(maxSimulationsOption);
            int maxRolloutDepth = parseResult.GetValue(maxRolloutDepthOption);
            string nnUrl = parseResult.GetValue(nnUrlOption)!;

            var aiKinds = new AiKind[aiNames.Length];
            for (var i = 0; i < aiNames.Length; i++)
            {
                if (!TryParseAiKind(aiNames[i], out var kind))
                {
                    Console.Error.WriteLine(
                        $"Unknown AI '{aiNames[i]}'. Available: {string.Join(", ", Enum.GetNames<AiKind>().Select(n => n.ToLowerInvariant()))}");
                    return;
                }
                aiKinds[i] = kind;
            }

            if (aiKinds.Length < 2)
            {
                Console.Error.WriteLine("At least 2 AIs are required for a benchmark.");
                return;
            }

            var options = new BenchmarkOptions
            {
                NumberOfGames = noOfGames,
                Seed = seed,
                MapConfig = mapConfig,
                Verbosity = verbosity ?? "normal",
                Players = aiKinds,
                OutputPath = output,
                SearchTimeMs = searchTimeMs,
                MaxSimulations = maxSimulations,
                MaxRolloutDepth = maxRolloutDepth,
                NnUrl = nnUrl,
            };

            var runner = new BenchmarkRunner(options);
            runner.Run();
        });

        return command;
    }

    private static bool TryParseAiKind(string name, out AiKind kind)
    {
        return Enum.TryParse(name, ignoreCase: true, out kind);
    }
}
