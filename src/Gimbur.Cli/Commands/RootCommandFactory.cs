using System.CommandLine;
using System.Text.Json;
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

        var exportTypeOption = new Option<ExportType>("--export-type")
        {
            Description = "Export schema: GameState (default) or InitialPlacement (placement-only actions)",
            DefaultValueFactory = _ => ExportType.GameState,
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

        var serverUrlOption = new Option<string>("--server-url")
        {
            Description = "Base URL of the Gimbur.Server (e.g. http://localhost:5123)",
            DefaultValueFactory = _ => "http://localhost:5123",
        };

        var serverPriorModeOption = new Option<string>("--server-prior-mode")
        {
            Description = "Prior mode for server-mcts-nn: 'state' or 'placement'",
            DefaultValueFactory = _ => "state",
        };

        var serverMaxPriorDepthOption = new Option<int?>("--server-max-prior-depth")
        {
            Description = "Max tree depth for NN prior requests (server-mcts-nn only)",
        };

        var placementOnlyOption = new Option<bool>("--placement-only")
        {
            Description = "Stop MCTS expansion at the placement/main-game boundary (RollDiceAction). " +
                          "The game loop ends when the best action is a HorizonAction.",
        };

        var maxPriorDepthOption = new Option<int>("--max-prior-depth")
        {
            Description = "Maximum tree depth for NN prior requests (default: unlimited). " +
                          "Nodes deeper than this use uniform priors, reducing wasted inference work.",
            DefaultValueFactory = _ => int.MaxValue,
        };

        var simulationsPerActionOption = new Option<int>("--simulations-per-action")
        {
            Description = "Run a fixed number of MCTS simulations per composite placement action. " +
                          "Ensures uniform evaluation coverage for all actions (placement export only, default: 0 = disabled).",
            DefaultValueFactory = _ => 0,
        };

        var parallelismOption = new Option<int>("--parallelism")
        {
            Description = "Maximum games simulated concurrently (default: 4 with NN priors, otherwise all processors)",
            DefaultValueFactory = _ => 0,
        };

        var command = new Command("simulate", "Run Settlers of Catan AI self-play simulations.")
        {
          noOfGamesOption,
          noOfPlayersOption,
          exportOption,
          exportFormatOption,
          exportTypeOption,
          searchTimeOption,
          maxSimulationsOption,
          maxRolloutDepthOption,
          actionRolloutLimitOption,
          noSymmetriesOption,
          priorOption,
          nnUrlOption,
          placementOnlyOption,
          maxPriorDepthOption,
          simulationsPerActionOption,
          parallelismOption,
        };

        command.SetAction(parseResult =>
        {
            // ── Load config file if provided ─────────────────────────
            JsonElement cfg = default;
            FileInfo? configFile = parseResult.GetValue(globals.Config);
            if (configFile is not null)
                cfg = ConfigLoader.Load(configFile);

            uint noOfGames = parseResult.GetValue(noOfGamesOption);
            int seed = parseResult.GetValue(globals.Seed) ?? new Random().Next();
            int noOfPlayers = parseResult.GetValue(noOfPlayersOption);
            string? mapConfig = parseResult.GetValue(globals.MapConfiguration);
            string? verbosity = ParseVerbosity(parseResult, globals);
            FileInfo? export = parseResult.GetValue(exportOption);
            ExportFormat exportFormat = parseResult.GetValue(exportFormatOption);
            ExportType exportType = parseResult.GetValue(exportTypeOption);
            int searchTimeMs = parseResult.GetValue(searchTimeOption);
            int maxSimulations = parseResult.GetValue(maxSimulationsOption);
            int maxRolloutDepth = parseResult.GetValue(maxRolloutDepthOption);
            int actionRolloutLimit = parseResult.GetValue(actionRolloutLimitOption);
            bool noSymmetries = parseResult.GetValue(noSymmetriesOption);
            bool prior = parseResult.GetValue(priorOption);
            string nnUrl = parseResult.GetValue(nnUrlOption)!;
            string serverUrl = parseResult.GetValue(serverUrlOption)!;
            string serverPriorMode = parseResult.GetValue(serverPriorModeOption)!;
            int? serverMaxPriorDepth = parseResult.GetValue(serverMaxPriorDepthOption);
            bool placementOnly = parseResult.GetValue(placementOnlyOption);
            int maxPriorDepth = parseResult.GetValue(maxPriorDepthOption);
            int simulationsPerAction = parseResult.GetValue(simulationsPerActionOption);
            int parallelism = parseResult.GetValue(parallelismOption);

            // ── Apply config file defaults for simulate ──────────────
            if (cfg.ValueKind == JsonValueKind.Object)
            {
                if (!WasProvided(parseResult, "--games", "-g"))
                    noOfGames = ConfigLoader.GetUInt(cfg, "games") ?? noOfGames;
                if (!WasProvided(parseResult, "--seed"))
                    seed = ConfigLoader.GetInt(cfg, "seed") ?? seed;
                if (!WasProvided(parseResult, "--players", "-p"))
                    noOfPlayers = ConfigLoader.GetInt(cfg, "players") ?? noOfPlayers;
                if (!WasProvided(parseResult, "--map-config"))
                    mapConfig = ConfigLoader.GetString(cfg, "mapConfig") ?? mapConfig;
                if (!WasProvided(parseResult, "--export"))
                {
                    var exportPath = ConfigLoader.GetString(cfg, "export");
                    if (exportPath is not null)
                        export = new FileInfo(exportPath);
                }
                if (!WasProvided(parseResult, "--export-format"))
                {
                    var fmt = ConfigLoader.GetString(cfg, "exportFormat");
                    if (fmt is not null && Enum.TryParse<ExportFormat>(fmt, ignoreCase: true, out var ef))
                        exportFormat = ef;
                }
                if (!WasProvided(parseResult, "--export-type"))
                {
                    var et = ConfigLoader.GetString(cfg, "exportType");
                    if (et is not null && Enum.TryParse<ExportType>(et, ignoreCase: true, out var etv))
                        exportType = etv;
                }
                if (!WasProvided(parseResult, "--search-time"))
                    searchTimeMs = ConfigLoader.GetInt(cfg, "searchTimeMs") ?? searchTimeMs;
                if (!WasProvided(parseResult, "--max-simulations"))
                    maxSimulations = ConfigLoader.GetInt(cfg, "maxSimulations") ?? maxSimulations;
                if (!WasProvided(parseResult, "--max-rollout-depth"))
                    maxRolloutDepth = ConfigLoader.GetInt(cfg, "maxRolloutDepth") ?? maxRolloutDepth;
                if (!WasProvided(parseResult, "--action-rollout-limit"))
                    actionRolloutLimit = ConfigLoader.GetInt(cfg, "actionRolloutLimit") ?? actionRolloutLimit;
                if (!WasProvided(parseResult, "--no-symmetries"))
                    noSymmetries = ConfigLoader.GetBool(cfg, "noSymmetries") ?? noSymmetries;
                if (!WasProvided(parseResult, "--prior"))
                    prior = ConfigLoader.GetBool(cfg, "prior") ?? prior;
                if (!WasProvided(parseResult, "--nn-url"))
                    nnUrl = ConfigLoader.GetString(cfg, "nnUrl") ?? nnUrl;
                if (!WasProvided(parseResult, "--server-url"))
                    serverUrl = ConfigLoader.GetString(cfg, "serverUrl") ?? serverUrl;
                if (!WasProvided(parseResult, "--server-prior-mode"))
                    serverPriorMode = ConfigLoader.GetString(cfg, "serverPriorMode") ?? serverPriorMode;
                if (!WasProvided(parseResult, "--server-max-prior-depth"))
                    serverMaxPriorDepth = ConfigLoader.GetInt(cfg, "serverMaxPriorDepth") ?? serverMaxPriorDepth;
                if (!WasProvided(parseResult, "--placement-only"))
                    placementOnly = ConfigLoader.GetBool(cfg, "placementOnly") ?? placementOnly;
                if (!WasProvided(parseResult, "--max-prior-depth"))
                    maxPriorDepth = ConfigLoader.GetInt(cfg, "maxPriorDepth") ?? maxPriorDepth;
                if (!WasProvided(parseResult, "--simulations-per-action"))
                    simulationsPerAction = ConfigLoader.GetInt(cfg, "simulationsPerAction") ?? simulationsPerAction;
                if (!WasProvided(parseResult, "--parallelism"))
                    parallelism = ConfigLoader.GetInt(cfg, "parallelism") ?? parallelism;
                if (!WasProvided(parseResult, "--verbosity", "-v") && !WasProvided(parseResult, "-q") && !WasProvided(parseResult, "--verbose"))
                    verbosity = ConfigLoader.GetString(cfg, "verbosity") ?? verbosity;
            }

            // Auto-enable placement-only mode when InitialPlacement export is selected.
            if (exportType == ExportType.InitialPlacement)
                placementOnly = true;

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
                ExportType = exportType,
                Verbosity = verbosity ?? "normal",
                SearchTimeMs = searchTimeMs,
                MaxSimulations = maxSimulations,
                MaxRolloutDepth = maxRolloutDepth,
                ActionRolloutLimit = actionRolloutLimit,
                Symmetries = !noSymmetries,
                Prior = prior,
                NnUrl = nnUrl,
                PlacementOnly = placementOnly,
                MaxPriorDepth = maxPriorDepth,
                SimulationsPerAction = simulationsPerAction,
                Parallelism = parallelism,
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
                          "Available: random, greedy, mcts, nn, nn-placement, nn-placement-random, nn-state, nn-state-random, server-mcts, server-mcts-nn, nn-mcts-placement, nn-mcts-placement-random, mcts-placement, mcts-placement-random",
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

        var serverUrlOption = new Option<string>("--server-url")
        {
            Description = "Base URL of the Gimbur.Server (e.g. http://localhost:5123)",
            DefaultValueFactory = _ => "http://localhost:5123",
        };

        var serverPriorModeOption = new Option<string>("--server-prior-mode")
        {
            Description = "Prior mode for server-mcts-nn: 'state' or 'placement'",
            DefaultValueFactory = _ => "state",
        };

        var serverMaxPriorDepthOption = new Option<int?>("--server-max-prior-depth")
        {
            Description = "Max tree depth for NN prior requests (server-mcts-nn only)",
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
            serverUrlOption,
            serverPriorModeOption,
            serverMaxPriorDepthOption,
        };

        command.SetAction(parseResult =>
        {
            // ── Load config file if provided ─────────────────────────
            JsonElement cfg = default;
            FileInfo? configFile = parseResult.GetValue(globals.Config);
            if (configFile is not null)
                cfg = ConfigLoader.Load(configFile);

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
            string serverUrl = parseResult.GetValue(serverUrlOption)!;
            string serverPriorMode = parseResult.GetValue(serverPriorModeOption)!;
            int? serverMaxPriorDepth = parseResult.GetValue(serverMaxPriorDepthOption);

            // ── Apply config file defaults for benchmark ─────────────
            if (cfg.ValueKind == JsonValueKind.Object)
            {
                if (!WasProvided(parseResult, "--games", "-g"))
                    noOfGames = ConfigLoader.GetUInt(cfg, "games") ?? noOfGames;
                if (!WasProvided(parseResult, "--seed"))
                    seed = ConfigLoader.GetInt(cfg, "seed") ?? seed;
                if (!WasProvided(parseResult, "--map-config"))
                    mapConfig = ConfigLoader.GetString(cfg, "mapConfig") ?? mapConfig;
                if (!WasProvided(parseResult, "--output", "-o"))
                {
                    var outputPath = ConfigLoader.GetString(cfg, "output");
                    if (outputPath is not null)
                        output = new FileInfo(outputPath);
                }
                if (!WasProvided(parseResult, "--ai"))
                {
                    var ai = ConfigLoader.GetStringArray(cfg, "ai");
                    if (ai is not null)
                        aiNames = ai;
                }
                if (!WasProvided(parseResult, "--search-time"))
                    searchTimeMs = ConfigLoader.GetInt(cfg, "searchTimeMs") ?? searchTimeMs;
                if (!WasProvided(parseResult, "--max-simulations"))
                    maxSimulations = ConfigLoader.GetInt(cfg, "maxSimulations") ?? maxSimulations;
                if (!WasProvided(parseResult, "--max-rollout-depth"))
                    maxRolloutDepth = ConfigLoader.GetInt(cfg, "maxRolloutDepth") ?? maxRolloutDepth;
                if (!WasProvided(parseResult, "--nn-url"))
                    nnUrl = ConfigLoader.GetString(cfg, "nnUrl") ?? nnUrl;
                if (!WasProvided(parseResult, "--server-url"))
                    serverUrl = ConfigLoader.GetString(cfg, "serverUrl") ?? serverUrl;
                if (!WasProvided(parseResult, "--server-prior-mode"))
                    serverPriorMode = ConfigLoader.GetString(cfg, "serverPriorMode") ?? serverPriorMode;
                if (!WasProvided(parseResult, "--server-max-prior-depth"))
                    serverMaxPriorDepth = ConfigLoader.GetInt(cfg, "serverMaxPriorDepth") ?? serverMaxPriorDepth;
                if (!WasProvided(parseResult, "--verbosity", "-v") && !WasProvided(parseResult, "-q") && !WasProvided(parseResult, "--verbose"))
                    verbosity = ConfigLoader.GetString(cfg, "verbosity") ?? verbosity;
            }

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
                ServerUrl = serverUrl,
                ServerPriorMode = serverPriorMode,
                ServerMaxPriorDepth = serverMaxPriorDepth,
            };

            var runner = new BenchmarkRunner(options);
            runner.Run();
        });

        return command;
    }

    private static bool TryParseAiKind(string name, out AiKind kind)
    {
        return Enum.TryParse(name.Replace("-", ""), ignoreCase: true, out kind);
    }

    /// <summary>
    /// Returns true if any of the given option names appear as tokens in the
    /// parse result, indicating the user explicitly provided the option on
    /// the command line.
    /// </summary>
    private static bool WasProvided(ParseResult parseResult, params string[] optionNames)
    {
        foreach (var token in parseResult.Tokens)
        {
            foreach (var name in optionNames)
            {
                if (token.Value == name) return true;
            }
        }

        return false;
    }
}
