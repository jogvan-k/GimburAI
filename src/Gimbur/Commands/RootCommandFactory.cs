using System.CommandLine;

namespace Gimbur.Commands;

internal static class RootCommandFactory
{
    private const string RootDescription = "Gimbur – Settlers of Catan simulations powered by Kjarni";

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
        rootCommand.Subcommands.Add(CreateEvaluateCommand());
        rootCommand.Subcommands.Add(CreateBenchmarkCommand());

        rootCommand.SetAction(parserResults => Console.WriteLine("Gimbur CLI placeholder – use --help to explore commands."));

        return rootCommand;
    }

    private static Command CreateSimulateCommand()
    {
        var gamesOption = new Option<int>("--games")
        {
          Description = "Number of games to simulate",
        };
        var searchTimeOption = new Option<TimeSpan>("--search-time")
        {
          Description = "Time budget per move",
        };
        var maxSimulationsOption = new Option<int>("--max-simulations")
        {
          Description = "Maximum simulations per turn (0 = unlimited)",
        };
        var seedOption = new Option<int?>("--seed")
        {
          Description = "Random seed to ensure reproducibility",
        };
        var playersOption = new Option<int>("--players")
        {
          Description = "Player count for the simulation",
        };
        var mapOption = new Option<string?>("--map")
        {
          Description = "Map layout identifier",
        };
        var logLevelOption = new Option<string>("--log-level")
        {
          Description = "Logging verbosity for simulation output",
        };
        var exportOption = new Option<FileInfo?>("--export")
        {
          Description = "Optional path to export game transcripts",
        };

        var command = new Command("simulate", "Run Settlers of Catan AI self-play simulations.")
        {
            gamesOption,
            searchTimeOption,
            maxSimulationsOption,
            seedOption,
            playersOption,
            mapOption,
            logLevelOption,
            exportOption
        };

        command.SetAction(parserResults => Console.WriteLine("TODO: implement simulation runner"));

        return command;
    }

    private static Command CreatePlayCommand()
    {
        var humanPositionOption = new Option<int?>("--human-position")
        {
          Description = "Board position (seat) for the human player",
        };

        var aiOption = new Option<string[]>("--ai")
        {
          Description = "AI identifiers for automated players",
          AllowMultipleArgumentsPerToken = true,
          DefaultValueFactory = _ => Array.Empty<string>()
        };

        var searchTimeOption = new Option<TimeSpan>("--search-time")
        {
          Description = "Time budget per move for AI players",
          DefaultValueFactory = _ => TimeSpan.FromSeconds(2),
        };

        var maxSimulationsOption = new Option<int>("--max-simulations")
        {
          Description = "Maximum simulations per AI turn (0 = unlimited)",
          DefaultValueFactory = _ => 0, 
        };

        var interactiveOption = new Option<bool>("--interactive")
        {
          Description = "Enable interactive prompts during play",
        };

        var command = new Command("play", "Play a Settlers of Catan match with human and AI players.")
        {
            humanPositionOption,
            aiOption,
            searchTimeOption,
            maxSimulationsOption,
            interactiveOption
        };

        command.SetAction(parserResults => Console.WriteLine("TODO: implement interactive play mode"));

        return command;
    }

    private static Command CreateEvaluateCommand()
    {
        var stateFileOption = new Option<FileInfo>("--state-file")
        {
          Description = "Path to a serialized Catan game state",
          Required = true
        };

        var searchTimeOption = new Option<TimeSpan>("--search-time")
        {
          Description = "Time budget per evaluation",
          DefaultValueFactory = _ => TimeSpan.FromSeconds(2)
        };

        var maxSimulationsOption = new Option<int>("--max-simulations")
        {
          Description = "Maximum simulations per evaluation (0 = unlimited)",
        };

        var metricsOption = new Option<string[]>("--metrics")
        {
          Description = "One or more evaluation metrics to compute",
          AllowMultipleArgumentsPerToken = true
        };

        var command = new Command("evaluate", "Evaluate saved game states using the Kjarni engine.")
        {
            stateFileOption,
            searchTimeOption,
            maxSimulationsOption,
            metricsOption
        };

        command.SetAction(parserResults => Console.WriteLine("TODO: implement evaluation pipeline"));

        return command;
    }

    private static Command CreateBenchmarkCommand()
    {
        var runsOption = new Option<int>("--runs")
        {
          Description = "Number of benchmark runs",
          DefaultValueFactory = _ => 5,
        };

        var searchTimesOption = new Option<TimeSpan[]>("--search-times")
        {
          Description = "One or more search times to benchmark",
          AllowMultipleArgumentsPerToken = true
        };

        var maxSimulationsOption = new Option<int[]>("--max-simulations")
        {
          Description = "One or more simulation caps to compare",
          AllowMultipleArgumentsPerToken = true
        };

        var outputOption = new Option<FileInfo?>("--output")
        {
          Description = "Optional output file for benchmark results",
        };

        var command = new Command("benchmark", "Benchmark different search configurations using Kjarni.")
        {
            runsOption,
            searchTimesOption,
            maxSimulationsOption,
            outputOption
        };

        command.SetAction(parserResults => Console.WriteLine("TODO: implement benchmarking suite"));

        return command;
    }
}
