namespace Gimbur.Cli;

internal static class AiKindNames
{
    public static string Format(AiKind kind) => kind switch
    {
        AiKind.NnPlacement => "nn-placement",
        AiKind.NnPlacementRandom => "nn-placement-random",
        AiKind.NnState => "nn-state",
        AiKind.NnStateRandom => "nn-state-random",
        AiKind.NnPlacementState => "nn-placement-state",
        AiKind.ServerMcts => "server-mcts",
        AiKind.ServerMctsNn => "server-mcts-nn",
        AiKind.NnMctsPlacement => "nn-mcts-placement",
        AiKind.NnMctsPlacementRandom => "nn-mcts-placement-random",
        AiKind.MctsPlacement => "mcts-placement",
        AiKind.MctsPlacementRandom => "mcts-placement-random",
        AiKind.NnMctsPlacementState => "nn-mcts-placement-state",
        AiKind.NnMctsState => "nn-mcts-state",
        _ => kind.ToString().ToLowerInvariant(),
    };
}
