namespace Gimbur.Cli;

internal static class AiKindNames
{
    public static string Format(AiKind kind) => kind switch
    {
        AiKind.NnPlacement => "nn-placement",
        AiKind.NnMainGame => "nn-main-game",
        AiKind.NnValue => "nn-value",
        AiKind.NnValuePlacement => "nn-value-placement",
        AiKind.NnValueMainGame => "nn-value-main-game",
        AiKind.NnOnePly => "nn-one-ply",
        AiKind.NnOnePlyPlacement => "nn-one-ply-placement",
        AiKind.NnOnePlyMainGame => "nn-one-ply-main-game",
        AiKind.ServerMcts => "server-mcts",
        AiKind.ServerMctsNn => "server-mcts-nn",
        _ => kind.ToString().ToLowerInvariant(),
    };
}
