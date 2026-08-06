namespace Gimbur.Cli;

internal static class AiKindNames
{
    public static string Format(AiKind kind) => kind switch
    {
        AiKind.ServerMcts => "server-mcts",
        AiKind.ServerMctsNn => "server-mcts-nn",
        _ => kind.ToString().ToLowerInvariant(),
    };
}
