using Kjarni;

namespace Gimbur;

/// <summary>
/// Placeholder action for the Settlers of Catan game.
/// Implements ICoreAction to integrate with the Kjarni MCTS engine.
/// </summary>
public class CatanAction : ICoreAction
{
    public CatanAction(ICoreState origin)
    {
        Origin = origin;
    }

    public ICoreState Origin { get; }

    public ICoreState DoCoreAction()
    {
        // TODO: Implement action execution
        return Origin;
    }

    public int CompareTo(object? obj)
    {
        if (obj is CatanAction other)
        {
            // TODO: Implement proper comparison
            return 0;
        }
        throw new ArgumentException("Cannot compare CatanAction with a different type", nameof(obj));
    }

    public override bool Equals(object? obj)
    {
        // TODO: Implement proper equality
        return obj is CatanAction;
    }

    public override int GetHashCode()
    {
        // TODO: Implement proper hash code
        return 0;
    }
}

/// <summary>
/// Placeholder state for the Settlers of Catan game.
/// Implements ICoreState to integrate with the Kjarni MCTS engine.
/// </summary>
public class CatanState : ICoreState
{
    public Player PlayerTurn => Player.Player1;

    public int TurnNumber => 0;

    public ICoreAction[] Actions()
    {
        // TODO: Implement action generation
        return [new CatanAction(this)];
    }
}
