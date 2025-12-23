"""Generate Pine Script from a trained strategy."""

from pathlib import Path

from jarvis.genetics.strategy import Strategy


def pinescript(strategy_id: str, output_path: str | None = None) -> str:
    """Generate Pine Script for a trained strategy.

    Args:
        strategy_id: Strategy ID (e.g., ETHUSDT_5bdb12c7) or path to JSON file
        output_path: Optional output path. If None, saves to strategies/{strategy_id}.pine

    Returns:
        Path to the generated Pine Script file
    """
    # Load strategy
    if strategy_id.endswith(".json"):
        strategy_path = strategy_id
    else:
        strategy_path = f"strategies/{strategy_id}.json"

    strategy = Strategy.load(strategy_path)

    # Generate Pine Script
    pine_code = strategy.individual.to_pine_script(strategy.id)

    # Determine output path
    if output_path is None:
        output_path = f"strategies/{strategy.id}.pine"

    # Write to file
    Path(output_path).write_text(pine_code)

    return output_path
