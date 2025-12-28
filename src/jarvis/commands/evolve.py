"""Evolve command - Evolve strategy after trade execution."""

import shutil
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from jarvis.commands.train import (
    _evaluate_individual_on_windows,
    _preload_window_data,
)
from jarvis.genetics.population import Population
from jarvis.genetics.strategy import Strategy, TrainingConfig
from jarvis.logging import logger
from jarvis.models import DEFAULT_LEVERAGE
from jarvis.settings import FUTURES_TAKER_FEE, notify


def evolve_strategy(
    strategy_id: str,
    interval: str = "4h",
    lookback_days: int = 60,
    generations: int = 30,
    population_size: int = 50,
    leverage: int = DEFAULT_LEVERAGE,
    strategies_dir: str = "strategies",
    history_dir: str = "strategies/history",
) -> Strategy | None:
    """Evolve a strategy using recent data.

    Uses the current strategy as seed and evolves for N generations.
    If new strategy is better, backs up old one and saves new one.

    Args:
        strategy_id: Current strategy ID (e.g., SOLUSDT_ef613e00)
        interval: Trading interval
        lookback_days: Days of recent data to use for evolution
        generations: Number of generations to evolve
        population_size: Population size for evolution
        leverage: Trading leverage
        strategies_dir: Directory containing strategies
        history_dir: Directory for backups

    Returns:
        New strategy if improved, None otherwise
    """
    # Load current strategy
    try:
        current_strategy = Strategy.load_by_id(strategy_id, strategies_dir)
        current_individual = current_strategy.individual
        symbol = current_strategy.symbol
        logger.info("Loaded strategy %s (fitness: %.2f)", strategy_id, current_individual.fitness)
    except Exception as e:
        logger.error("Failed to load strategy %s: %s", strategy_id, e)
        return None

    # Calculate date range
    end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=lookback_days)

    logger.info("Evolving %s with last %d days of data", symbol, lookback_days)
    logger.info("Period: %s to %s", start_dt.date(), end_dt.date())

    # Create single window for recent data
    windows = [(start_dt, end_dt)]

    # Preload data
    try:
        window_data = _preload_window_data(
            symbol=symbol,
            interval=interval,
            windows=windows,
            commission_ratio=FUTURES_TAKER_FEE,
            starting_margin=Decimal("100"),
        )
        if not window_data or not window_data[0].ohlcv_data:
            logger.error("No data available for evolution")
            return None
        logger.info("Loaded %d data points", len(window_data[0].ohlcv_data))
    except Exception as e:
        logger.error("Failed to preload data: %s", e)
        return None

    # Evaluate current strategy on recent data
    current_fitness, _ = _evaluate_individual_on_windows(
        individual=current_individual,
        windows=window_data,
        starting_margin=Decimal("100"),
        commission_ratio=FUTURES_TAKER_FEE,
        investment_ratio=Decimal("1.0"),
        leverage=leverage,
        funding_enabled=True,
    )
    logger.info("Current strategy fitness on recent data: %.2f", current_fitness)

    # Get price hint
    price_hint = None
    try:
        if window_data and window_data[0].ohlcv_data:
            price_hint = float(window_data[0].ohlcv_data[-1][1])
    except Exception:
        pass

    # Create population with current strategy as seed
    rules_per_individual = len(current_individual.rules)
    population = Population.create_random(
        population_size,
        rules_per_individual,
        price_hint=price_hint,
        seed_individual=current_individual,
        interval=interval,
    )

    # Evolution loop
    best_individual = current_individual
    best_fitness = current_fitness

    for gen in range(generations):
        gen_best_fitness = float("-inf")
        gen_best_individual = None

        for individual in population.individuals:
            fitness, _ = _evaluate_individual_on_windows(
                individual=individual,
                windows=window_data,
                starting_margin=Decimal("100"),
                commission_ratio=FUTURES_TAKER_FEE,
                investment_ratio=Decimal("1.0"),
                leverage=leverage,
                funding_enabled=True,
            )
            individual.fitness = fitness

            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_individual = individual

        # Track overall best
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual

        # Evolve population
        population = population.evolve()

        if (gen + 1) % 10 == 0:
            logger.info("Gen %d/%d: Best fitness=%.2f", gen + 1, generations, best_fitness)

    # Check if we found a better strategy
    improvement = best_fitness - current_fitness
    logger.info("Evolution complete. Improvement: %.2f", improvement)

    if improvement <= 0:
        logger.info("No improvement found. Keeping current strategy.")
        notify(f"🧬 {symbol} Evrim tamamlandı\nİyileşme yok, mevcut strateji korunuyor.")
        return None

    # Create new strategy with same ID (overwrite)
    logger.info("Found better strategy! Fitness: %.2f -> %.2f (+%.2f)", current_fitness, best_fitness, improvement)

    # Backup old strategy
    Path(history_dir).mkdir(parents=True, exist_ok=True)
    old_path = Path(strategies_dir) / f"{strategy_id}.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(history_dir) / f"{strategy_id}_{timestamp}.json"

    if old_path.exists():
        shutil.copy(old_path, backup_path)
        logger.info("Backed up old strategy to %s", backup_path)

    # Create new strategy with same ID
    new_strategy = Strategy(
        id=strategy_id,
        symbol=symbol,
        individual=best_individual,
        training=TrainingConfig(
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            interval=interval,
            generations=generations,
            population_size=population_size,
            rules_per_individual=rules_per_individual,
        ),
    )

    # Save new strategy
    new_strategy.save(strategies_dir)
    logger.info("Saved evolved strategy: %s", strategy_id)

    notify(
        f"🧬 {symbol} Evrim başarılı!\n"
        f"Fitness: {current_fitness:.2f} → {best_fitness:.2f} (+{improvement:.2f})\n"
        f"Yedek: {backup_path.name}"
    )

    return new_strategy
