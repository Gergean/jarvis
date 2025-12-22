"""Train command for training a strategy for a single symbol."""

from datetime import datetime
from decimal import Decimal

import enlighten
import pandas as pd

from jarvis.client import get_binance_client
from jarvis.ga.individual import Individual
from jarvis.ga.population import Population
from jarvis.ga.strategy import Strategy, TestResult, TrainingConfig
from jarvis.logging import logger
from jarvis.models import ActionType
from jarvis.utils import datetime_to_timestamp, dt_range, interval_to_timedelta


def run_backtest(
    individual: Individual,
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    starting_amount: Decimal = Decimal("100"),
    commission_ratio: Decimal = Decimal("0.001"),
    investment_ratio: Decimal = Decimal("0.2"),
) -> dict:
    """Run backtest for an individual and return metrics."""
    base_asset = "USDT"
    trade_asset = symbol[:-4] if symbol.endswith("USDT") else symbol[:-3]

    interval_td = interval_to_timedelta(interval)
    all_dts = list(dt_range(start_dt, end_dt, interval_td))

    client = get_binance_client(
        fake=True,
        extra_params={"assets": {base_asset: starting_amount}, "commission_ratio": commission_ratio},
    )

    assets: dict[str, Decimal] = {base_asset: starting_amount}
    peak_equity = starting_amount
    max_drawdown_pct = 0.0
    total_trades = 0

    for dt in all_dts:
        end_ts = datetime_to_timestamp(dt)
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=100, endTime=end_ts)
            if not klines or len(klines) < 50:
                continue
        except Exception:
            continue

        df = pd.DataFrame([k.model_dump() for k in klines])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        price = Decimal(str(df["close"].iloc[-1]))

        # Calculate equity
        equity = assets.get(base_asset, Decimal("0"))
        if assets.get(trade_asset, Decimal("0")) > 0:
            equity += assets[trade_asset] * price

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            drawdown_pct = float((peak_equity - equity) / peak_equity * 100)
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

        # Execute signal
        signal = individual.get_signal(df)

        if signal == ActionType.BUY:
            quote_balance = assets.get(base_asset, Decimal("0"))
            spend_amount = quote_balance * investment_ratio
            if spend_amount > 0 and price > 0:
                after_fee = spend_amount * (1 - commission_ratio)
                buy_qty = after_fee / price
                assets[base_asset] = quote_balance - spend_amount
                assets[trade_asset] = assets.get(trade_asset, Decimal("0")) + buy_qty
                total_trades += 1

        elif signal == ActionType.SELL:
            sell_qty = assets.get(trade_asset, Decimal("0"))
            if sell_qty > 0 and price > 0:
                proceeds = sell_qty * price
                after_fee = proceeds * (1 - commission_ratio)
                assets[trade_asset] = Decimal("0")
                assets[base_asset] = assets.get(base_asset, Decimal("0")) + after_fee
                total_trades += 1

    # Final equity
    final_equity = assets.get(base_asset, Decimal("0"))
    if assets.get(trade_asset, Decimal("0")) > 0:
        final_equity += assets[trade_asset] * price

    return_pct = float((final_equity - starting_amount) / starting_amount * 100)

    return {
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "final_equity": float(final_equity),
        "peak_equity": float(peak_equity),
    }


def train(
    symbol: str,
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    population_size: int = 100,
    generations: int = 30,
    rules_per_individual: int = 8,
    starting_amount: Decimal = Decimal("100"),
    commission_ratio: Decimal = Decimal("0.001"),
    investment_ratio: Decimal = Decimal("0.2"),
    strategies_dir: str = "strategies",
    results_dir: str = "results",
) -> tuple[Strategy, TestResult]:
    """Train a trading strategy for a single symbol.

    Returns:
        Tuple of (Strategy, TestResult) for the training run
    """
    # Default dates: last 6 months
    if end_dt is None:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt is None:
        start_dt = end_dt.replace(month=end_dt.month - 6) if end_dt.month > 6 else end_dt.replace(
            year=end_dt.year - 1, month=end_dt.month + 6
        )

    logger.info("Training strategy for %s", symbol)
    logger.info("Period: %s to %s", start_dt.date(), end_dt.date())
    logger.info("Population: %d, Generations: %d, Rules: %d", population_size, generations, rules_per_individual)

    bar_manager = enlighten.get_manager()
    gen_bar = bar_manager.counter(total=generations, desc=f"Training {symbol}", unit="gen")

    # Get approximate current price for proper target scaling
    price_hint = None
    try:
        client = get_binance_client(
            fake=True,
            extra_params={"assets": {"USDT": starting_amount}, "commission_ratio": commission_ratio},
        )
        end_ts = datetime_to_timestamp(end_dt)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=1, endTime=end_ts)
        if klines:
            price_hint = float(klines[-1].close)
            logger.info("Price hint for %s: %.2f", symbol, price_hint)
    except Exception as e:
        logger.warning("Could not get price hint: %s", e)

    # Create initial population
    population = Population.create_random(population_size, rules_per_individual, price_hint=price_hint)

    best_fitness = 0.0
    best_individual = None

    for gen in range(generations):
        # Evaluate fitness
        population.evaluate_fitness(
            symbol=symbol,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
            starting_amount=starting_amount,
            commission_ratio=commission_ratio,
            investment_ratio=investment_ratio,
        )

        current_best = population.get_best()
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_individual = current_best

        # Get top 3 elites
        sorted_pop = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:3]

        logger.info(
            "Gen %d: best=%.2f, avg=%.2f",
            gen,
            current_best.fitness,
            population.get_average_fitness(),
        )
        logger.info("  Elites:")
        for rank, elite in enumerate(elites, 1):
            logger.info("    #%d fitness=%.2f rules=%d", rank, elite.fitness, len(elite.rules))
            for rule in elite.rules:
                logger.info("       %s", rule)

        # Evolve (except last generation)
        if gen < generations - 1:
            population = population.evolve()

        gen_bar.update()

    gen_bar.close()
    bar_manager.stop()

    # Use the best individual found across all generations
    if best_individual is None:
        best_individual = population.get_best()

    # Calculate detailed metrics
    logger.info("Calculating performance metrics...")
    metrics = run_backtest(
        best_individual,
        symbol,
        interval,
        start_dt,
        end_dt,
        starting_amount,
        commission_ratio,
        investment_ratio,
    )

    # Create strategy
    training_config = TrainingConfig(
        interval=interval,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        generations=generations,
        population_size=population_size,
        rules_per_individual=rules_per_individual,
    )

    strategy = Strategy.create(
        symbol=symbol,
        individual=best_individual,
        training=training_config,
    )

    # Create test result for training
    result = TestResult(
        strategy_id=strategy.id,
        symbol=symbol,
        interval=interval,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        result_type="training",
        return_pct=metrics["return_pct"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
        total_trades=metrics["total_trades"],
        final_equity=metrics["final_equity"],
        peak_equity=metrics["peak_equity"],
    )

    # Save both
    strategy_path = strategy.save(strategies_dir)
    result_path = result.save(results_dir)

    logger.info("Strategy saved to %s", strategy_path)
    logger.info("Result saved to %s", result_path)

    # Log results
    logger.info("=== Training Complete ===")
    logger.info("Strategy ID: %s", strategy.id)
    logger.info("Return: %.2f%%", metrics["return_pct"])
    logger.info("Max Drawdown: %.2f%%", metrics["max_drawdown_pct"])
    logger.info("Total Trades: %d", metrics["total_trades"])
    logger.info("Rules: %d", len(best_individual.rules))

    return strategy, result
