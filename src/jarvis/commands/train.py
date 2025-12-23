"""Train command for training a futures strategy for a single symbol."""

from datetime import datetime
from decimal import Decimal

import enlighten
import numpy as np
from dateutil.relativedelta import relativedelta

from jarvis.client import get_binance_client
from jarvis.commands.download import download
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.individual import Individual
from jarvis.genetics.population import Population
from jarvis.genetics.strategy import Strategy, TestResult, TrainingConfig
from jarvis.logging import logger
from jarvis.models import (
    DEFAULT_LEVERAGE,
    FUNDING_FEE_RATE,
    FUNDING_INTERVAL_HOURS,
    FUTURES_TAKER_FEE,
    ActionType,
    PositionSide,
)
from jarvis.utils import datetime_to_timestamp


def run_backtest(
    individual: Individual,
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    starting_margin: Decimal = Decimal("100"),
    commission_ratio: Decimal = FUTURES_TAKER_FEE,
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    funding_enabled: bool = True,
) -> dict:
    """Run futures backtest for an individual and return metrics.

    Args:
        individual: The trading strategy individual
        symbol: Trading pair symbol
        interval: Kline interval
        start_dt: Backtest start datetime
        end_dt: Backtest end datetime
        starting_margin: Initial USDT margin
        commission_ratio: Trading fee ratio
        investment_ratio: Fraction of margin to use per trade
        leverage: Futures leverage (1-10)
        funding_enabled: Whether to simulate funding fees

    Returns:
        Dictionary with performance metrics
    """
    client = get_binance_client(
        fake=True,
        extra_params={"assets": {"USDT": starting_margin}, "commission_ratio": commission_ratio},
    )

    lookback = 200
    all_klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=datetime_to_timestamp(start_dt),
        endTime=datetime_to_timestamp(end_dt),
        limit=50000,
    )

    if not all_klines:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "final_equity": float(starting_margin),
            "peak_equity": float(starting_margin),
            "total_funding_paid": 0.0,
            "liquidation_count": 0,
        }

    # Convert to numpy arrays
    n = len(all_klines)
    open_arr = np.zeros(n, dtype=np.float64)
    high_arr = np.zeros(n, dtype=np.float64)
    low_arr = np.zeros(n, dtype=np.float64)
    close_arr = np.zeros(n, dtype=np.float64)
    volume_arr = np.zeros(n, dtype=np.float64)

    for i, k in enumerate(all_klines):
        open_arr[i] = float(k.open)
        high_arr[i] = float(k.high)
        low_arr[i] = float(k.low)
        close_arr[i] = float(k.close)
        volume_arr[i] = float(k.volume)

    # Calculate funding interval in candles
    interval_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}.get(interval, 1)
    funding_interval_candles = int(FUNDING_INTERVAL_HOURS / interval_hours)

    # Initialize state
    margin_balance = starting_margin
    position_side = PositionSide.NONE
    position_entry_price = Decimal(0)
    position_quantity = Decimal(0)
    position_margin = Decimal(0)

    peak_equity = starting_margin
    max_drawdown_pct = 0.0
    total_trades = 0
    total_funding_paid = Decimal(0)
    liquidation_count = 0
    last_funding_candle = 0
    price = Decimal(0)

    for i in range(lookback, n):
        start_idx = i - lookback + 1
        end_idx = i + 1
        ohlcv = OHLCV(
            open=open_arr[start_idx:end_idx],
            high=high_arr[start_idx:end_idx],
            low=low_arr[start_idx:end_idx],
            close=close_arr[start_idx:end_idx],
            volume=volume_arr[start_idx:end_idx],
        )
        price = Decimal(str(close_arr[i]))

        # Check liquidation if in position with leverage > 1
        if position_side != PositionSide.NONE and leverage > 1:
            liquidated = False
            if position_side == PositionSide.LONG:
                liq_price = position_entry_price * (1 - Decimal(1) / Decimal(leverage))
                if price <= liq_price:
                    liquidated = True
            else:  # SHORT
                liq_price = position_entry_price * (1 + Decimal(1) / Decimal(leverage))
                if price >= liq_price:
                    liquidated = True

            if liquidated:
                margin_balance -= position_margin
                position_side = PositionSide.NONE
                position_quantity = Decimal(0)
                position_margin = Decimal(0)
                liquidation_count += 1
                continue

        # Apply funding fee
        if funding_enabled and position_side != PositionSide.NONE:
            candles_since_funding = i - last_funding_candle
            if candles_since_funding >= funding_interval_candles:
                num_funding_periods = candles_since_funding // funding_interval_candles
                notional = position_quantity * price
                funding_payment = notional * FUNDING_FEE_RATE * num_funding_periods
                if position_side == PositionSide.LONG:
                    margin_balance -= funding_payment
                    total_funding_paid += funding_payment
                else:
                    margin_balance += funding_payment
                    total_funding_paid -= funding_payment
                last_funding_candle = i

        # Calculate current equity for drawdown tracking
        equity = margin_balance
        if position_side != PositionSide.NONE:
            if position_side == PositionSide.LONG:
                unrealized_pnl = position_quantity * (price - position_entry_price)
            else:
                unrealized_pnl = position_quantity * (position_entry_price - price)
            equity += position_margin + unrealized_pnl

        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            drawdown_pct = float((peak_equity - equity) / peak_equity * 100)
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

        # Get signal with position awareness
        signal = individual.get_signal(ohlcv, position_side)

        if signal == ActionType.LONG and position_side == PositionSide.NONE:
            margin_to_use = margin_balance * investment_ratio
            if margin_to_use > 0 and price > 0:
                position_size = (margin_to_use * leverage) / price
                fee = position_size * price * commission_ratio
                margin_balance -= fee
                position_side = PositionSide.LONG
                position_entry_price = price
                position_quantity = position_size
                position_margin = margin_to_use
                margin_balance -= margin_to_use
                total_trades += 1

        elif signal == ActionType.SHORT and position_side == PositionSide.NONE:
            margin_to_use = margin_balance * investment_ratio
            if margin_to_use > 0 and price > 0:
                position_size = (margin_to_use * leverage) / price
                fee = position_size * price * commission_ratio
                margin_balance -= fee
                position_side = PositionSide.SHORT
                position_entry_price = price
                position_quantity = position_size
                position_margin = margin_to_use
                margin_balance -= margin_to_use
                total_trades += 1

        elif signal == ActionType.CLOSE and position_side != PositionSide.NONE:
            if position_side == PositionSide.LONG:
                pnl = position_quantity * (price - position_entry_price)
            else:
                pnl = position_quantity * (position_entry_price - price)

            fee = position_quantity * price * commission_ratio
            margin_balance += position_margin + pnl - fee
            position_side = PositionSide.NONE
            position_quantity = Decimal(0)
            position_margin = Decimal(0)
            total_trades += 1

    # Final equity calculation
    final_equity = margin_balance
    if position_side != PositionSide.NONE and price > 0:
        if position_side == PositionSide.LONG:
            unrealized_pnl = position_quantity * (price - position_entry_price)
        else:
            unrealized_pnl = position_quantity * (position_entry_price - price)
        final_equity += position_margin + unrealized_pnl

    return_pct = float((final_equity - starting_margin) / starting_margin * 100)

    return {
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "final_equity": float(final_equity),
        "peak_equity": float(peak_equity),
        "total_funding_paid": float(total_funding_paid),
        "liquidation_count": liquidation_count,
    }


def train(
    symbol: str,
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    population_size: int = 100,
    generations: int = 30,
    rules_per_individual: int = 8,
    starting_margin: Decimal = Decimal("100"),
    commission_ratio: Decimal = FUTURES_TAKER_FEE,
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    funding_enabled: bool = True,
    strategies_dir: str = "strategies",
    results_dir: str = "results",
) -> tuple[Strategy, TestResult]:
    """Train a futures trading strategy for a single symbol.

    Args:
        symbol: Trading pair symbol
        interval: Kline interval
        start_dt: Training start datetime
        end_dt: Training end datetime
        population_size: Number of individuals in population
        generations: Number of generations to evolve
        rules_per_individual: Number of rules per individual
        starting_margin: Initial USDT margin
        commission_ratio: Trading fee ratio
        investment_ratio: Fraction of margin to use per trade
        leverage: Futures leverage (1-10)
        funding_enabled: Whether to simulate funding fees
        strategies_dir: Directory to save strategies
        results_dir: Directory to save results

    Returns:
        Tuple of (Strategy, TestResult) for the training run
    """
    # Default dates: last 6 months
    if end_dt is None:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt is None:
        start_dt = end_dt - relativedelta(months=6)

    logger.info("Training futures strategy for %s", symbol)
    logger.info("Period: %s to %s", start_dt.date(), end_dt.date())
    logger.info("Population: %d, Generations: %d, Rules: %d", population_size, generations, rules_per_individual)
    logger.info("Leverage: %dx, Funding: %s", leverage, "enabled" if funding_enabled else "disabled")

    # Download data if not available
    logger.info("Checking/downloading data...")
    download([symbol], interval, start_dt, end_dt)

    bar_manager = enlighten.get_manager()
    gen_bar = bar_manager.counter(total=generations, desc=f"Training {symbol}", unit="gen")

    # Get approximate current price for proper target scaling
    price_hint = None
    try:
        client = get_binance_client(
            fake=True,
            extra_params={"assets": {"USDT": starting_margin}, "commission_ratio": commission_ratio},
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

    best_fitness = float("-inf")
    best_individual = None
    preloaded_data = None
    buy_hold_return_pct = None

    for gen in range(generations):
        # Evaluate fitness with futures logic
        preloaded_data, buy_hold_return_pct = population.evaluate_fitness(
            symbol=symbol,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
            starting_margin=starting_margin,
            commission_ratio=commission_ratio,
            investment_ratio=investment_ratio,
            leverage=leverage,
            funding_enabled=funding_enabled,
            preloaded_data=preloaded_data,
            buy_hold_return_pct=buy_hold_return_pct,
        )

        current_best = population.get_best()
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_individual = current_best

        # Get top 3 elites
        sorted_pop = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:3]

        print(f"\nGen {gen}: best={current_best.fitness:.2f}, avg={population.get_average_fitness():.2f}")
        print("  Elites:")
        for rank, elite in enumerate(elites, 1):
            print(f"    #{rank} fitness={elite.fitness:.2f} rules={len(elite.rules)}")

        # Evolve (except last generation)
        if gen < generations - 1:
            population = population.evolve(price_hint=price_hint)

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
        starting_margin,
        commission_ratio,
        investment_ratio,
        leverage,
        funding_enabled,
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
    logger.info("Funding Paid: %.2f USDT", metrics["total_funding_paid"])
    logger.info("Liquidations: %d", metrics["liquidation_count"])
    logger.info("Rules: %d", len(best_individual.rules))

    return strategy, result
