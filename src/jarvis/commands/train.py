"""Train command for training a futures strategy for a single symbol."""

from dataclasses import dataclass
from datetime import datetime, timedelta
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
from jarvis.utils import datetime_to_timestamp, interval_to_timedelta, parse_period_to_days


@dataclass
class WindowResult:
    """Result of a single walk-forward window."""

    window_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_return_pct: float
    test_return_pct: float
    test_max_drawdown_pct: float
    test_trades: int


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

    # Need to fetch extra data for lookback period
    lookback_delta = interval_to_timedelta(interval) * lookback
    fetch_start = start_dt - lookback_delta

    all_klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=datetime_to_timestamp(fetch_start),
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
    interval_hours = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}.get(interval, 1)
    funding_interval_candles = max(1, int(FUNDING_INTERVAL_HOURS / interval_hours))

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


def _train_single_window(
    symbol: str,
    interval: str,
    train_start: datetime,
    train_end: datetime,
    population_size: int,
    generations: int,
    rules_per_individual: int,
    starting_margin: Decimal,
    commission_ratio: Decimal,
    investment_ratio: Decimal,
    leverage: int,
    funding_enabled: bool,
    price_hint: float | None,
    bar_manager: enlighten.Manager,
    window_label: str,
) -> Individual:
    """Train on a single window and return the best individual."""
    population = Population.create_random(population_size, rules_per_individual, price_hint=price_hint)

    gen_bar = bar_manager.counter(total=generations, desc=window_label, unit="gen", leave=False)

    best_fitness = float("-inf")
    best_individual = None
    preloaded_data = None
    buy_hold_return_pct = None

    for gen in range(generations):
        preloaded_data, buy_hold_return_pct = population.evaluate_fitness(
            symbol=symbol,
            interval=interval,
            start_dt=train_start,
            end_dt=train_end,
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

        if gen < generations - 1:
            population = population.evolve(price_hint=price_hint)

        gen_bar.update()

    gen_bar.close()

    return best_individual if best_individual else population.get_best()


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
    walk_forward: bool = True,
    train_period: str = "3M",
    test_period: str = "1M",
    step_period: str = "1M",
) -> tuple[Strategy, TestResult]:
    """Train a futures trading strategy using walk-forward validation.

    Walk-forward validation trains on rolling windows and tests on unseen data
    to prevent overfitting.

    Args:
        symbol: Trading pair symbol
        interval: Kline interval
        start_dt: Training start datetime (default: 1 year ago)
        end_dt: Training end datetime (default: now)
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
        walk_forward: Use walk-forward validation (default: True)
        train_period: Training period per window (e.g., "3M", "90d")
        test_period: Test period per window (e.g., "1M", "30d")
        step_period: Step size between windows (e.g., "1M", "30d")

    Returns:
        Tuple of (Strategy, TestResult) for the training run
    """
    # Parse periods to days
    train_days = parse_period_to_days(train_period)
    test_days = parse_period_to_days(test_period)
    step_days = parse_period_to_days(step_period)

    # Default dates: 1 year for walk-forward, 6 months for simple
    if end_dt is None:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt is None:
        start_dt = end_dt - relativedelta(years=1) if walk_forward else end_dt - relativedelta(months=6)

    total_days = (end_dt - start_dt).days

    logger.info("=" * 60)
    logger.info("JARVIS STRATEGY TRAINER")
    logger.info("=" * 60)
    logger.info("Symbol: %s | Interval: %s", symbol, interval)
    logger.info("Period: %s to %s (%d days)", start_dt.date(), end_dt.date(), total_days)
    logger.info("Population: %d | Generations: %d | Rules: %d", population_size, generations, rules_per_individual)
    logger.info("Leverage: %dx | Funding: %s", leverage, "ON" if funding_enabled else "OFF")

    if walk_forward:
        logger.info("-" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info(
            "Train: %s (%d days) | Test: %s (%d days) | Step: %s (%d days)",
            train_period,
            train_days,
            test_period,
            test_days,
            step_period,
            step_days,
        )

    logger.info("=" * 60)

    # Download data - include extra lookback period for indicators
    logger.info("[1/4] Downloading historical data...")
    lookback_bars = 200  # Required for indicator calculation
    lookback_delta = interval_to_timedelta(interval) * lookback_bars
    download_start = start_dt - lookback_delta
    logger.info("Including %d bars lookback (from %s)", lookback_bars, download_start.date())
    download([symbol], interval, download_start, end_dt)
    logger.info("Data download complete.")

    # Get price hint from last available data
    logger.info("[2/4] Getting price information...")
    price_hint = None
    try:
        client = get_binance_client(
            fake=True,
            extra_params={"assets": {"USDT": starting_margin}, "commission_ratio": commission_ratio},
        )
        # Use end_dt - 1 day to avoid missing file for end_dt itself
        price_check_dt = end_dt - timedelta(days=1)
        price_ts = datetime_to_timestamp(price_check_dt)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=1, endTime=price_ts)
        if klines:
            price_hint = float(klines[-1].close)
            logger.info("Current price: %.4f", price_hint)
    except Exception as e:
        logger.warning("Could not get price: %s", e)

    bar_manager = enlighten.get_manager()

    if not walk_forward:
        # Simple training (legacy mode)
        logger.info("[3/4] Training (simple mode - no walk-forward)...")
        logger.warning("Walk-forward disabled. Results may be overfit!")

        best_individual = _train_single_window(
            symbol=symbol,
            interval=interval,
            train_start=start_dt,
            train_end=end_dt,
            population_size=population_size,
            generations=generations,
            rules_per_individual=rules_per_individual,
            starting_margin=starting_margin,
            commission_ratio=commission_ratio,
            investment_ratio=investment_ratio,
            leverage=leverage,
            funding_enabled=funding_enabled,
            price_hint=price_hint,
            bar_manager=bar_manager,
            window_label=f"Training {symbol}",
        )

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

        window_results = []
        avg_test_return = metrics["return_pct"]
        avg_test_drawdown = metrics["max_drawdown_pct"]
        total_test_trades = metrics["total_trades"]
    else:
        # Walk-forward validation
        logger.info("[3/4] Running walk-forward validation...")

        # Calculate windows
        windows = []
        current_start = start_dt

        while True:
            train_end = current_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            if test_end > end_dt:
                break

            windows.append((current_start, train_end, test_start, test_end))
            current_start += timedelta(days=step_days)

        if not windows:
            raise ValueError(
                f"Not enough data for walk-forward. Need at least {train_days + test_days} days, "
                f"got {total_days} days. Try shorter periods or more data."
            )

        logger.info("Generated %d walk-forward windows:", len(windows))
        for i, (ts, te, vs, ve) in enumerate(windows):
            logger.info("  Window %d: Train %s-%s | Test %s-%s", i + 1, ts.date(), te.date(), vs.date(), ve.date())

        window_bar = bar_manager.counter(total=len(windows), desc="Walk-Forward", unit="win")
        window_results: list[WindowResult] = []
        all_individuals: list[Individual] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            window_num = i + 1
            logger.info("-" * 40)
            logger.info("WINDOW %d/%d", window_num, len(windows))
            logger.info("Training: %s to %s", train_start.date(), train_end.date())

            # Train on this window
            best_individual = _train_single_window(
                symbol=symbol,
                interval=interval,
                train_start=train_start,
                train_end=train_end,
                population_size=population_size,
                generations=generations,
                rules_per_individual=rules_per_individual,
                starting_margin=starting_margin,
                commission_ratio=commission_ratio,
                investment_ratio=investment_ratio,
                leverage=leverage,
                funding_enabled=funding_enabled,
                price_hint=price_hint,
                bar_manager=bar_manager,
                window_label=f"W{window_num} Train",
            )

            # Test on unseen data
            logger.info("Testing: %s to %s", test_start.date(), test_end.date())

            train_metrics = run_backtest(
                best_individual,
                symbol,
                interval,
                train_start,
                train_end,
                starting_margin,
                commission_ratio,
                investment_ratio,
                leverage,
                funding_enabled,
            )

            test_metrics = run_backtest(
                best_individual,
                symbol,
                interval,
                test_start,
                test_end,
                starting_margin,
                commission_ratio,
                investment_ratio,
                leverage,
                funding_enabled,
            )

            result = WindowResult(
                window_num=window_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_return_pct=train_metrics["return_pct"],
                test_return_pct=test_metrics["return_pct"],
                test_max_drawdown_pct=test_metrics["max_drawdown_pct"],
                test_trades=test_metrics["total_trades"],
            )
            window_results.append(result)
            all_individuals.append(best_individual)

            logger.info(
                "Results: Train %.2f%% | Test %.2f%% | Drawdown %.2f%% | Trades %d",
                result.train_return_pct,
                result.test_return_pct,
                result.test_max_drawdown_pct,
                result.test_trades,
            )

            window_bar.update()

        window_bar.close()

        # Select best individual based on test performance
        best_idx = max(range(len(window_results)), key=lambda i: window_results[i].test_return_pct)
        best_individual = all_individuals[best_idx]

        # Calculate aggregate metrics
        avg_test_return = sum(w.test_return_pct for w in window_results) / len(window_results)
        avg_test_drawdown = sum(w.test_max_drawdown_pct for w in window_results) / len(window_results)
        total_test_trades = sum(w.test_trades for w in window_results)

        # Log summary
        logger.info("=" * 60)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("=" * 60)
        logger.info("Window Results:")
        for w in window_results:
            status = "+" if w.test_return_pct > 0 else "-"
            logger.info(
                "  W%d: Train %+.2f%% | Test %+.2f%% %s", w.window_num, w.train_return_pct, w.test_return_pct, status
            )

        positive_windows = sum(1 for w in window_results if w.test_return_pct > 0)
        logger.info("-" * 40)
        logger.info(
            "Positive windows: %d/%d (%.0f%%)",
            positive_windows,
            len(window_results),
            100 * positive_windows / len(window_results),
        )
        logger.info("Average test return: %.2f%%", avg_test_return)
        logger.info("Average max drawdown: %.2f%%", avg_test_drawdown)
        logger.info("Best window: #%d (test return %.2f%%)", best_idx + 1, window_results[best_idx].test_return_pct)

    bar_manager.stop()

    # Create and save strategy
    logger.info("[4/4] Saving strategy...")

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

    result = TestResult(
        strategy_id=strategy.id,
        symbol=symbol,
        interval=interval,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        result_type="walk_forward" if walk_forward else "training",
        return_pct=avg_test_return,
        max_drawdown_pct=avg_test_drawdown,
        total_trades=total_test_trades,
        final_equity=float(starting_margin) * (1 + avg_test_return / 100),
        peak_equity=float(starting_margin) * (1 + avg_test_return / 100),
    )

    strategy_path = strategy.save(strategies_dir)
    result_path = result.save(results_dir)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Strategy ID: %s", strategy.id)
    logger.info("Strategy saved: %s", strategy_path)
    logger.info("Results saved: %s", result_path)
    logger.info("-" * 40)
    logger.info("Average Test Return: %.2f%%", avg_test_return)
    logger.info("Average Max Drawdown: %.2f%%", avg_test_drawdown)
    logger.info("Total Test Trades: %d", total_test_trades)
    logger.info("Rules: %d", len(best_individual.rules))
    logger.info("=" * 60)

    return strategy, result
