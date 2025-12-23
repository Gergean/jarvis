"""Test command for testing a trained futures strategy on out-of-sample data."""

from datetime import datetime
from decimal import Decimal

import enlighten
import numpy as np
from dateutil.relativedelta import relativedelta

from jarvis.client import get_binance_client
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.strategy import Strategy, TestResult
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


def test(
    strategy_id: str,
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    starting_margin: Decimal = Decimal("100"),
    commission_ratio: Decimal = FUTURES_TAKER_FEE,
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    funding_enabled: bool = True,
    strategies_dir: str = "strategies",
    results_dir: str = "results",
) -> TestResult:
    """Test a trained futures strategy on a specific time period.

    Args:
        strategy_id: ID of the strategy to test (e.g., "BTCUSDT_abc123")
        interval: Kline interval for testing
        start_dt: Test start date (default: 3 months ago)
        end_dt: Test end date (default: now)
        starting_margin: Starting USDT margin
        commission_ratio: Trading fee ratio
        investment_ratio: Fraction of margin to use per trade
        leverage: Futures leverage (1-10)
        funding_enabled: Whether to simulate funding fees
        strategies_dir: Directory containing strategy files
        results_dir: Directory to save results

    Returns:
        TestResult with performance metrics
    """
    # Load strategy
    strategy = Strategy.load_by_id(strategy_id, strategies_dir)
    symbol = strategy.symbol
    individual = strategy.individual

    logger.info("Testing futures strategy: %s", strategy_id)
    logger.info("Symbol: %s", symbol)
    logger.info("Leverage: %dx, Funding: %s", leverage, "enabled" if funding_enabled else "disabled")
    logger.info("Rules: %d", len(individual.rules))
    for rule in individual.rules:
        logger.info("  %s", rule)

    # Default dates: last 3 months (different from training period)
    if end_dt is None:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt is None:
        start_dt = end_dt - relativedelta(months=3)

    logger.info("Test period: %s to %s", start_dt.date(), end_dt.date())
    logger.info("Interval: %s", interval)

    client = get_binance_client(
        fake=True,
        extra_params={"assets": {"USDT": starting_margin}, "commission_ratio": commission_ratio},
    )

    # Load all klines at once
    lookback = 200
    all_klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=datetime_to_timestamp(start_dt),
        endTime=datetime_to_timestamp(end_dt),
        limit=50000,
    )

    if not all_klines:
        logger.warning("No klines found for %s", symbol)
        return TestResult(
            strategy_id=strategy_id,
            symbol=symbol,
            interval=interval,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            result_type="test",
            return_pct=0.0,
            max_drawdown_pct=0.0,
            total_trades=0,
            final_equity=float(starting_margin),
            peak_equity=float(starting_margin),
        )

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

    bar_manager = enlighten.get_manager()
    bar = bar_manager.counter(total=n - lookback, desc=f"Testing {symbol}", unit="bars")

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
    price = Decimal("0")

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
                bar.update()
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

        bar.update()

    bar.close()
    bar_manager.stop()

    # Final equity calculation
    final_equity = margin_balance
    if position_side != PositionSide.NONE and price > 0:
        if position_side == PositionSide.LONG:
            unrealized_pnl = position_quantity * (price - position_entry_price)
        else:
            unrealized_pnl = position_quantity * (position_entry_price - price)
        final_equity += position_margin + unrealized_pnl

    return_pct = float((final_equity - starting_margin) / starting_margin * 100)

    # Create and save result
    result = TestResult(
        strategy_id=strategy_id,
        symbol=symbol,
        interval=interval,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        result_type="test",
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        final_equity=float(final_equity),
        peak_equity=float(peak_equity),
    )

    result_path = result.save(results_dir)

    # Log results
    logger.info("=== Test Complete ===")
    logger.info("Strategy: %s", strategy_id)
    logger.info("Period: %s to %s (%s)", start_dt.date(), end_dt.date(), interval)
    logger.info("Return: %.2f%%", return_pct)
    logger.info("Max Drawdown: %.2f%%", max_drawdown_pct)
    logger.info("Total Trades: %d", total_trades)
    logger.info("Funding Paid: %.2f USDT", float(total_funding_paid))
    logger.info("Liquidations: %d", liquidation_count)
    logger.info("Final Equity: %.2f USDT", float(final_equity))
    logger.info("Result saved to %s", result_path)

    return result
