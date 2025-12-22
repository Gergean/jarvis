"""Test command for testing a trained strategy on out-of-sample data."""

from datetime import datetime
from decimal import Decimal

import enlighten
import pandas as pd

from jarvis.client import get_binance_client
from jarvis.ga.strategy import Strategy, TestResult
from jarvis.logging import logger
from jarvis.models import ActionType
from jarvis.utils import datetime_to_timestamp, dt_range, interval_to_timedelta


def test(
    strategy_id: str,
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    starting_amount: Decimal = Decimal("100"),
    commission_ratio: Decimal = Decimal("0.001"),
    investment_ratio: Decimal = Decimal("0.2"),
    strategies_dir: str = "strategies",
    results_dir: str = "results",
) -> TestResult:
    """Test a trained strategy on a specific time period.

    Args:
        strategy_id: ID of the strategy to test (e.g., "BTCUSDT_abc123")
        interval: Kline interval for testing
        start_dt: Test start date (default: 3 months ago)
        end_dt: Test end date (default: now)
        starting_amount: Starting USDT amount
        commission_ratio: Trading fee ratio
        investment_ratio: Portion of balance to trade
        strategies_dir: Directory containing strategy files
        results_dir: Directory to save results

    Returns:
        TestResult with performance metrics
    """
    # Load strategy
    strategy = Strategy.load_by_id(strategy_id, strategies_dir)
    symbol = strategy.symbol
    individual = strategy.individual

    logger.info("Testing strategy: %s", strategy_id)
    logger.info("Symbol: %s", symbol)
    logger.info("Rules: %d", len(individual.rules))
    for rule in individual.rules:
        logger.info("  %s", rule)

    # Default dates: last 3 months (different from training period)
    if end_dt is None:
        end_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_dt is None:
        start_dt = end_dt.replace(month=end_dt.month - 3) if end_dt.month > 3 else end_dt.replace(
            year=end_dt.year - 1, month=end_dt.month + 9
        )

    logger.info("Test period: %s to %s", start_dt.date(), end_dt.date())
    logger.info("Interval: %s", interval)

    base_asset = "USDT"
    trade_asset = symbol[:-4] if symbol.endswith("USDT") else symbol[:-3]

    interval_td = interval_to_timedelta(interval)
    all_dts = list(dt_range(start_dt, end_dt, interval_td))

    bar_manager = enlighten.get_manager()
    bar = bar_manager.counter(total=len(all_dts), desc=f"Testing {symbol}", unit="bars")

    client = get_binance_client(
        fake=True,
        extra_params={"assets": {base_asset: starting_amount}, "commission_ratio": commission_ratio},
    )

    assets: dict[str, Decimal] = {base_asset: starting_amount}
    peak_equity = starting_amount
    max_drawdown_pct = 0.0
    total_trades = 0
    price = Decimal("0")

    for dt in all_dts:
        end_ts = datetime_to_timestamp(dt)
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=100, endTime=end_ts)
            if not klines or len(klines) < 50:
                bar.update()
                continue
        except Exception:
            bar.update()
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
                logger.debug("%s BUY %.4f %s @ %.2f", dt, buy_qty, trade_asset, price)

        elif signal == ActionType.SELL:
            sell_qty = assets.get(trade_asset, Decimal("0"))
            if sell_qty > 0 and price > 0:
                proceeds = sell_qty * price
                after_fee = proceeds * (1 - commission_ratio)
                assets[trade_asset] = Decimal("0")
                assets[base_asset] = assets.get(base_asset, Decimal("0")) + after_fee
                total_trades += 1
                logger.debug("%s SELL %.4f %s @ %.2f", dt, sell_qty, trade_asset, price)

        bar.update()

    bar.close()
    bar_manager.stop()

    # Final equity
    final_equity = assets.get(base_asset, Decimal("0"))
    if assets.get(trade_asset, Decimal("0")) > 0:
        final_equity += assets[trade_asset] * price

    return_pct = float((final_equity - starting_amount) / starting_amount * 100)

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
    logger.info("Final Equity: %.2f USDT", float(final_equity))
    logger.info("Result saved to %s", result_path)

    return result
