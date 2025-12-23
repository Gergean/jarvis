"""Trade command for the Jarvis futures trading system."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

from jarvis.client import get_binance_client
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.strategy import Strategy
from jarvis.logging import logger
from jarvis.models import DEFAULT_LEVERAGE, ActionType, PositionSide
from jarvis.settings import notify
from jarvis.utils import datetime_to_timestamp


def get_signal_from_strategy(
    client: Any,
    strategy: Strategy,
    interval: str,
    dt: datetime,
    current_side: PositionSide = PositionSide.NONE,
) -> ActionType:
    """Get trading signal from a GA strategy.

    Args:
        client: Binance client (real or fake)
        strategy: GA strategy to use
        interval: Kline interval
        dt: Current datetime
        current_side: Current position direction

    Returns:
        ActionType signal (LONG, SHORT, CLOSE, STAY, or ERR)
    """
    symbol = strategy.symbol
    individual = strategy.individual

    # Get klines for signal calculation
    end_ts = datetime_to_timestamp(dt)
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=200, endTime=end_ts)
        if not klines or len(klines) < 50:
            logger.warning("Not enough klines for %s", symbol)
            return ActionType.STAY
    except Exception as e:
        logger.error("Failed to get klines for %s: %s", symbol, e)
        return ActionType.ERR

    # Convert to numpy arrays for OHLCV
    n = len(klines)
    open_arr = np.zeros(n, dtype=np.float64)
    high_arr = np.zeros(n, dtype=np.float64)
    low_arr = np.zeros(n, dtype=np.float64)
    close_arr = np.zeros(n, dtype=np.float64)
    volume_arr = np.zeros(n, dtype=np.float64)

    for i, k in enumerate(klines):
        open_arr[i] = float(k.open)
        high_arr[i] = float(k.high)
        low_arr[i] = float(k.low)
        close_arr[i] = float(k.close)
        volume_arr[i] = float(k.volume)

    ohlcv = OHLCV(
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
    )

    # Get signal from individual with position awareness
    return individual.get_signal(ohlcv, current_side)


def trade_with_strategies(
    strategy_ids: list[str],
    interval: str = "1h",
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    strategies_dir: str = "strategies",
    dry_run: bool = False,
) -> None:
    """Execute live futures trading using GA strategies.

    Args:
        strategy_ids: List of strategy IDs to use (e.g., ["BTCUSDT_fe43f298"])
        interval: Kline interval for signal calculation
        investment_ratio: Portion of margin to trade per signal
        leverage: Futures leverage (1-10)
        strategies_dir: Directory containing strategy files
        dry_run: If True, only show signals without executing trades
    """
    # In dry-run mode, use fake client with CSV data
    if dry_run:
        client = get_binance_client(fake=True, extra_params={"assets": {"USDT": Decimal("10000")}})
        # Use yesterday's date for dry-run (historical data available)
        dt = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
        print("=== DRY RUN MODE (Futures) ===")
        print(f"Using historical data from: {dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"Leverage: {leverage}x")
    else:
        client = get_binance_client()
        dt = datetime.now(UTC).replace(tzinfo=None)

    # Load strategies
    strategies: list[Strategy] = []
    for strategy_id in strategy_ids:
        try:
            strategy = Strategy.load_by_id(strategy_id, strategies_dir)
            strategies.append(strategy)
            logger.info("Loaded strategy: %s (%d rules)", strategy.id, len(strategy.individual.rules))
        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_id, e)

    if not strategies:
        logger.error("No strategies loaded!")
        return

    # Track positions per symbol (in real mode, would come from API)
    positions: dict[str, PositionSide] = {}

    # In dry-run mode, skip balance check
    if dry_run:
        margin_balance = Decimal("10000")
    else:
        # For real trading, get futures account info
        # Note: This requires futures API - not implemented yet
        logger.warning("Real futures trading not yet implemented!")
        logger.warning("Use --dry-run mode to test signals")
        return

    for strategy in strategies:
        symbol = strategy.symbol
        current_side = positions.get(symbol, PositionSide.NONE)

        # Get signal from strategy with position awareness
        signal = get_signal_from_strategy(client, strategy, interval, dt, current_side)
        logger.info("[%s] Position: %s, Signal: %s", symbol, current_side.value, signal.value)

        if signal == ActionType.STAY:
            message = f"[{symbol}] STAY - no action (position: {current_side.value})"
            logger.info(message)
            if dry_run:
                print(message)
            continue

        if signal == ActionType.ERR:
            message = f"[{symbol}] ERROR - skipping"
            logger.warning(message)
            if dry_run:
                print(message)
            continue

        # In dry-run mode, just print signals
        if dry_run:
            if signal == ActionType.LONG:
                margin_used = margin_balance * investment_ratio
                notional = margin_used * leverage
                print(f"[{symbol}] LONG: ${margin_used:.2f} margin x {leverage}x = ${notional:.2f} position")
                positions[symbol] = PositionSide.LONG
            elif signal == ActionType.SHORT:
                margin_used = margin_balance * investment_ratio
                notional = margin_used * leverage
                print(f"[{symbol}] SHORT: ${margin_used:.2f} margin x {leverage}x = ${notional:.2f} position")
                positions[symbol] = PositionSide.SHORT
            elif signal == ActionType.CLOSE:
                print(f"[{symbol}] CLOSE {current_side.value} position")
                positions[symbol] = PositionSide.NONE
            continue

        # Real trading would be implemented here
        # Would use client.futures_create_order() etc.

    # Print summary in dry-run mode
    if dry_run:
        print("\n=== Position Summary ===")
        for symbol, side in positions.items():
            if side != PositionSide.NONE:
                print(f"  {symbol}: {side.value}")
        if not any(s != PositionSide.NONE for s in positions.values()):
            print("  No open positions")


# Legacy trade function removed - futures only now
def trade(base_asset: str, trade_assets: list[str], interval: str, investment_ratio: Decimal) -> None:
    """Legacy spot trading - deprecated.

    This function has been deprecated. Use trade_with_strategies() for futures trading.
    """
    logger.error("Spot trading has been deprecated. Use trade-ga command for futures trading.")
    notify("Spot trading has been deprecated. Use trade-ga command for futures trading.")
