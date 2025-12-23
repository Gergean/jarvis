"""Paper trading commands for simulated trading."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.logging import logger

PAPER_DIR = Path("paper")
EVOLVE_POPULATION_SIZE = 30
EVOLVE_GENERATIONS = 10
EVOLVE_LOOKBACK_DAYS = 30


def paper_init(wallet_id: str, balance: float, config: list[str]) -> dict[str, Any]:
    """Initialize a new paper trading wallet.

    Args:
        wallet_id: Unique wallet identifier
        balance: Initial balance in USD
        config: List of "SYMBOL:INTERVAL" strings (e.g., ["BTCUSDT:1h", "ETHUSDT:4h"])

    Returns:
        Created wallet data
    """
    PAPER_DIR.mkdir(exist_ok=True)

    wallet_path = PAPER_DIR / f"{wallet_id}.json"
    if wallet_path.exists():
        raise ValueError(f"Wallet '{wallet_id}' already exists")

    # Parse config
    parsed_config = {}
    for item in config:
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid config format: {item}. Expected SYMBOL:INTERVAL")
        symbol, interval = parts
        parsed_config[symbol.upper()] = interval.lower()

    now = datetime.now(timezone.utc).isoformat()

    wallet = {
        "id": wallet_id,
        "created_at": now,
        "initial_balance": balance,
        "balance": balance,
        "config": parsed_config,
        "positions": {},
        "transactions": [
            {
                "type": "deposit",
                "amount": balance,
                "timestamp": now,
            }
        ],
        "last_run": None,
    }

    wallet_path.write_text(json.dumps(wallet, indent=2))
    logger.info(f"Created wallet '{wallet_id}' with ${balance} balance")

    return wallet


def paper_trade(wallet_id: str, end_dt: datetime | None = None) -> dict[str, Any]:
    """Run paper trading from last_run to end_dt (or now).

    Args:
        wallet_id: Wallet identifier
        end_dt: End datetime (default: now)

    Returns:
        Updated wallet data with trade results
    """
    wallet_path = PAPER_DIR / f"{wallet_id}.json"
    if not wallet_path.exists():
        raise ValueError(f"Wallet '{wallet_id}' not found")

    wallet = json.loads(wallet_path.read_text())

    # Import here to avoid circular imports
    from jarvis.client import get_klines_from_day_files
    from jarvis.genetics.indicators import OHLCV
    from jarvis.genetics.strategy import Strategy
    from jarvis.models import PositionSide
    from jarvis.utils import interval_to_timedelta

    # Determine start time
    if wallet["last_run"]:
        start_time = datetime.fromisoformat(wallet["last_run"])
    else:
        start_time = datetime.fromisoformat(wallet["created_at"])

    # Get the smallest interval from config to snap end time
    from jarvis.utils import floor_dt

    intervals = list(wallet["config"].values())
    min_interval_delta = min(interval_to_timedelta(i) for i in intervals)

    # Snap end time to last completed bar
    raw_now = end_dt or datetime.now(timezone.utc)
    now = floor_dt(raw_now.replace(tzinfo=None), min_interval_delta).replace(tzinfo=timezone.utc)

    logger.info(f"Paper trading '{wallet_id}' from {start_time} to {now}")

    from jarvis.utils import datetime_to_timestamp

    trades_made = []
    lookback = 200  # Bars needed for indicators

    for symbol, interval in wallet["config"].items():
        # Load current strategy
        strategy_path = Path(f"strategies/current/{symbol}_{interval}.json")
        if not strategy_path.exists():
            logger.warning(f"No current strategy for {symbol}_{interval}, skipping")
            continue

        strategy = Strategy.load(str(strategy_path))
        individual = strategy.individual

        # Get historical klines with lookback for indicators
        interval_delta = interval_to_timedelta(interval)
        lookback_time = start_time - (interval_delta * lookback)
        start_ts = datetime_to_timestamp(lookback_time)
        end_ts = datetime_to_timestamp(now)

        # Get data from lookback_time to now
        klines = get_klines_from_day_files(
            None,  # No client needed for cached files
            symbol,
            interval,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=10000,
        )

        if len(klines) < lookback:
            logger.info(f"Not enough data for {symbol} ({len(klines)} bars)")
            continue

        logger.info(f"{symbol}: Processing {len(klines)} bars")

        # Get current position
        current_position = wallet["positions"].get(symbol, {"side": "NONE"})
        position_side = PositionSide[current_position["side"]]

        # Process bars after lookback period
        for i in range(lookback, len(klines)):
            kline = klines[i]
            bar_time = kline.open_time.replace(tzinfo=timezone.utc)

            # Skip bars before start_time
            if bar_time < start_time:
                continue

            close_price = float(kline.close)

            # Build OHLCV with lookback for indicators
            window = klines[i - lookback + 1 : i + 1]
            ohlcv = OHLCV(
                open=np.array([float(k.open) for k in window]),
                high=np.array([float(k.high) for k in window]),
                low=np.array([float(k.low) for k in window]),
                close=np.array([float(k.close) for k in window]),
                volume=np.array([float(k.volume) for k in window]),
            )

            signal = individual.get_signal(ohlcv, position_side)

            # Check for evolve at 00:00 UTC
            if bar_time.hour == 0 and bar_time.minute == 0:
                logger.info(f"{symbol}: 00:00 UTC - evolving strategy...")
                _evolve_strategy(symbol, interval, bar_time, strategy_path)
                # Reload evolved strategy
                strategy = Strategy.load(str(strategy_path))
                individual = strategy.individual

            # Execute signal
            if signal.name in ["LONG", "SHORT"]:
                # Open position
                position_size = wallet["balance"] * 0.20  # 20% of balance
                if position_size > 0:
                    wallet["positions"][symbol] = {
                        "side": signal.name,
                        "entry_price": close_price,
                        "size": position_size,
                        "entry_time": bar_time.isoformat(),
                    }
                    position_side = PositionSide[signal.name]

                    trade = {
                        "type": "open",
                        "symbol": symbol,
                        "side": signal.name,
                        "price": close_price,
                        "size": position_size,
                        "timestamp": bar_time.isoformat(),
                    }
                    wallet["transactions"].append(trade)
                    trades_made.append(trade)
                    logger.info(f"{symbol}: Opened {signal.name} at ${close_price:.2f}")

            elif signal.name == "CLOSE" and symbol in wallet["positions"]:
                # Close position
                pos = wallet["positions"][symbol]
                entry_price = pos["entry_price"]
                size = pos["size"]

                if pos["side"] == "LONG":
                    pnl = (close_price - entry_price) / entry_price * size
                else:  # SHORT
                    pnl = (entry_price - close_price) / entry_price * size

                wallet["balance"] += pnl

                trade = {
                    "type": "close",
                    "symbol": symbol,
                    "side": pos["side"],
                    "entry_price": entry_price,
                    "exit_price": close_price,
                    "size": size,
                    "pnl": pnl,
                    "timestamp": bar_time.isoformat(),
                }
                wallet["transactions"].append(trade)
                trades_made.append(trade)

                del wallet["positions"][symbol]
                position_side = PositionSide.NONE
                logger.info(f"{symbol}: Closed {pos['side']} at ${close_price:.2f}, PnL: ${pnl:.2f}")

    wallet["last_run"] = now.isoformat()
    wallet_path.write_text(json.dumps(wallet, indent=2))

    logger.info(f"Paper trading complete. {len(trades_made)} trades made. Balance: ${wallet['balance']:.2f}")

    return wallet


def paper_info(wallet_id: str) -> dict[str, Any]:
    """Get paper wallet info and stats.

    Args:
        wallet_id: Wallet identifier

    Returns:
        Wallet data with calculated stats
    """
    wallet_path = PAPER_DIR / f"{wallet_id}.json"
    if not wallet_path.exists():
        raise ValueError(f"Wallet '{wallet_id}' not found")

    wallet = json.loads(wallet_path.read_text())

    # Calculate stats
    trades = [t for t in wallet["transactions"] if t["type"] == "close"]
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["pnl"] > 0])
    total_pnl = sum(t["pnl"] for t in trades)

    # Calculate open positions value
    open_positions_value = sum(pos["size"] for pos in wallet["positions"].values())

    stats = {
        "wallet_id": wallet_id,
        "initial_balance": wallet["initial_balance"],
        "current_balance": wallet["balance"],
        "total_pnl": total_pnl,
        "total_pnl_pct": (wallet["balance"] - wallet["initial_balance"]) / wallet["initial_balance"] * 100,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": winning_trades / total_trades * 100 if total_trades > 0 else 0,
        "open_positions": len(wallet["positions"]),
        "open_positions_value": open_positions_value,
        "config": wallet["config"],
        "created_at": wallet["created_at"],
        "last_run": wallet["last_run"],
    }

    return stats


def paper_list() -> list[str]:
    """List all paper wallets.

    Returns:
        List of wallet IDs
    """
    if not PAPER_DIR.exists():
        return []

    return [f.stem for f in PAPER_DIR.glob("*.json")]


def _evolve_strategy(
    symbol: str,
    interval: str,
    evolve_date: datetime,
    strategy_path: Path,
) -> None:
    """Evolve a strategy using recent data.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (e.g., 1h)
        evolve_date: The date of evolution (00:00 UTC)
        strategy_path: Path to current strategy file
    """
    from jarvis.client import get_klines_from_day_files
    from jarvis.genetics.indicators import OHLCV
    from jarvis.genetics.population import Population
    from jarvis.genetics.strategy import Strategy
    from jarvis.models import (
        FUNDING_FEE_RATE,
        FUNDING_INTERVAL_HOURS,
        FUTURES_TAKER_FEE,
        ActionType,
        PositionSide,
    )
    from jarvis.utils import datetime_to_timestamp, interval_to_timedelta

    # Load current strategy
    current_strategy = Strategy.load(str(strategy_path))
    current_individual = current_strategy.individual

    # Calculate data range (last N days before evolve_date)
    end_dt = evolve_date
    start_dt = evolve_date - timedelta(days=EVOLVE_LOOKBACK_DAYS)

    # Get historical data
    start_ts = datetime_to_timestamp(start_dt)
    end_ts = datetime_to_timestamp(end_dt)

    klines = get_klines_from_day_files(
        None,
        symbol,
        interval,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=10000,
    )

    if len(klines) < 200:
        logger.warning(f"Not enough data for evolution ({len(klines)} bars)")
        return

    # Build OHLCV arrays
    opens = np.array([float(k.open) for k in klines])
    highs = np.array([float(k.high) for k in klines])
    lows = np.array([float(k.low) for k in klines])
    closes = np.array([float(k.close) for k in klines])
    volumes = np.array([float(k.volume) for k in klines])

    price_hint = float(closes[-1])

    # Create population seeded from current strategy
    population = Population.create_random(
        population_size=EVOLVE_POPULATION_SIZE,
        rules_per_individual=len(current_individual.rules),
        price_hint=price_hint,
        seed_individual=current_individual,
    )

    # Prepare evaluation data
    lookback = 200
    interval_delta = interval_to_timedelta(interval)
    interval_hours = interval_delta.total_seconds() / 3600
    funding_interval_candles = max(1, int(FUNDING_INTERVAL_HOURS / interval_hours))

    ohlcv_data = []
    for i in range(lookback, len(klines)):
        ohlcv = OHLCV(
            open=opens[i - lookback + 1 : i + 1],
            high=highs[i - lookback + 1 : i + 1],
            low=lows[i - lookback + 1 : i + 1],
            close=closes[i - lookback + 1 : i + 1],
            volume=volumes[i - lookback + 1 : i + 1],
        )
        ohlcv_data.append((ohlcv, Decimal(str(closes[i])), i))

    def evaluate_individual(ind):
        """Evaluate a single individual on the data."""
        margin = Decimal("100")
        position_side = PositionSide.NONE
        entry_price = Decimal(0)
        quantity = Decimal(0)
        position_margin = Decimal(0)
        peak = margin
        max_dd = 0.0
        trades = 0

        for ohlcv, price, candle_idx in ohlcv_data:
            signal = ind.get_signal(ohlcv, position_side)

            # Funding fee
            if position_side != PositionSide.NONE and candle_idx % funding_interval_candles == 0:
                funding = position_margin * Decimal(str(FUNDING_FEE_RATE))
                margin -= funding

            # Execute signals
            if signal == ActionType.LONG and position_side == PositionSide.NONE:
                position_margin = margin * Decimal("0.2")
                commission = position_margin * Decimal(str(FUTURES_TAKER_FEE))
                margin -= commission
                quantity = position_margin / price
                entry_price = price
                position_side = PositionSide.LONG
                trades += 1
            elif signal == ActionType.SHORT and position_side == PositionSide.NONE:
                position_margin = margin * Decimal("0.2")
                commission = position_margin * Decimal(str(FUTURES_TAKER_FEE))
                margin -= commission
                quantity = position_margin / price
                entry_price = price
                position_side = PositionSide.SHORT
                trades += 1
            elif signal == ActionType.CLOSE and position_side != PositionSide.NONE:
                if position_side == PositionSide.LONG:
                    pnl = (price - entry_price) * quantity
                else:
                    pnl = (entry_price - price) * quantity
                commission = abs(pnl) * Decimal(str(FUTURES_TAKER_FEE))
                margin += pnl - commission
                position_side = PositionSide.NONE

            # Track drawdown
            if margin > peak:
                peak = margin
            dd = float((peak - margin) / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return_pct = float((margin - Decimal("100")) / Decimal("100") * 100)
        return return_pct, max_dd, trades

    # Evolution loop
    for gen in range(EVOLVE_GENERATIONS):
        # Evaluate all individuals
        for ind in population.individuals:
            ret, dd, trades = evaluate_individual(ind)
            ind.fitness = ret - dd  # Simple fitness

        population = population.evolve(price_hint=price_hint)

    # Evaluate final generation
    for ind in population.individuals:
        ret, dd, trades = evaluate_individual(ind)
        ind.fitness = ret - dd

    # Get best individual
    best = population.get_best()

    # Update strategy with new individual
    current_strategy.individual = best

    # Save directly to the path (not using save() which expects a directory)
    with open(strategy_path, "w") as f:
        json.dump(current_strategy.to_dict(), f, indent=2)

    logger.info(f"Evolved {symbol}_{interval}: fitness={best.fitness:.2f}")
