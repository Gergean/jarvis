"""Paper trading commands for simulated trading."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.logging import logger

PAPER_DIR = Path("paper")
ELITES_DIR = Path("strategies/elites")
EVOLVE_POPULATION_SIZE = 30
EVOLVE_GENERATIONS = 10
EVOLVE_LOOKBACK_DAYS = 30


def _get_elite_path(symbol: str, interval: str, dt: datetime) -> Path:
    """Get the path for an elite strategy file.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (e.g., 1h)
        dt: Datetime of the elite

    Returns:
        Path to elite file (e.g., strategies/elites/BTCUSDT/1h/20250101_000000.json)
    """
    filename = dt.strftime("%Y%m%d_%H%M%S") + ".json"
    return ELITES_DIR / symbol / interval / filename


def _find_elite(symbol: str, interval: str, dt: datetime) -> Path | None:
    """Find the most recent elite for a given datetime.

    Looks for an elite at or before the given datetime.

    Args:
        symbol: Trading pair
        interval: Candle interval
        dt: Target datetime

    Returns:
        Path to elite file, or None if not found
    """
    elite_dir = ELITES_DIR / symbol / interval
    if not elite_dir.exists():
        return None

    # Get all elite files sorted by name (which is chronological)
    elite_files = sorted(elite_dir.glob("*.json"))
    if not elite_files:
        return None

    # Find the most recent elite at or before dt
    target_name = dt.strftime("%Y%m%d_%H%M%S")
    best_match = None

    for f in elite_files:
        file_ts = f.stem  # e.g., "20250101_000000"
        if file_ts <= target_name:
            best_match = f
        else:
            break  # Files are sorted, no need to continue

    return best_match


def _get_or_create_elite(
    symbol: str,
    interval: str,
    evolve_dt: datetime,
    seed_individual: Any,
) -> Any:
    """Get existing elite or create new one via evolution.

    Args:
        symbol: Trading pair
        interval: Candle interval
        evolve_dt: Datetime for the elite (should be 00:00 UTC)
        seed_individual: Individual to seed evolution from

    Returns:
        The elite Individual
    """
    from jarvis.genetics.individual import Individual

    elite_path = _get_elite_path(symbol, interval, evolve_dt)

    # Check if elite already exists
    if elite_path.exists():
        logger.info(f"Loading existing elite: {elite_path}")
        with open(elite_path) as f:
            data = json.load(f)
        return Individual.from_dict(data)

    # Evolve new elite
    logger.info(f"Evolving new elite for {symbol}/{interval} at {evolve_dt}")
    new_elite = _evolve_elite(symbol, interval, evolve_dt, seed_individual)

    # Save elite
    elite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(elite_path, "w") as f:
        json.dump(new_elite.to_dict(), f, indent=2)

    logger.info(f"Saved elite: {elite_path} (fitness={new_elite.fitness:.2f})")
    return new_elite


def paper_init(
    wallet_id: str,
    balance: float,
    config: list[str],
    seed_strategy: str | None = None,
) -> dict[str, Any]:
    """Initialize a new paper trading wallet.

    Args:
        wallet_id: Unique wallet identifier
        balance: Initial balance in USD
        config: List of "SYMBOL:INTERVAL" strings (e.g., ["BTCUSDT:1h", "ETHUSDT:4h"])
        seed_strategy: Strategy ID to use as initial seed (e.g., "BTCUSDT_abc123")

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

    # Load seed strategies if provided
    seed_strategies = {}
    if seed_strategy:
        from jarvis.genetics.strategy import Strategy

        strategy = Strategy.load_by_id(seed_strategy)
        # Apply to all matching symbols
        for symbol, interval in parsed_config.items():
            if symbol == strategy.symbol:
                seed_strategies[f"{symbol}_{interval}"] = strategy.individual.to_dict()
                logger.info(f"Using seed strategy {seed_strategy} for {symbol}_{interval}")

    now = datetime.now(UTC).isoformat()

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
        "seed_strategies": seed_strategies,
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
    from jarvis.genetics.individual import Individual
    from jarvis.models import PositionSide
    from jarvis.utils import floor_dt, interval_to_timedelta

    # Determine start time
    if wallet["last_run"]:
        start_time = datetime.fromisoformat(wallet["last_run"])
    else:
        start_time = datetime.fromisoformat(wallet["created_at"])

    # Get the smallest interval from config to snap end time
    intervals = list(wallet["config"].values())
    min_interval_delta = min(interval_to_timedelta(i) for i in intervals)

    # Snap end time to last completed bar
    raw_now = end_dt or datetime.now(UTC)
    now = floor_dt(raw_now.replace(tzinfo=None), min_interval_delta).replace(tzinfo=UTC)

    logger.info(f"Paper trading '{wallet_id}' from {start_time} to {now}")

    from jarvis.utils import datetime_to_timestamp

    trades_made = []
    lookback = 200  # Bars needed for indicators

    for symbol, interval in wallet["config"].items():
        config_key = f"{symbol}_{interval}"

        # Get seed individual for this symbol/interval
        seed_dict = wallet.get("seed_strategies", {}).get(config_key)
        if not seed_dict:
            logger.warning(f"No seed strategy for {config_key}, skipping")
            continue

        seed_individual = Individual.from_dict(seed_dict)

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

        # Track current elite
        current_individual = None

        # Process bars after lookback period
        for i in range(lookback, len(klines)):
            kline = klines[i]
            bar_time = kline.open_time.replace(tzinfo=UTC)

            # Skip bars before start_time
            if bar_time < start_time:
                continue

            close_price = float(kline.close)

            # Check for elite update at 00:00 UTC
            if bar_time.hour == 0 and bar_time.minute == 0:
                # Get or create elite for this time
                current_individual = _get_or_create_elite(
                    symbol, interval, bar_time, seed_individual
                )
                # Update seed for next evolution
                seed_individual = current_individual

            # If no elite yet, find most recent one
            if current_individual is None:
                elite_path = _find_elite(symbol, interval, bar_time)
                if elite_path:
                    with open(elite_path) as f:
                        current_individual = Individual.from_dict(json.load(f))
                    logger.info(f"Loaded initial elite from {elite_path}")
                else:
                    # No elite exists, use seed and create first elite
                    current_individual = seed_individual
                    logger.info("Using seed strategy as initial elite")

            # Build OHLCV with lookback for indicators
            window = klines[i - lookback + 1 : i + 1]
            ohlcv = OHLCV(
                open=np.array([float(k.open) for k in window]),
                high=np.array([float(k.high) for k in window]),
                low=np.array([float(k.low) for k in window]),
                close=np.array([float(k.close) for k in window]),
                volume=np.array([float(k.volume) for k in window]),
            )

            signal = current_individual.get_signal(ohlcv, position_side)

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


def _evolve_elite(
    symbol: str,
    interval: str,
    evolve_dt: datetime,
    seed_individual: Any,
) -> Any:
    """Evolve a new elite using recent data.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (e.g., 1h)
        evolve_dt: The datetime of evolution (00:00 UTC)
        seed_individual: Individual to seed evolution from

    Returns:
        The evolved elite Individual
    """
    from jarvis.client import get_klines_from_day_files
    from jarvis.genetics.indicators import OHLCV
    from jarvis.genetics.population import Population
    from jarvis.models import (
        FUNDING_FEE_RATE,
        FUNDING_INTERVAL_HOURS,
        FUTURES_TAKER_FEE,
        ActionType,
        PositionSide,
    )
    from jarvis.utils import datetime_to_timestamp, interval_to_timedelta

    # Calculate data range (last N days before evolve_dt)
    end_dt = evolve_dt
    start_dt = evolve_dt - timedelta(days=EVOLVE_LOOKBACK_DAYS)

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
        logger.warning(f"Not enough data for evolution ({len(klines)} bars), returning seed")
        return seed_individual

    # Build OHLCV arrays
    opens = np.array([float(k.open) for k in klines])
    highs = np.array([float(k.high) for k in klines])
    lows = np.array([float(k.low) for k in klines])
    closes = np.array([float(k.close) for k in klines])
    volumes = np.array([float(k.volume) for k in klines])

    price_hint = float(closes[-1])

    # Create population seeded from current individual
    population = Population.create_random(
        population_size=EVOLVE_POPULATION_SIZE,
        rules_per_individual=len(seed_individual.rules),
        price_hint=price_hint,
        seed_individual=seed_individual,
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
    for _gen in range(EVOLVE_GENERATIONS):
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
    return population.get_best()
