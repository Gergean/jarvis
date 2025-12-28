"""Portfolio training and testing commands.

Train and test multi-coin portfolio strategies.
"""

import hashlib
from datetime import UTC, datetime, timedelta

import enlighten

from jarvis.genetics.portfolio import (
    COOLDOWN_CANDLES,
    DEFAULT_FEE_RATE,
    LOOKBACK_PERIOD,
    MIN_POSITION_SIZE,
    PortfolioPopulation,
    PortfolioStrategy,
)
from jarvis.logging import logger


def train_portfolio(
    symbols: list[str],
    interval: str = "4h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    population_size: int = 50,
    generations: int = 30,
    rules_per_coin: int = 5,
    max_allocation: float = 0.20,
    stop_loss: float = 20.0,
    seed_path: str | None = None,
) -> PortfolioStrategy:
    """Train a multi-coin portfolio strategy.

    Args:
        symbols: List of trading pair symbols (e.g., ["BTCUSDT", "ETHUSDT"])
        interval: Kline interval (e.g., "4h")
        start_dt: Training start date
        end_dt: Training end date
        population_size: Number of portfolios in population
        generations: Number of generations to evolve
        rules_per_coin: Number of rules per coin
        max_allocation: Maximum allocation per coin (0-1)
        stop_loss: Portfolio stop-loss percentage
        seed_path: Path to seed portfolio strategy

    Returns:
        Trained PortfolioStrategy
    """
    # Default to last 6 months
    if end_dt is None:
        end_dt = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=180)

    logger.info("=" * 60)
    logger.info("PORTFOLIO TRAINING")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Period: {start_dt.date()} to {end_dt.date()}")
    logger.info(f"Population: {population_size}")
    logger.info(f"Generations: {generations}")
    logger.info(f"Rules per coin: {rules_per_coin}")
    logger.info(f"Max allocation: {max_allocation * 100}%")
    logger.info(f"Stop-loss: {stop_loss}%")
    logger.info("=" * 60)

    # Load seed portfolio if provided
    seed_portfolio = None
    if seed_path:
        logger.info(f"Loading seed portfolio from {seed_path}")
        seed_strategy = PortfolioStrategy.load(seed_path)
        seed_portfolio = seed_strategy.portfolio

    # Create initial population
    logger.info("Creating initial population...")
    population = PortfolioPopulation.create_random(
        symbols=symbols,
        population_size=population_size,
        rules_per_coin=rules_per_coin,
        interval=interval,
        max_allocation_per_coin=max_allocation,
        stop_loss_pct=stop_loss,
        seed_portfolio=seed_portfolio,
    )

    # Progress bar
    manager = enlighten.get_manager()
    progress = manager.counter(total=generations, desc="Training", unit="gen")

    preloaded_data = None
    best_fitness = float("-inf")
    best_portfolio = None

    for gen in range(generations):
        # Evaluate fitness
        preloaded_data = population.evaluate_fitness(
            symbols=symbols,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
            starting_balance=1000.0,
            preloaded_data=preloaded_data,
        )

        # Track best
        current_best = population.get_best()
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_portfolio = current_best

        avg_fitness = population.get_average_fitness()
        logger.info(f"Gen {gen + 1}/{generations}: best={current_best.fitness:.2f}, avg={avg_fitness:.2f}")

        progress.update()

        # Evolve (except last generation)
        if gen < generations - 1:
            population = population.evolve(interval=interval)

    progress.close()
    manager.stop()

    if best_portfolio is None:
        raise RuntimeError("No valid portfolio found during training")

    # Generate strategy ID
    symbols_str = "_".join(sorted(symbols))
    hash_input = f"{symbols_str}_{interval}_{datetime.now().isoformat()}"
    strategy_id = f"PORTFOLIO_{hashlib.sha256(hash_input.encode()).hexdigest()[:8]}"

    # Create and save strategy
    strategy = PortfolioStrategy(
        id=strategy_id,
        symbols=symbols,
        created_at=datetime.now(UTC).replace(tzinfo=None),
        training_config={
            "interval": interval,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "generations": generations,
            "population_size": population_size,
            "rules_per_coin": rules_per_coin,
            "max_allocation": max_allocation,
            "stop_loss": stop_loss,
        },
        portfolio=best_portfolio,
    )

    # Save strategy
    filepath = f"strategies/{strategy_id}.json"
    strategy.save(filepath)
    logger.info(f"Strategy saved: {filepath}")

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Strategy ID: {strategy_id}")
    logger.info(f"Fitness: {best_fitness:.2f}")
    logger.info(f"Total rules: {best_portfolio.total_rules}")
    for symbol in symbols:
        if symbol in best_portfolio.coin_strategies:
            num_rules = len(best_portfolio.coin_strategies[symbol].rules)
            logger.info(f"  {symbol}: {num_rules} rules")
    logger.info("=" * 60)

    return strategy


def test_portfolio(
    strategy_path: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    starting_balance: float = 1000.0,
) -> dict[str, object]:
    """Test a portfolio strategy on historical data.

    Args:
        strategy_path: Path to portfolio strategy JSON file
        start_dt: Test start date
        end_dt: Test end date
        starting_balance: Initial balance

    Returns:
        Dict with test results
    """
    from typing import Any

    import numpy as np

    from jarvis.client import get_binance_client
    from jarvis.genetics.indicators import OHLCV
    from jarvis.genetics.portfolio import Position
    from jarvis.models import ActionType, PositionSide
    from jarvis.utils import datetime_to_timestamp, interval_to_timedelta

    # Load strategy
    strategy = PortfolioStrategy.load(strategy_path)
    portfolio = strategy.portfolio
    symbols = strategy.symbols
    interval = strategy.training_config.get("interval", "4h")

    # Default to last 6 months if not specified
    if end_dt is None:
        end_dt = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=180)

    logger.info("=" * 60)
    logger.info("PORTFOLIO BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Strategy: {strategy.id}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Period: {start_dt.date()} to {end_dt.date()}")
    logger.info(f"Starting balance: ${starting_balance:.2f}")
    logger.info("=" * 60)

    # Load data
    client = get_binance_client(fake=True)
    lookback_delta = interval_to_timedelta(interval) * LOOKBACK_PERIOD
    fetch_start = start_dt - lookback_delta

    klines_data = {}
    for symbol in symbols:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(fetch_start),
            endTime=datetime_to_timestamp(end_dt),
            limit=50000,
        )
        if klines:
            klines_data[symbol] = klines
            logger.info(f"Loaded {len(klines)} candles for {symbol}")

    if not klines_data:
        raise RuntimeError("No data loaded")

    # Prepare OHLCV arrays
    symbol_arrays: dict[str, dict[str, Any]] = {}
    for symbol, klines in klines_data.items():
        symbol_arrays[symbol] = {
            "open": np.array([float(k.open) for k in klines]),
            "high": np.array([float(k.high) for k in klines]),
            "low": np.array([float(k.low) for k in klines]),
            "close": np.array([float(k.close) for k in klines]),
            "volume": np.array([float(k.volume) for k in klines]),
            "times": [k.open_time for k in klines],
        }

    # Find common length
    min_len = min(len(arr["close"]) for arr in symbol_arrays.values())

    # Backtest
    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0.0
    positions: dict[str, Position] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []

    allocation_pct = portfolio.max_allocation_per_coin
    stop_loss_pct = portfolio.stop_loss_pct
    stopped_out = False
    cooldown = 0

    for i in range(LOOKBACK_PERIOD, min_len):
        # Get OHLCV and prices
        prices = {}
        ohlcv_data = {}
        current_time = None

        for symbol, arrays in symbol_arrays.items():
            ohlcv = OHLCV(
                open=arrays["open"][i - LOOKBACK_PERIOD + 1 : i + 1],
                high=arrays["high"][i - LOOKBACK_PERIOD + 1 : i + 1],
                low=arrays["low"][i - LOOKBACK_PERIOD + 1 : i + 1],
                close=arrays["close"][i - LOOKBACK_PERIOD + 1 : i + 1],
                volume=arrays["volume"][i - LOOKBACK_PERIOD + 1 : i + 1],
            )
            price = arrays["close"][i]
            prices[symbol] = price
            ohlcv_data[symbol] = ohlcv
            current_time = arrays["times"][i]

        # Calculate portfolio value
        portfolio_value = balance + sum(
            pos.unrealized_value(prices.get(symbol, pos.entry_price)) for symbol, pos in positions.items()
        )

        equity_curve.append({"time": current_time, "value": portfolio_value})

        # Track drawdown
        if portfolio_value > peak_balance:
            peak_balance = portfolio_value
        current_dd = (peak_balance - portfolio_value) / peak_balance * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        # Check stop-loss
        if current_dd >= stop_loss_pct and not stopped_out:
            for symbol, pos in list(positions.items()):
                price = prices[symbol]
                exit_value = pos.calculate_exit_value(price, DEFAULT_FEE_RATE)
                balance += exit_value
                trades.append(
                    {
                        "symbol": symbol,
                        "side": pos.side.value,
                        "pnl": exit_value - pos.allocated_amount,
                        "reason": "stop_loss",
                    }
                )
            positions.clear()
            stopped_out = True
            cooldown = COOLDOWN_CANDLES
            continue

        if cooldown > 0:
            cooldown -= 1
            if cooldown == 0:
                stopped_out = False
                peak_balance = balance
            continue

        # Process signals
        for symbol in portfolio.symbols:
            if symbol not in prices:
                continue

            price = prices[symbol]
            ohlcv = ohlcv_data[symbol]
            current_side = positions[symbol].side if symbol in positions else PositionSide.NONE
            signal = portfolio.get_signal(symbol, ohlcv, current_side)

            # Close position
            if symbol in positions:
                pos = positions[symbol]
                should_close = (
                    signal == ActionType.CLOSE
                    or (signal == ActionType.LONG and pos.side == PositionSide.SHORT)
                    or (signal == ActionType.SHORT and pos.side == PositionSide.LONG)
                )

                if should_close:
                    exit_value = pos.calculate_exit_value(price, DEFAULT_FEE_RATE)
                    balance += exit_value
                    trades.append(
                        {
                            "symbol": symbol,
                            "side": pos.side.value,
                            "pnl": exit_value - pos.allocated_amount,
                            "reason": "signal",
                        }
                    )
                    del positions[symbol]

            # Open position
            if symbol not in positions and signal in (ActionType.LONG, ActionType.SHORT):
                alloc = balance * allocation_pct
                if alloc > MIN_POSITION_SIZE:
                    quantity = alloc / price
                    fee = alloc * DEFAULT_FEE_RATE
                    balance -= alloc + fee
                    side = PositionSide.LONG if signal == ActionType.LONG else PositionSide.SHORT
                    positions[symbol] = Position(
                        side=side,
                        entry_price=price,
                        quantity=quantity,
                        allocated_amount=alloc,
                    )

    # Close remaining positions
    for symbol, pos in list(positions.items()):
        price = prices.get(symbol, pos.entry_price)
        exit_value = pos.calculate_exit_value(price, DEFAULT_FEE_RATE)
        balance += exit_value
        trades.append(
            {
                "symbol": symbol,
                "side": pos.side.value,
                "pnl": exit_value - pos.allocated_amount,
                "reason": "end",
            }
        )

    # Calculate results
    total_return = (balance - starting_balance) / starting_balance * 100
    total_days = (end_dt - start_dt).days
    monthly_return = total_return / (total_days / 30) if total_days > 0 else 0

    winning_trades = sum(1 for t in trades if t["pnl"] > 0)
    losing_trades = sum(1 for t in trades if t["pnl"] <= 0)
    win_rate = winning_trades / len(trades) * 100 if trades else 0

    # Per-symbol stats
    symbol_stats = {}
    for symbol in symbols:
        symbol_trades = [t for t in trades if t["symbol"] == symbol]
        if symbol_trades:
            symbol_profit = sum(t["pnl"] for t in symbol_trades)
            symbol_wins = sum(1 for t in symbol_trades if t["pnl"] > 0)
            symbol_stats[symbol] = {
                "trades": len(symbol_trades),
                "profit": symbol_profit,
                "win_rate": symbol_wins / len(symbol_trades) * 100,
            }

    results = {
        "strategy_id": strategy.id,
        "symbols": symbols,
        "period_days": total_days,
        "starting_balance": starting_balance,
        "ending_balance": balance,
        "total_return_pct": total_return,
        "monthly_return_pct": monthly_return,
        "max_drawdown_pct": max_drawdown,
        "total_trades": len(trades),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate,
        "symbol_stats": symbol_stats,
    }

    # Print results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Starting: ${starting_balance:.2f} → Ending: ${balance:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Monthly Return: {monthly_return:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info("-" * 60)
    logger.info(f"Total Trades: {len(trades)}")
    logger.info(f"Win Rate: {win_rate:.1f}%")
    logger.info("-" * 60)
    logger.info("Per-Symbol Performance:")
    for symbol, stats in symbol_stats.items():
        logger.info(f"  {symbol}: {stats['trades']} trades, ${stats['profit']:.2f}, {stats['win_rate']:.1f}% win")
    logger.info("=" * 60)

    return results
