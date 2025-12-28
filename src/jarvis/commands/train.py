"""Train command for training a futures strategy for a single symbol."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from multiprocessing import Pool, cpu_count

import enlighten
import numpy as np
from dateutil.relativedelta import relativedelta

from jarvis.client import get_binance_client
from jarvis.commands.download import download
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.individual import Individual
from jarvis.genetics.population import Population
from jarvis.genetics.strategy import (
    RuleContribution,
    Strategy,
    TestResult,
    TradeSignal,
    TrainingConfig,
)
from jarvis.genetics.strategy import (
    WindowResult as WindowResultModel,
)
from jarvis.logging import logger
from jarvis.settings import notify
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
class WindowData:
    """Preloaded data for a single test window."""

    window_num: int
    test_start: datetime
    test_end: datetime
    ohlcv_data: list[tuple[OHLCV, Decimal, int]]  # (ohlcv, price, candle_idx)
    timestamps: list[datetime]  # Timestamp for each candle in ohlcv_data
    funding_interval_candles: int


@dataclass
class WindowResult:
    """Result of testing an individual on a single window."""

    window_num: int
    return_pct: float
    max_drawdown_pct: float
    trades: int
    liquidated: bool
    buy_hold_pct: float = 0.0  # Buy & hold return for this window


def _run_backtest_on_preloaded(
    individual: Individual,
    ohlcv_data: list[tuple[OHLCV, Decimal, int]],
    funding_interval_candles: int,
    starting_margin: Decimal = Decimal("100"),
    commission_ratio: Decimal = FUTURES_TAKER_FEE,
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    funding_enabled: bool = True,
) -> dict:
    """Run backtest on preloaded OHLCV data.

    IMPORTANT: Only COMPLETED trades count toward return calculation.
    If a position is still open at the end, we DO NOT include its unrealized PnL.

    Why? Because:
    - We only measure what the STRATEGY decided, not what WE decided
    - The strategy opened a position but didn't close it = no decision made yet
    - Including unrealized PnL would reward/punish based on arbitrary end date
    - This keeps fitness evaluation fair and consistent

    The open position's margin stays "frozen" in position_margin, reducing
    available balance but not affecting realized return calculation.
    """
    # Empty data = no trades possible
    if not ohlcv_data:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "liquidated": False,
            "buy_hold_pct": 0.0,
        }

    # Calculate buy & hold return (first price to last price)
    first_price = ohlcv_data[0][1]  # price from first candle
    last_price = ohlcv_data[-1][1]  # price from last candle
    buy_hold_pct = float((last_price - first_price) / first_price * 100) if first_price > 0 else 0.0

    # === ACCOUNT STATE ===
    # margin_balance: Available cash (not locked in positions)
    margin_balance = starting_margin

    # === POSITION STATE ===
    # position_side: Current position direction (NONE = no position)
    position_side = PositionSide.NONE
    # position_entry_price: Price at which we entered the position
    position_entry_price = Decimal(0)
    # position_quantity: Size of position in base asset (e.g., BTC amount)
    position_quantity = Decimal(0)
    # position_margin: Collateral locked for this position
    position_margin = Decimal(0)

    # === TRACKING VARIABLES ===
    # peak_equity: Highest equity seen (for drawdown calculation)
    peak_equity = starting_margin
    # max_drawdown_pct: Largest peak-to-trough decline in equity
    max_drawdown_pct = 0.0
    # total_trades: Number of completed round-trip trades (entry + exit)
    total_trades = 0
    # last_funding_candle: Last candle index when funding was applied
    last_funding_candle = 0
    # price: Current candle's close price
    price = Decimal(0)
    # liquidated: Whether position was liquidated (margin call)
    liquidated = False

    # === MAIN LOOP: Process each candle ===
    for ohlcv, price, candle_idx in ohlcv_data:

        # --- STEP 1: Check for liquidation (only with leverage > 1) ---
        # Liquidation occurs when losses exceed margin (simplified model)
        if position_side != PositionSide.NONE and leverage > 1:
            if position_side == PositionSide.LONG:
                # Long liquidation: price drops below entry * (1 - 1/leverage)
                liq_price = position_entry_price * (1 - Decimal(1) / Decimal(leverage))
                if price <= liq_price:
                    liquidated = True
                    break
            else:
                # Short liquidation: price rises above entry * (1 + 1/leverage)
                liq_price = position_entry_price * (1 + Decimal(1) / Decimal(leverage))
                if price >= liq_price:
                    liquidated = True
                    break

        # --- STEP 2: Apply funding fee (every 8 hours in futures) ---
        # Funding is paid/received periodically when holding a position
        if funding_enabled and position_side != PositionSide.NONE:
            candles_since_funding = candle_idx - last_funding_candle
            if candles_since_funding >= funding_interval_candles:
                # Calculate how many funding periods passed
                num_funding_periods = candles_since_funding // funding_interval_candles
                # Funding is based on notional value (position size * price)
                notional = position_quantity * price
                funding_payment = notional * FUNDING_FEE_RATE * num_funding_periods
                if position_side == PositionSide.LONG:
                    # Longs PAY funding (cost)
                    margin_balance -= funding_payment
                else:
                    # Shorts RECEIVE funding (income)
                    margin_balance += funding_payment
                last_funding_candle = candle_idx

        # --- STEP 3: Calculate equity for drawdown tracking ---
        # Equity = available margin + locked margin + unrealized PnL
        # We track this for drawdown even though we don't use unrealized PnL for final return
        equity = margin_balance
        if position_side != PositionSide.NONE:
            if position_side == PositionSide.LONG:
                # Long profit: current price > entry price
                unrealized_pnl = position_quantity * (price - position_entry_price)
            else:
                # Short profit: entry price > current price
                unrealized_pnl = position_quantity * (position_entry_price - price)
            # Equity includes locked margin and paper profit/loss
            equity += position_margin + unrealized_pnl

        # Update peak equity (highest point seen)
        if equity > peak_equity:
            peak_equity = equity
        # Calculate drawdown from peak
        if peak_equity > 0:
            drawdown_pct = float((peak_equity - equity) / peak_equity * 100)
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

        # --- STEP 4: Get strategy signal and execute ---
        signal = individual.get_signal(ohlcv, position_side)

        # OPEN LONG: Strategy says go long and we have no position
        if signal == ActionType.LONG and position_side == PositionSide.NONE:
            # Use investment_ratio of available margin (e.g., 20%)
            margin_to_use = margin_balance * investment_ratio
            if margin_to_use > 0 and price > 0:
                # Position size = (margin * leverage) / price
                position_size = (margin_to_use * leverage) / price
                # Pay entry fee (taker fee on notional value)
                fee = position_size * price * commission_ratio
                margin_balance -= fee
                # Record position details
                position_side = PositionSide.LONG
                position_entry_price = price
                position_quantity = position_size
                position_margin = margin_to_use
                # Lock margin (move from available to position)
                margin_balance -= margin_to_use
                total_trades += 1

        # OPEN SHORT: Strategy says go short and we have no position
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

        # CLOSE POSITION: Strategy says close and we have a position
        elif signal == ActionType.CLOSE and position_side != PositionSide.NONE:
            # Calculate realized PnL
            if position_side == PositionSide.LONG:
                pnl = position_quantity * (price - position_entry_price)
            else:
                pnl = position_quantity * (position_entry_price - price)

            # Pay exit fee
            fee = position_quantity * price * commission_ratio
            # Return margin + PnL - fee to available balance
            margin_balance += position_margin + pnl - fee
            # Clear position state
            position_side = PositionSide.NONE
            position_quantity = Decimal(0)
            position_margin = Decimal(0)
            total_trades += 1

    # === FINAL RETURN CALCULATION ===
    # CRITICAL: We only count margin_balance (realized gains)
    # If position is still open, position_margin is "frozen" but NOT counted as profit/loss
    # This ensures we only measure what the strategy actually decided to realize
    final_equity = margin_balance

    # If still in position, add back the locked margin (but NOT unrealized PnL)
    # This way: final_equity = starting - fees - losses + gains
    # Open position margin is returned but its paper profit/loss is ignored
    if not liquidated and position_side != PositionSide.NONE:
        final_equity += position_margin

    # Calculate return percentage
    if liquidated:
        # Liquidation = total loss
        return_pct = -100.0
    else:
        return_pct = float((final_equity - starting_margin) / starting_margin * 100)

    return {
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "liquidated": liquidated,
        "buy_hold_pct": buy_hold_pct,
    }


def _run_backtest_with_signals(
    individual: Individual,
    ohlcv_data: list[tuple[OHLCV, Decimal, int]],
    timestamps: list[datetime],
    funding_interval_candles: int,
    starting_margin: Decimal = Decimal("100"),
    commission_ratio: Decimal = FUTURES_TAKER_FEE,
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    funding_enabled: bool = True,
) -> tuple[dict, list[TradeSignal]]:
    """Run backtest capturing all trading signals with rule contributions.

    Same as _run_backtest_on_preloaded but also captures detailed signal info.
    Used for final evaluation after training, not during evolution (too slow).
    """
    signals: list[TradeSignal] = []

    if not ohlcv_data:
        return {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "liquidated": False,
        }, signals

    margin_balance = starting_margin
    position_side = PositionSide.NONE
    position_entry_price = Decimal(0)
    position_quantity = Decimal(0)
    position_margin = Decimal(0)

    peak_equity = starting_margin
    max_drawdown_pct = 0.0
    total_trades = 0
    last_funding_candle = 0
    price = Decimal(0)
    liquidated = False

    for data_idx, (ohlcv, price, candle_idx) in enumerate(ohlcv_data):
        # Check liquidation
        if position_side != PositionSide.NONE and leverage > 1:
            if position_side == PositionSide.LONG:
                liq_price = position_entry_price * (1 - Decimal(1) / Decimal(leverage))
                if price <= liq_price:
                    liquidated = True
                    break
            else:
                liq_price = position_entry_price * (1 + Decimal(1) / Decimal(leverage))
                if price >= liq_price:
                    liquidated = True
                    break

        # Apply funding fee
        if funding_enabled and position_side != PositionSide.NONE:
            candles_since_funding = candle_idx - last_funding_candle
            if candles_since_funding >= funding_interval_candles:
                num_funding_periods = candles_since_funding // funding_interval_candles
                notional = position_quantity * price
                funding_payment = notional * FUNDING_FEE_RATE * num_funding_periods
                if position_side == PositionSide.LONG:
                    margin_balance -= funding_payment
                else:
                    margin_balance += funding_payment
                last_funding_candle = candle_idx

        # Calculate equity for drawdown
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

        # Get signal
        signal = individual.get_signal(ohlcv, position_side)

        # Capture signal details if there's an action
        if signal in (ActionType.LONG, ActionType.SHORT, ActionType.CLOSE):
            # Calculate total score and rule contributions
            rule_contributions = []
            total_score = 0.0
            for rule in individual.rules:
                value, target, contribution = rule.calculate_contribution_detailed(ohlcv)
                total_score += contribution
                rule_contributions.append(
                    RuleContribution(
                        rule_str=str(rule),
                        value=value,
                        target=target,
                        contribution=contribution,
                    )
                )

            # Get timestamp for this candle
            ts = timestamps[data_idx] if data_idx < len(timestamps) else datetime.utcnow()

            trade_signal = TradeSignal(
                timestamp=ts.isoformat(),
                action=signal.name,
                price=float(price),
                score=total_score,
                rule_contributions=rule_contributions,
            )

            # Only record if it's an actual trade (state change)
            will_trade = (
                (signal == ActionType.LONG and position_side == PositionSide.NONE)
                or (signal == ActionType.SHORT and position_side == PositionSide.NONE)
                or (signal == ActionType.CLOSE and position_side != PositionSide.NONE)
            )
            if will_trade:
                signals.append(trade_signal)

        # Execute trade logic (same as original)
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

    # === FINAL RETURN CALCULATION ===
    # Only count realized gains - open positions don't count toward PnL
    final_equity = margin_balance

    # Return locked margin but NOT unrealized PnL
    if not liquidated and position_side != PositionSide.NONE:
        final_equity += position_margin

    if liquidated:
        return_pct = -100.0
    else:
        return_pct = float((final_equity - starting_margin) / starting_margin * 100)

    return {
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "liquidated": liquidated,
    }, signals


from enum import Enum


class FitnessType(Enum):
    """Available fitness functions for strategy evaluation."""
    LEGACY = "legacy"      # sum(returns) - sum(drawdowns)
    ALPHA = "alpha"        # sum(alphas) - sum(drawdowns), alpha = return - buy_hold
    CALMAR = "calmar"      # sum(returns) / max(sum(drawdowns), 1)
    SHARPE = "sharpe"      # mean(returns) / std(returns)


def _calculate_fitness_legacy(
    total_return: float,
    total_drawdown: float,
    any_liquidation: bool,
) -> float:
    """Legacy fitness function: sum(returns) - sum(drawdowns).

    This was the original fitness function. Kept for reference/comparison.
    """
    if any_liquidation:
        return 0.0
    return total_return - total_drawdown


def _calculate_fitness_alpha(
    total_alpha: float,
    total_drawdown: float,
    any_liquidation: bool,
) -> float:
    """Alpha-based fitness: sum(alphas) - sum(drawdowns).

    Alpha = strategy_return - buy_hold_return for each window.
    This rewards strategies that beat buy & hold, not just absolute returns.
    """
    if any_liquidation:
        return 0.0
    return total_alpha - total_drawdown


def _calculate_fitness_calmar(
    total_return: float,
    total_drawdown: float,
    any_liquidation: bool,
) -> float:
    """Calmar ratio fitness: sum(returns) / sum(drawdowns).

    Rewards high returns with low drawdowns. A strategy with 20% return
    and 5% drawdown (calmar=4) beats one with 40% return and 20% drawdown (calmar=2).
    """
    if any_liquidation:
        return 0.0
    # Avoid division by zero - use minimum drawdown of 1%
    return total_return / max(total_drawdown, 1.0)


def _calculate_fitness_sharpe(
    window_returns: list[float],
    any_liquidation: bool,
) -> float:
    """Sharpe-like fitness: mean(returns) / std(returns).

    Rewards consistent returns. Penalizes volatile strategies even if
    they have high average returns.
    """
    if any_liquidation or len(window_returns) < 2:
        return 0.0

    import statistics
    mean_return = statistics.mean(window_returns)
    std_return = statistics.stdev(window_returns)

    # Avoid division by zero - use minimum std of 1%
    # Multiply by sqrt(n) to scale with number of windows
    n = len(window_returns)
    return (mean_return / max(std_return, 1.0)) * (n ** 0.5)


def _evaluate_individual_on_windows(
    individual: Individual,
    windows: list[WindowData],
    starting_margin: Decimal,
    commission_ratio: Decimal,
    investment_ratio: Decimal,
    leverage: int,
    funding_enabled: bool,
    fitness_type: FitnessType = FitnessType.ALPHA,
) -> tuple[float, list[WindowResult]]:
    """Evaluate an individual across all windows.

    Returns:
        Tuple of (fitness, list of window results)
        Fitness calculation depends on fitness_type parameter.
        If any window has liquidation, fitness = 0
    """
    results = []
    total_return = 0.0
    total_alpha = 0.0
    total_drawdown = 0.0
    window_returns: list[float] = []
    any_liquidation = False

    for window in windows:
        metrics = _run_backtest_on_preloaded(
            individual=individual,
            ohlcv_data=window.ohlcv_data,
            funding_interval_candles=window.funding_interval_candles,
            starting_margin=starting_margin,
            commission_ratio=commission_ratio,
            investment_ratio=investment_ratio,
            leverage=leverage,
            funding_enabled=funding_enabled,
        )

        result = WindowResult(
            window_num=window.window_num,
            return_pct=metrics["return_pct"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            trades=metrics["total_trades"],
            liquidated=metrics["liquidated"],
            buy_hold_pct=metrics["buy_hold_pct"],
        )
        results.append(result)

        if result.liquidated:
            any_liquidation = True
        else:
            total_return += result.return_pct
            total_alpha += result.return_pct - result.buy_hold_pct
            total_drawdown += result.max_drawdown_pct
            window_returns.append(result.return_pct)

    # Calculate fitness based on selected type
    if fitness_type == FitnessType.LEGACY:
        fitness = _calculate_fitness_legacy(total_return, total_drawdown, any_liquidation)
    elif fitness_type == FitnessType.ALPHA:
        fitness = _calculate_fitness_alpha(total_alpha, total_drawdown, any_liquidation)
    elif fitness_type == FitnessType.CALMAR:
        fitness = _calculate_fitness_calmar(total_return, total_drawdown, any_liquidation)
    elif fitness_type == FitnessType.SHARPE:
        fitness = _calculate_fitness_sharpe(window_returns, any_liquidation)
    else:
        fitness = _calculate_fitness_alpha(total_alpha, total_drawdown, any_liquidation)

    return fitness, results


# Global variable to hold window data for parallel workers
_parallel_window_data: list[WindowData] = []
_parallel_params: dict = {}


def _init_worker(window_data: list[WindowData], params: dict) -> None:
    """Initialize worker process with shared data."""
    global _parallel_window_data, _parallel_params
    _parallel_window_data = window_data
    _parallel_params = params


def _evaluate_individual_parallel(individual: Individual) -> tuple[float, list[WindowResult]]:
    """Evaluate a single individual - called by worker processes."""
    return _evaluate_individual_on_windows(
        individual=individual,
        windows=_parallel_window_data,
        starting_margin=_parallel_params["starting_margin"],
        commission_ratio=_parallel_params["commission_ratio"],
        investment_ratio=_parallel_params["investment_ratio"],
        leverage=_parallel_params["leverage"],
        funding_enabled=_parallel_params["funding_enabled"],
        fitness_type=_parallel_params.get("fitness_type", FitnessType.ALPHA),
    )


def _preload_window_data(
    symbol: str,
    interval: str,
    windows: list[tuple[datetime, datetime]],  # (test_start, test_end)
    commission_ratio: Decimal,
    starting_margin: Decimal,
) -> list[WindowData]:
    """Preload OHLCV data for all test windows."""
    client = get_binance_client(
        fake=True,
        extra_params={"assets": {"USDT": starting_margin}, "commission_ratio": commission_ratio},
    )

    lookback = 200
    lookback_delta = interval_to_timedelta(interval) * lookback

    interval_hours = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}.get(interval, 1)
    funding_interval_candles = max(1, int(FUNDING_INTERVAL_HOURS / interval_hours))

    window_data_list = []

    for i, (test_start, test_end) in enumerate(windows):
        fetch_start = test_start - lookback_delta

        all_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(fetch_start),
            endTime=datetime_to_timestamp(test_end),
            limit=50000,
        )

        ohlcv_data = []
        timestamps = []
        if all_klines:
            n = len(all_klines)
            open_arr = np.zeros(n, dtype=np.float64)
            high_arr = np.zeros(n, dtype=np.float64)
            low_arr = np.zeros(n, dtype=np.float64)
            close_arr = np.zeros(n, dtype=np.float64)
            volume_arr = np.zeros(n, dtype=np.float64)
            kline_timestamps = []

            for j, k in enumerate(all_klines):
                open_arr[j] = float(k.open)
                high_arr[j] = float(k.high)
                low_arr[j] = float(k.low)
                close_arr[j] = float(k.close)
                volume_arr[j] = float(k.volume)
                kline_timestamps.append(k.open_time)

            for j in range(lookback, n):
                start_idx = j - lookback + 1
                end_idx = j + 1
                ohlcv = OHLCV(
                    open=open_arr[start_idx:end_idx],
                    high=high_arr[start_idx:end_idx],
                    low=low_arr[start_idx:end_idx],
                    close=close_arr[start_idx:end_idx],
                    volume=volume_arr[start_idx:end_idx],
                )
                price = Decimal(str(close_arr[j]))
                ohlcv_data.append((ohlcv, price, j))
                timestamps.append(kline_timestamps[j])

        window_data_list.append(
            WindowData(
                window_num=i + 1,
                test_start=test_start,
                test_end=test_end,
                ohlcv_data=ohlcv_data,
                timestamps=timestamps,
                funding_interval_candles=funding_interval_candles,
            )
        )

    return window_data_list


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
    seed_strategy: str | None = None,
    fitness_type: FitnessType = FitnessType.ALPHA,
) -> tuple[Strategy, TestResult]:
    """Train a futures trading strategy using walk-forward validation.

    The GA evolves a single strategy that performs well across ALL test windows.
    Each individual is evaluated on every window, and fitness is calculated as:
    fitness = sum(returns) - sum(drawdowns)

    If any window causes liquidation, fitness = 0.

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
        train_period: Training period - used only for window spacing
        test_period: Test period per window (e.g., "1M", "30d")
        step_period: Step size between windows (e.g., "1M", "30d")

    Returns:
        Tuple of (Strategy, TestResult) for the training run
    """
    # Parse periods to days
    test_days = parse_period_to_days(test_period)
    step_days = parse_period_to_days(step_period)

    # Default dates
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
        logger.info("Test: %s (%d days) | Step: %s (%d days)", test_period, test_days, step_period, step_days)
        fitness_desc = {
            FitnessType.LEGACY: "Σ(return) - Σ(drawdown)",
            FitnessType.ALPHA: "Σ(alpha) - Σ(drawdown) where alpha=return-buyhold",
            FitnessType.CALMAR: "Σ(return) / Σ(drawdown)",
            FitnessType.SHARPE: "mean(returns) / std(returns)",
        }
        logger.info("Fitness [%s] = %s", fitness_type.value, fitness_desc.get(fitness_type, "unknown"))

    logger.info("=" * 60)

    # Calculate test windows
    windows = []  # (test_start, test_end)
    current_start = start_dt

    while True:
        test_end = current_start + timedelta(days=test_days)
        if test_end > end_dt:
            break
        windows.append((current_start, test_end))
        current_start += timedelta(days=step_days)

    if not windows:
        raise ValueError(
            f"Not enough data for walk-forward. Need at least {test_days} days, "
            f"got {total_days} days. Try shorter periods or more data."
        )

    # Warning for too few windows
    if len(windows) < 10:
        logger.warning("!" * 60)
        logger.warning("WARNING: Only %d windows. Results may not be reliable.", len(windows))
        logger.warning("Recommend at least 10 windows for robust validation.")
        logger.warning("!" * 60)

    logger.info("Generated %d test windows:", len(windows))
    for i, (ts, te) in enumerate(windows):
        logger.info("  Window %d: %s to %s", i + 1, ts.date(), te.date())

    # Download data
    logger.info("[1/4] Downloading historical data...")
    lookback_bars = 200
    lookback_delta = interval_to_timedelta(interval) * lookback_bars
    download_start = start_dt - lookback_delta
    logger.info("Including %d bars lookback (from %s)", lookback_bars, download_start.date())
    download([symbol], interval, download_start, end_dt)
    logger.info("Data download complete.")

    # Preload all window data
    logger.info("[2/4] Preloading window data...")
    window_data = _preload_window_data(
        symbol=symbol,
        interval=interval,
        windows=windows,
        commission_ratio=commission_ratio,
        starting_margin=starting_margin,
    )
    total_points = sum(len(w.ohlcv_data) for w in window_data)
    logger.info("Loaded %d evaluation points across %d windows", total_points, len(windows))

    # Get price hint
    price_hint = None
    try:
        if window_data and window_data[-1].ohlcv_data:
            price_hint = float(window_data[-1].ohlcv_data[-1][1])
            logger.info("Price hint: %.4f", price_hint)
    except Exception:
        pass

    # Load seed strategy if provided
    seed_individual = None
    if seed_strategy:
        from jarvis.genetics.rule import Rule
        from jarvis.genetics.strategy import Strategy as StrategyModel

        logger.info("Loading seed strategy: %s", seed_strategy)
        seed = StrategyModel.load(seed_strategy)
        seed_individual = seed.individual

        # Check if weights are in old format (small values between -1 and 1)
        # and convert to new format (scale by WEIGHT_SCALE)
        max_weight = max(abs(r.weight) for r in seed_individual.rules)
        if max_weight <= 10:  # Old format detected
            logger.info("Converting old weight format to new scale (x%d)", Rule.WEIGHT_SCALE)
            for rule in seed_individual.rules:
                rule.weight *= Rule.WEIGHT_SCALE

        logger.info("Seed fitness was: %.2f", seed_individual.fitness)

    # Initialize population
    logger.info("[3/4] Evolving population...")
    notify(f"🚀 {symbol} Training Started!\nPop: {population_size}, Gen: {generations}\nInterval: {interval}")
    population = Population.create_random(
        population_size, rules_per_individual, price_hint=price_hint, seed_individual=seed_individual, interval=interval
    )

    bar_manager = enlighten.get_manager()
    gen_bar = bar_manager.counter(total=generations, desc="Evolution", unit="gen")

    best_individual = None
    best_fitness = float("-inf")
    best_results: list[WindowResult] = []

    # Determine number of workers
    use_parallel = os.environ.get("JARVIS_NO_PARALLEL") != "1"
    num_workers = cpu_count() if use_parallel else 1
    logger.info("Using %d CPU cores for evaluation", num_workers)

    # Prepare params for parallel workers
    parallel_params = {
        "starting_margin": starting_margin,
        "commission_ratio": commission_ratio,
        "investment_ratio": investment_ratio,
        "leverage": leverage,
        "funding_enabled": funding_enabled,
        "fitness_type": fitness_type,
    }

    for gen in range(generations):
        # Evaluate all individuals on all windows
        gen_best_fitness = float("-inf")
        gen_best_individual = None
        gen_best_results = []

        if use_parallel and num_workers > 1:
            # Parallel evaluation
            with Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(window_data, parallel_params),
            ) as pool:
                eval_results = pool.map(_evaluate_individual_parallel, population.individuals)

            for individual, (fitness, results) in zip(population.individuals, eval_results):
                individual.fitness = fitness
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_individual = individual
                    gen_best_results = results
        else:
            # Serial evaluation
            for individual in population.individuals:
                fitness, results = _evaluate_individual_on_windows(
                    individual=individual,
                    windows=window_data,
                    starting_margin=starting_margin,
                    commission_ratio=commission_ratio,
                    investment_ratio=investment_ratio,
                    leverage=leverage,
                    funding_enabled=funding_enabled,
                    fitness_type=fitness_type,
                )
                individual.fitness = fitness

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_individual = individual
                    gen_best_results = results

        # Track overall best
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual
            best_results = gen_best_results

        # Log progress every generation
        avg_return = sum(r.return_pct for r in gen_best_results) / len(gen_best_results) if gen_best_results else 0
        logger.info(
            "Gen %d: Best fitness=%.2f, Avg monthly return=%.2f%%",
            gen + 1,
            gen_best_fitness,
            avg_return,
        )

        # Send Telegram notification for every generation
        yearly_return = 100 * ((1 + avg_return / 100) ** 12)
        notify(f"🧬 {symbol} Gen {gen + 1}/{generations}\nFitness: {gen_best_fitness:.2f}\nReturn: {avg_return:.2f}%\n$100 → ${yearly_return:.0f}")

        # Evolve (except last generation)
        if gen < generations - 1:
            population = population.evolve(price_hint=price_hint, interval=interval)

        gen_bar.update()

    gen_bar.close()
    bar_manager.stop()

    # Final results
    if best_individual is None:
        best_individual = population.get_best()
        _, best_results = _evaluate_individual_on_windows(
            individual=best_individual,
            windows=window_data,
            starting_margin=starting_margin,
            commission_ratio=commission_ratio,
            investment_ratio=investment_ratio,
            leverage=leverage,
            funding_enabled=funding_enabled,
            fitness_type=fitness_type,
        )

    # Calculate aggregate metrics
    avg_return = sum(r.return_pct for r in best_results) / len(best_results)
    avg_drawdown = sum(r.max_drawdown_pct for r in best_results) / len(best_results)
    total_trades = sum(r.trades for r in best_results)
    positive_windows = sum(1 for r in best_results if r.return_pct > 0)
    liquidations = sum(1 for r in best_results if r.liquidated)

    # Log summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Window Results:")
    for r in best_results:
        status = "X" if r.liquidated else ("+" if r.return_pct > 0 else "-")
        logger.info(
            "  W%d: Return %+.2f%% | Drawdown %.2f%% | Trades %d %s",
            r.window_num,
            r.return_pct,
            r.max_drawdown_pct,
            r.trades,
            status,
        )

    logger.info("-" * 40)
    logger.info(
        "Positive windows: %d/%d (%.0f%%)",
        positive_windows,
        len(best_results),
        100 * positive_windows / len(best_results),
    )
    logger.info("Liquidations: %d", liquidations)
    annualized_return = ((1 + avg_return / 100) ** 12 - 1) * 100
    logger.info("Average monthly return: %.2f%% (~%.0f%% annualized)", avg_return, annualized_return)
    logger.info("Average drawdown: %.2f%%", avg_drawdown)
    logger.info("Fitness: %.2f", best_fitness)
    logger.info("Total trades: %d", total_trades)

    # Re-run backtest with signal capture for debugging
    logger.info("Capturing detailed signals...")
    detailed_window_results: list[WindowResultModel] = []

    for window, basic_result in zip(window_data, best_results):
        # Run backtest with signal capture
        _, signals = _run_backtest_with_signals(
            individual=best_individual,
            ohlcv_data=window.ohlcv_data,
            timestamps=window.timestamps,
            funding_interval_candles=window.funding_interval_candles,
            starting_margin=starting_margin,
            commission_ratio=commission_ratio,
            investment_ratio=investment_ratio,
            leverage=leverage,
            funding_enabled=funding_enabled,
        )

        # Create detailed window result
        detailed_result = WindowResultModel(
            window_num=basic_result.window_num,
            start_date=window.test_start.strftime("%Y-%m-%d"),
            end_date=window.test_end.strftime("%Y-%m-%d"),
            return_pct=basic_result.return_pct,
            max_drawdown_pct=basic_result.max_drawdown_pct,
            trades=basic_result.trades,
            liquidated=basic_result.liquidated,
            signals=signals,
        )
        detailed_window_results.append(detailed_result)

    logger.info(
        "Captured %d signals across %d windows",
        sum(len(w.signals) for w in detailed_window_results),
        len(detailed_window_results),
    )

    # Save strategy
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
        result_type="walk_forward",
        return_pct=avg_return,
        max_drawdown_pct=avg_drawdown,
        total_trades=total_trades,
        final_equity=float(starting_margin) * (1 + avg_return / 100),
        peak_equity=float(starting_margin),
        windows=detailed_window_results,
    )

    strategy_path = strategy.save(strategies_dir)
    result_path = result.save(results_dir)

    logger.info("Strategy saved: %s", strategy_path)
    logger.info("Results saved: %s", result_path)
    logger.info("=" * 60)

    # Final notification
    avg_return = sum(r.return_pct for r in best_results) / len(best_results) if best_results else 0
    avg_dd = sum(r.max_drawdown_pct for r in best_results) / len(best_results) if best_results else 0
    yearly_return = 100 * ((1 + avg_return / 100) ** 12)
    notify(f"✅ {symbol} Training Complete!\nStrategy: {strategy.id}\nFitness: {best_fitness:.2f}\nReturn: {avg_return:.2f}%\nDrawdown: {avg_dd:.2f}%\n$100 → ${yearly_return:.0f}")

    return strategy, result
