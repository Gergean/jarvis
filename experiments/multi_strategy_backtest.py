#!/usr/bin/env python
"""
Multi-Strategy Portfolio Backtest

4 farklı stratejiyi aynı anda çalıştırarak portföy performansını test eder.
- Her coin için maksimum %25 allocation
- 1 yıllık backtest
- Toplam portföy getirisi hesaplanır
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from jarvis.client import get_binance_client
from jarvis.genetics.strategy import Strategy
from jarvis.genetics.individual import Individual
from jarvis.genetics.indicators import OHLCV
from jarvis.models import ActionType, PositionSide, FUTURES_TAKER_FEE
from jarvis.logging import logger


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    strategy_id: str
    symbol: str
    interval: str = "4h"


@dataclass
class Position:
    """An open position."""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: datetime
    allocated_amount: float  # How much USD was allocated


@dataclass
class Trade:
    """A completed trade."""
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    profit: float
    fee: float
    return_pct: float


@dataclass
class PortfolioBacktestConfig:
    """Portfolio backtest configuration."""
    strategies: list[StrategyConfig]
    start_date: datetime
    end_date: datetime
    starting_balance: float = 1000.0
    max_allocation_per_coin: float = 0.25  # 25% max per coin
    leverage: int = 1
    fee_rate: float = 0.0004  # Taker fee


class MultiStrategyBacktest:
    """Backtest multiple strategies as a portfolio."""

    def __init__(self, config: PortfolioBacktestConfig):
        self.config = config
        self.balance = config.starting_balance
        self.peak_balance = config.starting_balance
        self.max_drawdown = 0.0
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.trades: list[Trade] = []

        # Load strategies
        self.strategies: dict[str, tuple[Strategy, Individual]] = {}
        for sc in config.strategies:
            strategy = Strategy.load(f"strategies/{sc.strategy_id}.json")
            self.strategies[sc.symbol] = (strategy, strategy.individual)
            logger.info(f"Loaded strategy: {sc.strategy_id} for {sc.symbol}")

        # Get client for historical data
        self.client = get_binance_client(fake=True)

    def get_signal(self, symbol: str, ohlcv: OHLCV, position_side: PositionSide) -> tuple[ActionType, float]:
        """Get signal from strategy."""
        _, individual = self.strategies[symbol]
        signal = individual.get_signal(ohlcv, position_side)
        score = individual.get_total_score(ohlcv)
        return signal, score

    def get_allocation_amount(self) -> float:
        """Get amount to allocate for a new position."""
        return self.balance * self.config.max_allocation_per_coin * self.config.leverage

    def open_position(self, symbol: str, side: PositionSide, price: float, time: datetime) -> None:
        """Open a new position."""
        if symbol in self.positions:
            return  # Already have a position

        allocation = self.get_allocation_amount()
        if allocation <= 0:
            return  # No funds available

        quantity = allocation / price

        # Pay entry fee and lock allocation
        fee = allocation * self.config.fee_rate
        self.balance -= (allocation + fee)  # Lock allocation + pay fee

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_time=time,
            allocated_amount=allocation,
        )

        logger.debug(f"Opened {side.value} {symbol} @ {price:.4f}, qty={quantity:.4f}, alloc=${allocation:.2f}")

    def close_position(self, symbol: str, price: float, time: datetime) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Calculate P&L
        if pos.side == PositionSide.LONG:
            pnl = pos.quantity * (price - pos.entry_price)
        else:
            pnl = pos.quantity * (pos.entry_price - price)

        # Pay exit fee
        exit_notional = pos.quantity * price
        fee = exit_notional * self.config.fee_rate
        net_pnl = pnl - fee

        # Update balance
        self.balance += pos.allocated_amount + net_pnl

        # Track peak and drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Record trade
        return_pct = net_pnl / pos.allocated_amount * 100
        trade = Trade(
            symbol=symbol,
            side=pos.side.value,
            entry_time=pos.entry_time,
            exit_time=time,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            profit=net_pnl,
            fee=fee + (pos.allocated_amount * self.config.fee_rate),  # entry + exit fee
            return_pct=return_pct,
        )
        self.trades.append(trade)

        logger.debug(f"Closed {pos.side.value} {symbol} @ {price:.4f}, PnL=${net_pnl:.2f}")

        del self.positions[symbol]

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Get total portfolio value including open positions."""
        value = self.balance

        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            if pos.side == PositionSide.LONG:
                unrealized = pos.quantity * (price - pos.entry_price)
            else:
                unrealized = pos.quantity * (pos.entry_price - price)
            value += pos.allocated_amount + unrealized

        return value

    def run(self) -> dict[str, Any]:
        """Run the backtest."""
        logger.info("Starting Multi-Strategy Portfolio Backtest")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Strategies: {[s.strategy_id for s in self.config.strategies]}")
        logger.info(f"Starting balance: ${self.config.starting_balance}")
        logger.info(f"Max allocation per coin: {self.config.max_allocation_per_coin*100}%")

        # Load klines for all symbols
        klines_data: dict[str, list] = {}
        interval = self.config.strategies[0].interval

        for sc in self.config.strategies:
            klines = self.client.get_klines(
                symbol=sc.symbol,
                interval=interval,
                startTime=int(self.config.start_date.timestamp() * 1000),
                endTime=int(self.config.end_date.timestamp() * 1000),
                limit=10000,
            )
            klines_data[sc.symbol] = klines
            logger.info(f"Loaded {len(klines)} candles for {sc.symbol}")

        # Find common time range (all symbols must have data)
        min_len = min(len(k) for k in klines_data.values())
        if min_len < 200:
            logger.error("Not enough data (need 200+ candles)")
            return {}

        # Track portfolio value over time
        portfolio_history = []

        # Process each candle
        for i in range(200, min_len):
            current_prices = {}

            for symbol, klines in klines_data.items():
                # Build OHLCV for this symbol
                lookback = klines[i-199:i+1]
                ohlcv = OHLCV(
                    open=np.array([float(k.open) for k in lookback]),
                    high=np.array([float(k.high) for k in lookback]),
                    low=np.array([float(k.low) for k in lookback]),
                    close=np.array([float(k.close) for k in lookback]),
                    volume=np.array([float(k.volume) for k in lookback]),
                )

                current_kline = klines[i]
                current_price = float(current_kline.close)
                current_time = current_kline.open_time
                current_prices[symbol] = current_price

                # Get current position side
                if symbol in self.positions:
                    pos_side = self.positions[symbol].side
                else:
                    pos_side = PositionSide.NONE

                # Get signal
                signal, score = self.get_signal(symbol, ohlcv, pos_side)

                # Execute signal
                if signal == ActionType.LONG:
                    if symbol in self.positions:
                        if self.positions[symbol].side == PositionSide.SHORT:
                            # Close short, open long
                            self.close_position(symbol, current_price, current_time)
                            self.open_position(symbol, PositionSide.LONG, current_price, current_time)
                    else:
                        self.open_position(symbol, PositionSide.LONG, current_price, current_time)

                elif signal == ActionType.SHORT:
                    if symbol in self.positions:
                        if self.positions[symbol].side == PositionSide.LONG:
                            # Close long, open short
                            self.close_position(symbol, current_price, current_time)
                            self.open_position(symbol, PositionSide.SHORT, current_price, current_time)
                    else:
                        self.open_position(symbol, PositionSide.SHORT, current_price, current_time)

                elif signal == ActionType.CLOSE:
                    if symbol in self.positions:
                        self.close_position(symbol, current_price, current_time)

            # Record portfolio value
            portfolio_value = self.get_portfolio_value(current_prices)
            portfolio_history.append({
                "time": current_time,
                "value": portfolio_value,
                "balance": self.balance,
                "positions": len(self.positions),
            })

        # Close any remaining positions at end
        final_time = klines_data[list(klines_data.keys())[0]][-1].open_time
        for symbol in list(self.positions.keys()):
            final_price = current_prices[symbol]
            self.close_position(symbol, final_price, final_time)

        # Calculate results
        total_profit = self.balance - self.config.starting_balance
        total_return = total_profit / self.config.starting_balance * 100
        total_days = (self.config.end_date - self.config.start_date).days
        months = total_days / 30
        monthly_return = total_return / months if months > 0 else 0

        # Per-symbol stats
        symbol_stats = {}
        for symbol in klines_data.keys():
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            if symbol_trades:
                symbol_profit = sum(t.profit for t in symbol_trades)
                symbol_wins = sum(1 for t in symbol_trades if t.profit > 0)
                symbol_stats[symbol] = {
                    "trades": len(symbol_trades),
                    "profit": symbol_profit,
                    "wins": symbol_wins,
                    "win_rate": symbol_wins / len(symbol_trades) * 100,
                }

        results = {
            "strategies": [s.strategy_id for s in self.config.strategies],
            "symbols": list(klines_data.keys()),
            "period_days": total_days,
            "starting_balance": self.config.starting_balance,
            "ending_balance": self.balance,
            "total_profit": total_profit,
            "total_return_pct": total_return,
            "monthly_return_pct": monthly_return,
            "max_drawdown_pct": self.max_drawdown * 100,
            "total_trades": len(self.trades),
            "winning_trades": sum(1 for t in self.trades if t.profit > 0),
            "losing_trades": sum(1 for t in self.trades if t.profit <= 0),
            "total_fees": sum(t.fee for t in self.trades),
            "symbol_stats": symbol_stats,
            "portfolio_history": portfolio_history,
        }

        return results

    def print_results(self, results: dict[str, Any]) -> None:
        """Print backtest results."""
        print("\n" + "=" * 70)
        print("MULTI-STRATEGY PORTFOLIO BACKTEST RESULTS")
        print("=" * 70)
        print(f"Symbols: {', '.join(results['symbols'])}")
        print(f"Period: {results['period_days']} days")
        print(f"Max Allocation: {self.config.max_allocation_per_coin*100}% per coin")
        print(f"Leverage: {self.config.leverage}x")
        print("-" * 70)
        print(f"Starting Balance: ${results['starting_balance']:.2f}")
        print(f"Ending Balance: ${results['ending_balance']:.2f}")
        print(f"Total Profit: ${results['total_profit']:.2f}")
        print("-" * 70)
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Monthly Return: {results['monthly_return_pct']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print("-" * 70)
        print(f"Total Trades: {results['total_trades']}")
        win_rate = results['winning_trades'] / results['total_trades'] * 100 if results['total_trades'] > 0 else 0
        print(f"Winning: {results['winning_trades']} | Losing: {results['losing_trades']} | Win Rate: {win_rate:.1f}%")
        print(f"Total Fees: ${results['total_fees']:.2f}")
        print("=" * 70)

        # Per-symbol breakdown
        print("\nPer-Symbol Performance:")
        print("-" * 50)
        for symbol, stats in results['symbol_stats'].items():
            print(f"  {symbol}: {stats['trades']} trades, ${stats['profit']:.2f} profit, {stats['win_rate']:.1f}% win rate")

        # Monthly breakdown
        print("\nMonthly Portfolio Value:")
        print("-" * 50)
        history = results['portfolio_history']
        monthly_values = {}
        for entry in history:
            month_key = entry['time'].strftime("%Y-%m")
            monthly_values[month_key] = entry['value']

        prev_value = self.config.starting_balance
        for month in sorted(monthly_values.keys()):
            value = monthly_values[month]
            change = value - prev_value
            change_pct = change / prev_value * 100 if prev_value > 0 else 0
            print(f"  {month}: ${value:.2f} ({change_pct:+.1f}%)")
            prev_value = value


def main():
    """Run multi-strategy backtest."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Strategy Portfolio Backtest")
    parser.add_argument("--days", type=int, default=365, help="Backtest period in days")
    parser.add_argument("--balance", type=float, default=1000, help="Starting balance")
    parser.add_argument("--leverage", type=int, default=1, help="Leverage")
    parser.add_argument("--allocation", type=float, default=25, help="Max allocation per coin (%)")

    args = parser.parse_args()

    # Top 5 strategies (major coins only, ETH replaced with BNB)
    strategies = [
        StrategyConfig("LINKUSDT_8bad2afe", "LINKUSDT", "4h"),  # 8.33%
        StrategyConfig("SOLUSDT_ef613e00", "SOLUSDT", "4h"),    # 7.61%
        StrategyConfig("XRPUSDT_10dd1a39", "XRPUSDT", "4h"),    # 7.58%
        StrategyConfig("BNBUSDT_f1d3717e", "BNBUSDT", "4h"),    # 2.61% (low DD)
        StrategyConfig("BTCUSDT_07d8dc58", "BTCUSDT", "4h"),    # 3.96%
    ]

    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    start_date = end_date - timedelta(days=args.days)

    config = PortfolioBacktestConfig(
        strategies=strategies,
        start_date=start_date,
        end_date=end_date,
        starting_balance=args.balance,
        max_allocation_per_coin=args.allocation / 100,
        leverage=args.leverage,
    )

    backtest = MultiStrategyBacktest(config)
    results = backtest.run()

    if results:
        backtest.print_results(results)


if __name__ == "__main__":
    main()
