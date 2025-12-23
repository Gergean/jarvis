"""Portfolio model for managing multiple trading strategies."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from jarvis.genetics.strategy import Strategy
from jarvis.models import ActionType
from jarvis.utils import datetime_to_timestamp, dt_range, interval_to_timedelta


@dataclass
class SymbolAllocation:
    """Allocation configuration for a single symbol."""

    symbol: str
    weight: float  # 0.0 to 1.0
    strategy: Strategy | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolAllocation":
        return cls(
            symbol=data["symbol"],
            weight=data["weight"],
        )


@dataclass
class Portfolio:
    """Portfolio managing multiple trading strategies."""

    total_capital: Decimal
    allocation_strategy: str  # "equal", "risk_adjusted", "performance_based"
    symbols: list[SymbolAllocation] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at": self.updated_at.isoformat(),
            "total_capital": str(self.total_capital),
            "allocation_strategy": self.allocation_strategy,
            "symbols": [s.to_dict() for s in self.symbols],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Portfolio":
        return cls(
            updated_at=datetime.fromisoformat(data["updated_at"]),
            total_capital=Decimal(data["total_capital"]),
            allocation_strategy=data["allocation_strategy"],
            symbols=[SymbolAllocation.from_dict(s) for s in data["symbols"]],
        )

    def save(self, filepath: str | Path = "strategies/portfolio.json") -> Path:
        """Save portfolio configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.updated_at = datetime.utcnow()

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: str | Path = "strategies/portfolio.json") -> "Portfolio":
        """Load portfolio configuration from JSON file."""
        filepath = Path(filepath)

        with open(filepath) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def load_strategies(self, directory: str | Path = "strategies") -> None:
        """Load strategy files for all symbols."""
        for allocation in self.symbols:
            if Strategy.exists(allocation.symbol, directory):
                allocation.strategy = Strategy.load(allocation.symbol, directory)

    def add_symbol(self, symbol: str, weight: float | None = None) -> None:
        """Add a symbol to the portfolio."""
        # Check if already exists
        for alloc in self.symbols:
            if alloc.symbol == symbol:
                if weight is not None:
                    alloc.weight = weight
                return

        self.symbols.append(SymbolAllocation(symbol=symbol, weight=weight or 0.0))
        self._rebalance_weights()

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the portfolio."""
        self.symbols = [s for s in self.symbols if s.symbol != symbol]
        self._rebalance_weights()

    def _rebalance_weights(self) -> None:
        """Rebalance weights based on allocation strategy."""
        if not self.symbols:
            return

        if self.allocation_strategy == "equal":
            weight = 1.0 / len(self.symbols)
            for alloc in self.symbols:
                alloc.weight = weight

        elif self.allocation_strategy == "performance_based":
            # Weight by return percentage (higher return = higher weight)
            total_return = sum(
                max(0.01, alloc.strategy.performance.return_pct) if alloc.strategy else 1.0 for alloc in self.symbols
            )
            for alloc in self.symbols:
                ret = alloc.strategy.performance.return_pct if alloc.strategy else 1.0
                alloc.weight = max(0.01, ret) / total_return

        elif self.allocation_strategy == "risk_adjusted":
            # Weight inversely by drawdown (lower drawdown = higher weight)
            # Sharpe-like: return / drawdown
            scores = []
            for alloc in self.symbols:
                if alloc.strategy:
                    ret = alloc.strategy.performance.return_pct
                    dd = max(1.0, alloc.strategy.performance.max_drawdown_pct)
                    scores.append(ret / dd)
                else:
                    scores.append(1.0)

            total_score = sum(max(0.01, s) for s in scores)
            for alloc, score in zip(self.symbols, scores):
                alloc.weight = max(0.01, score) / total_score

    def backtest(
        self,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        commission_ratio: Decimal = Decimal("0.001"),
        investment_ratio: Decimal = Decimal("0.2"),
    ) -> dict[str, Any]:
        """Run portfolio backtest across all symbols.

        Returns:
            Dictionary with backtest results including equity curve and metrics.
        """
        from jarvis.client import get_binance_client

        # Initialize assets per symbol based on weights
        symbol_capital: dict[str, Decimal] = {}
        symbol_assets: dict[str, dict[str, Decimal]] = {}

        for alloc in self.symbols:
            capital = self.total_capital * Decimal(str(alloc.weight))
            symbol_capital[alloc.symbol] = capital

            base_asset = "USDT"
            trade_asset = alloc.symbol[:-4] if alloc.symbol.endswith("USDT") else alloc.symbol[:-3]
            symbol_assets[alloc.symbol] = {base_asset: capital, trade_asset: Decimal("0")}

        # Track portfolio equity over time
        interval_td = interval_to_timedelta(interval)
        all_dts = list(dt_range(start_dt, end_dt, interval_td))

        equity_history: list[tuple[datetime, float]] = []
        peak_equity = self.total_capital
        max_drawdown = Decimal("0")
        max_drawdown_pct = 0.0
        total_trades = 0

        # Create client for data
        client = get_binance_client(
            fake=True,
            extra_params={"assets": {"USDT": self.total_capital}, "commission_ratio": commission_ratio},
        )

        for dt in all_dts:
            end_ts = datetime_to_timestamp(dt)
            portfolio_equity = Decimal("0")

            for alloc in self.symbols:
                if not alloc.strategy:
                    continue

                symbol = alloc.symbol
                base_asset = "USDT"
                trade_asset = symbol[:-4] if symbol.endswith("USDT") else symbol[:-3]
                assets = symbol_assets[symbol]

                # Get klines
                try:
                    klines = client.get_klines(symbol=symbol, interval=interval, limit=100, endTime=end_ts)
                    if not klines or len(klines) < 50:
                        continue
                except Exception:
                    continue

                df = pd.DataFrame([k.model_dump() for k in klines])
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)

                price = Decimal(str(df["close"].iloc[-1]))

                # Calculate equity for this symbol
                equity = assets.get(base_asset, Decimal("0"))
                if assets.get(trade_asset, Decimal("0")) > 0:
                    equity += assets[trade_asset] * price
                portfolio_equity += equity

                # Get signal and execute
                signal = alloc.strategy.individual.get_signal(df)

                if signal == ActionType.BUY:
                    quote_balance = assets.get(base_asset, Decimal("0"))
                    spend_amount = quote_balance * investment_ratio
                    if spend_amount > 0 and price > 0:
                        after_fee = spend_amount * (1 - commission_ratio)
                        buy_qty = after_fee / price
                        assets[base_asset] = quote_balance - spend_amount
                        assets[trade_asset] = assets.get(trade_asset, Decimal("0")) + buy_qty
                        total_trades += 1

                elif signal == ActionType.SELL:
                    sell_qty = assets.get(trade_asset, Decimal("0"))
                    if sell_qty > 0 and price > 0:
                        proceeds = sell_qty * price
                        after_fee = proceeds * (1 - commission_ratio)
                        assets[trade_asset] = Decimal("0")
                        assets[base_asset] = assets.get(base_asset, Decimal("0")) + after_fee
                        total_trades += 1

            # Track drawdown
            if portfolio_equity > 0:
                if portfolio_equity > peak_equity:
                    peak_equity = portfolio_equity
                drawdown = peak_equity - portfolio_equity
                drawdown_pct = float(drawdown / peak_equity * 100)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct

                equity_history.append((dt, float(portfolio_equity)))

        # Final equity
        final_equity = Decimal("0")
        for alloc in self.symbols:
            symbol = alloc.symbol
            base_asset = "USDT"
            trade_asset = symbol[:-4] if symbol.endswith("USDT") else symbol[:-3]
            assets = symbol_assets[symbol]

            final_equity += assets.get(base_asset, Decimal("0"))
            # Add remaining trade assets at last known price
            if assets.get(trade_asset, Decimal("0")) > 0:
                # Get last price
                try:
                    klines = client.get_klines(
                        symbol=symbol, interval=interval, limit=1, endTime=datetime_to_timestamp(end_dt)
                    )
                    if klines:
                        price = Decimal(str(klines[-1].close))
                        final_equity += assets[trade_asset] * price
                except Exception:
                    pass

        return_pct = float((final_equity - self.total_capital) / self.total_capital * 100)

        return {
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "starting_capital": float(self.total_capital),
            "final_equity": float(final_equity),
            "return_pct": return_pct,
            "peak_equity": float(peak_equity),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": total_trades,
            "equity_history": equity_history,
        }

    def __repr__(self) -> str:
        symbols_str = ", ".join(f"{s.symbol}:{s.weight:.0%}" for s in self.symbols)
        return f"Portfolio(capital={self.total_capital}, strategy={self.allocation_strategy}, symbols=[{symbols_str}])"
