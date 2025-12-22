"""Technical indicators for the GA trading system.

Uses TA-Lib (C-based) for performance when available, falls back to `ta` (pure Python).
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Try to import talib (C-based, fast), fall back to ta (pure Python, slow)
try:
    import talib

    USE_TALIB = True
except ImportError:
    import ta

    USE_TALIB = False


@dataclass
class Indicator(ABC):
    """Base class for all technical indicators."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> float:
        """Calculate the indicator value from OHLCV dataframe.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            The calculated indicator value (single float).
        """
        pass

    @abstractmethod
    def mutate(self) -> "Indicator":
        """Return a mutated copy of this indicator."""
        pass

    @classmethod
    @abstractmethod
    def random(cls) -> "Indicator":
        """Create a random instance of this indicator."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Indicator":
        """Deserialize from dictionary."""
        pass


@dataclass
class RSI(Indicator):
    """Relative Strength Index indicator (0-100 range)."""

    period: int = 14

    def calculate(self, df: pd.DataFrame) -> float:
        if USE_TALIB:
            close = df["close"].to_numpy()
            result = talib.RSI(close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 50.0
        else:
            rsi = ta.momentum.RSIIndicator(df["close"], window=self.period)
            values = rsi.rsi()
            return float(values.iloc[-1]) if not values.empty else 50.0

    def mutate(self) -> "RSI":
        new_period = self.period + random.randint(-3, 3)
        new_period = max(5, min(30, new_period))  # clamp to 5-30
        return RSI(period=new_period)

    @classmethod
    def random(cls) -> "RSI":
        return cls(period=random.randint(5, 30))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "RSI", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RSI":
        return cls(period=data["params"]["period"])


@dataclass
class SMA(Indicator):
    """Simple Moving Average indicator."""

    period: int = 20

    def calculate(self, df: pd.DataFrame) -> float:
        if USE_TALIB:
            close = df["close"].to_numpy()
            result = talib.SMA(close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 0.0
        else:
            sma = ta.trend.SMAIndicator(df["close"], window=self.period)
            values = sma.sma_indicator()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self) -> "SMA":
        new_period = self.period + random.randint(-5, 5)
        new_period = max(5, min(200, new_period))
        return SMA(period=new_period)

    @classmethod
    def random(cls) -> "SMA":
        return cls(period=random.randint(5, 200))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "SMA", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SMA":
        return cls(period=data["params"]["period"])


@dataclass
class EMA(Indicator):
    """Exponential Moving Average indicator."""

    period: int = 20

    def calculate(self, df: pd.DataFrame) -> float:
        if USE_TALIB:
            close = df["close"].to_numpy()
            result = talib.EMA(close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 0.0
        else:
            ema = ta.trend.EMAIndicator(df["close"], window=self.period)
            values = ema.ema_indicator()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self) -> "EMA":
        new_period = self.period + random.randint(-5, 5)
        new_period = max(5, min(200, new_period))
        return EMA(period=new_period)

    @classmethod
    def random(cls) -> "EMA":
        return cls(period=random.randint(5, 200))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "EMA", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EMA":
        return cls(period=data["params"]["period"])


@dataclass
class MACD(Indicator):
    """MACD line indicator."""

    fast: int = 12
    slow: int = 26
    signal: int = 9

    def calculate(self, df: pd.DataFrame) -> float:
        if USE_TALIB:
            close = df["close"].to_numpy()
            macd_line, _, _ = talib.MACD(
                close, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal
            )
            return float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
        else:
            macd = ta.trend.MACD(
                df["close"],
                window_fast=self.fast,
                window_slow=self.slow,
                window_sign=self.signal,
            )
            values = macd.macd()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self) -> "MACD":
        new_fast = self.fast + random.randint(-2, 2)
        new_slow = self.slow + random.randint(-3, 3)
        new_signal = self.signal + random.randint(-2, 2)
        new_fast = max(5, min(20, new_fast))
        new_slow = max(20, min(50, new_slow))  # Allow up to 50 for slow
        new_signal = max(5, min(15, new_signal))
        if new_fast >= new_slow:
            new_slow = new_fast + 5
        return MACD(fast=new_fast, slow=new_slow, signal=new_signal)

    @classmethod
    def random(cls) -> "MACD":
        fast = random.randint(5, 20)
        slow = random.randint(fast + 5, 40)
        signal = random.randint(5, 15)
        return cls(fast=fast, slow=slow, signal=signal)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "MACD", "params": {"fast": self.fast, "slow": self.slow, "signal": self.signal}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MACD":
        return cls(
            fast=data["params"]["fast"],
            slow=data["params"]["slow"],
            signal=data["params"]["signal"],
        )


@dataclass
class MACD_HIST(Indicator):
    """MACD Histogram indicator."""

    fast: int = 12
    slow: int = 26
    signal: int = 9

    def calculate(self, df: pd.DataFrame) -> float:
        if USE_TALIB:
            close = df["close"].to_numpy()
            _, _, macd_hist = talib.MACD(
                close, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal
            )
            return float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
        else:
            macd = ta.trend.MACD(
                df["close"],
                window_fast=self.fast,
                window_slow=self.slow,
                window_sign=self.signal,
            )
            values = macd.macd_diff()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self) -> "MACD_HIST":
        new_fast = self.fast + random.randint(-2, 2)
        new_slow = self.slow + random.randint(-3, 3)
        new_signal = self.signal + random.randint(-2, 2)
        new_fast = max(5, min(20, new_fast))
        new_slow = max(20, min(40, new_slow))
        new_signal = max(5, min(15, new_signal))
        if new_fast >= new_slow:
            new_slow = new_fast + 5
        return MACD_HIST(fast=new_fast, slow=new_slow, signal=new_signal)

    @classmethod
    def random(cls) -> "MACD_HIST":
        fast = random.randint(5, 20)
        slow = random.randint(fast + 5, 40)
        signal = random.randint(5, 15)
        return cls(fast=fast, slow=slow, signal=signal)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "MACD_HIST", "params": {"fast": self.fast, "slow": self.slow, "signal": self.signal}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MACD_HIST":
        return cls(
            fast=data["params"]["fast"],
            slow=data["params"]["slow"],
            signal=data["params"]["signal"],
        )


@dataclass
class VOLUME(Indicator):
    """Volume indicator (current volume)."""

    period: int = 1  # Not used but kept for consistency

    def calculate(self, df: pd.DataFrame) -> float:
        return float(df["volume"].iloc[-1]) if not df.empty else 0.0

    def mutate(self) -> "VOLUME":
        return VOLUME(period=self.period)

    @classmethod
    def random(cls) -> "VOLUME":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "VOLUME", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VOLUME":
        return cls(period=data["params"].get("period", 1))


@dataclass
class PRICE(Indicator):
    """Current close price indicator."""

    period: int = 1  # Not used but kept for consistency

    def calculate(self, df: pd.DataFrame) -> float:
        return float(df["close"].iloc[-1]) if not df.empty else 0.0

    def mutate(self) -> "PRICE":
        return PRICE(period=self.period)

    @classmethod
    def random(cls) -> "PRICE":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "PRICE", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PRICE":
        return cls(period=data["params"].get("period", 1))


# Registry for indicator types
INDICATOR_TYPES: dict[str, type[Indicator]] = {
    "RSI": RSI,
    "SMA": SMA,
    "EMA": EMA,
    "MACD": MACD,
    "MACD_HIST": MACD_HIST,
    "VOLUME": VOLUME,
    "PRICE": PRICE,
}


def indicator_from_dict(data: dict[str, Any]) -> Indicator:
    """Create an indicator from a dictionary."""
    indicator_type = data["type"]
    if indicator_type not in INDICATOR_TYPES:
        raise ValueError(f"Unknown indicator type: {indicator_type}")
    return INDICATOR_TYPES[indicator_type].from_dict(data)


def random_indicator() -> Indicator:
    """Create a random indicator of any type."""
    indicator_class = random.choice(list(INDICATOR_TYPES.values()))
    return indicator_class.random()
