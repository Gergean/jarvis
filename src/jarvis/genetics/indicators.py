"""Technical indicators for the GA trading system.

Uses TA-Lib (C-based) for performance when available, falls back to `ta` (pure Python).

Design note - Period range based on interval:
Random period values are generated based on the interval, ranging from
MIN_INDICATOR_DAYS to MAX_INDICATOR_DAYS (configured in settings.py).
For example, at 1h interval (24 bars/day) with default 0.1-90 days:
  - min_period = 0.1 * 24 = 2 bars
  - max_period = 90 * 24 = 2160 bars
This ensures indicator periods make sense for the given timeframe.
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from jarvis.settings import (
    MACD_FAST_MAX_DAYS,
    MACD_FAST_MIN_DAYS,
    MACD_SIGNAL_MAX_DAYS,
    MACD_SIGNAL_MIN_DAYS,
    MACD_SLOW_MAX_DAYS,
    MACD_SLOW_MIN_DAYS,
    MAX_INDICATOR_DAYS,
    MIN_INDICATOR_BARS,
    MIN_INDICATOR_DAYS,
)
from jarvis.utils import interval_to_seconds


def _days_to_bars(days: float, interval: str) -> int:
    """Convert days to bar count for given interval.

    Args:
        days: Number of days (e.g., 0.1 to 90)
        interval: Interval string (e.g., "1h", "30m", "4h")

    Returns:
        Number of bars, minimum MIN_INDICATOR_BARS
    """
    seconds_per_bar = interval_to_seconds(interval)
    bars_per_day = 86400 / seconds_per_bar
    return max(MIN_INDICATOR_BARS, int(days * bars_per_day))


def _random_period(interval: str) -> int:
    """Generate random period (bar count) for MIN-MAX days range.

    Uses logarithmic distribution to favor shorter periods while
    still allowing long-term ones.
    """
    # Logarithmic distribution: 10^uniform(log10(min), log10(max))
    log_min = math.log10(MIN_INDICATOR_DAYS)
    log_max = math.log10(MAX_INDICATOR_DAYS)
    days = 10 ** random.uniform(log_min, log_max)
    return _days_to_bars(days, interval)


def _mutate_period(period: int, interval: str) -> int:
    """Mutate period by ±20%, clamped to MIN-MAX days range."""
    min_period = _days_to_bars(MIN_INDICATOR_DAYS, interval)
    max_period = _days_to_bars(MAX_INDICATOR_DAYS, interval)
    new_period = int(period * random.uniform(0.8, 1.2))
    return max(min_period, min(max_period, new_period))


def _random_macd_periods(interval: str) -> tuple[int, int, int]:
    """Generate random MACD periods (fast, slow, signal) ensuring fast < slow."""
    # Fast EMA: short-term
    fast_days = 10 ** random.uniform(
        math.log10(MACD_FAST_MIN_DAYS), math.log10(MACD_FAST_MAX_DAYS)
    )
    fast = _days_to_bars(fast_days, interval)

    # Slow EMA: must be > fast, use slow range but ensure minimum gap
    slow_min_days = max(MACD_SLOW_MIN_DAYS, fast_days * 1.5)  # At least 1.5x fast
    slow_days = 10 ** random.uniform(
        math.log10(slow_min_days), math.log10(MACD_SLOW_MAX_DAYS)
    )
    slow = _days_to_bars(slow_days, interval)

    # Ensure slow > fast (edge case protection)
    if slow <= fast:
        slow = fast + max(2, fast // 2)

    # Signal line: typically shorter
    signal_days = 10 ** random.uniform(
        math.log10(MACD_SIGNAL_MIN_DAYS), math.log10(MACD_SIGNAL_MAX_DAYS)
    )
    signal = _days_to_bars(signal_days, interval)

    return fast, slow, signal


def _mutate_macd_periods(fast: int, slow: int, signal: int, interval: str) -> tuple[int, int, int]:
    """Mutate MACD periods by ±20%, maintaining fast < slow constraint."""
    # Mutate each by ±20%
    new_fast = int(fast * random.uniform(0.8, 1.2))
    new_slow = int(slow * random.uniform(0.8, 1.2))
    new_signal = int(signal * random.uniform(0.8, 1.2))

    # Clamp to valid ranges
    fast_min = _days_to_bars(MACD_FAST_MIN_DAYS, interval)
    fast_max = _days_to_bars(MACD_FAST_MAX_DAYS, interval)
    slow_min = _days_to_bars(MACD_SLOW_MIN_DAYS, interval)
    slow_max = _days_to_bars(MACD_SLOW_MAX_DAYS, interval)
    signal_min = _days_to_bars(MACD_SIGNAL_MIN_DAYS, interval)
    signal_max = _days_to_bars(MACD_SIGNAL_MAX_DAYS, interval)

    new_fast = max(fast_min, min(fast_max, new_fast))
    new_slow = max(slow_min, min(slow_max, new_slow))
    new_signal = max(signal_min, min(signal_max, new_signal))

    # Ensure slow > fast
    if new_slow <= new_fast:
        new_slow = new_fast + max(2, new_fast // 2)

    return new_fast, new_slow, new_signal

# Try to import talib (C-based, fast), fall back to ta (pure Python, slow)
try:
    import talib

    USE_TALIB = True
except ImportError:
    import pandas as pd
    import ta

    USE_TALIB = False


class OHLCV(NamedTuple):
    """OHLCV data as numpy arrays - much faster than DataFrame."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


@dataclass
class Indicator(ABC):
    """Base class for all technical indicators."""

    @abstractmethod
    def calculate(self, ohlcv: OHLCV) -> float:
        """Calculate the indicator value from OHLCV arrays.

        Args:
            ohlcv: OHLCV named tuple with numpy arrays

        Returns:
            The calculated indicator value (single float).
        """
        pass

    @abstractmethod
    def mutate(self, interval: str = "1h") -> "Indicator":
        """Return a mutated copy of this indicator."""
        pass

    @classmethod
    @abstractmethod
    def random(cls, interval: str = "1h") -> "Indicator":
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

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        """Return the valid (min, max) range for this indicator's values.

        Used to constrain target values during rule generation/mutation.
        """
        return (0.0, 100.0)  # Default range


@dataclass
class RSI(Indicator):
    """Relative Strength Index indicator (0-100 range)."""

    period: int = 14

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (0.0, 100.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.RSI(ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 50.0
        else:
            rsi = ta.momentum.RSIIndicator(pd.Series(ohlcv.close), window=self.period)
            values = rsi.rsi()
            return float(values.iloc[-1]) if not values.empty else 50.0

    def mutate(self, interval: str = "1h") -> "RSI":
        return RSI(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "RSI":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "RSI", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RSI":
        return cls(period=data["params"]["period"])


@dataclass
class SMA(Indicator):
    """Simple Moving Average indicator."""

    period: int = 20

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.SMA(ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 0.0
        else:
            sma = ta.trend.SMAIndicator(pd.Series(ohlcv.close), window=self.period)
            values = sma.sma_indicator()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self, interval: str = "1h") -> "SMA":
        return SMA(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "SMA":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "SMA", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SMA":
        return cls(period=data["params"]["period"])


@dataclass
class EMA(Indicator):
    """Exponential Moving Average indicator."""

    period: int = 20

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.EMA(ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 0.0
        else:
            ema = ta.trend.EMAIndicator(pd.Series(ohlcv.close), window=self.period)
            values = ema.ema_indicator()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self, interval: str = "1h") -> "EMA":
        return EMA(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "EMA":
        return cls(period=_random_period(interval))

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

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            macd_line, _, _ = talib.MACD(
                ohlcv.close, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal
            )
            return float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
        else:
            macd = ta.trend.MACD(
                pd.Series(ohlcv.close),
                window_fast=self.fast,
                window_slow=self.slow,
                window_sign=self.signal,
            )
            values = macd.macd()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self, interval: str = "1h") -> "MACD":
        new_fast, new_slow, new_signal = _mutate_macd_periods(
            self.fast, self.slow, self.signal, interval
        )
        return MACD(fast=new_fast, slow=new_slow, signal=new_signal)

    @classmethod
    def random(cls, interval: str = "1h") -> "MACD":
        fast, slow, signal = _random_macd_periods(interval)
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

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            _, _, macd_hist = talib.MACD(
                ohlcv.close, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal
            )
            return float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
        else:
            macd = ta.trend.MACD(
                pd.Series(ohlcv.close),
                window_fast=self.fast,
                window_slow=self.slow,
                window_sign=self.signal,
            )
            values = macd.macd_diff()
            return float(values.iloc[-1]) if not values.empty else 0.0

    def mutate(self, interval: str = "1h") -> "MACD_HIST":
        new_fast, new_slow, new_signal = _mutate_macd_periods(
            self.fast, self.slow, self.signal, interval
        )
        return MACD_HIST(fast=new_fast, slow=new_slow, signal=new_signal)

    @classmethod
    def random(cls, interval: str = "1h") -> "MACD_HIST":
        fast, slow, signal = _random_macd_periods(interval)
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
    """Volume ratio indicator - current volume / average volume.

    Returns a normalized ratio (typically 0.5-2.0 range):
    - 1.0 = current volume equals average
    - >1.0 = above average volume
    - <1.0 = below average volume
    """

    period: int = 20  # Period for average volume calculation

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (0.0, 5.0)  # Ratio, typically 0.5-2.0 but can spike

    def calculate(self, ohlcv: OHLCV) -> float:
        if len(ohlcv.volume) < self.period:
            return 1.0  # Not enough data, return neutral
        avg_volume = np.mean(ohlcv.volume[-self.period :])
        if avg_volume == 0:
            return 1.0
        return float(ohlcv.volume[-1] / avg_volume)

    def mutate(self, interval: str = "1h") -> "VOLUME":
        return VOLUME(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "VOLUME":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "VOLUME", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VOLUME":
        return cls(period=data["params"].get("period", 20))


@dataclass
class PRICE(Indicator):
    """Current close price indicator. DEPRECATED - causes overfitting."""

    period: int = 1

    def calculate(self, ohlcv: OHLCV) -> float:
        return float(ohlcv.close[-1]) if len(ohlcv.close) > 0 else 0.0

    def mutate(self, interval: str = "1h") -> "PRICE":
        return PRICE(period=self.period)

    @classmethod
    def random(cls, interval: str = "1h") -> "PRICE":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "PRICE", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PRICE":
        return cls(period=data["params"].get("period", 1))


# ============================================================================
# NORMALIZED INDICATORS (price-independent, safe for long-term use)
# ============================================================================


@dataclass
class STOCH(Indicator):
    """Stochastic Oscillator %K (0-100 range).

    Measures where the close is relative to the high-low range.
    """

    fastk_period: int = 14
    slowk_period: int = 3

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (0.0, 100.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            slowk, _ = talib.STOCH(
                ohlcv.high,
                ohlcv.low,
                ohlcv.close,
                fastk_period=self.fastk_period,
                slowk_period=self.slowk_period,
                slowd_period=3,
            )
            return float(slowk[-1]) if not np.isnan(slowk[-1]) else 50.0
        else:
            stoch = ta.momentum.StochasticOscillator(
                pd.Series(ohlcv.high),
                pd.Series(ohlcv.low),
                pd.Series(ohlcv.close),
                window=self.fastk_period,
                smooth_window=self.slowk_period,
            )
            values = stoch.stoch()
            return float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else 50.0

    def mutate(self, interval: str = "1h") -> "STOCH":
        return STOCH(
            fastk_period=_mutate_period(self.fastk_period, interval),
            slowk_period=max(2, int(self.slowk_period * random.uniform(0.8, 1.2))),
        )

    @classmethod
    def random(cls, interval: str = "1h") -> "STOCH":
        return cls(
            fastk_period=_random_period(interval),
            slowk_period=random.randint(2, 5),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "STOCH", "params": {"fastk_period": self.fastk_period, "slowk_period": self.slowk_period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "STOCH":
        return cls(
            fastk_period=data["params"]["fastk_period"],
            slowk_period=data["params"].get("slowk_period", 3),
        )


@dataclass
class STOCHRSI(Indicator):
    """Stochastic RSI (0-100 range).

    RSI of RSI - more sensitive than regular RSI.
    """

    period: int = 14
    fastk_period: int = 5

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (0.0, 100.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            fastk, _ = talib.STOCHRSI(
                ohlcv.close,
                timeperiod=self.period,
                fastk_period=self.fastk_period,
                fastd_period=3,
            )
            return float(fastk[-1]) if not np.isnan(fastk[-1]) else 50.0
        else:
            stochrsi = ta.momentum.StochRSIIndicator(
                pd.Series(ohlcv.close),
                window=self.period,
                smooth1=self.fastk_period,
                smooth2=3,
            )
            values = stochrsi.stochrsi_k()
            val = float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else 0.5
            return val * 100  # ta returns 0-1, we want 0-100

    def mutate(self, interval: str = "1h") -> "STOCHRSI":
        return STOCHRSI(
            period=_mutate_period(self.period, interval),
            fastk_period=max(2, int(self.fastk_period * random.uniform(0.8, 1.2))),
        )

    @classmethod
    def random(cls, interval: str = "1h") -> "STOCHRSI":
        return cls(period=_random_period(interval), fastk_period=random.randint(3, 8))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "STOCHRSI", "params": {"period": self.period, "fastk_period": self.fastk_period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "STOCHRSI":
        return cls(
            period=data["params"]["period"],
            fastk_period=data["params"].get("fastk_period", 5),
        )


@dataclass
class WILLR(Indicator):
    """Williams %R (-100 to 0 range).

    Similar to Stochastic but inverted. -100 = oversold, 0 = overbought.
    """

    period: int = 14

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (-100.0, 0.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.WILLR(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else -50.0
        else:
            willr = ta.momentum.WilliamsRIndicator(
                pd.Series(ohlcv.high),
                pd.Series(ohlcv.low),
                pd.Series(ohlcv.close),
                lbp=self.period,
            )
            values = willr.williams_r()
            return float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else -50.0

    def mutate(self, interval: str = "1h") -> "WILLR":
        return WILLR(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "WILLR":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "WILLR", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WILLR":
        return cls(period=data["params"]["period"])


@dataclass
class ADX(Indicator):
    """Average Directional Index (0-100 range).

    Measures trend strength regardless of direction.
    <20 = weak/no trend, 20-40 = developing, 40-60 = strong, >60 = very strong.
    """

    period: int = 14

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (0.0, 100.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.ADX(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 25.0
        else:
            adx = ta.trend.ADXIndicator(
                pd.Series(ohlcv.high),
                pd.Series(ohlcv.low),
                pd.Series(ohlcv.close),
                window=self.period,
            )
            values = adx.adx()
            return float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else 25.0

    def mutate(self, interval: str = "1h") -> "ADX":
        return ADX(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "ADX":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ADX", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ADX":
        return cls(period=data["params"]["period"])


@dataclass
class MFI(Indicator):
    """Money Flow Index (0-100 range).

    Like RSI but incorporates volume. >80 = overbought, <20 = oversold.
    """

    period: int = 14

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (0.0, 100.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.MFI(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 50.0
        else:
            mfi = ta.volume.MFIIndicator(
                pd.Series(ohlcv.high),
                pd.Series(ohlcv.low),
                pd.Series(ohlcv.close),
                pd.Series(ohlcv.volume),
                window=self.period,
            )
            values = mfi.money_flow_index()
            return float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else 50.0

    def mutate(self, interval: str = "1h") -> "MFI":
        return MFI(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "MFI":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "MFI", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MFI":
        return cls(period=data["params"]["period"])


@dataclass
class CCI(Indicator):
    """Commodity Channel Index (typically -200 to +200).

    Measures deviation from statistical mean. >100 = overbought, <-100 = oversold.
    """

    period: int = 20

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (-200.0, 200.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.CCI(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 0.0
        else:
            cci = ta.trend.CCIIndicator(
                pd.Series(ohlcv.high),
                pd.Series(ohlcv.low),
                pd.Series(ohlcv.close),
                window=self.period,
            )
            values = cci.cci()
            return float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else 0.0

    def mutate(self, interval: str = "1h") -> "CCI":
        return CCI(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "CCI":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "CCI", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CCI":
        return cls(period=data["params"]["period"])


@dataclass
class ROC(Indicator):
    """Rate of Change (percentage).

    Measures percentage change over N periods. Typically -10 to +10 range.
    """

    period: int = 10

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (-30.0, 30.0)

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            result = talib.ROC(ohlcv.close, timeperiod=self.period)
            return float(result[-1]) if not np.isnan(result[-1]) else 0.0
        else:
            roc = ta.momentum.ROCIndicator(pd.Series(ohlcv.close), window=self.period)
            values = roc.roc()
            return float(values.iloc[-1]) if not values.empty and not pd.isna(values.iloc[-1]) else 0.0

    def mutate(self, interval: str = "1h") -> "ROC":
        return ROC(period=_mutate_period(self.period, interval))

    @classmethod
    def random(cls, interval: str = "1h") -> "ROC":
        return cls(period=_random_period(interval))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ROC", "params": {"period": self.period}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ROC":
        return cls(period=data["params"]["period"])


@dataclass
class BBANDS_B(Indicator):
    """Bollinger Bands %B (0-1 range, can exceed).

    Position within Bollinger Bands. 0 = at lower band, 1 = at upper band.
    <0 = below lower band (oversold), >1 = above upper band (overbought).
    """

    period: int = 20
    nbdev: float = 2.0

    @classmethod
    def get_range(cls) -> tuple[float, float]:
        return (-0.5, 1.5)  # Can exceed 0-1 in volatile markets

    def calculate(self, ohlcv: OHLCV) -> float:
        if USE_TALIB:
            upper, middle, lower = talib.BBANDS(
                ohlcv.close, timeperiod=self.period, nbdevup=self.nbdev, nbdevdn=self.nbdev
            )
            if np.isnan(upper[-1]) or np.isnan(lower[-1]):
                return 0.5
            band_width = upper[-1] - lower[-1]
            if band_width == 0:
                return 0.5
            return float((ohlcv.close[-1] - lower[-1]) / band_width)
        else:
            bb = ta.volatility.BollingerBands(
                pd.Series(ohlcv.close), window=self.period, window_dev=int(self.nbdev)
            )
            pband = bb.bollinger_pband()
            return float(pband.iloc[-1]) if not pband.empty and not pd.isna(pband.iloc[-1]) else 0.5

    def mutate(self, interval: str = "1h") -> "BBANDS_B":
        return BBANDS_B(
            period=_mutate_period(self.period, interval),
            nbdev=max(1.0, min(3.0, self.nbdev * random.uniform(0.9, 1.1))),
        )

    @classmethod
    def random(cls, interval: str = "1h") -> "BBANDS_B":
        return cls(period=_random_period(interval), nbdev=random.uniform(1.5, 2.5))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "BBANDS_B", "params": {"period": self.period, "nbdev": self.nbdev}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBANDS_B":
        return cls(
            period=data["params"]["period"],
            nbdev=data["params"].get("nbdev", 2.0),
        )


# Registry for indicator types
# Only normalized indicators that don't depend on absolute price levels.
# Removed: PRICE, SMA, EMA, MACD, MACD_HIST (all return absolute values that
# become stale as price moves away from training period levels).
# All normalized indicators (no absolute price-dependent ones)
INDICATOR_TYPES: dict[str, type[Indicator]] = {
    "RSI": RSI,
    "STOCH": STOCH,
    "STOCHRSI": STOCHRSI,
    "WILLR": WILLR,
    "ADX": ADX,
    "MFI": MFI,
    "CCI": CCI,
    "ROC": ROC,
    "BBANDS_B": BBANDS_B,
    "VOLUME": VOLUME,
}


def indicator_from_dict(data: dict[str, Any]) -> Indicator:
    """Create an indicator from a dictionary."""
    indicator_type = data["type"]
    if indicator_type not in INDICATOR_TYPES:
        raise ValueError(f"Unknown indicator type: {indicator_type}")
    return INDICATOR_TYPES[indicator_type].from_dict(data)


def random_indicator(interval: str = "1h") -> Indicator:
    """Create a random indicator of any type."""
    indicator_class = random.choice(list(INDICATOR_TYPES.values()))
    return indicator_class.random(interval)
