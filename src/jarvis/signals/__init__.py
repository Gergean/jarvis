"""Signal generators for the Jarvis trading system."""

from jarvis.signals.base import SignalGenerator
from jarvis.signals.consecutive import ConsecutiveUpDownSignalGenerator
from jarvis.signals.sma import SMASignalGenerator
from jarvis.signals.supertrend import SuperTrendSignalGenerator
from jarvis.signals.vwma import VWMASignalGenerator

__all__ = [
    "SignalGenerator",
    "SuperTrendSignalGenerator",
    "VWMASignalGenerator",
    "SMASignalGenerator",
    "ConsecutiveUpDownSignalGenerator",
]
