"""Jarvis - Cryptocurrency trading automation system."""

from jarvis.actions import ActionGenerator, AllInActionGenerator
from jarvis.client import CachedClient, get_binance_client
from jarvis.commands import backtest, trade
from jarvis.models import ActionType, Color, FakeResponse, Kline, Position
from jarvis.settings import Settings, get_settings, notify, settings
from jarvis.signals import (
    ConsecutiveUpDownSignalGenerator,
    SignalGenerator,
    SMASignalGenerator,
    SuperTrendSignalGenerator,
    VWMASignalGenerator,
)

__all__ = [
    # Models
    "ActionType",
    "Color",
    "FakeResponse",
    "Kline",
    "Position",
    # Settings
    "Settings",
    "get_settings",
    "notify",
    "settings",
    # Client
    "CachedClient",
    "get_binance_client",
    # Signals
    "ConsecutiveUpDownSignalGenerator",
    "SignalGenerator",
    "SMASignalGenerator",
    "SuperTrendSignalGenerator",
    "VWMASignalGenerator",
    # Actions
    "ActionGenerator",
    "AllInActionGenerator",
    # Commands
    "backtest",
    "trade",
]
