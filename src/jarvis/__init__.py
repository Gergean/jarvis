"""Jarvis - Cryptocurrency trading automation system."""

from jarvis.client import CachedClient, get_binance_client
from jarvis.commands import (
    download,
    paper_info,
    paper_init,
    paper_list,
    paper_trade,
    pinescript,
    plot,
    test,
    trade,
    trade_with_strategies,
    train,
)
from jarvis.models import ActionType, Color, FakeResponse, Kline, Position
from jarvis.settings import Settings, get_settings, notify, settings

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
    # Commands
    "download",
    "paper_info",
    "paper_init",
    "paper_list",
    "paper_trade",
    "pinescript",
    "plot",
    "test",
    "trade",
    "trade_with_strategies",
    "train",
]
