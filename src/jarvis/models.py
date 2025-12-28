"""Data models for the Jarvis trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from jarvis.utils import timestamp_to_datetime

__all__ = [
    # Enums
    "ActionType",
    "Color",
    "PositionSide",
    # Models
    "FakeResponse",
    "FuturesPosition",
    "Kline",
    "Position",
    # Constants (re-exported from settings)
    "DEFAULT_LEVERAGE",
    "FUNDING_FEE_RATE",
    "FUNDING_INTERVAL_HOURS",
    "FUTURES_MAKER_FEE",
    "FUTURES_TAKER_FEE",
    "MAX_LEVERAGE",
    # Utilities (re-exported)
    "validate_leverage",
]


class Kline(BaseModel):
    """Binance kline (candlestick) data.

    >>> raw = [
    ...     1499040000000, "0.01634790", "0.80000000", "0.01575800",
    ...     "0.01577100", "148976.11427815", 1499644799999, "2434.19055334",
    ...     308, "1756.87402397", "28.46694368", "17928899.62484339"
    ... ]
    >>> kline = Kline.from_raw(raw)
    >>> kline.open_time
    datetime.datetime(2017, 7, 3, 0, 0)
    >>> kline.open
    Decimal('0.01634790')
    >>> kline.volume
    Decimal('148976.11427815')
    >>> kline.num_of_trades
    308
    """

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_asset_volume: Decimal
    num_of_trades: int
    taker_buy_base_asset_volume: Decimal
    taker_buy_quote_asset_volume: Decimal
    ignore: Decimal

    @classmethod
    def from_raw(cls, raw: list[Any]) -> "Kline":
        """Create Kline from raw Binance API response list."""
        return cls(
            open_time=timestamp_to_datetime(raw[0]),
            open=Decimal(raw[1]),
            high=Decimal(raw[2]),
            low=Decimal(raw[3]),
            close=Decimal(raw[4]),
            volume=Decimal(raw[5]),
            close_time=timestamp_to_datetime(raw[6]),
            quote_asset_volume=Decimal(raw[7]),
            num_of_trades=int(raw[8]),
            taker_buy_base_asset_volume=Decimal(raw[9]),
            taker_buy_quote_asset_volume=Decimal(raw[10]),
            ignore=Decimal(raw[11]),
        )


class Color(Enum):
    RED = "RED"
    GREEN = "GREEN"


class ActionType(Enum):
    LONG = "LONG"    # Open long position
    SHORT = "SHORT"  # Open short position
    CLOSE = "CLOSE"  # Close current position
    STAY = "STAY"    # Keep current position
    ERR = "ERR"      # Error state


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Position:
    """Represents a trading position for a symbol.

    >>> pos = Position(symbol='BTCUSDT', spent=Decimal('100'), amount=Decimal('0.01'))
    >>> pos.symbol
    'BTCUSDT'
    >>> pos.spent
    Decimal('100')
    """

    symbol: str
    spent: Decimal
    amount: Decimal


# Re-export constants from settings for backwards compatibility
from jarvis.settings import (  # noqa: E402, F401
    DEFAULT_LEVERAGE,
    FUNDING_FEE_RATE,
    FUNDING_INTERVAL_HOURS,
    FUTURES_MAKER_FEE,
    FUTURES_TAKER_FEE,
    MAX_LEVERAGE,
)
from jarvis.utils import validate_leverage  # noqa: E402, F401


@dataclass
class FuturesPosition:
    """Represents a futures trading position.

    >>> pos = FuturesPosition(
    ...     symbol='BTCUSDT',
    ...     side=PositionSide.LONG,
    ...     entry_price=Decimal('50000'),
    ...     quantity=Decimal('0.1'),
    ...     margin=Decimal('500'),
    ...     leverage=10
    ... )
    >>> pos.notional_value
    Decimal('5000.0')
    >>> pos.liquidation_price  # Long at 10x: ~10% drop
    Decimal('45000.0')
    """

    symbol: str
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal  # Always positive
    margin: Decimal    # USDT margin used
    leverage: int

    @property
    def notional_value(self) -> Decimal:
        """Position value = quantity * entry_price."""
        return self.quantity * self.entry_price

    @property
    def liquidation_price(self) -> Decimal:
        """Calculate approximate liquidation price based on leverage.

        Simplified formula - real exchanges use more complex calculations.
        """
        if self.leverage <= 1:
            # No liquidation at 1x
            return Decimal("0") if self.side == PositionSide.LONG else Decimal("999999999")

        margin_ratio = Decimal("1") / Decimal(self.leverage)
        if self.side == PositionSide.LONG:
            # Long liquidates when price drops by margin ratio
            return self.entry_price * (1 - margin_ratio)
        else:
            # Short liquidates when price rises by margin ratio
            return self.entry_price * (1 + margin_ratio)


class FakeResponse:
    """Fake HTTP response for simulating Binance API errors.

    >>> resp = FakeResponse(400, {'code': -1102, 'msg': 'error'})
    >>> resp.status_code
    400
    >>> resp.json()
    {'code': -1102, 'msg': 'error'}
    """

    def __init__(self, status_code: int, _dict: dict[str, Any]) -> None:
        self.status_code = status_code
        self._dict = _dict
        self.text = str(_dict)

    def json(self) -> dict[str, Any]:
        return self._dict
