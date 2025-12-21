"""Data models for the Jarvis trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from jarvis.utils import timestamp_to_datetime


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
    BUY = "BUY"
    SELL = "SELL"
    STAY = "STAY"
    ERR = "ERR"


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
