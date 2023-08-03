from _decimal import Decimal
from typing import Literal
from dataclasses import dataclass
from datetime import datetime

from .enums import ActionType

Assets = dict[str: Decimal]

Interval = Literal['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h',
                   '8h', '12h', '1d', '3d', '1w', '1M']


@dataclass
class Kline:
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
    ignore: Decimal


@dataclass
class Signal:
    action: ActionType
    klines: list[Kline]
    reason: str

