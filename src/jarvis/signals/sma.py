"""SMA signal generator."""

from datetime import datetime
from typing import TYPE_CHECKING

from pandas import DataFrame
from ta.trend import SMAIndicator

from jarvis.models import ActionType, Kline
from jarvis.signals.base import SignalGenerator
from jarvis.utils import datetime_to_timestamp, decimal_as_str, floor_dt, interval_to_timedelta

if TYPE_CHECKING:
    from jarvis.client import CachedClient


class SMASignalGenerator(SignalGenerator):
    """Signal generator using Simple Moving Average."""

    def __init__(self, client: "CachedClient", signal_length: int = 20) -> None:
        self.client = client
        self.length = signal_length

    def get_signal(self, dt: datetime, symbol: str, interval: str) -> tuple[ActionType, list[Kline], str]:
        needed_num_of_candles = 2 * self.length - 1
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta * needed_num_of_candles
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(start_dt),
            endTime=datetime_to_timestamp(end_dt),
        )
        trade_asset = symbol.replace("USDT", "")
        buy_price = self.client.positions[trade_asset]["avg_buy_price"]

        if not klines:
            return ActionType.STAY, klines, "No new signal generated"
        if klines:
            klines.pop(-1)  ## current kline not closed

        df = DataFrame([k.model_dump() for k in klines])
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["sma"] = SMAIndicator(close=df["close"], window=self.length, fillna=False).sma_indicator()

        max_buy_price = df.tail(1).sma.item() + df.tail(1).sma.item() * 0.01
        min_sell_price = df.tail(1).sma.item() - df.tail(1).sma.item() * 0.01

        if df.tail(1).sma.item() <= df.tail(1).close.item() <= max_buy_price and buy_price == 0:
            return (
                ActionType.BUY,
                klines,
                "Close: %s is greater than SMA: %s"
                % (decimal_as_str(df.tail(1).close.item()), decimal_as_str(max_buy_price)),
            )

        if df.tail(1).close.item() <= min_sell_price and df.tail(1).close.item() < df.tail(1).open.item():
            return (
                ActionType.SELL,
                klines,
                "Close: %s is smaller than SMA: %s"
                % (decimal_as_str(df.tail(1).close.item()), decimal_as_str(min_sell_price)),
            )

        return ActionType.STAY, klines, "No new signal generated"
