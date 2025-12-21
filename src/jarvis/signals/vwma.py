"""VWMA signal generator."""

from datetime import datetime
from typing import TYPE_CHECKING

from pandas import Series
from pandas_ta import vwma

from jarvis.models import ActionType, Kline
from jarvis.settings import settings
from jarvis.signals.base import SignalGenerator
from jarvis.utils import datetime_to_timestamp, decimal_as_str, floor_dt, interval_to_timedelta

if TYPE_CHECKING:
    from jarvis.client import CachedClient


class VWMASignalGenerator(SignalGenerator):
    """Signal generator using Volume-Weighted Moving Average."""

    def __init__(self, client: "CachedClient", signal_length: int = 20, base_asset: str = "USDT") -> None:
        self.client = client
        self.length = signal_length
        self.base_asset = base_asset

    def get_signal(self, dt: datetime, symbol: str, interval: str) -> tuple[ActionType, list[Kline], str]:
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(settings.indicator_warmup_start),
            endTime=datetime_to_timestamp(end_dt),
        )
        buy_price = 0

        if klines:
            klines.pop(-1)  ## current kline not closed

        ohlc4 = Series((float(x.close) + float(x.open) + float(x.high) + float(x.low)) / 4 for x in klines)

        vwma_arr = vwma(ohlc4, Series(float(x.volume) for x in klines), self.length)

        action = ActionType.STAY
        actions = [action]

        for i in range(self.length, len(ohlc4)):
            highest_vwma = max(vwma_arr[i : -self.length + i : -1])
            lowest_vwma = min(vwma_arr[i : -self.length + i : -1])
            close = ohlc4.iat[i]
            if close > highest_vwma and actions[-1] != ActionType.BUY and buy_price == 0:
                action = ActionType.BUY
                buy_price = close
            elif close < lowest_vwma and actions[-1] != ActionType.SELL and buy_price != 0:
                action = ActionType.SELL
                buy_price = 0
            else:
                action = ActionType.STAY

            actions.append(action)

        if actions[-1] == ActionType.BUY:
            return (
                ActionType.BUY,
                klines,
                "Close: %s is greater than VWMA: %s" % (decimal_as_str(ohlc4.iat[-1]), decimal_as_str(highest_vwma)),
            )

        if actions[-1] == ActionType.SELL:
            return (
                ActionType.SELL,
                klines,
                "Close: %s is smaller than VWMA: %s" % (decimal_as_str(ohlc4.iat[-1]), decimal_as_str(lowest_vwma)),
            )

        return ActionType.STAY, klines, "No new signal generated"
