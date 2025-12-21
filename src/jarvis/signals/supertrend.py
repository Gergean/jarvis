"""SuperTrend signal generator."""

from datetime import datetime
from typing import TYPE_CHECKING

from pandas import Series
from pandas_ta import supertrend

from jarvis.models import ActionType, Kline
from jarvis.settings import settings
from jarvis.signals.base import SignalGenerator
from jarvis.utils import datetime_to_timestamp, floor_dt, interval_to_timedelta

if TYPE_CHECKING:
    from jarvis.client import CachedClient


class SuperTrendSignalGenerator(SignalGenerator):
    """Signal generator using SuperTrend indicator with ATR."""

    def __init__(self, client: "CachedClient", factor: int = 3, atr_period: int = 10) -> None:
        self.client = client
        self.factor = factor
        self.atr_period = atr_period

    def get_signal(self, dt: datetime, symbol: str, interval: str) -> tuple[ActionType, list[Kline], str]:
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(settings.indicator_warmup_start),
            endTime=datetime_to_timestamp(end_dt),
        )

        if klines:
            klines.pop(-1)  ## current kline not closed

        ind = supertrend(
            Series(float(x.high) for x in klines),
            Series(float(x.low) for x in klines),
            Series(float(x.close) for x in klines),
            self.atr_period,
            self.factor,
        )
        directions = ind.iloc[:, 1]

        try:
            if directions.iat[-2] > directions.iat[-1]:
                return (
                    ActionType.SELL,
                    klines,
                    "Direction changed from: %s to %s" % (directions.iat[-2], directions.iat[-1]),
                )

            if directions.iat[-1] > directions.iat[-2]:
                return (
                    ActionType.BUY,
                    klines,
                    "Direction changed from: %s to %s" % (directions.iat[-2], directions.iat[-1]),
                )

        except IndexError:
            return ActionType.STAY, klines, "Error"

        return ActionType.STAY, klines, f"No new signal generated {directions.iat[-2]} {directions.iat[-1]}"
