"""Consecutive up/down signal generator."""

from datetime import datetime
from typing import TYPE_CHECKING

from jarvis.logging import logger
from jarvis.models import ActionType, Color, Kline
from jarvis.signals.base import SignalGenerator
from jarvis.utils import datetime_to_timestamp, floor_dt, interval_to_timedelta

if TYPE_CHECKING:
    from jarvis.client import CachedClient


class ConsecutiveUpDownSignalGenerator(SignalGenerator):
    """Signal generator based on consecutive red/green candles."""

    def __init__(self, client: "CachedClient", num_of_reds_to_sell: int = 4, num_of_greens_to_buy: int = 3) -> None:
        self.client = client
        self.num_of_reds_to_sell = num_of_reds_to_sell
        self.num_of_greens_to_buy = num_of_greens_to_buy

    @staticmethod
    def get_colors(klines: list[Kline]) -> list[Color]:
        """Classify klines as RED (bearish) or GREEN (bullish).

        >>> from unittest.mock import Mock
        >>> k1, k2 = Mock(open=100, close=90), Mock(open=100, close=110)
        >>> ConsecutiveUpDownSignalGenerator.get_colors([k1, k2])
        [<Color.RED: 'RED'>, <Color.GREEN: 'GREEN'>]
        """
        return [Color.RED if kline.open > kline.close else Color.GREEN for kline in klines]

    def get_signal(self, dt: datetime, symbol: str, interval: str) -> tuple[ActionType, list[Kline], str]:
        needed_num_of_candles = max([self.num_of_reds_to_sell, self.num_of_greens_to_buy]) + 1
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta * needed_num_of_candles
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(start_dt),
            endTime=datetime_to_timestamp(end_dt),
        )

        # Binance gives non closed kline as last item. We should remove it
        # Before calculation.
        if klines:
            klines.pop(-1)

        if len(klines) < needed_num_of_candles:
            return ActionType.ERR, klines, f"Requested {needed_num_of_candles} klines, {len(klines)} returned"

        colors = self.get_colors(klines)
        logger.debug(
            "Kline colors between %s - %s: " + ("%s, " * needed_num_of_candles)[:-2],
            *[start_dt, end_dt] + [color.value for color in colors],
        )

        sell_colors = colors[-self.num_of_reds_to_sell :]
        if all([c == Color.RED for c in sell_colors]):
            return (
                ActionType.SELL,
                klines,
                "%s reds last %s klines." % (self.num_of_reds_to_sell, self.num_of_reds_to_sell),
            )

        buy_colors = colors[-self.num_of_greens_to_buy :]
        if all([c == Color.GREEN for c in buy_colors]):
            return (
                ActionType.BUY,
                klines,
                "%s greens last %s klines." % (self.num_of_greens_to_buy, self.num_of_greens_to_buy),
            )

        return (
            ActionType.STAY,
            klines,
            f"{self.num_of_greens_to_buy} greens or {self.num_of_reds_to_sell} reds are not matched.",
        )
