"""Base signal generator class."""

from datetime import datetime
from typing import TYPE_CHECKING

from jarvis.models import ActionType, Kline

if TYPE_CHECKING:
    pass


class SignalGenerator:
    """Signal generators responsible for generating Signal objects when it's
    get_signal method is called.
    """

    def get_signal(self, dt: datetime, symbol: str, interval: str) -> tuple[ActionType, list[Kline], str]:
        raise NotImplementedError(
            "Signal classes must have get_signal method that returns Signal, used klines and reason."
        )
