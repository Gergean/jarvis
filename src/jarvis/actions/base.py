"""Base action generator class."""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import ring

from jarvis.models import ActionType
from jarvis.signals.base import SignalGenerator

if TYPE_CHECKING:
    from jarvis.client import CachedClient


class ActionGenerator:
    """Action generators responsible for generating decisions by using
    registered SIGNAL_GENERATORS when its get_decision method is called.
    """

    def __init__(self, client: "CachedClient", signal_generators: dict[str, SignalGenerator] | None = None) -> None:
        self.client = client
        self.signal_generators: dict[str, SignalGenerator] = signal_generators or {}

    def __str__(self) -> str:
        """Needed by cache library to create cache key."""
        return "ActionGenerator"

    @ring.lru()  # type: ignore[untyped-decorator]
    def get_symbol_filter(self, symbol: str, filter_type: str) -> dict[str, Any]:
        """
        TODO: Doctests.
        """
        filters: list[dict[str, Any]] = self.client.get_symbol_info(symbol)["filters"]
        result: dict[str, Any] = {}
        for _filter in filters:
            result = _filter
            if _filter["filterType"] == filter_type:
                return result
        return result

    def get_action(
        self, dt: datetime, symbol: str, interval: str
    ) -> tuple[ActionType, Decimal | None, Decimal | None, str]:
        raise NotImplementedError(
            "DecisionGenerator classes must have get_decision method that "
            "returns Action, Quantity, Quote Asset Quantity and Reason"
        )
