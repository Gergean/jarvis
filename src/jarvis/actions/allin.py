"""All-in action generator."""

from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from jarvis.actions.base import ActionGenerator
from jarvis.logging import logger
from jarvis.models import ActionType
from jarvis.signals import SignalGenerator
from jarvis.utils import decimal_as_str, floor_to_step

if TYPE_CHECKING:
    from jarvis.client import CachedClient


class AllInActionGenerator(ActionGenerator):
    """Action generator that invests a percentage of available balance per trade."""

    def __init__(
        self,
        client: "CachedClient",
        signal_generators: dict[str, SignalGenerator] | None = None,
        investment_multiplier: Decimal = Decimal(1),
    ) -> None:
        super().__init__(client, signal_generators=signal_generators)
        self.investment_multiplier = investment_multiplier

    def get_action(
        self, dt: datetime, symbol: str, interval: str
    ) -> tuple[ActionType, Decimal | None, Decimal | None, str]:
        symbol_info = self.client.get_symbol_info(symbol)
        base_asset = symbol_info["baseAsset"]
        quote_asset = symbol_info["quoteAsset"]

        signals = []
        for name, generator in self.signal_generators.items():
            signal, klines, reason = generator.get_signal(dt, symbol, interval)
            logger.debug("%s returned %s on %s. Reason: %s", name, signal.value, dt, reason)
            signals.append(signal)
        most_common_signal = Counter(signals).most_common(1)[0][0]

        base_asset_quantity = None
        quote_asset_quantity = None
        market_lot_info = self.get_symbol_filter(symbol, "MARKET_LOT_SIZE")
        market_min_quantity = Decimal(market_lot_info["minQty"])

        lot_info = self.get_symbol_filter(symbol, "LOT_SIZE")
        min_quantity = Decimal(lot_info["minQty"])
        step_size = Decimal(lot_info["stepSize"])

        min_notional_info = self.get_symbol_filter(symbol, "NOTIONAL")
        min_notional = Decimal(min_notional_info["minNotional"])

        if most_common_signal == ActionType.SELL:
            base_asset_quantity = Decimal(self.client.get_asset_balance(asset=base_asset)["free"])
            base_asset_quantity = floor_to_step(base_asset_quantity, step_size)
            avg_price = self.client.get_avg_price(symbol=symbol).get("price")

            if avg_price is None:
                return ActionType.ERR, None, None, "Average price problem"

            base_asset_value = base_asset_quantity * Decimal(avg_price)

            if base_asset_quantity <= market_min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would sell but there are not enough "
                    f"{base_asset} in wallet (" + decimal_as_str(base_asset_quantity) + ") (MARKET_LOT_SIZE)",
                )

            if base_asset_quantity <= min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would sell but there are not enough "
                    f"{base_asset} in wallet (" + decimal_as_str(base_asset_quantity) + ") (LOT_SIZE)",
                )

            if base_asset_value <= min_notional:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would sell but there are not enough "
                    f"{base_asset} in wallet (" + decimal_as_str(base_asset_quantity) + ") (MIN_NOTIONAL)",
                )

        if most_common_signal == ActionType.BUY:
            tradable_asset_quantity = (
                Decimal(self.client.get_asset_balance(asset=quote_asset)["free"]) * self.investment_multiplier
            )

            quote_asset_quantity = int(tradable_asset_quantity / step_size) * step_size

            if quote_asset_quantity <= market_min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would buy but there are not enough "
                    f"{quote_asset} in wallet (" + decimal_as_str(quote_asset_quantity) + ") (MARKET_LOT_SIZE)",
                )

            if quote_asset_quantity <= min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would buy but there are not enough "
                    f"{quote_asset} in wallet (" + decimal_as_str(quote_asset_quantity) + ") (LOT_SIZE)",
                )

            if quote_asset_quantity <= min_notional:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would buy but there are not enough "
                    f"{quote_asset} in wallet (" + decimal_as_str(quote_asset_quantity) + ") (MIN_NOTIONAL)",
                )

        return (
            most_common_signal,
            base_asset_quantity,
            quote_asset_quantity,
            f"All signals that I have says {most_common_signal.value}",
        )
