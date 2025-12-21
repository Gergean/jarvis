"""Trade command for the Jarvis trading system."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

from jarvis.actions import AllInActionGenerator
from jarvis.client import assets_to_usdt, get_binance_client
from jarvis.logging import logger
from jarvis.models import ActionType
from jarvis.settings import notify, settings
from jarvis.signals import VWMASignalGenerator
from jarvis.utils import decimal_as_str


def trade(base_asset: str, trade_assets: list[str], interval: str, investment_ratio: Decimal) -> None:
    """Execute live trading based on signal generators."""
    # Use fake client for klines (reads from cached CSV files)
    # and real client for actual trading operations
    client = get_binance_client()
    action_generator = AllInActionGenerator(
        client, signal_generators={"VWMA": VWMASignalGenerator(client)}, investment_multiplier=investment_ratio
    )
    dt = datetime.now(UTC).replace(tzinfo=None)
    grouped_actions: dict[ActionType, str] = {ActionType.BUY: "", ActionType.SELL: ""}

    for trade_asset in trade_assets:
        symbol = f"{trade_asset}{base_asset}"
        action, base_asset_quantity, quote_asset_quantity, reason = action_generator.get_action(dt, symbol, interval)

        if action not in (ActionType.BUY, ActionType.SELL):
            message = f"Decided to {action.value} for {trade_asset}, Reason: {reason}"
            logger.info(message)
            if settings.debug:
                notify(message)
            continue

        order_side = SIDE_BUY if action == ActionType.BUY else SIDE_SELL

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": order_side,
            "type": ORDER_TYPE_MARKET,
        }

        if quote_asset_quantity:
            params.update({"quoteOrderQty": decimal_as_str(quote_asset_quantity)})

        if base_asset_quantity:
            params.update({"quantity": decimal_as_str(base_asset_quantity)})

        try:
            order = client.create_order(**params)
        except Exception as e:
            logger.info(e)
            continue

        if order is None:
            continue

        # * 100 ETH for 20 USDT (BTCUSDT)

        grouped_actions[action] += "• %s %s for %s %s\n" % (
            order["executedQty"],
            trade_asset,
            order["cummulativeQuoteQty"],
            base_asset,
        )

    message = ""
    if grouped_actions[ActionType.BUY]:
        message += "*I've Bought:*\n\n" + grouped_actions[ActionType.BUY]

    if grouped_actions[ActionType.SELL]:
        message += "*I've Sold:*\n" + grouped_actions[ActionType.SELL]

    if message:
        assets: dict[str, Decimal] = dict(
            [
                (asset["asset"], Decimal(asset["free"]))
                for asset in client.get_account()["balances"]
                if Decimal(asset["free"]) > 0
            ]
        )

        assets_as_usdt = assets_to_usdt(client, assets)
        message += "\n💰 *Your current assets*:\n"
        for asset, value in assets.items():
            message += f"• {asset}: {decimal_as_str(value)}\n"

        message += "\n🤑 *Total Worth*:\n"
        message += f"• {decimal_as_str(assets_as_usdt)} USDT (TWT not Included)"
        notify(message)
