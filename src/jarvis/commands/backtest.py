"""Backtest command for the Jarvis trading system."""

from datetime import datetime
from decimal import Decimal

import enlighten
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from freezegun import freeze_time

from jarvis.actions import AllInActionGenerator
from jarvis.client import get_binance_client
from jarvis.logging import logger
from jarvis.models import ActionType
from jarvis.signals import VWMASignalGenerator
from jarvis.utils import dt_range, interval_to_timedelta, num_of_intervals


def backtest(
    base_asset: str,
    starting_amount: Decimal,
    trade_assets: list[str],
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    commission_ratio: Decimal,
    investment_ratio: Decimal,
) -> None:
    """Run backtest simulation on historical data."""
    bar_manager = enlighten.get_manager()

    client = get_binance_client(
        fake=True, extra_params={"assets": {base_asset: starting_amount}, "commission_ratio": commission_ratio}
    )
    action_generator = AllInActionGenerator(
        client,
        signal_generators={
            "VWMA": VWMASignalGenerator(client),
        },
        investment_multiplier=investment_ratio,
    )
    interval_as_timedelta = interval_to_timedelta(interval)

    total_intervals = num_of_intervals(start_dt, end_dt, interval_as_timedelta)

    bar = bar_manager.counter(total=total_intervals, desc="Trading", unit="Intervals")

    for idx, dt in enumerate(dt_range(start_dt, end_dt, interval_as_timedelta)):
        for idx2, trade_asset in enumerate(trade_assets):
            with freeze_time(dt):
                logger.debug("Datetime frozen to %s", dt)
                symbol = f"{trade_asset}{base_asset}"
                action, base_asset_quantity, quote_asset_quantity, reason = action_generator.get_action(
                    dt, symbol, interval
                )

                traded = action in (ActionType.BUY, ActionType.SELL)

                log_func = logger.info if traded else logger.debug
                log_func("For %s, on %s Decided to %s, reason: %s", symbol, dt, action.value, reason)

                if traded:
                    order_side = SIDE_BUY if action == ActionType.BUY else SIDE_SELL

                    params = {
                        "symbol": symbol,
                        "side": order_side,
                        "type": ORDER_TYPE_MARKET,
                        "quantity": 0,
                        "quoteOrderQty": 0,
                    }

                    if quote_asset_quantity:
                        params.update({"quoteOrderQty": quote_asset_quantity})

                    if base_asset_quantity:
                        params.update({"quantity": base_asset_quantity})
                    try:
                        client.create_order(**params)
                    except Exception as e:
                        logger.info(e)
                if idx == total_intervals - 1 and idx2 == len(trade_assets) - 1:
                    logger.info("Total worth: %s", client.get_total_usdt())
        bar.update()
    logger.debug(client.asset_report)
    logger.debug(client.get_total_usdt())
    for trade_asset in trade_assets:
        symbol = f"{trade_asset}{base_asset}"
        client.generate_order_chart(symbol, end_dt, interval, base_asset)
