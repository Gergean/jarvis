#!../venv/bin/python
import argparse
import doctest
import errno
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from os.path import exists, isfile

import enlighten
from binance.enums import (
    ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)
from dotenv import load_dotenv
from freezegun import freeze_time

from jarvis.client import get_binance_client
from jarvis.enums import ActionType
from jarvis.helpers import (
    interval_to_timedelta,
    dc_to_str,
    num_of_intervals,
    dt_range,
    assets_to_usdt,
)
from jarvis.types import Interval
from jarvis.signals import ActionGeneratorBase

load_dotenv(verbose=True)

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

COMMISSION_RATIO = Decimal(0.001)
INVESTMENT_RATIO = Decimal(0.2)

logger = logging.getLogger(__name__)


def backtest(
    base_asset: str,
    starting_amount: Decimal,
    trade_assets: list[str],
    interval: Interval,
    start_dt: datetime,
    end_dt: datetime,
    commission_ratio: Decimal,
    investment_ratio: Decimal,
    action_generator
):
    bar_manager = enlighten.get_manager()
    client = get_binance_client(
        fake=True,
        extra_params={
            "assets": {base_asset: starting_amount},
            "commission_ratio": commission_ratio,
        },
    )
    if hasattr(client, 'prefetch'):
        client.prefetch(start_dt, end_dt, interval)

    interval: timedelta = interval_to_timedelta(interval)
    total_intervals: int = num_of_intervals(start_dt, end_dt, interval)
    bar = bar_manager.counter(total=total_intervals, desc="Trading",
                              unit="Intervals")

    for idx, dt in enumerate(dt_range(start_dt, end_dt, interval)):
        for idx2, trade_asset in enumerate(trade_assets):
            with freeze_time(dt):
                logger.debug("Datetime frozen to %s", dt)
                symbol = f"{trade_asset}{base_asset}"
                (
                    action,
                    base_asset_quantity,
                    quote_asset_quantity,
                    reason,
                ) = action_generator.get_action(dt, symbol, interval)

                traded = action in (ActionType.BUY, ActionType.SELL)

                logging.log(
                    logging.INFO if traded else logging.DEBUG,
                    logger.info if traded else logger.debug,
                    "For %s, on %s Decided to %s, reason: %s",
                )

                if traded:
                    order_side = (
                        SIDE_BUY if action == ActionType.BUY else SIDE_SELL
                    )

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
                if (
                        idx == total_intervals - 1
                        and idx2 == len(trade_assets) - 1
                ):
                    logger.info("Total worth: %s", client.get_total_usdt())
        bar.update()
    logger.debug(client.asset_report)
    logger.debug(client.get_total_usdt())
    for trade_asset in trade_assets:
        symbol = f"{trade_asset}{base_asset}"
        client.generate_order_chart(symbol, start_dt, end_dt, interval,
                                    base_asset)


def trade(
        base_asset: str,
        trade_assets: list[str],
        interval: Interval,
        investment_ratio: Decimal,
        action_generator: ActionGeneratorBase
):
    """
    trade(USDT, (BTC, LTC), 1H, ActionGenerator)
    """

    client = get_binance_client()
    dt = datetime.utcnow()
    grouped_actions = {ActionType.BUY: "", ActionType.SELL: ""}

    for trade_asset in trade_assets:
        symbol = f"{trade_asset}{base_asset}"
        (
            action,
            base_asset_quantity,
            quote_asset_quantity,
            reason,
        ) = action_generator.get_action(dt, symbol, interval)

        if action not in (ActionType.BUY, ActionType.SELL):
            logger.info(
                "Decided to %s for %s, Reason: %s",
                action.value,
                trade_asset,
                reason,
            )
            continue

        order_side = SIDE_BUY if action == ActionType.BUY else SIDE_SELL

        params = {
            "symbol": symbol,
            "side": order_side,
            "type": ORDER_TYPE_MARKET,
        }

        if quote_asset_quantity:
            params.update(
                {"quoteOrderQty": dc_to_str(quote_asset_quantity)}
            )

        if base_asset_quantity:
            params.update({"quantity": dc_to_str(base_asset_quantity)})

        try:
            order = client.create_order(**params)
        except Exception as e:
            logger.info(e)
            continue

        # * 100 ETH for 20 USDT (BTCUSDT)

        grouped_actions[action] += "• %s %s for %s %s\n" % (
            order["executedQty"],
            trade_asset,
            order["cumulativeQuoteQty"],
            base_asset,
        )

    message = ""
    if grouped_actions[ActionType.BUY]:
        message += "*I've Bought:*\n\n" + grouped_actions[ActionType.BUY]

    if grouped_actions[ActionType.SELL]:
        message += "*I've Sold:*\n" + grouped_actions[ActionType.SELL]

    if message:
        assets = dict(
            [
                (asset["asset"], Decimal(asset["free"]))
                for asset in client.get_account()["balances"]
                if Decimal(asset["free"]) > 0
            ]
        )

        assets_as_usdt = assets_to_usdt(client, assets)
        message += f"\n💰 *Your current assets*:\n"
        for asset, value in assets.items():
            message += f"• {asset}: {dc_to_str(value)}\n"

        message += f"\n🤑 *Total Worth*:\n"
        message += (
            f"• {dc_to_str(assets_as_usdt)} USDT " f"(TWT not Included)"
        )
        notify(message)


def main():
    parser = argparse.ArgumentParser(
        description="Example: "
                    "python src.py -st 2020-01-01T00:00:00 -et "
                    "2020-12-01T00:00:00 -ba USDT -ta BTC ETH -i 1h"
    )

    parser.add_argument(
        "-ep",
        "--env-path",
        default=".env",
        type=str,
        dest="env_path",
        help="File that we can load environment variables from %(default)s",
    )

    subparsers = parser.add_subparsers(dest="subparser")

    backtest_parser = subparsers.add_parser("backtest")
    doctest_parser = subparsers.add_parser("doctest")
    trade_parser = subparsers.add_parser("trade")

    dt_type = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    backtest_parser.add_argument(
        "-ba",
        dest="base_asset",
        metavar="BASE_ASSET",
        default="USDT",
        type=str,
        help="The base asset that you want to trade. " "Default: %(default)s",
    )

    backtest_parser.add_argument(
        "-sa",
        dest="starting_amount",
        default=Decimal(100.0),
        type=Decimal,
        metavar="STARTING_AMOUNT",
        help="Amount of base asset when you start "
             "testing. Default: %(default)s",
    )

    backtest_parser.add_argument(
        "-ta",
        dest="trade_assets",
        nargs="+",
        metavar="TRADE_ASSET",
        help="List of assets that you want to trade against base asset.",
        type=str,
        required=True,
    )

    backtest_parser.add_argument(
        "-st",
        dest="start_dt",
        default="2020-01-01T00:00:00",
        type=dt_type,
        metavar="START_TIME",
        help="The time that trade will start. " "Default: %(default)s.",
    )

    backtest_parser.add_argument(
        "-et",
        dest="end_dt",
        default="2020-12-01T00:00:00",
        type=dt_type,
        metavar="END_TIME",
        help="The time that trade end. " "Default: %(default)s.",
    )

    backtest_parser.add_argument(
        "-i",
        dest="interval",
        default="1h",
        metavar="INTERVAL",
        type=str,
        help="Interval of klines to check.",
    )

    backtest_parser.add_argument(
        "-cr",
        dest="commission_ratio",
        default=COMMISSION_RATIO,
        type=Decimal,
        metavar="COMMISSION_RATIO",
        help="Commission ratio of platform. Default: %s"
             % dc_to_str(COMMISSION_RATIO),
    )

    backtest_parser.add_argument(
        "-ir",
        dest="investment_ratio",
        default=INVESTMENT_RATIO,
        type=Decimal,
        metavar="INVESTMENT_RATIO",
        help="Investment ratio of platform. Default: %s"
             % dc_to_str(INVESTMENT_RATIO),
    )

    doctest_parser.add_argument(
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="Gives verbose output when set.",
    )

    trade_parser.add_argument(
        "-ba",
        dest="base_asset",
        metavar="BASE_ASSET",
        default="USDT",
        type=str,
        help="The base asset that you want to trade. " "Default: %(default)s",
    )

    trade_parser.add_argument(
        "-ta",
        dest="trade_assets",
        nargs="+",
        metavar="TRADE_ASSET",
        help="List of assets that you want to trade against base asset.",
        type=str,
        required=True,
    )

    trade_parser.add_argument(
        "-i",
        dest="interval",
        default="1h",
        metavar="INTERVAL",
        type=str,
        help="Interval of klines to check.",
    )

    trade_parser.add_argument(
        "-ir",
        dest="investment_ratio",
        default=INVESTMENT_RATIO,
        type=Decimal,
        metavar="INVESTMENT_RATIO",
        help="Investment ratio of platform. Default: %s"
             % dc_to_str(INVESTMENT_RATIO),
    )

    kwargs = parser.parse_args()

    if kwargs.env_path:
        if not exists(kwargs.env_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), kwargs.env_path
            )
        load_dotenv(dotenv_path=kwargs.env_path, verbose=True, override=True)
        logger.info("Config loaded from %s", kwargs.env_path)

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

    if kwargs.subparser == "doctest":
        doctest.testmod(verbose=kwargs.verbose)
    elif kwargs.subparser == "backtest":
        backtest(
            kwargs.base_asset,
            kwargs.starting_amount,
            kwargs.trade_assets,
            kwargs.interval,
            kwargs.start_dt,
            kwargs.end_dt,
            kwargs.commission_ratio,
            kwargs.investment_ratio,
        )
        print("Output written to backtest.log")
    elif kwargs.subparser == "trade":
        trade(
            kwargs.base_asset,
            kwargs.trade_assets,
            kwargs.interval,
            kwargs.investment_ratio,
        )


if __name__ == "__main__":
    main()
