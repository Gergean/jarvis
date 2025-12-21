#!/usr/bin/env python
"""Jarvis CLI - Cryptocurrency trading automation system."""

import argparse
import doctest
import os
from datetime import datetime
from decimal import Decimal
from os.path import exists

import sentry_sdk

from jarvis import backtest, trade
from jarvis.logging import logger
from jarvis.settings import get_settings, settings


def main() -> None:
    """Main CLI entry point."""
    import sys

    this_module = sys.modules[__name__]

    parser = argparse.ArgumentParser(
        description="Example: "
        "python jarvis.py -st 2020-01-01T00:00:00 -et "
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

    def dt_type(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    backtest_parser.add_argument(
        "-ba",
        dest="base_asset",
        metavar="BASE_ASSET",
        default="USDT",
        type=str,
        help="The base asset that you want to trade. Default: %(default)s",
    )

    backtest_parser.add_argument(
        "-sa",
        dest="starting_amount",
        default=Decimal(100.0),
        type=Decimal,
        metavar="STARTING_AMOUNT",
        help="Amount of base asset when you start testing. Default: %(default)s",
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
        help="The time that trade will start. Default: %(default)s.",
    )

    backtest_parser.add_argument(
        "-et",
        dest="end_dt",
        default="2020-12-01T00:00:00",
        type=dt_type,
        metavar="END_TIME",
        help="The time that trade end. Default: %(default)s.",
    )

    backtest_parser.add_argument(
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval of klines to check."
    )

    backtest_parser.add_argument(
        "-cr",
        dest="commission_ratio",
        default=Decimal("0.001"),
        type=Decimal,
        metavar="COMMISSION_RATIO",
        help="Commission ratio of platform. Default: %(default)s",
    )

    backtest_parser.add_argument(
        "-ir",
        dest="investment_ratio",
        default=Decimal("0.2"),
        type=Decimal,
        metavar="INVESTMENT_RATIO",
        help="Investment ratio of platform. Default: %(default)s",
    )

    doctest_parser.add_argument(
        "-v", dest="verbose", default=False, action="store_true", help="Gives verbose output when set."
    )

    trade_parser.add_argument(
        "-ba",
        dest="base_asset",
        metavar="BASE_ASSET",
        default="USDT",
        type=str,
        help="The base asset that you want to trade. Default: %(default)s",
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
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval of klines to check."
    )

    trade_parser.add_argument(
        "-ir",
        dest="investment_ratio",
        default=Decimal("0.2"),
        type=Decimal,
        metavar="INVESTMENT_RATIO",
        help="Investment ratio of platform. Default: %(default)s",
    )

    kwargs = parser.parse_args()

    # Load settings from env file if specified
    if kwargs.env_path and exists(kwargs.env_path):
        os.environ["ENV_FILE"] = kwargs.env_path
        get_settings.cache_clear()
        # Re-import settings after clearing cache
        from jarvis import settings as new_settings

        this_module.settings = new_settings  # type: ignore[attr-defined]
        logger.info("Config loaded from %s", kwargs.env_path)

    # Initialize Sentry
    if settings.sentry_dsn:
        sentry_sdk.init(settings.sentry_dsn, traces_sample_rate=1.0)

    if kwargs.subparser == "doctest":
        # Run doctests on all modules
        import jarvis.client
        import jarvis.models
        import jarvis.signals.consecutive
        import jarvis.utils

        results = []
        results.append(doctest.testmod(jarvis.utils, verbose=kwargs.verbose))
        results.append(doctest.testmod(jarvis.models, verbose=kwargs.verbose))
        results.append(doctest.testmod(jarvis.client, verbose=kwargs.verbose))
        results.append(doctest.testmod(jarvis.signals.consecutive, verbose=kwargs.verbose))

        total_failures = sum(r.failed for r in results)
        total_tests = sum(r.attempted for r in results)
        print(f"Ran {total_tests} doctests, {total_failures} failures")

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
        trade(kwargs.base_asset, kwargs.trade_assets, kwargs.interval, kwargs.investment_ratio)


if __name__ == "__main__":
    main()
