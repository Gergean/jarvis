#!/usr/bin/env python
"""Jarvis CLI - Cryptocurrency trading automation system."""

import argparse
import doctest
import os
from datetime import datetime
from decimal import Decimal  # Used by trade_parser
from os.path import exists

import sentry_sdk

from jarvis import download, test, trade, trade_with_strategies, train
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

    doctest_parser = subparsers.add_parser("doctest")
    download_parser = subparsers.add_parser("download")
    test_parser = subparsers.add_parser("test")
    trade_parser = subparsers.add_parser("trade")
    trade_ga_parser = subparsers.add_parser("trade-ga")
    train_parser = subparsers.add_parser("train")

    def dt_type(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

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

    # Train parser arguments
    train_parser.add_argument(
        "-s",
        dest="symbol",
        metavar="SYMBOL",
        type=str,
        required=True,
        help="Trading pair to train (e.g., BTCUSDT)",
    )

    train_parser.add_argument(
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval. Default: %(default)s"
    )

    train_parser.add_argument(
        "-st",
        dest="start_dt",
        default=None,
        type=dt_type,
        metavar="START_TIME",
        help="Training start date. Default: 6 months ago.",
    )

    train_parser.add_argument(
        "-et",
        dest="end_dt",
        default=None,
        type=dt_type,
        metavar="END_TIME",
        help="Training end date. Default: now.",
    )

    train_parser.add_argument(
        "-ps",
        dest="population_size",
        default=100,
        type=int,
        metavar="POPULATION_SIZE",
        help="Number of individuals. Default: %(default)s",
    )

    train_parser.add_argument(
        "-g",
        dest="generations",
        default=30,
        type=int,
        metavar="GENERATIONS",
        help="Number of generations. Default: %(default)s",
    )

    train_parser.add_argument(
        "-r",
        dest="rules_per_individual",
        default=8,
        type=int,
        metavar="RULES",
        help="Rules per individual. Default: %(default)s",
    )

    train_parser.add_argument(
        "-l",
        dest="leverage",
        default=1,
        type=int,
        metavar="LEVERAGE",
        help="Futures leverage (1-10). Default: %(default)s",
    )

    train_parser.add_argument(
        "--no-funding",
        dest="funding_enabled",
        action="store_false",
        default=True,
        help="Disable funding fee simulation.",
    )

    train_parser.add_argument(
        "--train-period",
        dest="train_period",
        default="3M",
        type=str,
        metavar="PERIOD",
        help="Training period per window (e.g., 3M, 90d, 12w). Default: %(default)s",
    )

    train_parser.add_argument(
        "--test-period",
        dest="test_period",
        default="1M",
        type=str,
        metavar="PERIOD",
        help="Test period per window (e.g., 1M, 30d, 4w). Default: %(default)s",
    )

    train_parser.add_argument(
        "--step-period",
        dest="step_period",
        default="1M",
        type=str,
        metavar="PERIOD",
        help="Step size between windows (e.g., 1M, 30d, 4w). Default: %(default)s",
    )

    train_parser.add_argument(
        "--no-walk-forward",
        dest="walk_forward",
        action="store_false",
        default=True,
        help="Disable walk-forward validation (not recommended).",
    )

    # Trade-GA parser arguments (trade with GA strategies)
    trade_ga_parser.add_argument(
        "-s",
        dest="strategy_ids",
        nargs="+",
        metavar="STRATEGY_ID",
        type=str,
        required=True,
        help="Strategy IDs to use (e.g., BTCUSDT_fe43f298 ETHUSDT_abc123)",
    )

    trade_ga_parser.add_argument(
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval. Default: %(default)s"
    )

    trade_ga_parser.add_argument(
        "-ir",
        dest="investment_ratio",
        default=Decimal("0.2"),
        type=Decimal,
        metavar="INVESTMENT_RATIO",
        help="Investment ratio. Default: %(default)s",
    )

    trade_ga_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Dry run: show signals without executing trades.",
    )

    # Test parser arguments
    test_parser.add_argument(
        "-s",
        dest="strategy_id",
        metavar="STRATEGY_ID",
        type=str,
        required=True,
        help="Strategy ID to test (e.g., BTCUSDT_abc123)",
    )

    test_parser.add_argument(
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval. Default: %(default)s"
    )

    test_parser.add_argument(
        "-st",
        dest="start_dt",
        default=None,
        type=dt_type,
        metavar="START_TIME",
        help="Test start date. Default: 3 months ago.",
    )

    test_parser.add_argument(
        "-et",
        dest="end_dt",
        default=None,
        type=dt_type,
        metavar="END_TIME",
        help="Test end date. Default: now.",
    )

    test_parser.add_argument(
        "-l",
        dest="leverage",
        default=1,
        type=int,
        metavar="LEVERAGE",
        help="Futures leverage (1-10). Default: %(default)s",
    )

    test_parser.add_argument(
        "--no-funding",
        dest="funding_enabled",
        action="store_false",
        default=True,
        help="Disable funding fee simulation.",
    )

    # Download parser arguments
    download_parser.add_argument(
        "-s",
        dest="symbols",
        nargs="+",
        metavar="SYMBOL",
        type=str,
        required=True,
        help="Trading pairs to download (e.g., BTCUSDT ETHUSDT)",
    )

    download_parser.add_argument(
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval. Default: %(default)s"
    )

    download_parser.add_argument(
        "-st",
        dest="start_dt",
        default=None,
        type=dt_type,
        metavar="START_TIME",
        help="Start date. Default: 1 year ago.",
    )

    download_parser.add_argument(
        "-et",
        dest="end_dt",
        default=None,
        type=dt_type,
        metavar="END_TIME",
        help="End date. Default: now.",
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

    elif kwargs.subparser == "trade":
        trade(kwargs.base_asset, kwargs.trade_assets, kwargs.interval, kwargs.investment_ratio)

    elif kwargs.subparser == "trade-ga":
        trade_with_strategies(
            kwargs.strategy_ids,
            kwargs.interval,
            kwargs.investment_ratio,
            dry_run=kwargs.dry_run,
        )

    elif kwargs.subparser == "train":
        strategy, result = train(
            kwargs.symbol,
            kwargs.interval,
            kwargs.start_dt,
            kwargs.end_dt,
            kwargs.population_size,
            kwargs.generations,
            kwargs.rules_per_individual,
            leverage=kwargs.leverage,
            funding_enabled=kwargs.funding_enabled,
            walk_forward=kwargs.walk_forward,
            train_period=kwargs.train_period,
            test_period=kwargs.test_period,
            step_period=kwargs.step_period,
        )
        print(f"Strategy: {strategy.id}")
        print(f"Return: {result.return_pct:.2f}%")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")

    elif kwargs.subparser == "test":
        result = test(
            kwargs.strategy_id,
            kwargs.interval,
            kwargs.start_dt,
            kwargs.end_dt,
            leverage=kwargs.leverage,
            funding_enabled=kwargs.funding_enabled,
        )
        print(f"Strategy: {result.strategy_id}")
        print(f"Return: {result.return_pct:.2f}%")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")

    elif kwargs.subparser == "download":
        result = download(
            kwargs.symbols,
            kwargs.interval,
            kwargs.start_dt,
            kwargs.end_dt,
        )
        for symbol, count in result.items():
            print(f"{symbol}: {count} klines")


if __name__ == "__main__":
    main()
