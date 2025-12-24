#!/usr/bin/env python
"""Jarvis CLI - Cryptocurrency trading automation system."""

import argparse
import doctest
import os
import webbrowser
from datetime import datetime
from decimal import Decimal  # Used by trade_parser
from os.path import exists

import sentry_sdk

from jarvis import (
    download,
    paper_info,
    paper_init,
    paper_list,
    paper_trade,
    pinescript,
    plot,
    test,
    trade,
    trade_with_strategies,
    train,
)
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
    paper_parser = subparsers.add_parser("paper")
    pinescript_parser = subparsers.add_parser("pinescript")
    plot_parser = subparsers.add_parser("plot")
    test_parser = subparsers.add_parser("test")
    trade_parser = subparsers.add_parser("trade")
    trade_ga_parser = subparsers.add_parser("trade-ga")
    train_parser = subparsers.add_parser("train")

    def dt_type(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    doctest_parser.add_argument(
        "-v", dest="verbose", default=False, action="store_true", help="Gives verbose output when set."
    )

    # Paper trading parser arguments
    paper_subparsers = paper_parser.add_subparsers(dest="paper_command")

    paper_init_parser = paper_subparsers.add_parser("init", help="Create a new paper wallet")
    paper_init_parser.add_argument(
        "wallet_id",
        type=str,
        help="Unique wallet identifier",
    )
    paper_init_parser.add_argument(
        "-b",
        "--balance",
        dest="balance",
        type=float,
        default=100.0,
        help="Initial balance in USD. Default: %(default)s",
    )
    paper_init_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        action="append",
        required=True,
        help="Trading config as SYMBOL:INTERVAL (e.g., -c BTCUSDT:1h -c ETHUSDT:4h)",
    )
    paper_init_parser.add_argument(
        "-s",
        "--seed",
        dest="seed_strategy",
        type=str,
        default=None,
        help="Seed strategy ID (e.g., ETHUSDT_abc123). Required for paper trading.",
    )

    paper_trade_parser = paper_subparsers.add_parser("trade", help="Run paper trading")
    paper_trade_parser.add_argument(
        "wallet_id",
        type=str,
        help="Wallet identifier",
    )
    paper_trade_parser.add_argument(
        "-et",
        dest="end_dt",
        default=None,
        type=dt_type,
        metavar="END_TIME",
        help="End date (for testing with historical data). Default: now.",
    )

    paper_info_parser = paper_subparsers.add_parser("info", help="Show wallet info and stats")
    paper_info_parser.add_argument(
        "wallet_id",
        type=str,
        help="Wallet identifier",
    )

    paper_subparsers.add_parser("list", help="List all paper wallets")

    # Pinescript parser arguments
    pinescript_parser.add_argument(
        "-s",
        dest="strategy_id",
        metavar="STRATEGY_ID",
        type=str,
        required=True,
        help="Strategy ID (e.g., ETHUSDT_5bdb12c7) or path to JSON file",
    )

    pinescript_parser.add_argument(
        "-o",
        dest="output_path",
        metavar="OUTPUT_PATH",
        type=str,
        default=None,
        help="Output path for Pine Script file. Default: strategies/{strategy_id}.pine",
    )

    # Plot parser arguments
    plot_parser.add_argument(
        "-s",
        dest="strategy_id",
        metavar="STRATEGY_ID",
        type=str,
        required=True,
        help="Strategy ID (e.g., BTCUSDT_abc123) or path to JSON file",
    )

    plot_parser.add_argument(
        "-i", dest="interval", default="1h", metavar="INTERVAL", type=str, help="Interval. Default: %(default)s"
    )

    plot_parser.add_argument(
        "-st",
        dest="start_dt",
        default=None,
        type=dt_type,
        metavar="START_TIME",
        help="Chart start date. Default: 3 months ago.",
    )

    plot_parser.add_argument(
        "-et",
        dest="end_dt",
        default=None,
        type=dt_type,
        metavar="END_TIME",
        help="Chart end date. Default: now.",
    )

    plot_parser.add_argument(
        "-o",
        dest="output_path",
        metavar="OUTPUT_PATH",
        type=str,
        default=None,
        help="Output HTML file path. Default: charts/{strategy_id}.html",
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

    train_parser.add_argument(
        "--seed",
        dest="seed_strategy",
        default=None,
        type=str,
        metavar="STRATEGY_PATH",
        help="Path to strategy JSON to use as seed (e.g., strategies/ETHUSDT_abc123.json)",
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
            seed_strategy=kwargs.seed_strategy,
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

    elif kwargs.subparser == "pinescript":
        output_path = pinescript(kwargs.strategy_id, kwargs.output_path)
        print(f"Pine Script saved: {output_path}")

    elif kwargs.subparser == "plot":
        output_path = plot(
            kwargs.strategy_id,
            kwargs.interval,
            kwargs.start_dt,
            kwargs.end_dt,
            kwargs.output_path,
        )
        print(f"Chart saved: {output_path}")
        webbrowser.open(f"file://{os.path.abspath(output_path)}")

    elif kwargs.subparser == "paper":
        if kwargs.paper_command == "init":
            wallet = paper_init(kwargs.wallet_id, kwargs.balance, kwargs.config, kwargs.seed_strategy)
            print(f"Wallet '{wallet['id']}' created with ${wallet['balance']} balance")
            print(f"Config: {wallet['config']}")
            if wallet.get("seed_strategies"):
                print(f"Seed strategies: {list(wallet['seed_strategies'].keys())}")

        elif kwargs.paper_command == "trade":
            end_dt = kwargs.end_dt.replace(tzinfo=datetime.now().astimezone().tzinfo) if kwargs.end_dt else None
            wallet = paper_trade(kwargs.wallet_id, end_dt=end_dt)
            print(f"Balance: ${wallet['balance']:.2f}")
            print(f"Open positions: {len(wallet['positions'])}")

        elif kwargs.paper_command == "info":
            stats = paper_info(kwargs.wallet_id)
            print(f"Wallet: {stats['wallet_id']}")
            print(f"Initial: ${stats['initial_balance']:.2f}")
            print(f"Current: ${stats['current_balance']:.2f}")
            print(f"PnL: ${stats['total_pnl']:.2f} ({stats['total_pnl_pct']:.2f}%)")
            print(f"Trades: {stats['total_trades']} ({stats['win_rate']:.1f}% win rate)")
            print(f"Open positions: {stats['open_positions']}")

        elif kwargs.paper_command == "list":
            wallets = paper_list()
            if wallets:
                print("Paper wallets:")
                for w in wallets:
                    print(f"  - {w}")
            else:
                print("No paper wallets found")


if __name__ == "__main__":
    main()
