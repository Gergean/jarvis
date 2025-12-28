#!/usr/bin/env python
"""Jarvis CLI - Cryptocurrency trading automation system."""

import argparse
import doctest
import os
import webbrowser
from datetime import datetime
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
    status,
    test,
    trade,
    train,
)
from jarvis.commands.train import FitnessType
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
    message_parser = subparsers.add_parser("message")
    paper_parser = subparsers.add_parser("paper")
    pinescript_parser = subparsers.add_parser("pinescript")
    plot_parser = subparsers.add_parser("plot")
    status_parser = subparsers.add_parser("status")
    test_parser = subparsers.add_parser("test")
    trade_parser = subparsers.add_parser("trade")
    train_parser = subparsers.add_parser("train")

    def dt_type(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    doctest_parser.add_argument(
        "-v", dest="verbose", default=False, action="store_true", help="Gives verbose output when set."
    )

    # Message parser arguments
    message_parser.add_argument(
        "account_name",
        type=str,
        help="Account name to send message to (e.g., mirat)",
    )
    message_parser.add_argument(
        "message",
        type=str,
        help="Message to send via Telegram",
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
        "--end", dest="end_dt", default=None, type=dt_type,
        help="End date for historical simulation. Default: now.",
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
        "strategy_id",
        metavar="STRATEGY",
        type=str,
        help="Strategy ID (e.g., ETHUSDT_5bdb12c7) or path to JSON file",
    )

    pinescript_parser.add_argument(
        "-o", "--output", dest="output_path", type=str, default=None,
        help="Output path. Default: strategies/{strategy_id}.pine",
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
        "-a", "--account",
        dest="account_name",
        default=None,
        type=str,
        help="Specific account to trade. Default: all accounts in accounts/ directory.",
    )

    trade_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Show signals without executing trades.",
    )

    # Status parser arguments
    status_parser.add_argument(
        "-a", "--account",
        dest="account_name",
        default=None,
        type=str,
        help="Specific account to show status for. Default: all accounts.",
    )
    status_parser.add_argument(
        "--notify",
        dest="send_notification",
        action="store_true",
        default=False,
        help="Send summary to Telegram.",
    )

    # Train parser arguments
    train_parser.add_argument(
        "symbol",
        metavar="SYMBOL",
        type=str,
        help="Trading pair to train (e.g., BTCUSDT)",
    )

    train_parser.add_argument(
        "-i", "--interval", dest="interval", default="1h", type=str, help="Interval. Default: %(default)s"
    )

    train_parser.add_argument(
        "--start", dest="start_dt", default=None, type=dt_type, help="Start date (YYYY-MM-DDTHH:MM:SS). Default: 6 months ago."
    )

    train_parser.add_argument(
        "--end", dest="end_dt", default=None, type=dt_type, help="End date (YYYY-MM-DDTHH:MM:SS). Default: now."
    )

    train_parser.add_argument(
        "-p", "--population", dest="population_size", default=100, type=int, help="Population size. Default: %(default)s"
    )

    train_parser.add_argument(
        "-g", "--generations", dest="generations", default=30, type=int, help="Generations. Default: %(default)s"
    )

    train_parser.add_argument(
        "-r", "--rules", dest="rules_per_individual", default=8, type=int, help="Rules per individual. Default: %(default)s"
    )

    train_parser.add_argument(
        "-l", "--leverage", dest="leverage", default=1, type=int, help="Leverage (1-10). Default: %(default)s"
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

    train_parser.add_argument(
        "--fitness",
        dest="fitness_type",
        default="alpha",
        type=str,
        choices=["legacy", "alpha", "calmar", "sharpe"],
        help="Fitness function: legacy (return-dd), alpha (vs buy&hold), calmar (return/dd), sharpe (consistency). Default: alpha",
    )

    # Test parser arguments
    test_parser.add_argument(
        "strategy_id",
        metavar="STRATEGY",
        type=str,
        help="Strategy ID to test (e.g., BTCUSDT_abc123)",
    )

    test_parser.add_argument(
        "-i", "--interval", dest="interval", default="1h", type=str, help="Interval. Default: %(default)s"
    )

    test_parser.add_argument(
        "--start", dest="start_dt", default=None, type=dt_type, help="Start date. Default: 3 months ago."
    )

    test_parser.add_argument(
        "--end", dest="end_dt", default=None, type=dt_type, help="End date. Default: now."
    )

    test_parser.add_argument(
        "-l", "--leverage", dest="leverage", default=1, type=int, help="Leverage (1-10). Default: %(default)s"
    )

    test_parser.add_argument(
        "--no-funding", dest="funding_enabled", action="store_false", default=True, help="Disable funding fee simulation."
    )

    # Download parser arguments
    download_parser.add_argument(
        "symbols",
        nargs="+",
        metavar="SYMBOL",
        type=str,
        help="Trading pairs to download (e.g., BTCUSDT ETHUSDT)",
    )

    download_parser.add_argument(
        "-i", "--interval", dest="interval", default="1h", type=str, help="Interval. Default: %(default)s"
    )

    download_parser.add_argument(
        "--start", dest="start_dt", default=None, type=dt_type, help="Start date. Default: 1 year ago."
    )

    download_parser.add_argument(
        "--end", dest="end_dt", default=None, type=dt_type, help="End date. Default: now."
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
        import jarvis.genetics.indicators
        import jarvis.models
        import jarvis.utils

        results = []
        results.append(doctest.testmod(jarvis.utils, verbose=kwargs.verbose))
        results.append(doctest.testmod(jarvis.models, verbose=kwargs.verbose))
        results.append(doctest.testmod(jarvis.client, verbose=kwargs.verbose))
        results.append(doctest.testmod(jarvis.genetics.indicators, verbose=kwargs.verbose))

        total_failures = sum(r.failed for r in results)
        total_tests = sum(r.attempted for r in results)
        print(f"Ran {total_tests} doctests, {total_failures} failures")

    elif kwargs.subparser == "trade":
        trade(
            account_name=kwargs.account_name,
            dry_run=kwargs.dry_run,
        )

    elif kwargs.subparser == "status":
        status(account_name=kwargs.account_name, send_notification=kwargs.send_notification)

    elif kwargs.subparser == "message":
        from jarvis.accounts import load_account

        try:
            account = load_account(kwargs.account_name)
            if not account.telegram_dm_id:
                print(f"Account {kwargs.account_name} has no TELEGRAM_DM_ID configured")
            else:
                account.notify(kwargs.message)
                print(f"Message sent to {kwargs.account_name}")
        except FileNotFoundError:
            print(f"Account not found: {kwargs.account_name}")
        except Exception as e:
            print(f"Error: {e}")

    elif kwargs.subparser == "train":
        fitness_type = FitnessType(kwargs.fitness_type)
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
            fitness_type=fitness_type,
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
