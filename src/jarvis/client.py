"""Binance client wrapper with caching and offline support."""

import csv
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import wraps
from os import makedirs
from os.path import dirname, exists
from pathlib import Path
from typing import Any, TypeVar

import mplfinance as mpf
import pandas as pd
import ring
from binance.client import Client
from binance.enums import (
    KLINE_INTERVAL_1MINUTE,
    ORDER_RESP_TYPE_RESULT,
    ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)
from binance.exceptions import BinanceAPIException

from jarvis.logging import logger
from jarvis.models import FakeResponse, Kline
from jarvis.settings import settings
from jarvis.utils import (
    DAY_AS_TIMEDELTA,
    assets_to_str,
    calculate_avg_buy_price,
    ceil_dt,
    datetime_to_timestamp,
    decimal_as_str,
    dt_range,
    floor_dt,
    interval_to_seconds,
    interval_to_timedelta,
    ratio_as_str,
    timestamp_to_datetime,
)

T = TypeVar("T")

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_day_file_path(symbol: str, interval: str, day: datetime) -> str:
    """
    >>> get_day_file_path('BTCUSDT', '1h', datetime(2020, 1, 1)).endswith('data/binance/BTCUSDT/1h/20200101.csv')
    True
    """
    file_name = day.strftime("%Y%m%d.csv")
    return str(PROJECT_ROOT / "data" / "binance" / symbol / interval / file_name)


def create_day_file(client: Client | None, symbol: str, interval: str, day: datetime) -> str:
    """Fetch klines from given symbol and interval and write to day file.

    TODO: Doctests.
    """
    if client is None:
        file_path = get_day_file_path(symbol, interval, day)
        raise FileNotFoundError(f"CSV file not found and no client available to fetch data: {file_path}")

    interval_delta = interval_to_timedelta(interval)
    file_path = get_day_file_path(symbol, interval, day)

    start = datetime(year=day.year, month=day.month, day=day.day)
    end = start + timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)

    # Binance gives maximum 500 klines at one time.
    num_of_klines = (end - start) / interval_delta
    num_of_iterations = round(max(num_of_klines / 500, 1))
    limit_delta = interval_delta * 500

    klines: list[Any] = []
    for idx in range(num_of_iterations):
        page_start = start + (interval_delta * idx * 500)
        page_end = min(page_start + limit_delta, end)
        logger.debug(
            "Fetching kline data from Binance (%s / %s) (%s / %s).", idx + 1, num_of_iterations, page_start, page_end
        )
        klines.extend(
            client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=datetime_to_timestamp(page_start),
                endTime=datetime_to_timestamp(page_end),
            )
        )

    file_dir = dirname(file_path)
    if not exists(file_dir):
        makedirs(file_dir)
        logger.debug(f"{file_dir} created.")

    with open(file_path, "w", newline="") as csv_file:
        field_names = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset volume",
        ]
        writer = csv.writer(csv_file)
        writer.writerow(field_names)
        for kline in klines:
            writer.writerow(kline)
    logger.debug(f"{file_path} written with {len(klines)} records.")
    return file_path


@ring.lru()  # type: ignore[untyped-decorator]
def load_day_file(symbol: str, interval: str, day: datetime, file_path: str | None = None) -> list[list[Any]]:
    """Load day file, make type conversions and return list of lists."""
    file_path = file_path or get_day_file_path(symbol, interval, day)
    lines: list[list[Any]] = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header.
        for _line in reader:
            open_time = int(_line[0])
            close_time = int(_line[6])
            line: list[Any] = list(map(float, _line))
            line[0] = open_time
            line[6] = close_time
            lines.append(line)
    logger.debug("Loaded day file %s", file_path)
    return lines


def get_expected_num_of_lines_in_day_file(
    day: datetime, interval_delta: timedelta, utc_now: datetime | None = None
) -> int:
    """Calculate expected number of klines in a day file based on interval.

    >>> day = datetime(2020, 1, 1)
    >>> now = datetime(2020, 1, 1, 12, 0)
    >>> get_expected_num_of_lines_in_day_file(day, timedelta(hours=1), now)
    12
    >>> get_expected_num_of_lines_in_day_file(day, timedelta(hours=4), now)
    3
    """
    utc_now = utc_now or datetime.now(UTC)
    delta = min(utc_now - day, DAY_AS_TIMEDELTA)
    return int(delta / interval_delta)


def get_klines_from_day_files(
    client: Client | None,
    symbol: str,
    interval: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    limit: int = 500,
) -> list[Kline]:
    """
    1. If end_ts is not given, make it timestamp of utcnow.
    2. Floor end_ts to the given interval.

       For example, if end_ts 2020-1-1 14:38 and interval is 1h, it will be
       floored to 2020-1-1 14:00.

    2. If start_ts is not given, calculate it by going back limit * interval

       For example if end_ts is 2020-1-2 00:00:00, interval is 4h and limit
       is 3, start_ts will be:

       2020-1-2 - (4h * 3) = 2020-1-1 12:00:00

    3. If start_ts is given, ceil start_ts.

    3. Day of start_dt is start_day, day of end_ts is end_day

    4. Loop over days from start_day to end_day.

    5. Try to get day_file, create it by calling create_day_file.

    6. Accumulate kline data from day_files by excluding the records that are
       not between start_ts and end_ts.

    7. Return Klines.

              0     4     8     12    16    20
    Intervals \\.....\\.....\\.....\\.....\\.....\\
              start_ts →\\...........\\← end_ts
                   (ceil)→\\.....\\←(floor)

    Note: These tests require CSV data files to be present.
    They are skipped by default as test data may not be available.

    >>> sts = datetime_to_timestamp(datetime(2020, 1, 1))  # doctest: +SKIP
    >>> ets = datetime_to_timestamp(datetime(2020, 1, 1, 5))  # doctest: +SKIP
    >>> klines = get_klines_from_day_files(None, 'BNBBTC', '1h', start_ts=sts, end_ts=ets, limit=1)  # doctest: +SKIP
    >>> len(klines)  # doctest: +SKIP
    1
    """

    interval_as_timedelta = timedelta(seconds=interval_to_seconds(interval))

    if not end_ts:
        end_ts = datetime_to_timestamp(datetime.now(UTC))
    end_dt_orig = timestamp_to_datetime(end_ts)
    end_dt = floor_dt(end_dt_orig, interval_as_timedelta)
    logger.debug("End time snapped by %s: %s -> %s", interval, end_dt_orig, end_dt)

    if start_ts:
        start_dt_orig = timestamp_to_datetime(start_ts)
        start_dt = ceil_dt(start_dt_orig, interval_as_timedelta)
        logger.debug("Start time snapped by %s: %s -> %s", interval, start_dt_orig, start_dt)
    else:
        # If start time is not defined step back interval * limit hours to
        # find start time of request.
        start_dt = end_dt - (interval_as_timedelta * (limit - 1))
        logger.debug("Start time calculated as: %s", start_dt)

    start_day = floor_dt(start_dt, DAY_AS_TIMEDELTA)
    end_day = floor_dt(end_dt, DAY_AS_TIMEDELTA)
    day_dts = list(dt_range(start_day, end_day, DAY_AS_TIMEDELTA))

    utc_now = datetime.now(UTC).replace(tzinfo=None)
    today = floor_dt(utc_now, DAY_AS_TIMEDELTA)

    results = []

    for day_dt in day_dts:
        file_path = get_day_file_path(symbol, interval, day_dt)

        if not exists(file_path):
            file_path = create_day_file(client, symbol, interval, day_dt)

        lines = load_day_file(symbol, interval, day_dt, file_path)

        if day_dt == today:
            expected_num_of_lines = get_expected_num_of_lines_in_day_file(
                day_dt, interval_as_timedelta, utc_now=utc_now
            )

            if len(lines) < expected_num_of_lines:  # Fetch from API again.
                logger.debug(
                    "Day file %s has %s records which are below expected %s",
                    file_path,
                    len(lines),
                    expected_num_of_lines,
                )

                file_path = create_day_file(client, symbol, interval, day_dt)
                load_day_file.delete(symbol, interval, day_dt, file_path)
                lines = load_day_file(symbol, interval, day_dt, file_path)

        for line in lines:
            open_ts, close_ts = line[0], line[6]
            open_dt = timestamp_to_datetime(open_ts)
            if start_dt <= open_dt <= end_dt:
                results.append(line)
            # else: This is too much even for debug logs. But can be needed.
            #   logger.debug('Ignored line between %s - %s', open_dt,
            #   close_dt)
    return [Kline.from_raw(k) for k in results[:limit]]


def raise_binance_api_exception(status_code: int, code: int, msg: str) -> None:
    """Helper to raise BinanceAPIException with correct signature for current library version."""
    response = FakeResponse(status_code, {"code": code, "msg": msg})
    raise BinanceAPIException(response, status_code, response.text)


def binance_api_exception_on_missing_params(*required_params: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that raises BinanceAPIException if required params are missing.

    >>> def fun(a=None, b=None):
    ...     print('OK')
    >>> fun = binance_api_exception_on_missing_params('b')(fun)
    >>> fun(a=True)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    binance.exceptions.BinanceAPIException: ...Mandatory parameter b was not sent...
    >>> fun(b=True)
    OK
    """

    def inner(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for key in required_params:
                if key not in kwargs:
                    raise_binance_api_exception(
                        400, -1102, f"Mandatory parameter {key} was not sent, was empty / null, or malformed."
                    )
            return func(*args, **kwargs)

        return wrapper

    return inner


def assets_to_usdt(client: "CachedClient", assets: dict[str, Decimal]) -> Decimal:
    """Convert asset holdings to total USDT value."""
    total = Decimal(0.0)
    for asset, count in assets.items():
        if asset == "TWT":
            continue
        if asset == "USDT":
            total += count
        else:
            symbol = f"{asset}USDT"
            price = client.get_avg_price(symbol=symbol).get("price")
            if price is None:
                continue
            total += Decimal(price) * count
    return total


class CachedClient:
    """A caching proxy for Binance client that reads klines from local CSV files.

    When client is provided, falls back to real API for missing data.
    When client is None, works fully offline with default values.

    >>> client = CachedClient()
    >>> client.get_symbol_info('BTCUSDT')['filters'][0]['filterType']
    'LOT_SIZE'
    """

    # Default symbol info for offline mode
    DEFAULT_SYMBOL_INFO = {
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.00001000", "maxQty": "9000.00000000", "stepSize": "0.00001000"},
            {
                "filterType": "MARKET_LOT_SIZE",
                "minQty": "0.00000000",
                "maxQty": "1000.00000000",
                "stepSize": "0.00000000",
            },
            {
                "filterType": "NOTIONAL",
                "minNotional": "5.00000000",
                "applyMinToMarket": True,
                "maxNotional": "9000000.00000000",
                "applyMaxToMarket": False,
            },
        ],
    }

    def __init__(
        self,
        client: Client | None = None,
        assets: dict[str, Decimal] | None = None,
        commission_ratio: Decimal | None = None,
    ) -> None:
        self.client = client
        self.assets: defaultdict[str, Decimal] = defaultdict(Decimal)
        self.assets.update(assets or {})
        self.commission_ratio = commission_ratio
        self.__order_id = 0
        self.positions: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {
                "avg_buy_price": 0,
            }
        )
        self.order_history: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self.asset_report: defaultdict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "successful_trades": 0,
                "total_trades": 0,
                "max_drawdown": 0,
                "max_profit": 0,
                "total_profit": 0,
                "ratio": 0,
            }
        )
        self.successful_trades = 0
        self.total_trades = 0
        logger.debug("Cached Binance client initialized.")

    def __str__(self) -> str:
        """Needed by cache library to create cache key."""
        return "CachedClient"

    @ring.lru()  # type: ignore[untyped-decorator]
    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol info from real client or return defaults for offline mode.

        For offline mode, extracts base/quote assets from symbol name.
        Assumes standard Binance symbol format (e.g., BTCUSDT, BNBBTC).
        """
        if self.client:
            result: dict[str, Any] = self.client.get_symbol_info(symbol)
            return result

        # Parse symbol to extract base and quote assets
        # Common quote assets in order of length (longest first to match correctly)
        quote_assets = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]
        base_asset = symbol
        quote_asset = "USDT"
        for qa in quote_assets:
            if symbol.endswith(qa):
                base_asset = symbol[: -len(qa)]
                quote_asset = qa
                break

        return {
            **self.DEFAULT_SYMBOL_INFO,
            "symbol": symbol,
            "baseAsset": base_asset,
            "quoteAsset": quote_asset,
        }

    @ring.lru()  # type: ignore[untyped-decorator]
    def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange info from real client or return empty dict for offline mode."""
        if self.client:
            result: dict[str, Any] = self.client.get_exchange_info()
            return result
        return {}

    @binance_api_exception_on_missing_params("symbol", "interval")
    def get_klines(self, **params: Any) -> list[Kline]:
        return get_klines_from_day_files(
            self.client,
            params["symbol"],
            params["interval"],
            start_ts=params.get("startTime"),
            end_ts=params.get("endTime"),
            limit=params.get("limit") or 500,
        )

    @binance_api_exception_on_missing_params("symbol")
    def get_avg_price(self, interval: str = KLINE_INTERVAL_1MINUTE, **params: Any) -> dict[str, Any]:
        """
        Get average price by using latest 5minute kline.
        TODO: Check if there's more reliable way to this. There's a little
        difference between original API response and ours.

        The test below compares real API vs cached client results.
        Skipped by default as it requires API credentials and network access.

        >>> c1 = get_binance_client()  # doctest: +SKIP
        >>> c2 = get_binance_client(fake=True)  # doctest: +SKIP
        >>> real_client_result = c1.get_avg_price(symbol='BNBUSDT')['price']  # doctest: +SKIP
        >>> fake_client_result = c2.get_avg_price(symbol='BNBUSDT')['price']  # doctest: +SKIP
        >>> abs(Decimal(fake_client_result) - Decimal(real_client_result)) < 1  # doctest: +SKIP
        True
        """

        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(datetime.now(UTC).replace(tzinfo=None), interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta
        logger.debug("Calculating average price between %s - %s", start_dt, end_dt)
        latest_klines = get_klines_from_day_files(
            self.client,
            params["symbol"],
            interval,
            start_ts=datetime_to_timestamp(start_dt),
            end_ts=datetime_to_timestamp(end_dt),
            limit=1,
        )
        try:
            latest_kline = latest_klines[-1]
        except IndexError:
            return {"interval": interval, "price": None}
        avg_price = (latest_kline.close + latest_kline.open) / 2
        logger.debug(
            "Average %s price of %s is %s (calculated from kline %s - %s)",
            interval,
            params["symbol"],
            decimal_as_str(avg_price),
            latest_kline.open_time,
            latest_kline.close_time,
        )
        return {"interval": interval, "price": avg_price}

    def get_asset_balance(self, asset: str, recvWindow: int | None = None) -> dict[str, str]:
        """
        >>> extra_params = {'assets': {'USDT': Decimal(100)}}
        >>> client = get_binance_client(fake=True, extra_params=extra_params)
        >>> client.get_asset_balance('USDT')
        {'asset': 'USDT', 'free': '100.00000000', 'locked': '0.00000000'}

        Locked: the amount of tokens that has been used in any outstanding
        orders. Once the order terminates (either filled, canceled or
        expired), the locked amount will decrease.
        """
        balance = self.assets.get(asset, Decimal(0))
        return {"asset": asset, "free": decimal_as_str(balance), "locked": decimal_as_str(Decimal(0))}

    def get_account(self) -> dict[str, Any]:
        """Get account info from real client or return cached assets for offline mode."""
        if self.client:
            result: dict[str, Any] = self.client.get_account()
            return result
        # For offline mode, return assets as balances
        balances = [{"asset": k, "free": str(v), "locked": "0"} for k, v in self.assets.items()]
        return {"balances": balances}

    @binance_api_exception_on_missing_params("symbol", "side", "type")
    def create_order(self, **params: Any) -> dict[str, Any] | None:
        """Create a simulated order for backtesting.

        Note: This test requires CSV kline data to calculate avg price.
        Skipped by default as test data may not be available.

        >>> extra_params = {'assets': {'USDT': Decimal(100)}}  # doctest: +SKIP
        >>> extra_params['commission_ratio'] = Decimal(0.001)  # doctest: +SKIP
        >>> client = get_binance_client(fake=True, extra_params=extra_params)  # doctest: +SKIP
        >>> order = client.create_order(symbol='BNBUSDT', side=SIDE_BUY, quantity=1, type=ORDER_TYPE_MARKET)  # doctest: +SKIP
        >>> Decimal(client.get_asset_balance('USDT')['free']) < Decimal(100)  # doctest: +SKIP
        True
        >>> Decimal(client.get_asset_balance('BNB')['free']) > Decimal(0)  # doctest: +SKIP
        True
        """
        if "newOrderRespType" in params and params["newOrderRespType"] is not ORDER_RESP_TYPE_RESULT:
            raise_binance_api_exception(400, 666, "CachedClient only accepts RESULT as newOrderRespType")
        if params["type"] is not ORDER_TYPE_MARKET:
            raise_binance_api_exception(400, 666, "CachedClient only accepts MARKET as order type")
        """
        Quantity represents is base asset of symbol. For example if your
        symbol is BNBUSDT, and quantity is 100, this means you want to buy
        100BNB.

        On the other hand we have quoteOrderQty which is represents quote
        asset of the symbol. If our symbol is BNBBTC and quoteOrderQty is 100
        this means I want to spend 100BTC to buy BNB.

        One of these parameters must be given to call create_order method.
        """
        quantity = params.get("quantity")
        quote_order_quantity = params.get("quoteOrderQty")

        if not any([quantity, quote_order_quantity]):
            raise_binance_api_exception(
                400, -1102, "Param 'quantity' or 'quoteOrderQty' must be sent, but both were empty/null!"
            )

        symbol_info = self.get_symbol_info(params["symbol"])

        base_asset = symbol_info["baseAsset"]
        quote_asset = symbol_info["quoteAsset"]

        # TODO: Method is too long. Can we split this method for BUY and
        #       SELL side?

        avg_price_result = self.get_avg_price(symbol=params["symbol"]).get("price")
        if avg_price_result is None:
            logger.debug("Average price calculation problem")
            return None
        price = Decimal(str(avg_price_result))
        fee = Decimal(0)
        commission = self.commission_ratio or Decimal(0)

        if quantity:
            quantity = Decimal(str(quantity))
            quote_order_quantity = price * quantity
            fee = quote_order_quantity * commission
            quote_order_quantity -= fee
            logger.debug("Calculated quote order quantity: %s", decimal_as_str(quote_order_quantity))
        elif quote_order_quantity:
            quote_order_quantity = Decimal(str(quote_order_quantity))
            quantity = quote_order_quantity / price
            fee = (quantity * commission) * price
            quote_order_quantity -= fee
            logger.debug("Calculated quantity: %s", decimal_as_str(quantity))
        else:
            quantity = Decimal(0)
            quote_order_quantity = Decimal(0)

        logger.debug("Calculated quote order fee: %s", decimal_as_str(fee))

        base_asset_balance = Decimal(self.get_asset_balance(asset=base_asset)["free"])

        quote_asset_balance = Decimal(self.get_asset_balance(asset=quote_asset)["free"])

        if params["side"] == SIDE_BUY and quote_order_quantity > quote_asset_balance:
            raise_binance_api_exception(400, -1102, f"You don't have enough {quote_asset}")

        if params["side"] == SIDE_SELL and quantity > base_asset_balance:
            raise_binance_api_exception(400, -1102, f"You don't have enough {base_asset}")

        quantity_as_str = decimal_as_str(quantity)
        now_as_utc = datetime.now(UTC)
        transaction_time = datetime_to_timestamp(now_as_utc.replace(tzinfo=None))

        result = {
            "symbol": params["symbol"],
            "orderId": self.__order_id,
            "orderListId": -1,
            "clientOrderId": "rBaDuImczsKfIrO8gSPI0S",
            "transactTime": transaction_time,
            "price": "0.00000000",
            "origQty": quantity_as_str,
            "executedQty": quantity_as_str,
            "cummulativeQuoteQty": quote_order_quantity,
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": params["type"],
            "side": params["side"],
        }
        if params["side"] == SIDE_BUY:
            if self.assets[base_asset]:
                self.positions[base_asset]["avg_buy_price"] = calculate_avg_buy_price(
                    float(quantity),
                    float(price),
                    float(self.assets[base_asset]),
                    self.positions[base_asset]["avg_buy_price"],
                )
            else:
                self.positions[base_asset]["avg_buy_price"] = float(price)
            self.order_history[params["symbol"]].append({"side": "buy", "open_time": now_as_utc, "quantity": quantity})
            self.assets[base_asset] += quantity
            self.assets[quote_asset] -= quote_order_quantity
        else:
            previous_asset_worth = Decimal(str(self.positions[base_asset]["avg_buy_price"])) * quantity
            new_asset_worth = quote_order_quantity
            diff = float(new_asset_worth - previous_asset_worth)
            self.asset_report[base_asset]["total_profit"] += diff
            if diff > 0:
                self.successful_trades += 1
                self.asset_report[base_asset]["successful_trades"] += 1
                if diff > self.asset_report[base_asset]["max_profit"]:
                    self.asset_report[base_asset]["max_profit"] = diff
            if diff < 0:
                if diff < self.asset_report[base_asset]["max_drawdown"]:
                    self.asset_report[base_asset]["max_drawdown"] = diff
            self.asset_report[base_asset]["total_trades"] += 1
            self.total_trades += 1
            logger.debug(
                "Symbol: %s Buy Avg Price: %s Sell Price: %s",
                params["symbol"],
                self.positions[base_asset]["avg_buy_price"],
                price,
            )
            self.asset_report[base_asset]["ratio"] = (
                100 * self.asset_report[base_asset]["successful_trades"] / self.asset_report[base_asset]["total_trades"]
            )
            self.order_history[params["symbol"]].append({"side": "sell", "open_time": now_as_utc, "quantity": quantity})
            self.positions[base_asset]["avg_buy_price"] = 0
            self.assets[base_asset] -= quantity
            self.assets[quote_asset] += quote_order_quantity
        new_worth = assets_to_usdt(self, dict(self.assets))
        logger.debug(
            "Created order on %s: SM: %s, SD: %s, T: %s, Q: %s, QQ: %s, AvgP: %s, F: %s",
            now_as_utc,
            params["symbol"],
            params["side"],
            params["type"],
            decimal_as_str(params.get("quantity", 0)),
            decimal_as_str(params.get("quoteOrderQty", 0)),
            decimal_as_str(price),
            decimal_as_str(fee),
        )
        logger.debug(assets_to_str(self.assets, "Assets after operation: "))
        self.__order_id += 1
        try:
            ratio = 100 * self.successful_trades / self.total_trades
        except ZeroDivisionError:
            ratio = 0
        logger.info(
            "On %s Total Worth: %s USDT Success Ratio: %s", now_as_utc, decimal_as_str(new_worth), ratio_as_str(ratio)
        )
        return result

    def get_total_usdt(self) -> Decimal:
        """Calculate total portfolio value in USDT."""
        return assets_to_usdt(self, dict(self.assets))

    def generate_order_chart(self, symbol: str, dt: datetime, interval: str, base_asset: str) -> None:
        """Generate candlestick chart with buy/sell markers."""
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        klines = self.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(settings.indicator_warmup_start),
            endTime=datetime_to_timestamp(end_dt),
        )

        if klines:
            klines.pop(-1)  ## current kline not closed

        file_path = get_day_file_path(symbol, interval, end_dt)
        os.remove(file_path)
        load_day_file.delete(symbol, interval, end_dt, file_path)

        df = pd.DataFrame([k.model_dump() for k in klines])
        df.set_index("open_time", inplace=True)
        if not self.order_history[symbol]:
            return
        order_df = pd.DataFrame(self.order_history[symbol])
        order_df.set_index("open_time", inplace=True)

        def calculate_marker_price(row: "pd.Series[Any]") -> float | None:
            if row["side"] == "buy":
                return float(row["low"]) * 0.95
            elif row["side"] == "sell":
                return float(row["high"]) * 1.05
            else:
                return None

        df = pd.concat([df, order_df], axis=1)

        df["marker"] = df.apply(calculate_marker_price, axis=1)
        mpf.plot(
            df,
            type="candle",
            style="charles",
            title=f"{symbol} chart",
            ylabel=f"{base_asset}",
            savefig=f"charts/{symbol}_{interval}.png",
            figscale=5,
            addplot=mpf.make_addplot(df["marker"], scatter=True, color="blue", marker="o", markersize=50),
        )


def get_binance_client(fake: bool = False, extra_params: dict[str, Any] | None = None) -> CachedClient:
    """Get a Binance client.

    Args:
        fake: If True, returns a CachedClient that works offline with CSV files.
              If False, returns a real Binance Client.
        extra_params: Additional parameters for CachedClient (assets, commission_ratio).

    Returns:
        CachedClient for offline/backtest mode, or real Client for live trading.
    """
    if fake:
        # For backtest/offline mode, no real client needed
        return CachedClient(**extra_params or {})

    # For live trading, create real client and wrap in CachedClient for kline caching
    client = Client(settings.binance_api_key, settings.binance_secret_key)
    return CachedClient(client, **extra_params or {})
