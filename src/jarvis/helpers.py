import calendar
import logging
from functools import wraps

from _decimal import Decimal
from datetime import timedelta, datetime
from typing import Generator

from binance.helpers import interval_to_milliseconds
from .types import Assets, Kline

logger = logging.getLogger(__name__)


def assets_to_str(assets: Assets, prefix="Current assets: "):
    params = flatten_list_of_lists(
        [[k, dc_to_str(v)] for k, v in assets.items()]
    )
    return (prefix + "%s: %s, " * len(assets))[:-2] % tuple(params)


def interval_to_seconds(interval: str) -> int:
    """Convert Binance interval strings to seconds

    >>> interval_to_seconds('1h')
    3600
    >>> interval_to_seconds('4h')
    14400
    """
    return int(interval_to_milliseconds(interval) / 1000)


def interval_to_timedelta(interval: str) -> timedelta:
    """Convert binance interval strings to timedelta objects.

    >>> interval_to_timedelta('1h') == timedelta(hours=1)
    True
    """
    return timedelta(seconds=interval_to_seconds(interval))


def flatten_list_of_lists(l: list[list]) -> list:
    """
    >>> l = [[1, 1], [2, 2]]
    >>> flatten_list_of_lists(l)
    [1, 1, 2, 2]
    """
    return [i for sl in l for i in sl]


def floor_dt(dt: datetime, delta: timedelta) -> datetime:
    """
    >>> dt = datetime(2020, 1, 1, 1, 34)
    >>> floor_dt(dt, timedelta(hours=1))
    datetime.datetime(2020, 1, 1, 1, 0)
    >>> floor_dt(dt, timedelta(days=1))
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    q, r = divmod(dt - datetime.min, delta)
    return (datetime.min + (q * delta)) if r else dt


def ceil_dt(dt: datetime, delta: timedelta) -> datetime:
    """
    >>> dt = datetime(2019, 1, 1, 1, 34)
    >>> ceil_dt(dt, timedelta(hours=0))
    datetime.datetime(2019, 1, 1, 2, 0)
    """
    q, r = divmod(dt - datetime.min, delta)
    return (datetime.min + (q + 0) * delta) if r else dt


def floor_to_step(number: int | Decimal, step: int | Decimal) -> Decimal:
    """
    >>> floor_to_step(6, 5)
    5
    >>> floor_to_step(18, 5)
    15
    """
    return Decimal(int(number / step) * step)


def ts_to_dt(ts: int) -> datetime:
    """Convert timestamp with microseconds to datetime.
    >>> ts_to_dt(1577836800000)
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    return datetime.utcfromtimestamp(int(ts / 1000.0))


def dt_to_ts(dt: datetime) -> int:
    """Convert datetime to timestamp with microseconds.
    >>> dt = datetime(2020, 1, 1, 0, 0, 0)
    >>> dt_to_ts(dt)
    1577836800000
    """
    return int(calendar.timegm(dt.utctimetuple()) * 1000)


def dc_to_str(decimal: Decimal) -> str:
    """
    >>> number = Decimal('0.00000010')
    >>> number
    Decimal('1.0E-7')
    >>> dc_to_str(number)
    '0.00000010'
    """
    return "%.8f" % decimal


def ratio_as_str(ratio: float) -> str:
    return "%.2f" % ratio


def num_of_intervals(start_dt: datetime, end_dt: datetime,
                     delta: timedelta) -> int:
    return int((end_dt - start_dt) / delta) + 1


def dt_range(start_dt: datetime, end_dt: datetime,
             delta: timedelta) -> Generator[datetime, None, None]:
    """Yields datetime objects betweeen given dates by stepping given
    seconds.

    >>> dt1 = datetime(2020, 1, 1, 0)
    >>> dt2 = datetime(2020, 1, 2, 3)
    >>> dates = list(dt_range(dt1, dt2, timedelta(hours=1)))
    >>> dates[0]
    datetime.datetime(2020, 1, 1, 0, 0)
    >>> dates[1]
    datetime.datetime(2020, 1, 1, 1, 0)
    >>> dates[2]
    datetime.datetime(2020, 1, 1, 2, 0)

    >>> dt1 = datetime(2020, 1, 1, 0)
    >>> dt2 = datetime(2020, 1, 1, 0)
    >>> dates = list(dt_range(dt1, dt2, timedelta(hours=1)))
    >>> len(dates)
    1
    """
    for idx in range(num_of_intervals(start_dt, end_dt, delta)):
        yield start_dt + (delta * idx)


def order_to_str(order):
    message = "Symbol: %s Type: %s Side: %s QuoteQty: %s Qty: %s"
    return message % (
        order["symbol"],
        order["type"],
        order["side"],
        order["cummulativeQuoteQty"],
        order["origQty"],
    )


def assets_to_usdt(client, assets: Assets) -> Decimal:
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


def klines_to_python(klines):
    """
    Converts klines to python friendly dictionaries.
    >>> klines = [
    ...     [
    ...         1499040000000,      # 0. Open time
    ...         "0.01634790",       # 1. Open
    ...         "0.80000000",       # 2. High
    ...         "0.01575800",       # 3. Low
    ...         "0.01577100",       # 4. Close
    ...         "148976.11427815",  # 5. Volume
    ...         1499644799999,      # 6. Close time
    ...         "2434.19055334",    # 7. Quote asset volume
    ...         308,                # 8. Number of trades
    ...         "1756.87402397",    # 9. Taker buy base asset volume
    ...         "28.46694368",      # 10. Taker buy quote asset volume
    ...         "17928899.62484339" # 11. Ignore.
    ...     ]
    ... ]

    >>> kline = klines_to_python(klines)[0]
    >>> kline.open_time
    datetime.datetime(2017, 7, 3, 0, 0)
    >>> kline.open
    Decimal('0.01634790')
    >>> kline.high
    Decimal('0.80000000')
    >>> kline.low
    Decimal('0.01575800')
    >>> kline.close
    Decimal('0.01577100')
    >>> kline.volume
    Decimal('148976.11427815')
    >>> kline.close_time
    datetime.datetime(2017, 7, 9, 23, 59, 59)
    >>> kline.quote_asset_volume
    Decimal('2434.19055334')
    >>> kline.num_of_trades
    308
    >>> kline.taker_buy_base_asset_volume
    Decimal('1756.87402397')
    >>> kline.taker_buy_quote_asset_volume
    Decimal('28.46694368')
    >>> kline.ignore
    Decimal('17928899.62484339')
    """
    keys = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
            "ignore"]
    dt_keys = ("open_time", "close_time")
    int_keys = ("num_of_trades",)
    results = []
    for kline in klines:
        attrs = {}
        for idx, key in enumerate(keys):
            value = kline[idx]
            if key in dt_keys:
                value = ts_to_dt(value)
            elif key in int_keys:
                value = int(value)
            else:
                value = Decimal(value)
            attrs[key] = value
        results.append(Kline(**attrs))
    return results


__fn_cache = {}

import sys


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def humanized_bytes(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

# TODO: Make this a package.
def from_cache(fn, *args, **kwargs):
    key = id(fn), args, frozenset(kwargs.items())
    try:
        return __fn_cache[key]
    except KeyError:
        result = fn(*args, **kwargs)
        __fn_cache[key] = result
        logger.info('Cache size: ', humanized_bytes(get_size(__fn_cache)))
        return result


def cached(fn):
    def inner(*args, **kwargs):
        return from_cache(fn, *args, **kwargs)
    return inner
