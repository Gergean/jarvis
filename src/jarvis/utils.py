"""Utility functions for the Jarvis trading system."""

import calendar
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, TypeVar

from binance.helpers import interval_to_milliseconds

T = TypeVar("T")

DAY_AS_TIMEDELTA = timedelta(days=1)


def interval_to_seconds(interval: str) -> int:
    """Convert Binance interval strings to seconds

    >>> interval_to_seconds('1h')
    3600
    >>> interval_to_seconds('4h')
    14400
    """
    result = interval_to_milliseconds(interval)
    if result is None:
        raise ValueError(f"Invalid interval: {interval}")
    return int(result / 1000)


def interval_to_timedelta(interval: str) -> timedelta:
    """Convert Binance interval strings to timedelta objects.

    >>> interval_to_timedelta('1h') == timedelta(hours=1)
    True
    """
    return timedelta(seconds=interval_to_seconds(interval))


def flatten_list_of_lists(lst: list[list[T]]) -> list[T]:
    """
    >>> l = [[1, 1], [2, 2]]
    >>> flatten_list_of_lists(l)
    [1, 1, 2, 2]
    """
    return [i for sl in lst for i in sl]


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


def floor_to_step(number: Decimal | float, step: Decimal | float) -> Decimal:
    """
    >>> floor_to_step(6, 5)
    Decimal('5')
    >>> floor_to_step(18, 5)
    Decimal('15')
    >>> floor_to_step(Decimal('0.123'), Decimal('0.01'))
    Decimal('0.12')
    """
    return Decimal(int(Decimal(str(number)) / Decimal(str(step)))) * Decimal(str(step))


def ceil_dt(dt: datetime, delta: timedelta) -> datetime:
    """
    >>> dt = datetime(2020, 1, 1, 1, 34)
    >>> ceil_dt(dt, timedelta(hours=1))
    datetime.datetime(2020, 1, 1, 2, 0)
    """
    q, r = divmod(dt - datetime.min, delta)
    return (datetime.min + (q + 1) * delta) if r else dt


def timestamp_to_datetime(ts: int) -> datetime:
    """Convert timestamp with milliseconds to datetime (UTC, naive).
    >>> timestamp_to_datetime(1577836800000)
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    return datetime.fromtimestamp(int(ts / 1000.0), tz=UTC).replace(tzinfo=None)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to timestamp with milliseconds.
    >>> dt = datetime(2020, 1, 1, 0, 0, 0)
    >>> datetime_to_timestamp(dt)
    1577836800000
    """
    return int(calendar.timegm(dt.utctimetuple()) * 1000)


def decimal_as_str(value: Decimal | float) -> str:
    """
    >>> decimal = Decimal('0.00000010')
    >>> decimal
    Decimal('1.0E-7')
    >>> decimal_as_str(decimal)
    '0.00000010'
    """
    return "%.8f" % value


def ratio_as_str(ratio: float) -> str:
    """Format ratio as percentage string with 2 decimal places.

    >>> ratio_as_str(0.5)
    '0.50'
    >>> ratio_as_str(99.999)
    '100.00'
    """
    return "%.2f" % ratio


def num_of_intervals(start: datetime, end: datetime, delta: timedelta) -> int:
    """Calculate number of intervals between two datetimes.

    >>> start = datetime(2020, 1, 1, 0, 0)
    >>> end = datetime(2020, 1, 1, 3, 0)
    >>> num_of_intervals(start, end, timedelta(hours=1))
    4
    """
    return int((end - start) / delta) + 1


def dt_range(start: datetime, end: datetime, delta: timedelta) -> Iterator[datetime]:
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
    for idx in range(num_of_intervals(start, end, delta)):
        yield start + (delta * idx)


def order_to_str(order: dict[str, Any]) -> str:
    """Format order dict as human-readable string.

    >>> order = {'symbol': 'BTCUSDT', 'type': 'MARKET', 'side': 'BUY',
    ...          'cummulativeQuoteQty': '100.00', 'origQty': '0.001'}
    >>> order_to_str(order)
    'Symbol: BTCUSDT Type: MARKET Side: BUY QuoteQty: 100.00 Qty: 0.001'
    """
    message = "Symbol: %s Type: %s Side: %s QuoteQty: %s Qty: %s"
    return message % (order["symbol"], order["type"], order["side"], order["cummulativeQuoteQty"], order["origQty"])


def assets_to_str(assets: dict[str, Decimal], prefix: str = "Current assets: ") -> str:
    """Format asset holdings as human-readable string.

    >>> assets_to_str({'USDT': Decimal('100.5'), 'BTC': Decimal('0.001')})
    'Current assets: USDT: 100.50000000, BTC: 0.00100000'
    >>> assets_to_str({'ETH': Decimal('2.5')}, prefix="Holdings: ")
    'Holdings: ETH: 2.50000000'
    """
    params = flatten_list_of_lists([[k, decimal_as_str(v)] for k, v in assets.items()])
    return (prefix + "%s: %s, " * len(assets))[:-2] % tuple(params)


def parse_period_to_days(period: str) -> int:
    """Parse period string to number of days.

    Supported formats:
    - Nd: N days (e.g., '7d', '30d', '90d')
    - Nw: N weeks (e.g., '1w', '2w', '4w')
    - NM: N months (e.g., '1M', '3M', '6M') - assumes 30 days/month

    >>> parse_period_to_days('7d')
    7
    >>> parse_period_to_days('1w')
    7
    >>> parse_period_to_days('2w')
    14
    >>> parse_period_to_days('1M')
    30
    >>> parse_period_to_days('3M')
    90
    >>> parse_period_to_days('90d')
    90
    """
    if not period:
        raise ValueError("Period string cannot be empty")

    unit = period[-1]
    try:
        value = int(period[:-1])
    except ValueError as e:
        raise ValueError(f"Invalid period format: {period}") from e

    if unit == "d":
        return value
    elif unit == "w":
        return value * 7
    elif unit == "M":
        return value * 30
    else:
        raise ValueError(f"Unknown period unit: {unit}. Use 'd' (days), 'w' (weeks), or 'M' (months)")


def calculate_avg_buy_price(quantity: float, price: float, last_quantity: float, last_avg_price: float) -> float:
    """Calculate weighted average buy price after adding new position.

    >>> calculate_avg_buy_price(1, 100, 0, 0)
    100.0
    >>> calculate_avg_buy_price(1, 200, 1, 100)
    150.0
    >>> calculate_avg_buy_price(2, 300, 1, 100)
    233.33333333333334
    """
    return (price * quantity + last_quantity * last_avg_price) / (last_quantity + quantity)
