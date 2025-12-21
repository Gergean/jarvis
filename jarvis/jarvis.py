#!../venv/bin/python
import argparse
import calendar
import csv
import doctest
import errno
import logging
import os
import simplejson as json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from functools import wraps
from logging.handlers import RotatingFileHandler
from os import makedirs
from os.path import dirname, exists, isfile, join
from dataclasses import dataclass, asdict
from pathlib import Path
import enlighten
import requests
import ring
import sentry_sdk
from binance.client import Client
from binance.enums import (ORDER_RESP_TYPE_RESULT,
                           ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL, KLINE_INTERVAL_1MINUTE)
from binance.exceptions import BinanceAPIException
from binance.helpers import interval_to_milliseconds
from dotenv import load_dotenv
from freezegun import freeze_time
from pandas import Series, DataFrame
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
from pandas_ta.overlap.vwma import vwma
from pandas_ta.overlap import supertrend
import pandas as pd
import mplfinance as mpf
import random

load_dotenv(verbose=True)

SENTRY_DSN = os.getenv("SENTRY_DSN")

if SENTRY_DSN is not None:
    sentry_sdk.init(SENTRY_DSN, traces_sample_rate=1.0)

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
DEBUG = os.getenv('DEBUG')
POSITIONS_FILE = ""

COMMISSION_RATIO = Decimal(0.001)
INVESTMENT_RATIO = Decimal(0.2)
DAY_AS_TIMEDELTA = timedelta(days=1)


class Color(Enum):
    RED = 'RED'
    GREEN = 'GREEN'


class ActionType(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    STAY = 'STAY'
    ERR = 'ERR'


def get_logger(filename='logs/backtest.log'):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(filename, mode='a', backupCount=5)
    if isfile(filename):
        file_handler.doRollover()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s: %(funcName)s : %(message)s")
    )


    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


logger = get_logger()


def __notify(token, chat_id, message):
    send_text = 'https://api.telegram.org/bot' + token + \
                '/sendMessage?chat_id=' + str(chat_id) + \
                '&parse_mode=Markdown&text=' + str(message)
    response = requests.get(send_text)
    return response.json()


def notify(message):
    telegram_dm_id = os.getenv("TELEGRAM_DM_ID")
    telegram_gm_id = os.getenv("TELEGRAM_GM_ID")
    telegram_gm_prefix = os.getenv("TELEGRAM_GM_PREFIX")
    telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not telegram_bot_token:
        return
    if telegram_dm_id:
        __notify(telegram_bot_token, telegram_dm_id, message)
    if telegram_gm_id:
        __notify(
            telegram_bot_token, telegram_gm_id, telegram_gm_prefix + message)
    logger.debug('Sent telegram message:\n%s', message)


def interval_to_seconds(interval):
    """Convert Binance interval strings to seconds

    >>> interval_to_seconds('1h')
    3600
    >>> interval_to_seconds('4h')
    14400
    """
    return int(interval_to_milliseconds(interval) / 1000)


def interval_to_timedelta(interval):
    """Convert binance interval strings to timedelta objects.

    >>> interval_to_timedelta('1h') == timedelta(hours=1)
    True
    """
    return timedelta(seconds=interval_to_seconds(interval))


def flatten_list_of_lists(l):
    """
    >>> l = [[1, 1], [2, 2]]
    >>> flatten_list_of_lists(l)
    [1, 1, 2, 2]
    """
    return [i for sl in l for i in sl]


def floor_dt(dt, delta):
    """
    >>> dt = datetime(2020, 1, 1, 1, 34)
    >>> floor_dt(dt, timedelta(hours=1))
    datetime.datetime(2020, 1, 1, 1, 0)
    >>> floor_dt(dt, timedelta(days=1))
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    q, r = divmod(dt - datetime.min, delta)
    return (datetime.min + (q * delta)) if r else dt


def floor_to_step(number, step):
    """
    >>> floor_to_step(6, 5)
    5
    >>> floor_to_step(18, 5)
    15
    """
    return int(number / step) * step


def ceil_dt(dt, delta):
    """
    >>> dt = datetime(2020, 1, 1, 1, 34)
    >>> ceil_dt(dt, timedelta(hours=1))
    datetime.datetime(2020, 1, 1, 2, 0)
    """
    q, r = divmod(dt - datetime.min, delta)
    return (datetime.min + (q + 1) * delta) if r else dt


def timestamp_to_datetime(ts):
    """Convert timestamp with microseconds to datetime.
    >>> timestamp_to_datetime(1577836800000)
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    return datetime.utcfromtimestamp(int(ts / 1000.0))


def datetime_to_timestamp(dt):
    """Convert datetime to timestamp with microseconds.
    >>> dt = datetime(2020, 1, 1, 0, 0, 0)
    >>> datetime_to_timestamp(dt)
    1577836800000
    """
    return int(calendar.timegm(dt.utctimetuple()) * 1000)


def decimal_as_str(decimal):
    """
    >>> decimal = Decimal('0.00000010')
    >>> decimal
    Decimal('1.0E-7')
    >>> decimal_as_str(decimal)
    '0.00000010'
    """
    return '%.8f' % decimal


def ratio_as_str(ratio):
    return '%.2f' % ratio


def num_of_intervals(start_dt, end_dt, delta):
    return int((end_dt - start_dt) / delta) + 1


def dt_range(start_dt, end_dt, delta):
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


def get_day_file_path(symbol, interval, day_dt):
    """
    >>> get_day_file_path('BTCUSDT', '1h', datetime(2020, 1, 1))
    '../data/binance/BTCUSDT/1h/20200101.csv'
    """
    file_name = day_dt.strftime("%Y%m%d.csv")
    return join("../", "data", "binance", symbol, interval, file_name)


def order_to_str(order):
    message = "Symbol: %s Type: %s Side: %s QuoteQty: %s Qty: %s"
    return message % (order['symbol'], order['type'], order['side'],
                      order['cummulativeQuoteQty'], order['origQty'])


def assets_to_usdt(client, assets):
    total = Decimal(0.0)
    for asset, count in assets.items():
        if asset == 'TWT':
            continue
        if asset == 'USDT':
            total += count
        else:
            symbol = f"{asset}USDT"
            price = client.get_avg_price(symbol=symbol).get('price')
            if price is None:
                continue
            total += Decimal(price) * count
    return total


def calculate_avg_buy_price(quantity, price, last_quantity, last_avg_price):
    return (price * quantity + last_quantity * last_avg_price) / \
            (last_quantity + quantity)


def create_day_file(client, symbol, interval, day_dt):
    """ Fetch klines from given symbol and interval and write to day file.

    TODO: Doctests.
    """
    interval_as_timedelta = interval_to_timedelta(interval)
    file_path = get_day_file_path(symbol, interval, day_dt)

    start_dt = datetime(year=day_dt.year, month=day_dt.month, day=day_dt.day)
    end_dt = start_dt + timedelta(hours=23, minutes=59, seconds=59,
                                  microseconds=999999)

    # Binance gives maximum 500 klines at one time.
    num_of_klines = (end_dt - start_dt) / interval_as_timedelta
    num_of_iterations = round(max(num_of_klines / 500, 1))
    limit_as_timedelta = interval_as_timedelta * 500

    klines = []
    for idx in range(num_of_iterations):
        page_start_dt = start_dt + (interval_as_timedelta * idx * 500)
        page_end_dt = min(page_start_dt + limit_as_timedelta, end_dt)
        logger.debug('Fetching kline data from Binance (%s / %s) (%s / %s).',
                     idx + 1, num_of_iterations, page_start_dt, page_end_dt)
        klines.extend(client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(page_start_dt),
            endTime=datetime_to_timestamp(page_end_dt)
        ))

    file_dir = dirname(file_path)
    if not exists(file_dir):
        makedirs(file_dir)
        logger.debug(f'{file_dir} created.')

    with open(file_path, 'w', newline='') as csv_file:
        field_names = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                       'Close Time', 'Quote Asset Volume', 'Number of trades',
                       'Taker Buy Base Asset Volume',
                       'Taker Buy Quote Asset volume']
        writer = csv.writer(csv_file)
        writer.writerow(field_names)
        for kline in klines:
            writer.writerow(kline)
    logger.debug(f"{file_path} written with {len(klines)} records.")
    return file_path


@ring.lru()
def load_day_file(symbol, interval, day_dt, file_path=None):
    """Load day file, make type conversions and return list of lists.
    """
    file_path = file_path or get_day_file_path(symbol, interval, day_dt)
    lines = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header.
        for _line in reader:
            open_time = int(_line[0])
            close_time = int(_line[6])
            line = list(map(float, _line))
            line[0] = open_time
            line[6] = close_time
            lines.append(line)
    logger.debug('Loaded day file %s', file_path)
    return lines


def get_expected_num_of_lines_in_day_file(day_dt, interval_as_timedelta,
                                          utc_now=None):
    utc_now = utc_now or datetime.utcnow()
    delta = min(utc_now - day_dt, DAY_AS_TIMEDELTA)
    return int(delta / interval_as_timedelta)


def get_klines_from_day_files(client, symbol, interval, start_ts=None,
                              end_ts=None, limit=500):
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
    Intervals \.....\.....\.....\.....\.....\
              start_ts →\...........\← end_ts
                   (ceil)→\.....\←(floor)

    >>> client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    >>> klines = client.get_klines(symbol='BTCUSDT', interval='1h', limit=1)
    >>> len(klines)
    1

    >>> sts = datetime_to_timestamp(datetime(2020, 1, 1))
    >>> ets = datetime_to_timestamp(datetime(2020, 1, 1, 5))
    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '1h', \
start_ts=sts, end_ts=ets, limit=1)
    >>> len(klines)
    1
    >>> timestamp_to_datetime(klines[0][0])
    datetime.datetime(2020, 1, 1, 0, 0)
    >>> timestamp_to_datetime(klines[0][6])
    datetime.datetime(2020, 1, 1, 0, 59, 59)

    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '4h', \
start_ts=sts, end_ts=ets, limit=10)
    >>> len(klines)
    2

    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '4h', limit=5)
    >>> len(klines)
    5

    >>> sts = datetime_to_timestamp(datetime(2020, 1, 1))
    >>> ets = datetime_to_timestamp(datetime(2020, 1, 2))
    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '1h', \
start_ts=sts, end_ts=ets)
    >>> timestamp_to_datetime(klines[0][0])
    datetime.datetime(2020, 1, 1, 0, 0)
    >>> timestamp_to_datetime(klines[24][0])
    datetime.datetime(2020, 1, 2, 0, 0)
    >>> len(klines)
    25
    >>> klines = get_klines_from_day_files(
    ...     client, 'BNBBTC', '6h', start_ts=sts, end_ts=ets)
    >>> len(klines)
    5
    """

    interval_as_timedelta = timedelta(seconds=interval_to_seconds(interval))

    if not end_ts:
        end_ts = datetime_to_timestamp(datetime.utcnow())
    end_dt_orig = timestamp_to_datetime(end_ts)
    end_dt = floor_dt(end_dt_orig, interval_as_timedelta)
    logger.debug('End time snapped by %s: %s -> %s',
                 interval, end_dt_orig, end_dt)

    if start_ts:
        start_dt_orig = timestamp_to_datetime(start_ts)
        start_dt = ceil_dt(start_dt_orig, interval_as_timedelta)
        logger.debug('Start time snapped by %s: %s -> %s',
                     interval, start_dt_orig, start_dt)
    else:
        # If start time is not defined step back interval * limit hours to
        # find start time of request.
        start_dt = end_dt - (interval_as_timedelta * (limit - 1))
        logger.debug('Start time calculated as: %s', start_dt)

    start_day = floor_dt(start_dt, DAY_AS_TIMEDELTA)
    end_day = floor_dt(end_dt, DAY_AS_TIMEDELTA)
    day_dts = list(dt_range(start_day, end_day, DAY_AS_TIMEDELTA))

    utc_now = datetime.utcnow()
    today = floor_dt(utc_now, DAY_AS_TIMEDELTA)

    results = []

    for day_dt in day_dts:
        file_path = get_day_file_path(symbol, interval, day_dt)

        if not exists(file_path):
            file_path = create_day_file(client, symbol, interval, day_dt)

        lines = load_day_file(symbol, interval, day_dt, file_path)

        if day_dt == today:

            expected_num_of_lines = get_expected_num_of_lines_in_day_file(
                day_dt, interval_as_timedelta, utc_now=utc_now)

            if len(lines) < expected_num_of_lines:  # Fetch from API again.

                logger.debug(f'Day file %s has %s records '
                             f'which are below expected %s',
                             file_path, len(lines), expected_num_of_lines)

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
    return results[:limit]


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
    >>> kline['open_time']
    datetime.datetime(2017, 7, 3, 0, 0)
    >>> kline['open']
    Decimal('0.01634790')
    >>> kline['high']
    Decimal('0.80000000')
    >>> kline['low']
    Decimal('0.01575800')
    >>> kline['close']
    Decimal('0.01577100')
    >>> kline['volume']
    Decimal('148976.11427815')
    >>> kline['close_time']
    datetime.datetime(2017, 7, 9, 23, 59, 59)
    >>> kline['quote_asset_volume']
    Decimal('2434.19055334')
    >>> kline['num_of_trades']
    308
    >>> kline['taker_buy_base_asset_volume']
    Decimal('1756.87402397')
    >>> kline['taker_buy_quote_asset_volume']
    Decimal('28.46694368')
    >>> kline['ignore']
    Decimal('17928899.62484339')
    """
    keys = ['open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'ignore']
    dt_keys = ('open_time', 'close_time')
    int_keys = ('num_of_trades',)
    results = []
    for kline in klines:
        result = {}
        for idx, key in enumerate(keys):
            value = kline[idx]
            if key in dt_keys:
                value = timestamp_to_datetime(value)
            elif key in int_keys:
                value = int(value)
            else:
                value = float(value)
            result[key] = value
        results.append(result)
    return results


# ============================================================================
# GENETIC ALGORITHM TRADING SYSTEM
# ============================================================================

class Indicator:
    """Base class for all technical indicators"""

    def calculate(self, klines):
        """
        Calculate indicator value from klines data
        Returns: float (single value)
        """
        raise NotImplementedError

    def mutate(self):
        """
        Return a mutated copy of this indicator
        Mutation: slightly adjust parameters (±10-20%)
        Returns: new Indicator instance
        """
        raise NotImplementedError

    @classmethod
    def random(cls):
        """
        Class method: create random instance with random parameters
        Returns: new Indicator instance
        """
        raise NotImplementedError


class RSI(Indicator):
    def __init__(self, period=14):
        self.period = period

    def calculate(self, klines):
        if len(klines) < self.period:
            return 50.0  # Neutral value
        closes = Series([k['close'] for k in klines])
        rsi = RSIIndicator(close=closes, window=self.period).rsi()
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def mutate(self):
        new_period = int(self.period * random.uniform(0.8, 1.2))
        new_period = max(5, min(30, new_period))
        return RSI(period=new_period)

    @classmethod
    def random(cls):
        return cls(period=random.randint(10, 20))

    def __repr__(self):
        return f"RSI({self.period})"


class SMA(Indicator):
    def __init__(self, period=20):
        self.period = period

    def calculate(self, klines):
        if len(klines) < self.period:
            return klines[-1]['close']
        closes = Series([k['close'] for k in klines])
        sma = SMAIndicator(close=closes, window=self.period).sma_indicator()
        return float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else klines[-1]['close']

    def mutate(self):
        new_period = int(self.period * random.uniform(0.8, 1.2))
        new_period = max(5, min(100, new_period))
        return SMA(period=new_period)

    @classmethod
    def random(cls):
        return cls(period=random.randint(10, 50))

    def __repr__(self):
        return f"SMA({self.period})"


class EMA(Indicator):
    def __init__(self, period=20):
        self.period = period

    def calculate(self, klines):
        if len(klines) < self.period:
            return klines[-1]['close']
        closes = Series([k['close'] for k in klines])
        ema = EMAIndicator(close=closes, window=self.period).ema_indicator()
        return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else klines[-1]['close']

    def mutate(self):
        new_period = int(self.period * random.uniform(0.8, 1.2))
        new_period = max(5, min(100, new_period))
        return EMA(period=new_period)

    @classmethod
    def random(cls):
        return cls(period=random.randint(10, 50))

    def __repr__(self):
        return f"EMA({self.period})"


class WMA(Indicator):
    def __init__(self, period=20):
        self.period = period

    def calculate(self, klines):
        if len(klines) < self.period:
            return klines[-1]['close']
        closes = [k['close'] for k in klines[-self.period:]]
        weights = list(range(1, self.period + 1))
        wma = sum(c * w for c, w in zip(closes, weights)) / sum(weights)
        return float(wma)

    def mutate(self):
        new_period = int(self.period * random.uniform(0.8, 1.2))
        new_period = max(5, min(100, new_period))
        return WMA(period=new_period)

    @classmethod
    def random(cls):
        return cls(period=random.randint(10, 50))

    def __repr__(self):
        return f"WMA({self.period})"


class MACD(Indicator):
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, klines):
        min_required = max(self.fast, self.slow, self.signal)
        if len(klines) < min_required:
            return 0.0
        closes = Series([k['close'] for k in klines])
        macd_indicator = MACDIndicator(
            close=closes,
            window_slow=self.slow,
            window_fast=self.fast,
            window_sign=self.signal
        )
        macd_line = macd_indicator.macd()
        return float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0

    def mutate(self):
        choice = random.choice(['fast', 'slow', 'signal'])
        if choice == 'fast':
            new_fast = int(self.fast * random.uniform(0.8, 1.2))
            return MACD(fast=max(5, min(20, new_fast)), slow=self.slow, signal=self.signal)
        elif choice == 'slow':
            new_slow = int(self.slow * random.uniform(0.8, 1.2))
            return MACD(fast=self.fast, slow=max(15, min(40, new_slow)), signal=self.signal)
        else:
            new_signal = int(self.signal * random.uniform(0.8, 1.2))
            return MACD(fast=self.fast, slow=self.slow, signal=max(5, min(15, new_signal)))

    @classmethod
    def random(cls):
        return cls(
            fast=random.randint(8, 15),
            slow=random.randint(20, 35),
            signal=random.randint(7, 12)
        )

    def __repr__(self):
        return f"MACD({self.fast},{self.slow},{self.signal})"


class PRICE(Indicator):
    """Current price - no parameters"""
    def __init__(self):
        pass

    def calculate(self, klines):
        return klines[-1]['close']

    def mutate(self):
        return PRICE()

    @classmethod
    def random(cls):
        return cls()

    def __repr__(self):
        return "PRICE"


class VOLUME(Indicator):
    """Current volume - no parameters"""
    def __init__(self):
        pass

    def calculate(self, klines):
        return klines[-1]['volume']

    def mutate(self):
        return VOLUME()

    @classmethod
    def random(cls):
        return cls()

    def __repr__(self):
        return "VOLUME"


class Rule:
    def __init__(self, indicator, operator, threshold, weight):
        """
        Args:
            indicator: Indicator instance (RSI, SMA, etc.)
            operator: str ('>', '<', '>=', '<=')
            threshold: float (value to compare against)
            weight: float (-1.0 to +1.0, positive=BUY, negative=SELL)
        """
        self.indicator = indicator
        self.operator = operator
        self.threshold = threshold
        self.weight = weight

    def evaluate(self, klines):
        """
        Evaluate this rule against klines data
        Returns: weight if triggered, 0 otherwise
        """
        value = self.indicator.calculate(klines)

        triggered = False
        if self.operator == '>':
            triggered = value > self.threshold
        elif self.operator == '<':
            triggered = value < self.threshold
        elif self.operator == '>=':
            triggered = value >= self.threshold
        elif self.operator == '<=':
            triggered = value <= self.threshold

        return self.weight if triggered else 0

    def mutate(self):
        """
        Mutate this rule - returns new Rule instance
        Randomly mutate one component
        """
        choice = random.random()

        if choice < 0.1:
            # Mutate indicator
            return Rule(
                self.indicator.mutate(),
                self.operator,
                self.threshold,
                self.weight
            )
        elif choice < 0.2:
            # Flip operator
            new_op = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}[self.operator]
            return Rule(
                self.indicator,
                new_op,
                self.threshold,
                self.weight
            )
        elif choice < 0.6:
            # Mutate threshold
            new_threshold = self.threshold * random.uniform(0.8, 1.2)
            return Rule(
                self.indicator,
                self.operator,
                new_threshold,
                self.weight
            )
        else:
            # Mutate weight
            new_weight = self.weight + random.uniform(-0.2, 0.2)
            new_weight = max(-1.0, min(1.0, new_weight))
            return Rule(
                self.indicator,
                self.operator,
                self.threshold,
                new_weight
            )

    @classmethod
    def random(cls, klines):
        """
        Create random rule
        Args:
            klines: historical data (needed to set reasonable threshold)
        """
        # Random indicator
        indicator_class = random.choice([RSI, SMA, EMA, WMA, MACD, PRICE, VOLUME])
        indicator = indicator_class.random()

        # Random operator
        operator = random.choice(['>', '<', '>=', '<='])

        # Random threshold (based on current values)
        current_value = indicator.calculate(klines)
        threshold = current_value * random.uniform(0.8, 1.2)

        # Random weight
        weight = random.uniform(-1.0, 1.0)

        return cls(indicator, operator, threshold, weight)

    def __repr__(self):
        return f"Rule({self.indicator} {self.operator} {self.threshold:.2f}, w={self.weight:.2f})"


class Individual:
    def __init__(self, rules):
        """
        Args:
            rules: list of Rule instances
        """
        self.rules = rules
        self.fitness = None

    def get_signal(self, klines):
        """
        Evaluate all rules and return trading signal
        Returns: ActionType.BUY, ActionType.SELL, or ActionType.STAY
        """
        score = sum(rule.evaluate(klines) for rule in self.rules)

        if score > 0.5:
            return ActionType.BUY
        elif score < -0.5:
            return ActionType.SELL
        else:
            return ActionType.STAY

    @classmethod
    def random(cls, klines, num_rules=10):
        """
        Create random individual
        Args:
            klines: historical data
            num_rules: number of rules (default 10)
        """
        rules = [Rule.random(klines) for _ in range(num_rules)]
        return cls(rules)

    def __repr__(self):
        return f"Individual({len(self.rules)} rules, fitness={self.fitness})"


def mutate_individual(individual, klines):
    """
    Mutate individual - returns new Individual

    Mutation types (equal probability):
    1. Mutate random rule (40%)
    2. Add new random rule (20%)
    3. Remove random rule (20%)
    4. Replace random rule (20%)
    """
    choice = random.random()
    new_rules = individual.rules.copy()

    if choice < 0.4 and new_rules:
        # Mutate random rule
        idx = random.randint(0, len(new_rules) - 1)
        new_rules[idx] = new_rules[idx].mutate()

    elif choice < 0.6:
        # Add new rule
        new_rule = Rule.random(klines)
        new_rules.append(new_rule)

    elif choice < 0.8 and len(new_rules) > 3:
        # Remove random rule (keep at least 3 rules)
        idx = random.randint(0, len(new_rules) - 1)
        new_rules.pop(idx)

    else:
        # Replace random rule
        if new_rules:
            idx = random.randint(0, len(new_rules) - 1)
            new_rules[idx] = Rule.random(klines)

    return Individual(new_rules)


def crossover(parent1, parent2):
    """
    Single-point crossover
    Returns: two children
    """
    # Find crossover point
    min_len = min(len(parent1.rules), len(parent2.rules))
    if min_len < 2:
        return parent1, parent2

    point = random.randint(1, min_len - 1)

    # Create children
    child1_rules = parent1.rules[:point] + parent2.rules[point:]
    child2_rules = parent2.rules[:point] + parent1.rules[point:]

    return Individual(child1_rules), Individual(child2_rules)


def tournament_selection(population, k=3):
    """
    Select best individual from k random individuals
    Args:
        population: list of Individual instances
        k: tournament size
    Returns: selected Individual
    """
    tournament = random.sample(population, min(k, len(population)))
    return max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))


class GeneticSignalGenerator(SignalGenerator):
    """Signal generator that uses a genetic algorithm individual"""

    def __init__(self, client, individual):
        self.client = client
        self.individual = individual

    def get_signal(self, dt, symbol, interval):
        """Get signal from the genetic individual"""
        needed_num_of_candles = 100  # Get enough history for indicators
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta * needed_num_of_candles

        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=datetime_to_timestamp(start_dt),
            endTime=datetime_to_timestamp(end_dt)
        )

        klines = klines_to_python(klines)
        if klines:
            klines.pop(-1)  # Current kline not closed

        if not klines:
            return ActionType.STAY, klines, 'No klines available'

        signal = self.individual.get_signal(klines)
        return signal, klines, f'Genetic algorithm decision: {signal.value}'


def evaluate_individual(individual, base_asset, trade_assets, interval, start_dt, end_dt):
    """
    Backtest individual and return profit

    Args:
        individual: Individual instance
        base_asset: base asset (e.g., 'USDT')
        trade_assets: list of trade assets
        interval: interval string
        start_dt: start datetime
        end_dt: end datetime

    Returns:
        float: final profit in base asset
    """
    # Create fake client for backtest
    client = get_binance_client(fake=True, extra_params={
        'assets': {base_asset: Decimal(100)},
        'commission_ratio': Decimal(0.001)
    })

    # Create signal generator from individual
    signal_generator = GeneticSignalGenerator(client, individual)

    # Create action generator
    action_generator = AllInActionGenerator(
        client,
        signal_generators={'genetic': signal_generator},
        investment_multiplier=1
    )

    # Run backtest
    interval_as_timedelta = interval_to_timedelta(interval)

    for dt in dt_range(start_dt, end_dt, interval_as_timedelta):
        for trade_asset in trade_assets:
            symbol = f"{trade_asset}{base_asset}"

            try:
                action, base_qty, quote_qty, reason = action_generator.get_action(dt, symbol, interval)

                if action in (ActionType.BUY, ActionType.SELL):
                    order_side = SIDE_BUY if action == ActionType.BUY else SIDE_SELL

                    params = {
                        'symbol': symbol,
                        'side': order_side,
                        'type': ORDER_TYPE_MARKET,
                    }

                    if quote_qty:
                        params['quoteOrderQty'] = quote_qty

                    if base_qty:
                        params['quantity'] = base_qty

                    try:
                        client.create_order(**params)
                    except Exception as e:
                        logger.debug(f'Order failed: {e}')
            except Exception as e:
                logger.debug(f'Error evaluating individual: {e}')

    # Return final worth
    return float(client.get_total_usdt())


def evolve(base_asset, trade_assets, interval, start_dt, end_dt,
           population_size=100, generations=50):
    """
    Main evolution loop

    Args:
        base_asset: 'USDT'
        trade_assets: ['BTC', 'ETH', ...]
        interval: '1h'
        start_dt: datetime
        end_dt: datetime
        population_size: 100
        generations: 50

    Returns:
        best_individual: Individual with highest fitness
    """
    logger.info('Starting genetic algorithm evolution')
    logger.info(f'Population size: {population_size}, Generations: {generations}')
    logger.info(f'Trading {trade_assets} against {base_asset}')
    logger.info(f'Period: {start_dt} to {end_dt}, Interval: {interval}')

    # Get historical data for threshold calculations
    client = get_binance_client(fake=True)
    symbol = f"{trade_assets[0]}{base_asset}"

    # Get sample klines for creating random rules
    sample_klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=datetime_to_timestamp(start_dt),
        endTime=datetime_to_timestamp(start_dt + interval_to_timedelta(interval) * 100),
        limit=100
    )
    sample_klines = klines_to_python(sample_klines)

    logger.info('Initializing population...')
    # Initialize population
    population = [Individual.random(sample_klines, num_rules=random.randint(5, 15))
                  for _ in range(population_size)]

    # Evaluate initial population
    logger.info('Evaluating initial population...')
    bar_manager = enlighten.get_manager()
    bar = bar_manager.counter(total=population_size, desc='Initial Evaluation', unit='individuals')

    for individual in population:
        individual.fitness = evaluate_individual(
            individual, base_asset, trade_assets, interval, start_dt, end_dt
        )
        bar.update()
    bar.close()

    best_ever = max(population, key=lambda ind: ind.fitness)
    logger.info(f'Initial best fitness: ${best_ever.fitness:.2f}')

    # Evolution loop
    for gen in range(generations):
        logger.info(f'Generation {gen+1}/{generations}')

        # Create new generation
        new_population = []

        # Elitism: keep top 10%
        elite_size = population_size // 10
        elite = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:elite_size]
        new_population.extend(elite)
        logger.debug(f'Kept {elite_size} elite individuals')

        # Fill rest with crossover + mutation
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover (50% chance)
            if random.random() < 0.5:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = Individual(parent1.rules.copy()), Individual(parent2.rules.copy())

            # Mutation (20% chance per child)
            if random.random() < 0.2:
                child1 = mutate_individual(child1, sample_klines)
            if random.random() < 0.2:
                child2 = mutate_individual(child2, sample_klines)

            new_population.extend([child1, child2])

        # Trim to population size
        population = new_population[:population_size]

        # Evaluate new generation
        logger.info('Evaluating new generation...')
        bar = bar_manager.counter(total=population_size, desc=f'Gen {gen+1}', unit='individuals')

        for individual in population:
            if individual.fitness is None:
                individual.fitness = evaluate_individual(
                    individual, base_asset, trade_assets, interval, start_dt, end_dt
                )
            bar.update()
        bar.close()

        # Track best
        gen_best = max(population, key=lambda ind: ind.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = gen_best
            logger.info(f'*** NEW BEST: ${best_ever.fitness:.2f} ***')

        # Log progress
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        logger.info(f'Best: ${gen_best.fitness:.2f}, Avg: ${avg_fitness:.2f}, All-time best: ${best_ever.fitness:.2f}')

    bar_manager.stop()

    logger.info('Evolution complete!')
    logger.info(f'Best individual fitness: ${best_ever.fitness:.2f}')
    logger.info(f'Best individual has {len(best_ever.rules)} rules')

    return best_ever


# ============================================================================
# END GENETIC ALGORITHM TRADING SYSTEM
# ============================================================================


class SignalGenerator:
    """Signal generators responsible for generating Signal objects when it's
    get_signal method is called.
    """

    def get_signal(self, dt, symbol, interval):
        raise NotImplemented("Signal classes must have get_signal method "
                             "that returns Signal, used klines and reason.")


class SuperTrendSignalGenerator(SignalGenerator):
    def __init__(self, client, factor=3, atr_period=10) -> None:
        self.client = client
        self.factor = factor
        self.atr_period = atr_period

    def get_signal(self, dt, symbol, interval):

        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = datetime(2017, 1, 1)
        klines = self.client.get_klines(
            symbol=symbol, 
            interval=interval, 
            startTime=datetime_to_timestamp(start_dt), 
            endTime=datetime_to_timestamp(end_dt))

        klines = klines_to_python(klines)
        if klines:
            klines.pop(-1) ## current kline not closed

        file_path = get_day_file_path(symbol, interval, end_dt)
        os.remove(file_path)
        load_day_file.delete(symbol, interval, end_dt, file_path)

        ind = supertrend(
            Series(x["high"] for x in klines),
            Series(x["low"] for x in klines),
            Series(x["close"] for x in klines),
            self.atr_period,
            self.factor
        )
        directions = ind.iloc[:,1]

        try:

            if directions.iat[-2] > directions.iat[-1]:
                return ActionType.SELL, klines, "Direction changed from: %s to %s" % (
                    directions.iat[-2], directions.iat[-1]
                )
            
            if directions.iat[-1] > directions.iat[-2]:
                return ActionType.BUY, klines, "Direction changed from: %s to %s" % (
                    directions.iat[-2], directions.iat[-1]
                )
        
        except IndexError:
            return ActionType.STAY, klines, "Error"
        
        return ActionType.STAY, klines, \
               f'No new signal generated {directions.iat[-2]} {directions.iat[-1]}'

class VWMASignalGenerator(SignalGenerator):
    def __init__(self, client, signal_length=20, base_asset="USDT"):
        self.client = client
        self.length = signal_length
        self.base_asset = base_asset

    def get_signal(self, dt, symbol, interval):
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = datetime(2017, 1, 1)
        klines = self.client.get_klines(
            symbol=symbol, 
            interval=interval, 
            startTime=datetime_to_timestamp(start_dt), 
            endTime=datetime_to_timestamp(end_dt))
        buy_price = 0

        klines = klines_to_python(klines)
        if klines:
            klines.pop(-1) ## current kline not closed
        file_path = get_day_file_path(symbol, interval, end_dt)
        os.remove(file_path)
        load_day_file.delete(symbol, interval, end_dt, file_path)

        ohlc4 = Series((x['close'] + x['open'] + x['high'] + x['low']) / 4 for x in klines)

        vwma_arr = vwma(
            ohlc4,
            Series(x['volume'] for x in klines),
            self.length)

        action = ActionType.STAY
        actions = [action]

        for i in range(self.length, len(ohlc4)):
            highest_vwma = max(vwma_arr[i:-self.length + i:-1])
            lowest_vwma = min(vwma_arr[i:-self.length + i:-1])
            close = ohlc4.iat[i]
            if close > highest_vwma and actions[-1] != ActionType.BUY and buy_price == 0:
                action = ActionType.BUY
                buy_price = close
            elif close < lowest_vwma and actions[-1] != ActionType.SELL and buy_price != 0:
                action = ActionType.SELL
                buy_price = 0
            else:
                action = ActionType.STAY
            
            actions.append(action)
        
        if actions[-1] == ActionType.BUY:
            return ActionType.BUY, klines, "Close: %s is greater than VWMA: %s" % (
                decimal_as_str(ohlc4.iat[-1]), decimal_as_str(highest_vwma)
            )

        if actions[-1] == ActionType.SELL:
            return ActionType.SELL, klines, "Close: %s is smaller than VWMA: %s" % (
                decimal_as_str(ohlc4.iat[-1]), decimal_as_str(lowest_vwma)
            )

        return ActionType.STAY, klines, \
               f'No new signal generated'


class SMASignalGenerator(SignalGenerator):
    def __init__(self, client, signal_length=20):
        self.client = client
        self.length = signal_length

    def get_signal(self, dt, symbol, interval):
        needed_num_of_candles = 2 * self.length - 1
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta * needed_num_of_candles
        klines = self.client.get_klines(
            symbol=symbol, interval=interval,
            startTime=datetime_to_timestamp(start_dt),
            endTime=datetime_to_timestamp(end_dt))
        trade_asset = symbol.replace("USDT", "")
        buy_price = self.client.positions[trade_asset]['avg_buy_price']

        klines = klines_to_python(klines)
        if not klines:
            return ActionType.STAY, klines, \
               f'No new signal generated'
        if klines:
            klines.pop(-1) ## current kline not closed

        df = DataFrame(klines)
        df[f"sma"] = SMAIndicator(
            close=df["close"], window=self.length, fillna=False
        ).sma_indicator()

        max_buy_price = df.tail(1).sma.item() + df.tail(1).sma.item() * 0.01
        min_sell_price = df.tail(1).sma.item() - df.tail(1).sma.item() * 0.01

        if df.tail(1).sma.item() <= df.tail(1).close.item() <= max_buy_price and buy_price == 0:
            return ActionType.BUY, klines, "Close: %s is greater than SMA: %s" % (
                decimal_as_str(df.tail(1).close.item()), decimal_as_str(max_buy_price)
            )

        if df.tail(1).close.item() <= min_sell_price and df.tail(1).close.item() < df.tail(1).open.item():
            return ActionType.SELL, klines, "Close: %s is smaller than SMA: %s" % (
                decimal_as_str(df.tail(1).close.item()), decimal_as_str(min_sell_price)
            )

        return ActionType.STAY, klines, \
               f'No new signal generated'


class ConsecutiveUpDownSignalGenerator(SignalGenerator):

    def __init__(self, client, num_of_reds_to_sell=4, num_of_greens_to_buy=3):
        self.client = client
        self.num_of_reds_to_sell = num_of_reds_to_sell
        self.num_of_greens_to_buy = num_of_greens_to_buy

    @staticmethod
    def get_colors(klines):
        return [Color.RED if kline['open'] > kline['close'] else Color.GREEN
                for kline in klines]

    def get_signal(self, dt, symbol, interval):
        needed_num_of_candles = max(
            [self.num_of_reds_to_sell, self.num_of_greens_to_buy]) + 1
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta * needed_num_of_candles
        klines = self.client.get_klines(
            symbol=symbol, interval=interval,
            startTime=datetime_to_timestamp(start_dt),
            endTime=datetime_to_timestamp(end_dt))

        # Binance gives non closed kline as last item. We should remove it
        # Before calculation.
        if klines:
            klines.pop(-1)

        klines = klines_to_python(klines)

        if len(klines) < needed_num_of_candles:
            return ActionType.ERR, klines, \
                   f'Requested {needed_num_of_candles} klines, ' \
                   f'{len(klines)} returned'

        colors = self.get_colors(klines)
        logger.debug('Kline colors between %s - %s: ' + ('%s, ' *
                                                         needed_num_of_candles)[
                                                        :-2],
                     *[start_dt, end_dt] + [color.value for color in colors])

        sell_colors = colors[-self.num_of_reds_to_sell:]
        if all([c == Color.RED for c in sell_colors]):
            return ActionType.SELL, klines, "%s reds last %s klines." % (
                self.num_of_reds_to_sell, self.num_of_reds_to_sell
            )

        buy_colors = colors[-self.num_of_greens_to_buy:]
        if all([c == Color.GREEN for c in buy_colors]):
            return ActionType.BUY, klines, "%s greens last %s klines." % (
                self.num_of_greens_to_buy, self.num_of_greens_to_buy
            )

        return ActionType.STAY, klines, \
               f'{self.num_of_greens_to_buy} greens or ' \
               f'{self.num_of_reds_to_sell} reds are not matched.'


class ActionGenerator:
    """Action generators responsible for generating decisions by using
    registered SIGNAL_GENERATORS when its get_decision method is called.
    """

    def __init__(self, client, signal_generators=None):
        self.client = client
        self.signal_generators = signal_generators or {}

    def __str__(self):
        """Needed by cache library to create cache key."""
        return 'ActionGenerator'

    @ring.lru()
    def get_symbol_filter(self, symbol, filter_type):
        """
        TODO: Doctests.
        """
        filters = self.client.get_symbol_info(symbol)['filters']
        for _filter in filters:
            if _filter['filterType'] == filter_type:
                return _filter
        return _filter

    def get_action(self, dt, symbol, interval):
        raise NotImplemented(
            "DecisionGenerator classes must have get_decision method that "
            "returns Action, Quantity, Quote Asset Quantity and Reason"
        )

@dataclass
class Position:
    symbol: str
    spent: Decimal
    amount: Decimal

class AllInActionGenerator(ActionGenerator):
    def set_positions(self):
        if POSITIONS_FILE.stat().st_size != 0:
            with open(POSITIONS_FILE, "r") as f:
                data = json.load(f)
            self.positions = [Position(**pos) for pos in data]

    def save_positions(self):
        with open(POSITIONS_FILE, "w") as f:
            json.dump([asdict(pos) for pos in self.positions], f)

    def __init__(self, client, signal_generators=None,
                 investment_multiplier=1):
        super().__init__(client, signal_generators=signal_generators)
        self.investment_multiplier = investment_multiplier
        self.positions = []

    def get_action(self, dt, symbol, interval):
        symbol_info = self.client.get_symbol_info(symbol)
        base_asset = symbol_info['baseAsset']
        quote_asset = symbol_info['quoteAsset']

        signals = []
        for name, generator in self.signal_generators.items():
            signal, klines, reason = \
                generator.get_signal(dt, symbol, interval)
            logger.debug('%s returned %s on %s. Reason: %s',
                         name, signal.value, dt, reason)
            signals.append(signal)
        most_common_signal = Counter(signals).most_common(1)[0][0]

        base_asset_quantity = None
        quote_asset_quantity = None
        market_lot_info = self.get_symbol_filter(symbol, 'MARKET_LOT_SIZE')
        market_min_quantity = Decimal(market_lot_info['minQty'])

        lot_info = self.get_symbol_filter(symbol, 'LOT_SIZE')
        min_quantity = Decimal(lot_info['minQty'])
        step_size = Decimal(lot_info['stepSize'])

        min_notional_info = self.get_symbol_filter(symbol, 'NOTIONAL')
        min_notional = Decimal(min_notional_info['minNotional'])
        # max_quantity = Decimal(lot_info['maxQty'])
        # I hope some day we have rich enough to calculate max quantity of
        # orders.

        if most_common_signal == ActionType.SELL:
            base_asset_quantity = Decimal(
                self.client.get_asset_balance(
                    asset=base_asset)['free'])
            base_asset_quantity = floor_to_step(base_asset_quantity, step_size)
            avg_price = self.client.get_avg_price(symbol=symbol).get('price')

            if avg_price is None:
                return ActionType.ERR, None, None, 'Average price problem'

            base_asset_value = base_asset_quantity * Decimal(avg_price)

            if base_asset_quantity <= market_min_quantity:
                return ActionType.STAY, None, None, \
                       'I would sell but there are not enough ' \
                       f'{base_asset} in wallet (' + \
                       decimal_as_str(
                           base_asset_quantity) + ') (MARKET_LOT_SIZE)'

            if base_asset_quantity <= min_quantity:
                return ActionType.STAY, None, None, \
                       'I would sell but there are not enough ' \
                       f'{base_asset} in wallet (' + \
                       decimal_as_str(base_asset_quantity) + ') (LOT_SIZE)'

            if base_asset_value <= min_notional:
                return ActionType.STAY, None, None, \
                       'I would sell but there are not enough ' \
                       f'{base_asset} in wallet (' + \
                       decimal_as_str(base_asset_quantity) + ') (MIN_NOTIONAL)'
            self.positions = [pos for pos in self.positions if pos.symbol != symbol]

        if most_common_signal == ActionType.BUY:
            if not self.positions:
                tradable_asset_quantity = Decimal(
                    self.client.get_asset_balance(
                        asset=quote_asset)['free']) * \
                                    self.investment_multiplier
            else:
                tradable_asset_quantity = self.positions[0].spent

            quote_asset_quantity = \
                int(tradable_asset_quantity / step_size) * step_size

            if quote_asset_quantity <= market_min_quantity:
                return ActionType.STAY, None, None, \
                       'I would buy but there are not enough ' \
                       f'{quote_asset} in wallet (' + \
                       decimal_as_str(quote_asset_quantity) + \
                       ') (MARKET_LOT_SIZE)'

            if quote_asset_quantity <= min_quantity:
                return ActionType.STAY, None, None, \
                       'I would buy but there are not enough ' \
                       f'{quote_asset} in wallet (' + \
                       decimal_as_str(quote_asset_quantity) + ') (LOT_SIZE)'

            if quote_asset_quantity <= min_notional:
                return ActionType.STAY, None, None, \
                       'I would buy but there are not enough ' \
                       f'{quote_asset} in wallet (' + \
                       decimal_as_str(
                           quote_asset_quantity) + ') (MIN_NOTIONAL)'

            self.positions.append(
                Position(
                    symbol=symbol,
                    spent=tradable_asset_quantity,
                    amount=quote_asset_quantity
                )
            )

        return most_common_signal, base_asset_quantity, quote_asset_quantity, \
               f'All signals that I have says {most_common_signal.value}'


def binance_api_exception_on_missing_params(*required_params):
    """
    >>> def fun(a=None, b=None):
    ...     print('OK')
    >>> fun = binance_api_exception_on_missing_params('b')(fun)
    >>> fun(a=True)
    Traceback (most recent call last):
        ...
    binance.exceptions.BinanceAPIException: APIError(code=-1102): \
Mandatory parameter b was not sent, was empty / null, or malformed.
    >>> fun(b=True)
    OK
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for key in required_params:
                if key not in kwargs:
                    raise BinanceAPIException(
                        FakeResponse(400, {
                            'code': -1102,
                            'msg': f'Mandatory parameter {key} was not sent, '
                                   'was empty / null, or malformed.'
                        })
                    )
            return func(*args, **kwargs)

        return wrapper

    return inner


class FakeResponse:

    def __init__(self, status_code, _dict):
        self.status_code = status_code
        self._dict = _dict

    def json(self):
        return self._dict


def assets_to_str(assets, prefix="Current assets: "):
    params = flatten_list_of_lists([
        [k, decimal_as_str(v)] for k, v in assets.items()
    ])
    return (prefix + "%s: %s, " * len(assets))[:-2] % tuple(params)


class FakeClient:
    """
    >>> client = FakeClient(Client(BINANCE_API_KEY, BINANCE_SECRET_KEY))
    """

    def __init__(self, client, assets=None, commission_ratio=None):
        self.client = client
        self.assets = defaultdict(Decimal)
        self.assets.update(assets or {})
        self.commission_ratio = commission_ratio
        self.__order_id = 0
        self.positions = defaultdict(lambda: {
            "avg_buy_price": 0,
        })
        self.order_history = defaultdict(list)
        self.asset_report = defaultdict(lambda: {
            "successful_trades": 0,
            "total_trades": 0,
            "max_drawdown": 0,
            "max_profit": 0,
            "total_profit": 0,
            "ratio": 0
        })
        self.successful_trades = 0
        self.total_trades = 0
        logger.info('Fake binance client initialized.')

    def __str__(self):
        """Needed by cache library to create cache key."""
        return 'FakeClient'

    @ring.lru()
    def get_exchange_info(self):
        return self.client.get_exchange_info()

    @binance_api_exception_on_missing_params('symbol', 'interval')
    def get_klines(self, **params):
        return get_klines_from_day_files(
            self.client, params['symbol'], params['interval'],
            start_ts=params.get('startTime'), end_ts=params.get('endTime'),
            limit=params.get('limit')
        )

    @binance_api_exception_on_missing_params('symbol')
    def get_avg_price(self, interval=KLINE_INTERVAL_1MINUTE, **params):
        """
        Get average price by using latest 5minute kline.
        TODO: Check if there's more reliable way to this. There's a little
        difference between original API response and ours.

        >>> c1 = get_binance_client()
        >>> c2 = get_binance_client(fake=True)
        >>> real_client_result = c1.get_avg_price(symbol='BNBUSDT')['price']
        >>> fake_client_result = c1.get_avg_price(symbol='BNBUSDT')['price']
        >>> abs(Decimal(fake_client_result) - Decimal(real_client_result)) < 1
        True
        """

        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(datetime.utcnow(), interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta
        logger.debug('Calculating average price between %s - %s',
                     start_dt, end_dt)
        latest_klines = klines_to_python(
            get_klines_from_day_files(
                self.client, params['symbol'], interval,
                start_ts=datetime_to_timestamp(start_dt),
                end_ts=datetime_to_timestamp(end_dt),
                limit=1
            )
        )
        try:
            latest_kline = latest_klines[-1]
        except IndexError:
            return {'interval': interval, 'price': None}
        avg_price = (latest_kline['close'] + latest_kline['open']) / 2
        logger.debug(
            'Average %s price of %s is %s (calculated from kline '
            '%s - %s)', interval, params['symbol'],
            decimal_as_str(avg_price), latest_kline['open_time'],
            latest_kline['close_time'],
        )
        return {'interval': interval, 'price': avg_price}

    def get_asset_balance(self, asset, recvWindow=None):
        """
        >>> extra_params = {'assets': {'USDT': Decimal(100)}}
        >>> client = get_binance_client(fake=True, extra_params=extra_params)
        >>> client.get_asset_balance('USDT')
        {'asset': 'USDT', 'free': '100.00000000', 'locked': '0.00000000'}

        Locked: the amount of tokens that has been used in any outstanding
        orders. Once the order terminates (either filled, canceled or
        expired), the locked amount will decrease.
        """
        balance = self.assets.get(asset, 0)
        return {
            'asset': asset,
            'free': decimal_as_str(Decimal(balance)),
            'locked': decimal_as_str(Decimal(0))
        }

    @ring.lru()
    def get_symbol_info(self, symbol):
        """
        TODO:
        Create caching mechanism like get_klines_from_files. Restructure
        data folder as this:

        - data
            - binance
                - klines
                    -{interval}
                        - day file
                        - day file
                        - day file
                - symbol_info
                    - symbol file
                    - symbol file
                    - symbol file
        """
        return self.client.get_symbol_info(symbol)

    @binance_api_exception_on_missing_params('symbol', 'side', 'type')
    def create_order(self, **params):
        """

        >>> extra_params = {'assets': {'USDT': Decimal(100)}}
        >>> extra_params['commission_ratio'] = Decimal(0.001)
        >>> client = get_binance_client(fake=True, extra_params=extra_params)
        >>> order = client.create_order(symbol='BNBUSDT', side=SIDE_BUY, \
quantity=1, type=ORDER_TYPE_MARKET)
        >>> Decimal(client.get_asset_balance('USDT')['free']) < Decimal(100)
        True
        >>> Decimal(client.get_asset_balance('BNB')['free']) > Decimal(0)
        True
        """
        if 'newOrderRespType' in params and \
                params['newOrderRespType'] is not ORDER_RESP_TYPE_RESULT:
            raise BinanceAPIException(
                FakeResponse(400, {
                    'code': 666,
                    'msg': f"Fake client only accepts RESULT as"
                           f"newOrderRespType"
                })
            )
        if params['type'] is not ORDER_TYPE_MARKET:
            raise BinanceAPIException(
                FakeResponse(400, {
                    'code': 666,
                    'msg': f"Fake client only accepts MARKET as"
                           f"order type"
                })
            )
        """
        Quantity represents is base asset of symbol. For example if your
        symbol is BNBUSDT, and quantity is 100, this means you want to buy
        100BNB.

        On the other hand we have quoteOrderQty which is represents quote
        asset of the symbol. If our symbol is BNBBTC and quoteOrderQty is 100
        this means I want to spend 100BTC to buy BNB.

        One of these parameters must be given to call create_order method.
        """
        quantity = params.get('quantity')
        quote_order_quantity = params.get('quoteOrderQty')

        if not any([quantity, quote_order_quantity]):
            raise BinanceAPIException(
                FakeResponse(400, {
                    'code': -1102,
                    'msg': "Param 'quantity' or 'quoteOrderQty' must be "
                           "sent, but both were empty/null!"
                })
            )

        symbol_info = self.get_symbol_info(params['symbol'])

        base_asset = symbol_info['baseAsset']
        quote_asset = symbol_info['quoteAsset']

        # TODO: Method is too long. Can we split this method for BUY and
        #       SELL side?

        price = self.get_avg_price(symbol=params['symbol']).get('price')
        if price is None:
            logger.debug('Average price calculation problem')
            return
        price = Decimal(price)
        fee = 0

        if quantity:
            quote_order_quantity = price * quantity
            fee = quote_order_quantity * self.commission_ratio
            quote_order_quantity -= fee
            logger.debug('Calculated quote order quantity: %s',
                         decimal_as_str(quote_order_quantity))
        elif quote_order_quantity:
            quantity = quote_order_quantity / price
            fee = (quantity * self.commission_ratio) * price
            quote_order_quantity -= fee
            logger.debug('Calculated quantity: %s', decimal_as_str(quantity))

        logger.debug('Calculated quote order fee: %s', decimal_as_str(fee))

        base_asset_balance = Decimal(self.get_asset_balance(
            asset=base_asset)['free'])

        quote_asset_balance = Decimal(self.get_asset_balance(
            asset=quote_asset)['free'])

        if params['side'] == SIDE_BUY and quote_order_quantity > \
                quote_asset_balance:
            raise BinanceAPIException(
                FakeResponse(400, {
                    'code': -1102,
                    'msg': f'You don\'t have enough {quote_asset}'
                    # TODO: Fix with proper binance message.
                })
            )

        if params['side'] == SIDE_SELL and quantity > \
                base_asset_balance:
            raise BinanceAPIException(
                FakeResponse(400, {
                    'code': -1102,
                    'msg': f'You don\'t have enough {base_asset}'
                    # TODO: Fix with proper binance message.
                })
            )

        quantity_as_str = decimal_as_str(quantity)
        now_as_utc = datetime.utcnow()
        transaction_time = datetime_to_timestamp(now_as_utc)

        result = {
            'symbol': params['symbol'],
            'orderId': self.__order_id,
            'orderListId': -1,
            'clientOrderId': 'rBaDuImczsKfIrO8gSPI0S',
            'transactTime': transaction_time,
            'price': '0.00000000',
            'origQty': quantity_as_str,
            'executedQty': quantity_as_str,
            'cummulativeQuoteQty': quote_order_quantity,
            'status': 'FILLED',
            'timeInForce': 'GTC',
            'type': params['type'],
            'side': params['side']
        }
        if params['side'] == SIDE_BUY:
            if self.assets[base_asset]:
                self.positions[base_asset]["avg_buy_price"] = \
                    calculate_avg_buy_price(
                        quantity, price, self.assets[base_asset],
                        self.positions[base_asset]["avg_buy_price"])
            else:
                self.positions[base_asset]["avg_buy_price"] = price
            self.order_history[params["symbol"]].append({
                "side": "buy", "open_time": now_as_utc, "quantity": quantity})
            self.assets[base_asset] += quantity
            self.assets[quote_asset] -= quote_order_quantity
        else:
            previous_asset_worth = \
                self.positions[base_asset]["avg_buy_price"] * quantity
            new_asset_worth = quote_order_quantity
            diff = new_asset_worth - previous_asset_worth
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
                'Symbol: %s Buy Avg Price: %s Sell Price: %s',
                params['symbol'], self.positions[base_asset]["avg_buy_price"],
                price)
            self.asset_report[base_asset]["ratio"] = \
                100 * self.asset_report[base_asset]["successful_trades"] / \
                self.asset_report[base_asset]["total_trades"]
            self.order_history[params["symbol"]].append({
                "side": "sell", "open_time": now_as_utc, "quantity": quantity
            })
            self.positions[base_asset]["avg_buy_price"] = 0
            self.assets[base_asset] -= quantity
            self.assets[quote_asset] += quote_order_quantity
        new_worth = assets_to_usdt(self, self.assets)
        logger.debug('Created order on %s: SM: %s, SD: %s, '
                     'T: %s, Q: %s, QQ: %s, '
                     'AvgP: %s, F: %s', now_as_utc, params['symbol'],
                     params['side'], params['type'],
                     decimal_as_str(params.get('quantity', 0)),
                     decimal_as_str(params.get('quoteOrderQty', 0)),
                     decimal_as_str(price), decimal_as_str(fee))
        logger.debug(assets_to_str(self.assets, "Assets after operation: "))
        self.__order_id += 1
        try:
            ratio = 100 * self.successful_trades / self.total_trades
        except ZeroDivisionError:
            ratio = 0
        logger.info('On %s Total Worth: %s USDT Success Ratio: %s',
                    now_as_utc, decimal_as_str(new_worth),
                    ratio_as_str(ratio))
        return result

    def get_total_usdt(self):
        return assets_to_usdt(self, self.assets)

    def generate_order_chart(self, symbol, dt, interval, base_asset):
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = datetime(2017, 1, 1)
        klines = self.get_klines(
            symbol=symbol, 
            interval=interval, 
            startTime=datetime_to_timestamp(start_dt), 
            endTime=datetime_to_timestamp(end_dt))

        klines = klines_to_python(klines)
        if klines:
            klines.pop(-1) ## current kline not closed

        file_path = get_day_file_path(symbol, interval, end_dt)
        os.remove(file_path)
        load_day_file.delete(symbol, interval, end_dt, file_path)
        
        df = pd.DataFrame(klines)
        df.set_index("open_time", inplace=True)
        if not self.order_history[symbol]:
            return
        order_df = pd.DataFrame(self.order_history[symbol])
        order_df.set_index("open_time", inplace=True)

        def calculate_marker_price(row):
            if row["side"] == "buy":
                return row["low"] * 0.95
            elif row["side"] == "sell":
                return row["high"] * 1.05
            else:
                return None

        df = pd.concat([df, order_df], axis=1)

        df["marker"] = df.apply(calculate_marker_price, axis=1)
        mpf.plot(
            df, type="candle", style="charles", title=f"{symbol} chart", 
            ylabel=f"{base_asset}", savefig=f"charts/{symbol}_{interval}.png", figscale=5,
            addplot=mpf.make_addplot(df["marker"], scatter=True, color="blue", marker="o", markersize=50))


def get_binance_client(fake=False, extra_params=None):
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    if fake:
        return FakeClient(client, **extra_params or {})
    return client


def backtest(base_asset, starting_amount, trade_assets, interval, start_dt,
             end_dt, commission_ratio, investment_ratio):
    bar_manager = enlighten.get_manager()

    client = get_binance_client(fake=True, extra_params={
        'assets': {base_asset: starting_amount},
        'commission_ratio': commission_ratio
    })
    action_generator = AllInActionGenerator(
        client,
        signal_generators={
            'VWMA':
                VWMASignalGenerator(client),
        },
        investment_multiplier=investment_ratio
    )
    interval_as_timedelta = interval_to_timedelta(interval)

    total_intervals = num_of_intervals(start_dt, end_dt, interval_as_timedelta)

    bar = bar_manager.counter(
        total=total_intervals,
        desc='Trading', unit='Intervals')

    for idx, dt in enumerate(dt_range(start_dt, end_dt, interval_as_timedelta)):
        for idx2, trade_asset in enumerate(trade_assets):
            with freeze_time(dt):
                logger.debug('Datetime frozen to %s', dt)
                symbol = f"{trade_asset}{base_asset}"
                action, base_asset_quantity, quote_asset_quantity, reason = \
                    action_generator.get_action(dt, symbol, interval)

                traded = action in (ActionType.BUY, ActionType.SELL)

                logging.log(
                    logging.INFO if traded else logging.DEBUG,
                    logger.info if traded else logger.debug,
                    'For %s, on %s Decided to %s, reason: %s',
                )

                if traded:

                    order_side = SIDE_BUY if action == ActionType.BUY else \
                        SIDE_SELL

                    params = {
                        'symbol': symbol,
                        'side': order_side,
                        'type': ORDER_TYPE_MARKET,
                        'quantity': 0,
                        'quoteOrderQty': 0,
                    }

                    if quote_asset_quantity:
                        params.update({'quoteOrderQty': quote_asset_quantity})

                    if base_asset_quantity:
                        params.update({'quantity': base_asset_quantity})
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

def trade(base_asset, trade_assets, interval, investment_ratio):
    fake_client = get_binance_client(fake=True)
    client = get_binance_client()
    client.get_klines = fake_client.get_klines
    action_generator = AllInActionGenerator(
        client,
        signal_generators={
            'VWMA':
                VWMASignalGenerator(client)
        },
        investment_multiplier=investment_ratio
    )
    action_generator.set_positions()
    dt = datetime.utcnow()
    grouped_actions = {ActionType.BUY: "", ActionType.SELL: ""}

    for trade_asset in trade_assets:
        symbol = f"{trade_asset}{base_asset}"
        action, base_asset_quantity, quote_asset_quantity, reason = \
            action_generator.get_action(dt, symbol, interval)

        if action not in (ActionType.BUY, ActionType.SELL):
            message = f'Decided to {action.value} for {trade_asset}, Reason: {reason}'
            logger.info(message)
            if DEBUG:
                notify(message)
            continue

        order_side = SIDE_BUY if action == ActionType.BUY else \
            SIDE_SELL

        params = {
            'symbol': symbol,
            'side': order_side,
            'type': ORDER_TYPE_MARKET,
        }

        if quote_asset_quantity:
            params.update({
                'quoteOrderQty': decimal_as_str(quote_asset_quantity)
            })

        if base_asset_quantity:
            params.update({
                'quantity': decimal_as_str(base_asset_quantity)
            })

        try:
            order = client.create_order(**params)
        except Exception as e:
            logger.info(e)
            continue

        # * 100 ETH for 20 USDT (BTCUSDT)

        grouped_actions[action] += \
            '• %s %s for %s %s\n' % (
                order['executedQty'], trade_asset,
                order['cummulativeQuoteQty'], base_asset)

    message = ""
    if grouped_actions[ActionType.BUY]:
        message += "*I've Bought:*\n\n" + grouped_actions[ActionType.BUY]

    if grouped_actions[ActionType.SELL]:
        message += "*I've Sold:*\n" + grouped_actions[ActionType.SELL]

    if message:
        assets = dict([(asset['asset'], Decimal(asset['free']))
                       for asset in client.get_account()['balances']
                       if Decimal(asset['free']) > 0])

        assets_as_usdt = assets_to_usdt(client, assets)
        message += f"\n💰 *Your current assets*:\n"
        for asset, value in assets.items():
            message += f"• {asset}: {decimal_as_str(value)}\n"

        message += f"\n🤑 *Total Worth*:\n"
        message += f"• {decimal_as_str(assets_as_usdt)} USDT " \
                   f"(TWT not Included)"
        notify(message)

    action_generator.save_positions()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Example: "
                    "python jarvis.py -st 2020-01-01T00:00:00 -et "
                    "2020-12-01T00:00:00 -ba USDT -ta BTC ETH -i 1h")

    parser.add_argument(
        '-ep', '--env-path', default='.env', type=str, dest="env_path",
        help='File that we can load environment variables from %(default)s')

    subparsers = parser.add_subparsers(dest='subparser')

    backtest_parser = subparsers.add_parser('backtest')
    doctest_parser = subparsers.add_parser('doctest')
    trade_parser = subparsers.add_parser('trade')
    evolve_parser = subparsers.add_parser('evolve')

    dt_type = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')

    backtest_parser.add_argument(
        '-ba', dest="base_asset", metavar="BASE_ASSET", default='USDT',
        type=str, help='The base asset that you want to trade. '
                       'Default: %(default)s')

    backtest_parser.add_argument(
        '-sa', dest="starting_amount", default=Decimal(100.0), type=Decimal,
        metavar="STARTING_AMOUNT", help='Amount of base asset when you start '
                                        'testing. Default: %(default)s')

    backtest_parser.add_argument(
        '-ta', dest="trade_assets", nargs="+", metavar="TRADE_ASSET",
        help="List of assets that you want to trade against base asset.",
        type=str, required=True)

    backtest_parser.add_argument(
        '-st', dest="start_dt", default="2020-01-01T00:00:00", type=dt_type,
        metavar="START_TIME", help='The time that trade will start. '
                                   'Default: %(default)s.')

    backtest_parser.add_argument(
        '-et', dest="end_dt", default="2020-12-01T00:00:00", type=dt_type,
        metavar="END_TIME", help='The time that trade end. '
                                 'Default: %(default)s.')

    backtest_parser.add_argument(
        '-i', dest="interval", default="1h", metavar="INTERVAL",
        type=str, help='Interval of klines to check.')

    backtest_parser.add_argument(
        '-cr', dest="commission_ratio", default=COMMISSION_RATIO, type=Decimal,
        metavar="COMMISSION_RATIO",
        help='Commission ratio of platform. Default: %s'
             % decimal_as_str(COMMISSION_RATIO)
    )

    backtest_parser.add_argument(
        '-ir', dest="investment_ratio", default=INVESTMENT_RATIO, type=Decimal,
        metavar="INVESTMENT_RATIO",
        help='Investment ratio of platform. Default: %s'
             % decimal_as_str(INVESTMENT_RATIO)
    )

    doctest_parser.add_argument(
        '-v', dest="verbose", default=False,
        action="store_true", help="Gives verbose output when set.")

    trade_parser.add_argument(
        '-ba', dest="base_asset", metavar="BASE_ASSET", default='USDT',
        type=str, help='The base asset that you want to trade. '
                       'Default: %(default)s')

    trade_parser.add_argument(
        '-ta', dest="trade_assets", nargs="+", metavar="TRADE_ASSET",
        help="List of assets that you want to trade against base asset.",
        type=str, required=True)

    trade_parser.add_argument(
        '-i', dest="interval", default="1h", metavar="INTERVAL",
        type=str, help='Interval of klines to check.')

    trade_parser.add_argument(
        '-ir', dest="investment_ratio", default=INVESTMENT_RATIO, type=Decimal,
        metavar="INVESTMENT_RATIO",
        help='Investment ratio of platform. Default: %s'
             % decimal_as_str(INVESTMENT_RATIO)
    )

    # Evolve parser arguments
    evolve_parser.add_argument(
        '-ba', dest="base_asset", metavar="BASE_ASSET", default='USDT',
        type=str, help='The base asset that you want to trade. '
                       'Default: %(default)s')

    evolve_parser.add_argument(
        '-ta', dest="trade_assets", nargs="+", metavar="TRADE_ASSET",
        help="List of assets that you want to trade against base asset.",
        type=str, required=True)

    evolve_parser.add_argument(
        '-st', dest="start_dt", default="2020-01-01T00:00:00", type=dt_type,
        metavar="START_TIME", help='The time that evolution will start. '
                                   'Default: %(default)s.')

    evolve_parser.add_argument(
        '-et', dest="end_dt", default="2020-12-01T00:00:00", type=dt_type,
        metavar="END_TIME", help='The time that evolution will end. '
                                 'Default: %(default)s.')

    evolve_parser.add_argument(
        '-i', dest="interval", default="1h", metavar="INTERVAL",
        type=str, help='Interval of klines to check.')

    evolve_parser.add_argument(
        '-ps', dest="population_size", default=100, type=int,
        metavar="POPULATION_SIZE",
        help='Population size for genetic algorithm. Default: %(default)s')

    evolve_parser.add_argument(
        '-gen', dest="generations", default=50, type=int,
        metavar="GENERATIONS",
        help='Number of generations to evolve. Default: %(default)s')

    kwargs = parser.parse_args()

    if kwargs.env_path:
        if not exists(kwargs.env_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), kwargs.env_path)
        load_dotenv(dotenv_path=kwargs.env_path, verbose=True, override=True)
        logger.info('Config loaded from %s', kwargs.env_path)

    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    DEBUG = os.getenv('DEBUG') == 'True'
    POSITIONS_FILE = Path(os.getenv('POSITIONS_FILE'))
    POSITIONS_FILE.touch(exist_ok=True)

    if kwargs.subparser == 'doctest':
        doctest.testmod(verbose=kwargs.verbose)
    elif kwargs.subparser == 'backtest':
        backtest(kwargs.base_asset, kwargs.starting_amount,
                 kwargs.trade_assets, kwargs.interval, kwargs.start_dt,
                 kwargs.end_dt, kwargs.commission_ratio, kwargs.investment_ratio)
        print('Output written to backtest.log')
    elif kwargs.subparser == 'trade':
        trade(kwargs.base_asset, kwargs.trade_assets,
              kwargs.interval, kwargs.investment_ratio)
    elif kwargs.subparser == 'evolve':
        best_individual = evolve(
            kwargs.base_asset,
            kwargs.trade_assets,
            kwargs.interval,
            kwargs.start_dt,
            kwargs.end_dt,
            kwargs.population_size,
            kwargs.generations
        )
        logger.info('='*80)
        logger.info('EVOLUTION COMPLETE!')
        logger.info(f'Best fitness: ${best_individual.fitness:.2f}')
        logger.info(f'Number of rules: {len(best_individual.rules)}')
        logger.info('='*80)
        logger.info('Best individual rules:')
        for i, rule in enumerate(best_individual.rules, 1):
            logger.info(f'  {i}. {rule}')
        logger.info('='*80)
        print('Output written to backtest.log')
