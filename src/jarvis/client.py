import csv
import logging
from _decimal import Decimal
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from os import makedirs
from os.path import exists, dirname, join
from appdirs import user_cache_dir
import ring
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1MINUTE, SIDE_BUY, \
    ORDER_RESP_TYPE_RESULT, ORDER_TYPE_MARKET, SIDE_SELL
from binance.exceptions import BinanceAPIException

from .constants import DAY_AS_TIMEDELTA, APP_NAME, APP_AUTHOR
from .helpers import ts_to_dt, floor_dt, dt_range, \
    interval_to_timedelta, dt_to_ts, interval_to_seconds, \
    ceil_dt, dc_to_str, ratio_as_str, assets_to_str, assets_to_usdt, \
    klines_to_python
from .types import Assets

logger = logging.getLogger(__name__)


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
                        FakeResponse(
                            400,
                            {
                                "code": -1102,
                                "msg": f"Mandatory parameter {key} was not "
                                       "sent, was empty / null, or malformed.",
                            },
                        )
                    )
            return func(*args, **kwargs)
        return wrapper
    return inner


def calculate_avg_buy_price(quantity, price, last_quantity, last_avg_price):
    return (price * quantity + last_quantity * last_avg_price) / (
            last_quantity + quantity
    )


def get_day_file_path(symbol: str, interval: str, day_dt: datetime):
    """
    >>> get_day_file_path('BTCUSDT', '1h', datetime(2020, 1, 1))
    '/Users/.../Library/Caches/Jarvis/BTCUSDT/1d/20200101.csv'
    """
    file_name = day_dt.strftime("%Y%m%d.csv")
    cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)
    return join(cache_dir, symbol.capitalize(), interval.lower(), file_name)


def create_day_file(client, symbol, interval, day_dt):
    """Fetch klines from given symbol and interval and write to day file.

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
        logger.debug(
            "Fetching kline data from Binance (%s / %s) (%s / %s).",
            idx + 1, num_of_iterations, page_start_dt, page_end_dt, )
        klines.extend(client.get_klines(
            symbol=symbol, interval=interval,
            startTime=dt_to_ts(page_start_dt),
            endTime=dt_to_ts(page_end_dt)
        ))

    file_dir = dirname(file_path)
    if not exists(file_dir):
        makedirs(file_dir)
        logger.debug(f"{file_dir} created.")

    with open(file_path, "w", newline="") as csv_file:
        field_names = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                       "Close Time", "Quote Asset Volume", "Number of trades",
                       "Taker Buy Base Asset Volume",
                       "Taker Buy Quote Asset volume", ]
        writer = csv.writer(csv_file)
        writer.writerow(field_names)
        for kline in klines:
            writer.writerow(kline)
    logger.debug(f"{file_path} written with {len(klines)} records.")
    return file_path


@ring.lru()
def load_day_file(symbol: str, interval: str, day_dt, file_path=None):
    """Load day file, make type conversions and return list of lists."""
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
    logger.debug("Loaded day file %s", file_path)
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

    >>> sts = dt_to_ts(datetime(2020, 1, 1))
    >>> ets = dt_to_ts(datetime(2020, 1, 1, 5))
    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '1h', \
start_ts=sts, end_ts=ets, limit=1)
    >>> len(klines)
    1
    >>> ts_to_dt(klines[0][0])
    datetime.datetime(2020, 1, 1, 0, 0)
    >>> ts_to_dt(klines[0][6])
    datetime.datetime(2020, 1, 1, 0, 59, 59)

    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '4h', \
start_ts=sts, end_ts=ets, limit=10)
    >>> len(klines)
    2

    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '4h', limit=5)
    >>> len(klines)
    5

    >>> sts = dt_to_ts(datetime(2020, 1, 1))
    >>> ets = dt_to_ts(datetime(2020, 1, 2))
    >>> klines = get_klines_from_day_files(client, 'BNBBTC', '1h', \
start_ts=sts, end_ts=ets)
    >>> ts_to_dt(klines[0][0])
    datetime.datetime(2020, 1, 1, 0, 0)
    >>> ts_to_dt(klines[24][0])
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
        end_ts = dt_to_ts(datetime.utcnow())
    end_dt_orig = ts_to_dt(end_ts)
    end_dt = floor_dt(end_dt_orig, interval_as_timedelta)
    logger.debug("End time snapped by %s: %s -> %s", interval, end_dt_orig,
                 end_dt)

    if start_ts:
        start_dt_orig = ts_to_dt(start_ts)
        start_dt = ceil_dt(start_dt_orig, interval_as_timedelta)
        logger.debug("Start time snapped by %s: %s -> %s", interval,
                     start_dt_orig, start_dt, )
    else:
        # If start time is not defined step back interval * limit hours to
        # find start time of request.
        start_dt = end_dt - (interval_as_timedelta * (limit - 1))
        logger.debug("Start time calculated as: %s", start_dt)

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
                logger.debug("Day file %s has %s records "
                             "which are below expected %s", file_path,
                             len(lines), expected_num_of_lines, )

                file_path = create_day_file(client, symbol, interval, day_dt)
                load_day_file.delete(symbol, interval, day_dt, file_path)
                lines = load_day_file(symbol, interval, day_dt, file_path)

        for line in lines:
            open_ts = line[0]
            open_dt = ts_to_dt(open_ts)
            if start_dt <= open_dt <= end_dt:
                results.append(line)
    return results[:limit]


class FakeResponse:
    def __init__(self, status_code, _dict):
        self.status_code = status_code
        self._dict = _dict

    def json(self):
        return self._dict


class FakeClient:
    """
    >>> client = FakeClient(Client(BINANCE_API_KEY, BINANCE_SECRET_KEY))
    """

    def __init__(self, client: Client, assets=None,
                 commission_ratio: Decimal = Decimal(0)):
        if assets is None:
            assets = {}
        self.client = client
        self.assets: Assets = defaultdict(Decimal)
        self.assets.update(assets or {})
        self.commission_ratio = commission_ratio
        self.__order_id = 0
        self.positions = defaultdict(lambda: {"avg_buy_price": Decimal(0), })
        self.order_history = defaultdict(list)
        self.asset_report = defaultdict(
            lambda: {
                "successful_trades": 0,
                "total_trades": 0,
                "max_drawdown": 0,
                "max_profit": 0,
                "total_profit": 0,
                "ratio": float(0)
            })
        self.successful_trades = 0
        self.total_trades = 0
        logger.info("Fake binance client initialized.")

    def __str__(self):
        """Needed by cache library to create cache key."""
        return "FakeClient"

    @ring.lru()
    def get_exchange_info(self):
        return self.client.get_exchange_info()

    @binance_api_exception_on_missing_params("symbol", "interval")
    def get_klines(self, **params):
        return get_klines_from_day_files(self.client, params["symbol"],
                                         params["interval"],
                                         start_ts=params.get("startTime"),
                                         end_ts=params.get("endTime"),
                                         limit=params.get("limit"), )

    @binance_api_exception_on_missing_params("symbol")
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
        logger.debug("Calculating average price between %s - %s", start_dt,
                     end_dt)
        latest_klines = klines_to_python(
            get_klines_from_day_files(self.client, params["symbol"], interval,
                                      start_ts=dt_to_ts(start_dt),
                                      end_ts=dt_to_ts(end_dt),
                                      limit=1, ))
        try:
            latest_kline = latest_klines[-1]
        except IndexError:
            return {"interval": interval, "price": None}
        avg_price = (latest_kline["close"] + latest_kline["open"]) / 2
        logger.debug(
            "Average %s price of %s is %s (calculated from kline " "%s - %s)",
            interval, params["symbol"], dc_to_str(avg_price),
            latest_kline["open_time"], latest_kline["close_time"], )
        return {"interval": interval, "price": avg_price}

    def get_asset_balance(self, asset):
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
        return {"asset": asset, "free": dc_to_str(Decimal(balance)),
                "locked": dc_to_str(Decimal(0)), }

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

    @binance_api_exception_on_missing_params("symbol", "side", "type")
    def create_order(self, **params):
        """

        >>> extra_params = {'assets': {'USDT': Decimal(100)}, 'commission_ratio': Decimal(0.001)}
        >>> client = get_binance_client(fake=True, extra_params=extra_params)
        >>> order = client.create_order(symbol='BNBUSDT', side=SIDE_BUY, \
quantity=1, type=ORDER_TYPE_MARKET)
        >>> Decimal(client.get_asset_balance('USDT')['free']) < Decimal(100)
        True
        >>> Decimal(client.get_asset_balance('BNB')['free']) > Decimal(0)
        True
        """
        if ("newOrderRespType" in params and
                params["newOrderRespType"] is not ORDER_RESP_TYPE_RESULT):
            raise BinanceAPIException(FakeResponse(
                400, {
                    "code": 666,
                    "msg": "Fake client only accepts RESULT as"
                           "newOrderRespType"
                }))
        if params["type"] is not ORDER_TYPE_MARKET:
            raise BinanceAPIException(FakeResponse(
                400, {
                    "code": 666,
                    "msg": "Fake client only accepts MARKET as"
                           "order type"
                }))
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
            raise BinanceAPIException(FakeResponse(
                400, {
                    "code": -1102,
                    "msg": "Param 'quantity' or 'quoteOrderQty' must be "
                           "sent, but both were empty/null!"
                }))

        symbol_info = self.get_symbol_info(params["symbol"])

        base_asset = symbol_info["baseAsset"]
        quote_asset = symbol_info["quoteAsset"]

        # TODO: Method is too long. Can we split this method for BUY and
        #       SELL side?

        price = self.get_avg_price(symbol=params["symbol"]).get("price")
        if price is None:
            logger.debug("Average price calculation problem")
            return
        price = Decimal(price)
        fee = 0

        if quantity:
            quote_order_quantity = price * quantity
            fee = quote_order_quantity * self.commission_ratio
            quote_order_quantity -= fee
            logger.debug("Calculated quote order quantity: %s",
                         dc_to_str(quote_order_quantity), )
        elif quote_order_quantity:
            quantity = quote_order_quantity / price
            fee = (quantity * self.commission_ratio) * price
            quote_order_quantity -= fee
            logger.debug("Calculated quantity: %s",
                         dc_to_str(quantity))

        logger.debug("Calculated quote order fee: %s",
                     dc_to_str(fee))

        base_asset_balance = Decimal(
            self.get_asset_balance(asset=base_asset)["free"])

        quote_asset_balance = Decimal(
            self.get_asset_balance(asset=quote_asset)["free"])

        if params["side"] == SIDE_BUY and \
                quote_order_quantity > quote_asset_balance:
            raise BinanceAPIException(FakeResponse(
                400, {
                    "code": -1102,
                    "msg": f"You don't have enough {quote_asset}"
                }))

        if params["side"] == SIDE_SELL and quantity > base_asset_balance:
            raise BinanceAPIException(FakeResponse(
                400, {
                    "code": -1102,
                    "msg": f"You don't have enough {base_asset}"
                }))

        quantity_as_str = dc_to_str(quantity)
        now_as_utc = datetime.utcnow()
        transaction_time = dt_to_ts(now_as_utc)

        result = {"symbol": params["symbol"], "orderId": self.__order_id,
                  "orderListId": -1, "clientOrderId": "rBaDuImczsKfIrO8gSPI0S",
                  "transactTime": transaction_time, "price": "0.00000000",
                  "origQty": quantity_as_str, "executedQty": quantity_as_str,
                  "cumulativeQuoteQty": quote_order_quantity,
                  "status": "FILLED",
                  "timeInForce": "GTC", "type": params["type"],
                  "side": params["side"], }
        if params["side"] == SIDE_BUY:
            if self.assets[base_asset]:
                self.positions[base_asset]["avg_buy_price"] = \
                    calculate_avg_buy_price(
                        quantity, price,
                        self.assets[base_asset],
                        self.positions[base_asset]["avg_buy_price"]
                    )
            else:
                self.positions[base_asset]["avg_buy_price"] = price
            self.order_history[params["symbol"]].append(
                {"side": "buy", "open_time": now_as_utc, "quantity": quantity})
            self.assets[base_asset] += quantity
            self.assets[quote_asset] -= quote_order_quantity
        else:
            previous_asset_worth = (
                    self.positions[base_asset]["avg_buy_price"] * quantity)
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
            logger.debug("Symbol: %s Buy Avg Price: %s Sell Price: %s",
                         params["symbol"],
                         self.positions[base_asset]["avg_buy_price"],
                         price, )
            self.asset_report[base_asset]["ratio"] = \
                100.0 * \
                self.asset_report[base_asset]["successful_trades"] / \
                self.asset_report[base_asset]["total_trades"]
            self.order_history[params["symbol"]].append(
                {"side": "sell", "open_time": now_as_utc,
                 "quantity": quantity})
            self.positions[base_asset]["avg_buy_price"] = Decimal(0)
            self.assets[base_asset] -= quantity
            self.assets[quote_asset] += quote_order_quantity
        new_worth = assets_to_usdt(self, self.assets)
        logger.debug("Created order on %s: SM: %s, SD: %s, "
                     "T: %s, Q: %s, QQ: %s, "
                     "AvgP: %s, F: %s", now_as_utc, params["symbol"],
                     params["side"], params["type"],
                     dc_to_str(params.get("quantity", 0)),
                     dc_to_str(params.get("quoteOrderQty", 0)),
                     dc_to_str(price), dc_to_str(fee), )
        logger.debug(assets_to_str(self.assets, "Assets after operation: "))
        self.__order_id += 1
        try:
            ratio = 100 * self.successful_trades / self.total_trades
        except ZeroDivisionError:
            ratio = 0
        logger.info("On %s Total Worth: %s USDT Success Ratio: %s", now_as_utc,
                    dc_to_str(new_worth), ratio_as_str(ratio), )
        return result


def get_binance_client(fake=False, extra_params=None):
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    if fake:
        return FakeClient(client, **extra_params or {})
    return client
