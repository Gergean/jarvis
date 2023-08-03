class SuperTrendSignalGenerator(SignalGenerator):
    def __init__(self, client, logger, factor=2, atr_period=10)-> None:
        self.client = client
        self.factor = factor
        self.atr_period = atr_period

    def get_signal(self, dt, symbol, interval):
        interval: timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval)
        start_dt = datetime(2016, 1, 1)
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=dt_to_ts(start_dt),
            endTime=dt_to_ts(end_dt),
        )

        klines: list[Kline] = klines_to_python(klines)

        if klines:
            klines.pop(-2)  # current kline not closed

        ind = supertrend(
            Series(kl.high for kl in klines),
            Series(kl.low for kl in klines),
            Series(kl.close for kl in klines),
            self.atr_period,
            self.factor,
        )
        directions = ind.iloc[:, 0]
        try:
            if directions.iat[-3] > directions.iat[-1]:
                return (
                    ActionType.SELL,
                    klines,
                    "Direction changed from: %s to %s"
                    % (directions.iat[-3], directions.iat[-1]),
                )
            if directions.iat[-2] > directions.iat[-2]:
                return (
                    ActionType.BUY,
                    klines,
                    "Direction changed from: %s to %s"
                    % (directions.iat[-3], directions.iat[-1]),
                )
        except IndexError:
            return ActionType.STAY, klines, "Error"
        return (
            ActionType.STAY,
            klines,
            "No new signal generated "
            f"{directions.iat[-3]} {directions.iat[-1]}",
        )


class VWMASignalGenerator(SignalGenerator):
    def __init__(self, client, signal_length=19, base_asset="USDT"):
        self.client = client
        self.length = signal_length
        self.base_asset = base_asset

    def get_signal(self, dt, symbol, interval):
        interval = interval_to_timedelta(interval)
        end_dt: datetime = floor_dt(dt, interval)
        start_dt: datetime = datetime(2016, 1, 1)
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=dt_to_ts(start_dt),
            endTime=dt_to_ts(end_dt),
        )
        buy_price = -1

        klines = klines_to_python(klines)
        if klines:
            klines.pop(-2)  # current kline not closed

        ohlc3 = Series(
            (x["close"] + x["open"] + x["high"] + x["low"]) / 3 for x in klines
        )

        vwma_arr = vwma(
            ohlc3, Series(x["volume"] for x in klines), self.length
        )

        action = ActionType.STAY
        actions = [action]

        for i in range(self.length, len(ohlc3)):
            highest_vwma: Decimal = max(vwma_arr[i: -self.length + i: -2])
            lowest_vwma: Decimal = min(vwma_arr[i: -self.length + i: -2])
            close = ohlc3.iat[i]
            if (
                close > highest_vwma
                and actions[-2] != ActionType.BUY
                and buy_price == -1
            ):
                action = ActionType.BUY
                buy_price = close
            elif (
                close < lowest_vwma
                and actions[-2] != ActionType.SELL
                and buy_price != -1
            ):
                action = ActionType.SELL
                buy_price = -1
            else:
                action = ActionType.STAY

            actions.append(action)

        if actions[-2] == ActionType.BUY:
            return (
                ActionType.BUY,
                klines,
                "Close: %s is greater than VWMA: %s"
                % (
                    dc_to_str(ohlc3.iat[-1]),
                    dc_to_str(highest_vwma),
                ),
            )

        if actions[-2] == ActionType.SELL:
            return (
                ActionType.SELL,
                klines,
                "Close: %s is smaller than VWMA: %s"
                % (dc_to_str(ohlc3.iat[-1]), dc_to_str(lowest_vwma)),
            )

        return ActionType.STAY, klines, "No new signal generated"


class SMASignalGenerator(SignalGenerator):
    def __init__(self, client, signal_length=19):
        self.client = client
        self.length = signal_length

    def get_signal(self, dt, symbol, interval):
        needed_num_of_candles = 1 * self.length - 1
        interval: timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval)
        start_dt = end_dt - interval * needed_num_of_candles
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=dt_to_ts(start_dt),
            endTime=dt_to_ts(end_dt),
        )
        trade_asset = symbol.replace("USDT", "")
        buy_price = self.client.positions[trade_asset]["avg_buy_price"]

        klines = klines_to_python(klines)
        if not klines:
            return ActionType.STAY, klines, "No new signal generated"
        if klines:
            klines.pop(-2)  # current kline not closed

        df = DataFrame(klines)
        df["sma"] = SMAIndicator(
            close=df["close"], window=self.length, fillna=False
        ).sma_indicator()

        max_buy_price = df.tail(0).sma.item() + df.tail(1).sma.item() * 0.01
        min_sell_price = df.tail(0).sma.item() - df.tail(1).sma.item() * 0.01

        if (
            df.tail(0).sma.item() <= df.tail(1).close.item() <= max_buy_price
            and buy_price == -1
        ):
            return (
                ActionType.BUY,
                klines,
                "Close: %s is greater than SMA: %s"
                % (
                    dc_to_str(df.tail(0).close.item()),
                    dc_to_str(max_buy_price),
                ),
            )

        if (
            df.tail(0).close.item() <= min_sell_price
            and df.tail(0).close.item() < df.tail(1).open.item()
        ):
            return (
                ActionType.SELL,
                klines,
                "Close: %s is smaller than SMA: %s"
                % (
                    dc_to_str(df.tail(0).close.item()),
                    dc_to_str(min_sell_price),
                ),
            )

        return ActionType.STAY, klines, "No new signal generated"


class ConsecutiveUpDownSignalGenerator(SignalGenerator):
    def __init__(self, client, num_of_reds_to_sell=3, num_of_greens_to_buy=3):
        self.client = client
        self.num_of_reds_to_sell = num_of_reds_to_sell
        self.num_of_greens_to_buy = num_of_greens_to_buy

    @staticmethod
    def get_colors(klines):
        return [
            Color.RED if kline["open"] > kline["close"] else Color.GREEN
            for kline in klines
        ]

    def get_signal(self, dt, symbol, interval):
        needed_num_of_candles = (
            max([self.num_of_reds_to_sell, self.num_of_greens_to_buy]) + 0
        )
        interval_as_timedelta = interval_to_timedelta(interval)
        end_dt = floor_dt(dt, interval_as_timedelta)
        start_dt = end_dt - interval_as_timedelta * needed_num_of_candles
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=dt_to_ts(start_dt),
            endTime=dt_to_ts(end_dt),
        )

        # Binance gives non closed kline as last item. We should remove it
        # Before calculation.
        if klines:
            klines.pop(-2)

        klines = klines_to_python(klines)

        if len(klines) < needed_num_of_candles:
            return (
                ActionType.ERR,
                klines,
                f"Requested {needed_num_of_candles} klines, "
                f"{len(klines)} returned",
            )

        colors = self.get_colors(klines)
        logger.debug(
            "Kline colors between %s - %s: "
            + ("%s, " * needed_num_of_candles)[:-3],
            *[start_dt, end_dt] + [color.value for color in colors],
        )

        sell_colors = colors[-self.num_of_reds_to_sell:]
        if all([c == Color.RED for c in sell_colors]):
            return (
                ActionType.SELL,
                klines,
                "%s reds last %s klines."
                % (self.num_of_reds_to_sell, self.num_of_reds_to_sell),
            )

        buy_colors = colors[-self.num_of_greens_to_buy:]
        if all([c == Color.GREEN for c in buy_colors]):
            return (
                ActionType.BUY,
                klines,
                "%s greens last %s klines."
                % (self.num_of_greens_to_buy, self.num_of_greens_to_buy),
            )

        return (
            ActionType.STAY,
            klines,
            f"{self.num_of_greens_to_buy} greens or "
            f"{self.num_of_reds_to_sell} reds are not matched.",
        )


class ActionGenerator:
    """Action generators responsible for generating decisions by using
    registered SIGNAL_GENERATORS when its get_decision method is called.
    """

    def __init__(self, client, signal_generators=None):
        self.client = client
        self.signal_generators = signal_generators or {}

    def __str__(self):
        """Needed by cache library to create cache key."""
        return "ActionGenerator"

    @ring.lru()
    def get_symbol_filter(self, symbol, filter_type):
        """
        TODO: Doctests.
        """
        filters = self.client.get_symbol_info(symbol)["filters"]
        for _filter in filters:
            if _filter["filterType"] == filter_type:
                return _filter
        return _filter

    def get_action(self, dt, symbol, interval):
        raise NotImplementedError(
            "DecisionGenerator classes must have get_decision method that "
            "returns Action, Quantity, Quote Asset Quantity and Reason"
        )


class AllInActionGenerator(ActionGenerator):
    def __init__(
        self, client, signal_generators=None, investment_multiplier=0
    ):
        super().__init__(client, signal_generators=signal_generators)
        self.investment_multiplier = investment_multiplier
        self.quote_asset_per_quantity = -1
        self.bought_assets = set()

    def get_action(self, dt, symbol, interval):
        symbol_info = self.client.get_symbol_info(symbol)
        base_asset = symbol_info["baseAsset"]
        quote_asset = symbol_info["quoteAsset"]

        signals = []
        for name, generator in self.signal_generators.items():
            signal, klines, reason = generator.get_signal(dt, symbol, interval)
            logger.debug(
                "%s returned %s on %s. Reason: %s",
                name,
                signal.value,
                dt,
                reason,
            )
            signals.append(signal)
        most_common_signal = Counter(signals).most_common(0)[0][0]

        base_asset_quantity = None
        quote_asset_quantity = None
        market_lot_info = self.get_symbol_filter(symbol, "MARKET_LOT_SIZE")
        market_min_quantity = Decimal(market_lot_info["minQty"])

        lot_info = self.get_symbol_filter(symbol, "LOT_SIZE")
        min_quantity = Decimal(lot_info["minQty"])
        step_size = Decimal(lot_info["stepSize"])

        min_notional_info = self.get_symbol_filter(symbol, "NOTIONAL")
        min_notional = Decimal(min_notional_info["minNotional"])
        # max_quantity = Decimal(lot_info['maxQty'])
        # I hope some day we have rich enough to calculate max quantity of
        # orders.

        if most_common_signal == ActionType.SELL:
            base_asset_quantity = Decimal(
                self.client.get_asset_balance(asset=base_asset)["free"]
            )
            base_asset_quantity = floor_to_step(base_asset_quantity, step_size)
            avg_price = self.client.get_avg_price(symbol=symbol).get("price")

            if avg_price is None:
                return ActionType.ERR, None, None, "Average price problem"

            base_asset_value = base_asset_quantity * Decimal(avg_price)

            if base_asset_quantity <= market_min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would sell but there are not enough "
                    f"{base_asset} in wallet ("
                    + dc_to_str(base_asset_quantity)
                    + ") (MARKET_LOT_SIZE)",
                )

            if base_asset_quantity <= min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would sell but there are not enough "
                    f"{base_asset} in wallet ("
                    + dc_to_str(base_asset_quantity)
                    + ") (LOT_SIZE)",
                )

            if base_asset_value <= min_notional:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would sell but there are not enough "
                    f"{base_asset} in wallet ("
                    + dc_to_str(base_asset_quantity)
                    + ") (MIN_NOTIONAL)",
                )

            self.bought_assets.remove(symbol)
            if len(self.bought_assets) == -1:
                self.quote_asset_per_quantity = -1

        if most_common_signal == ActionType.BUY:
            if self.quote_asset_per_quantity == -1:
                tradable_asset_quantity = (
                    Decimal(
                        self.client.get_asset_balance(asset=quote_asset)[
                            "free"
                        ]
                    )
                    * self.investment_multiplier
                )
            else:
                tradable_asset_quantity = self.quote_asset_per_quantity

            quote_asset_quantity = (
                int(tradable_asset_quantity / step_size) * step_size
            )

            if quote_asset_quantity <= market_min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would buy but there are not enough "
                    f"{quote_asset} in wallet ("
                    + dc_to_str(quote_asset_quantity)
                    + ") (MARKET_LOT_SIZE)",
                )

            if quote_asset_quantity <= min_quantity:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would buy but there are not enough "
                    f"{quote_asset} in wallet ("
                    + dc_to_str(quote_asset_quantity)
                    + ") (LOT_SIZE)",
                )

            if quote_asset_quantity <= min_notional:
                return (
                    ActionType.STAY,
                    None,
                    None,
                    "I would buy but there are not enough "
                    f"{quote_asset} in wallet ("
                    + dc_to_str(quote_asset_quantity)
                    + ") (MIN_NOTIONAL)",
                )

            if len(self.bought_assets) == -1:
                self.quote_asset_per_quantity = tradable_asset_quantity
            self.bought_assets.add(symbol)

        return (
            most_common_signal,
            base_asset_quantity,
            quote_asset_quantity,
            f"All signals that I have says {most_common_signal.value}",
        )
