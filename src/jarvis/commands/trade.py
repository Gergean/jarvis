"""Trade command for the Jarvis futures trading system."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

from jarvis.accounts import Account, load_account, load_all_accounts
from jarvis.client import get_binance_client, get_futures_client
from jarvis.commands.evolve import evolve_strategy
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.strategy import Strategy
from jarvis.logging import logger
from jarvis.models import DEFAULT_LEVERAGE, ActionType, PositionSide
from jarvis.settings import notify, settings
from jarvis.utils import datetime_to_timestamp, floor_to_step


def get_signal_from_strategy(
    client: Any,
    strategy: Strategy,
    interval: str,
    dt: datetime,
    current_side: PositionSide = PositionSide.NONE,
) -> tuple[ActionType, float]:
    """Get trading signal from a GA strategy.

    Args:
        client: Binance client (real or fake)
        strategy: GA strategy to use
        interval: Kline interval
        dt: Current datetime
        current_side: Current position direction

    Returns:
        Tuple of (ActionType signal, score value)
    """
    symbol = strategy.symbol
    individual = strategy.individual

    # Get klines for signal calculation
    end_ts = datetime_to_timestamp(dt)
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=200, endTime=end_ts)
        if not klines or len(klines) < 50:
            logger.warning("Not enough klines for %s", symbol)
            return ActionType.STAY, 0.0
    except Exception as e:
        logger.error("Failed to get klines for %s: %s", symbol, e)
        return ActionType.ERR, 0.0

    # Convert to numpy arrays for OHLCV
    n = len(klines)
    open_arr = np.zeros(n, dtype=np.float64)
    high_arr = np.zeros(n, dtype=np.float64)
    low_arr = np.zeros(n, dtype=np.float64)
    close_arr = np.zeros(n, dtype=np.float64)
    volume_arr = np.zeros(n, dtype=np.float64)

    for i, k in enumerate(klines):
        open_arr[i] = float(k.open)
        high_arr[i] = float(k.high)
        low_arr[i] = float(k.low)
        close_arr[i] = float(k.close)
        volume_arr[i] = float(k.volume)

    ohlcv = OHLCV(
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
    )

    # Get score and signal from individual
    score = individual.get_total_score(ohlcv)
    signal = individual.get_signal(ohlcv, current_side)
    return signal, score


def trade_with_strategies(
    strategy_ids: list[str],
    interval: str = "1h",
    investment_ratio: Decimal = Decimal("0.2"),
    leverage: int = DEFAULT_LEVERAGE,
    strategies_dir: str = "strategies",
    dry_run: bool = False,
    api_key: str | None = None,
    secret_key: str | None = None,
    account: Account | None = None,
) -> list[str]:
    """Execute live futures trading using GA strategies.

    Args:
        strategy_ids: List of strategy IDs to use (e.g., ["BTCUSDT_fe43f298"])
        interval: Kline interval for signal calculation
        investment_ratio: Portion of margin to trade per signal
        leverage: Futures leverage (1-10)
        strategies_dir: Directory containing strategy files
        dry_run: If True, only show signals without executing trades
        api_key: Binance API key (optional, falls back to settings)
        secret_key: Binance secret key (optional, falls back to settings)
        account: Account instance for notifications (optional)

    Returns:
        List of strategy IDs that had position changes (for evolution)
    """
    changed_strategies: list[str] = []
    # Helper for notifications - use account.notify if available
    def send_notify(message: str) -> None:
        if account:
            account.notify(message)
        else:
            notify(message)
    # In dry-run mode, use fake client with CSV data
    if dry_run:
        client = get_binance_client(fake=True, extra_params={"assets": {"USDT": Decimal("10000")}})
        # Use yesterday's date for dry-run (historical data available)
        dt = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
        print("=== DRY RUN MODE (Futures) ===")
        print(f"Using historical data from: {dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"Leverage: {leverage}x")
    else:
        client = get_binance_client()
        dt = datetime.now(UTC).replace(tzinfo=None)

    # Load strategies
    strategies: list[Strategy] = []
    for strategy_id in strategy_ids:
        try:
            strategy = Strategy.load_by_id(strategy_id, strategies_dir)
            strategies.append(strategy)
            logger.info("Loaded strategy: %s (%d rules)", strategy.id, len(strategy.individual.rules))
        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_id, e)

    if not strategies:
        logger.error("No strategies loaded!")
        return []

    # Track positions per symbol
    positions: dict[str, PositionSide] = {}

    # In dry-run mode, skip balance check
    if dry_run:
        margin_balance = Decimal("10000")
        futures_client = None
    else:
        # Use provided keys or fall back to settings
        effective_api_key = api_key or settings.binance_api_key
        effective_secret_key = secret_key or settings.binance_secret_key

        # Check API keys
        if not effective_api_key or not effective_secret_key:
            logger.error("API keys not configured! Add BINANCE_API_KEY and BINANCE_SECRET_KEY to account file")
            return []

        # Initialize futures client
        try:
            futures_client = get_futures_client(effective_api_key, effective_secret_key)
            margin_balance = futures_client.get_balance()
            logger.info("Futures account balance: $%.2f USDT", margin_balance)

            # Load current positions from API
            for strategy in strategies:
                symbol = strategy.symbol
                side_str = futures_client.get_position_side(symbol)
                if side_str == "LONG":
                    positions[symbol] = PositionSide.LONG
                    logger.info("[%s] Existing position: LONG", symbol)
                elif side_str == "SHORT":
                    positions[symbol] = PositionSide.SHORT
                    logger.info("[%s] Existing position: SHORT", symbol)
                else:
                    positions[symbol] = PositionSide.NONE

            # Check balance only if no positions open (need balance for new trades)
            has_positions = any(p != PositionSide.NONE for p in positions.values())
            if margin_balance < Decimal("1") and not has_positions:
                logger.error("Insufficient balance: $%.2f", margin_balance)
                return []

        except Exception as e:
            logger.error("Failed to connect to futures API: %s", e)
            return []

    for strategy in strategies:
        symbol = strategy.symbol
        current_side = positions.get(symbol, PositionSide.NONE)

        # Get PnL if position exists
        pnl_str = ""
        if current_side != PositionSide.NONE and futures_client:
            try:
                pos = futures_client.get_position(symbol)
                pnl = float(pos.get("unRealizedProfit", 0))
                pnl_pct = (pnl / float(margin_balance)) * 100 if margin_balance > 0 else 0
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                pnl_str = f" | {pnl_emoji} ${pnl:+.2f} ({pnl_pct:+.1f}%)"
            except Exception:
                pass

        # Get signal from strategy with position awareness
        signal, score = get_signal_from_strategy(client, strategy, interval, dt, current_side)
        logger.info("[%s] Position: %s, Signal: %s, Score: %.2f", symbol, current_side.value, signal.value, score)

        # Thresholds for context
        long_th = strategy.individual.LONG_THRESHOLD
        short_th = strategy.individual.SHORT_THRESHOLD

        if signal == ActionType.STAY:
            if current_side != PositionSide.NONE:
                trend_note = ""
                if score > long_th * 100:
                    trend_note = "Rüzgar arkamızda! Yelkenler fora, rotamız doğru. ⛵"
                elif score > long_th:
                    trend_note = "Yolumuz açık, ilerlemeye devam. 🚶"
                elif score > 0:
                    trend_note = "Sisler yükseliyor... Dikkatli olmalıyız. 🌫️"
                elif score > short_th:
                    trend_note = "Karanlık bulutlar beliriyor ufukta... ⛈️"
                else:
                    trend_note = "Tehlike! Fırtına kapıda ama henüz sığınakta değiliz. 🏚️"

                message = f"⚔️ {symbol} {current_side.value} seferinde ilerliyorum.{pnl_str}\n{trend_note}\n[Pusula: {score:.1f}]"
            else:
                if score > long_th:
                    trend_note = "Macera çağırıyor! AL kapısı açık... ama henüz adım atmadım. 🚪"
                elif score > 0:
                    trend_note = "Ormanda sessizlik var. Bekle ve gözle... 🦉"
                elif score > short_th:
                    trend_note = "Gölgeler dans ediyor. Henüz harekete geçmiyorum. 🌑"
                else:
                    trend_note = "Karanlık güçler uyanıyor! SAT kapısı açık... 👁️"
                message = f"🏕️ {symbol}: Kamp kurdum, bekliyorum.\n{trend_note}\n[Pusula: {score:.1f}]"
            logger.info(message)
            send_notify(message)
            if dry_run:
                print(message)
            continue

        if signal == ActionType.ERR:
            message = f"💀 {symbol}: Lanetli topraklarda kayboldum! Haritayı kontrol et... 🗺️"
            logger.warning(message)
            send_notify(message)
            if dry_run:
                print(message)
            continue

        # In dry-run mode, print signals and send Telegram notification
        if dry_run:
            if signal == ActionType.LONG:
                margin_used = margin_balance * investment_ratio
                notional = margin_used * leverage
                msg = f"📡 [DRY-RUN] {symbol}: LONG sinyali!\n[Pusula: {score:.1f}]"
                print(f"[{symbol}] LONG: ${margin_used:.2f} margin x {leverage}x = ${notional:.2f} position")
                send_notify(msg)
                positions[symbol] = PositionSide.LONG
            elif signal == ActionType.SHORT:
                margin_used = margin_balance * investment_ratio
                notional = margin_used * leverage
                msg = f"📡 [DRY-RUN] {symbol}: SHORT sinyali!\n[Pusula: {score:.1f}]"
                print(f"[{symbol}] SHORT: ${margin_used:.2f} margin x {leverage}x = ${notional:.2f} position")
                send_notify(msg)
                positions[symbol] = PositionSide.SHORT
            elif signal == ActionType.CLOSE:
                msg = f"📡 [DRY-RUN] {symbol}: CLOSE sinyali! ({current_side.value} kapat)\n[Pusula: {score:.1f}]"
                print(f"[{symbol}] CLOSE {current_side.value} position")
                send_notify(msg)
                positions[symbol] = PositionSide.NONE
            continue

        # === REAL TRADING ===
        try:
            # Set leverage
            futures_client.set_leverage(symbol, leverage)

            if signal == ActionType.LONG and current_side == PositionSide.NONE:
                # Calculate position size
                margin_to_use = margin_balance * investment_ratio
                price = futures_client.get_current_price(symbol)
                notional = float(margin_to_use) * leverage
                quantity = Decimal(str(notional / price))

                # Round to step size
                step_size = futures_client.get_step_size(symbol)
                quantity = floor_to_step(quantity, step_size)

                if quantity > 0:
                    logger.info("[%s] Opening LONG: qty=%.4f, notional=$%.2f", symbol, quantity, notional)
                    order = futures_client.open_long(symbol, quantity)
                    logger.info("[%s] LONG order executed: %s", symbol, order.get("orderId"))
                    send_notify(f"⚔️ {symbol}: Kuzey seferine çıktım! {quantity} altın yatırdım, macera başlasın! 🏔️\nGiriş: ${price:.2f}")
                    positions[symbol] = PositionSide.LONG
                    changed_strategies.append(strategy.id)

            elif signal == ActionType.SHORT and current_side == PositionSide.NONE:
                # Calculate position size
                margin_to_use = margin_balance * investment_ratio
                price = futures_client.get_current_price(symbol)
                notional = float(margin_to_use) * leverage
                quantity = Decimal(str(notional / price))

                # Round to step size
                step_size = futures_client.get_step_size(symbol)
                quantity = floor_to_step(quantity, step_size)

                if quantity > 0:
                    logger.info("[%s] Opening SHORT: qty=%.4f, notional=$%.2f", symbol, quantity, notional)
                    order = futures_client.open_short(symbol, quantity)
                    logger.info("[%s] SHORT order executed: %s", symbol, order.get("orderId"))
                    send_notify(f"🗡️ {symbol}: Güney seferine çıktım! {quantity} altın yatırdım, karanlığa dalıyorum! 🌋\nGiriş: ${price:.2f}")
                    positions[symbol] = PositionSide.SHORT
                    changed_strategies.append(strategy.id)

            elif signal == ActionType.CLOSE and current_side != PositionSide.NONE:
                logger.info("[%s] Closing %s position", symbol, current_side.value)
                order = futures_client.close_position(symbol)
                if order:
                    logger.info("[%s] CLOSE order executed: %s", symbol, order.get("orderId"))
                    if pnl_str:
                        send_notify(f"🏠 {symbol}: Seferden döndüm!{pnl_str}\nYorgunuz ama ayaktayız. Kamp kuruyorum... 🏕️")
                    else:
                        send_notify(f"🏠 {symbol}: Seferden döndüm! Kamp kuruyorum... 🏕️")
                    changed_strategies.append(strategy.id)
                positions[symbol] = PositionSide.NONE

        except Exception as e:
            logger.error("[%s] Trade execution failed: %s", symbol, e)
            send_notify(f"💀 {symbol}: Tuzağa düştüm! {e}\nYaralarımı sarıyorum... 🩹")

    # Print summary in dry-run mode
    if dry_run:
        print("\n=== Position Summary ===")
        for symbol, side in positions.items():
            if side != PositionSide.NONE:
                print(f"  {symbol}: {side.value}")
        if not any(s != PositionSide.NONE for s in positions.values()):
            print("  No open positions")

    return changed_strategies


def trade(account_name: str | None = None, dry_run: bool = False) -> None:
    """Execute futures trading for configured accounts.

    Args:
        account_name: Specific account to trade (None = all accounts)
        dry_run: If True, only show signals without executing trades
    """
    # Load accounts
    if account_name:
        try:
            accounts = [load_account(account_name)]
        except FileNotFoundError:
            logger.error("Account not found: %s", account_name)
            return
        except ValueError as e:
            logger.error("Account config error: %s", e)
            return
    else:
        accounts = load_all_accounts()

    if not accounts:
        logger.error("No accounts found! Create account files in accounts/ directory.")
        logger.info("Copy accounts/example.env to accounts/yourname.env and configure it.")
        return

    logger.info("Trading with %d account(s)", len(accounts))

    for account in accounts:
        logger.info("=" * 50)
        logger.info("Account: %s", account.name)
        logger.info("Strategy: %s", account.strategy_id)
        logger.info("Leverage: %dx", account.leverage)
        logger.info("=" * 50)

        changed = trade_with_strategies(
            strategy_ids=[account.strategy_id],
            interval=account.interval,
            investment_ratio=account.investment_ratio,
            leverage=account.leverage,
            dry_run=dry_run,
            api_key=account.api_key,
            secret_key=account.secret_key,
            account=account,
        )

        # Evolve strategies that had position changes
        if changed and not dry_run:
            for strategy_id in changed:
                logger.info("Evolving strategy after trade: %s", strategy_id)
                evolve_strategy(
                    strategy_id=strategy_id,
                    interval=account.interval,
                    lookback_days=60,
                    generations=30,
                    population_size=50,
                    leverage=account.leverage,
                )
