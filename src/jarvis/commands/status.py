"""Status command - Display account positions and PnL."""

from datetime import datetime

from binance.client import Client

from jarvis.accounts import Account, load_account, load_all_accounts
from jarvis.logging import logger
from jarvis.settings import notify


def get_account_status(account: Account) -> dict:
    """Get position and PnL status for an account."""
    client = Client(account.api_key, account.secret_key)

    # Get account info
    account_info = client.futures_account()
    total_balance = float(account_info["totalWalletBalance"])
    unrealized_pnl = float(account_info["totalUnrealizedProfit"])
    margin_balance = float(account_info["totalMarginBalance"])

    # Get position info
    symbol = account.strategy_id.split("_")[0] if account.strategy_id else None
    position_data = None

    if symbol:
        positions = client.futures_position_information(symbol=symbol)
        for p in positions:
            if float(p["positionAmt"]) != 0:
                qty = float(p["positionAmt"])
                entry = float(p["entryPrice"])
                mark = float(p["markPrice"])
                pnl = float(p["unRealizedProfit"])
                notional = abs(qty) * entry
                pnl_pct = (pnl / notional) * 100 if notional > 0 else 0

                position_data = {
                    "symbol": symbol,
                    "side": "LONG" if qty > 0 else "SHORT",
                    "quantity": abs(qty),
                    "entry_price": entry,
                    "mark_price": mark,
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct": pnl_pct,
                    "notional": notional,
                }
                break

    # Get recent income history
    income_history = []
    if symbol:
        try:
            income = client.futures_income_history(symbol=symbol, limit=20)
            for i in income[-10:]:
                ts = datetime.fromtimestamp(i["time"] / 1000)
                income_history.append({
                    "time": ts,
                    "type": i["incomeType"],
                    "amount": float(i["income"]),
                    "symbol": i.get("symbol", ""),
                })
        except Exception as e:
            logger.warning("Could not fetch income history: %s", e)

    return {
        "account_name": account.name,
        "strategy_id": account.strategy_id,
        "total_balance": total_balance,
        "unrealized_pnl": unrealized_pnl,
        "margin_balance": margin_balance,
        "position": position_data,
        "income_history": income_history,
    }


def status(account_name: str | None = None, send_notification: bool = False) -> list[dict]:
    """Display status for all accounts or a specific account.

    Args:
        account_name: Specific account name, or None for all accounts
        send_notification: If True, send summary to Telegram

    Returns:
        List of account status dictionaries
    """
    if account_name:
        accounts = [load_account(account_name)]
    else:
        accounts = load_all_accounts()

    if not accounts:
        logger.error("No accounts found!")
        return []

    results = []
    notification_lines = ["📊 *Günlük Özet*", ""]

    for account in accounts:
        try:
            status_data = get_account_status(account)
            results.append(status_data)

            # Print status
            print("=" * 60)
            print(f"Account: {status_data['account_name']}")
            print(f"Strategy: {status_data['strategy_id']}")
            print("-" * 60)

            # Account summary
            print(f"Total Balance:   ${status_data['total_balance']:.2f}")
            print(f"Unrealized PnL:  ${status_data['unrealized_pnl']:.4f}")
            print(f"Margin Balance:  ${status_data['margin_balance']:.2f}")

            # Position
            pos = status_data["position"]
            if pos:
                print("-" * 60)
                print(f"Position: {pos['side']} {pos['quantity']} {pos['symbol']}")
                print(f"Entry:    ${pos['entry_price']:.2f}")
                print(f"Mark:     ${pos['mark_price']:.2f}")
                pnl = pos["unrealized_pnl"]
                pnl_pct = pos["unrealized_pnl_pct"]
                pnl_sign = "+" if pnl >= 0 else ""
                print(f"PnL:      {pnl_sign}${pnl:.4f} ({pnl_sign}{pnl_pct:.2f}%)")
            else:
                print("-" * 60)
                print("Position: NONE")

            # Recent income
            history = status_data["income_history"]
            if history:
                print("-" * 60)
                print("Recent Activity:")
                for item in history[-5:]:
                    ts = item["time"].strftime("%m-%d %H:%M")
                    amount = item["amount"]
                    sign = "+" if amount >= 0 else ""
                    print(f"  {ts} | {item['type']:15} | {sign}${amount:.6f}")

            print("=" * 60)
            print()

            # Build notification message
            if send_notification:
                notification_lines.append(f"*{status_data['account_name']}*")
                notification_lines.append(f"Bakiye: ${status_data['total_balance']:.2f}")

                pos = status_data["position"]
                if pos:
                    pnl = pos["unrealized_pnl"]
                    pnl_pct = pos["unrealized_pnl_pct"]
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    notification_lines.append(
                        f"{emoji} {pos['side']} {pos['quantity']} {pos['symbol']}"
                    )
                    notification_lines.append(
                        f"Giriş: ${pos['entry_price']:.2f} → ${pos['mark_price']:.2f}"
                    )
                    sign = "+" if pnl >= 0 else ""
                    notification_lines.append(f"PnL: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)")
                else:
                    notification_lines.append("Pozisyon: YOK")

                notification_lines.append("")

        except Exception as e:
            logger.error("Error getting status for %s: %s", account.name, e)
            results.append({
                "account_name": account.name,
                "error": str(e),
            })

    # Send notification if requested
    if send_notification and results:
        notify("\n".join(notification_lines))
        logger.info("Sent daily summary notification")

    return results
