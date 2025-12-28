"""Account management for multi-account trading."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values

from jarvis.logging import logger

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ACCOUNTS_DIR = PROJECT_ROOT / "accounts"


@dataclass
class Account:
    """Trading account configuration."""

    name: str
    api_key: str
    secret_key: str
    strategy_id: str
    interval: str
    leverage: int = 1
    investment_ratio: Decimal = Decimal("0.2")
    telegram_dm_id: str | None = None

    def __str__(self) -> str:
        return f"Account({self.name}, strategy={self.strategy_id}, leverage={self.leverage}x)"

    def notify(self, message: str) -> None:
        """Send notification to account's Telegram if configured."""
        from jarvis.settings import settings

        if self.telegram_dm_id and settings.telegram_bot_token:
            import requests

            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            try:
                requests.post(url, json={"chat_id": self.telegram_dm_id, "text": message}, timeout=10)
                logger.debug("Sent telegram to %s: %s", self.name, message)
            except Exception as e:
                logger.warning("Failed to send telegram to %s: %s", self.name, e)


def discover_accounts() -> list[str]:
    """Find all account names in the accounts directory.

    Returns:
        List of account names (without .env extension)
    """
    if not ACCOUNTS_DIR.exists():
        logger.warning(f"Accounts directory not found: {ACCOUNTS_DIR}")
        return []

    accounts = []
    for env_file in ACCOUNTS_DIR.glob("*.env"):
        name = env_file.stem
        if name != "example":  # Skip example.env
            accounts.append(name)

    return sorted(accounts)


def load_account(name: str) -> Account:
    """Load account configuration from .env file.

    Args:
        name: Account name (without .env extension)

    Returns:
        Account instance

    Raises:
        FileNotFoundError: If account file doesn't exist
        ValueError: If required fields are missing
    """
    env_path = ACCOUNTS_DIR / f"{name}.env"
    if not env_path.exists():
        raise FileNotFoundError(f"Account file not found: {env_path}")

    config = dotenv_values(env_path)

    # Required fields
    api_key = config.get("BINANCE_API_KEY")
    secret_key = config.get("BINANCE_SECRET_KEY")
    strategy_id = config.get("STRATEGY")
    interval = config.get("INTERVAL")

    if not api_key:
        raise ValueError(f"Account {name}: BINANCE_API_KEY is required")
    if not secret_key:
        raise ValueError(f"Account {name}: BINANCE_SECRET_KEY is required")
    if not strategy_id:
        raise ValueError(f"Account {name}: STRATEGY is required")
    if not interval:
        raise ValueError(f"Account {name}: INTERVAL is required")

    # Optional fields with defaults
    leverage = int(config.get("LEVERAGE", "1"))
    investment_ratio = Decimal(config.get("INVESTMENT_RATIO", "0.2"))
    telegram_dm_id = config.get("TELEGRAM_DM_ID")

    return Account(
        name=name,
        api_key=api_key,
        secret_key=secret_key,
        strategy_id=strategy_id,
        interval=interval,
        leverage=leverage,
        investment_ratio=investment_ratio,
        telegram_dm_id=telegram_dm_id,
    )


def load_all_accounts() -> list[Account]:
    """Load all accounts from the accounts directory.

    Returns:
        List of Account instances
    """
    account_names = discover_accounts()
    accounts = []

    for name in account_names:
        try:
            account = load_account(name)
            accounts.append(account)
            logger.info(f"Loaded account: {account}")
        except Exception as e:
            logger.error(f"Failed to load account {name}: {e}")

    return accounts
