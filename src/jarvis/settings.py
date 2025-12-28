"""Application settings for the Jarvis trading system."""

import os
from datetime import datetime
from decimal import Decimal
from functools import cache
from pathlib import Path

import requests
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis.logging import logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === API Credentials ===
    binance_api_key: str | None = None
    binance_secret_key: str | None = None

    # === General Settings ===
    debug: bool = False
    sentry_dsn: str | None = None

    # === Telegram Notifications ===
    telegram_bot_token: str | None = None
    telegram_dm_id: str | None = None
    telegram_gm_id: str | None = None
    telegram_gm_prefix: str = ""

    # === Indicator Settings ===
    # Technical indicators (SuperTrend, VWMA, SMA) need historical data to "warm up"
    # before producing valid signals. Data is cached in memory and on disk (CSV files).
    indicator_warmup_start: datetime = Field(default=datetime(2023, 1, 1))

    # === Trading Settings ===
    commission_ratio: Decimal = Decimal("0.001")
    investment_ratio: Decimal = Decimal("0.2")

@cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Module-level settings instance
settings = get_settings()

# === Futures Trading Constants ===
# Binance default fees (VIP0): Maker 0.02%, Taker 0.05%
# Verified against actual trades: 2024-12-26
FUTURES_TAKER_FEE = Decimal("0.0005")  # 0.05%
FUTURES_MAKER_FEE = Decimal("0.0002")  # 0.02%
FUNDING_FEE_RATE = Decimal("0.0001")   # 0.01% per 8 hours
FUNDING_INTERVAL_HOURS = 8
DEFAULT_LEVERAGE = 1
MAX_LEVERAGE = 10

# === Paper Trading Constants ===
PAPER_DIR = Path("paper")
ELITES_DIR = Path("strategies/elites")
EVOLVE_POPULATION_SIZE = 30
EVOLVE_GENERATIONS = 10
EVOLVE_LOOKBACK_DAYS = 30

# === Download Constants ===
RATE_LIMIT_DELAY = 0.1  # 100ms between requests

# === Indicator Period Constants ===
# Random indicator periods are generated within this range (in days)
# Using logarithmic distribution: 10^uniform(log10(MIN), log10(MAX))
MIN_INDICATOR_DAYS = 0.1   # ~2.4 hours
MAX_INDICATOR_DAYS = 90.0  # 3 months
MIN_INDICATOR_BARS = 2     # Absolute minimum bars (regardless of interval)

# MACD uses shorter periods since it compares two EMAs
# Standard MACD is 12/26/9 which at 1h = 0.5/1.1/0.4 days
MACD_FAST_MIN_DAYS = 0.1   # ~2.4 hours
MACD_FAST_MAX_DAYS = 5.0   # 5 days
MACD_SLOW_MIN_DAYS = 0.5   # 12 hours
MACD_SLOW_MAX_DAYS = 15.0  # 15 days
MACD_SIGNAL_MIN_DAYS = 0.1 # ~2.4 hours
MACD_SIGNAL_MAX_DAYS = 3.0 # 3 days


def __notify(token: str, chat_id: str, message: str) -> dict[str, object]:
    """Send a message via Telegram bot API."""
    send_text = (
        "https://api.telegram.org/bot"
        + token
        + "/sendMessage?chat_id="
        + str(chat_id)
        + "&parse_mode=Markdown&text="
        + str(message)
    )
    response = requests.get(send_text)
    result: dict[str, object] = response.json()
    return result


def notify(message: str) -> None:
    """Send notification to configured Telegram channels."""
    if not settings.telegram_bot_token:
        return
    if settings.telegram_dm_id:
        __notify(settings.telegram_bot_token, settings.telegram_dm_id, message)
    if settings.telegram_gm_id:
        __notify(settings.telegram_bot_token, settings.telegram_gm_id, settings.telegram_gm_prefix + message)
    logger.debug("Sent telegram message:\n%s", message)
