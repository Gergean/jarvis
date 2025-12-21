"""Application settings for the Jarvis trading system."""

import os
from datetime import datetime
from decimal import Decimal
from functools import cache

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

    binance_api_key: str | None = None
    binance_secret_key: str | None = None
    debug: bool = False
    sentry_dsn: str | None = None
    telegram_bot_token: str | None = None
    telegram_dm_id: str | None = None
    telegram_gm_id: str | None = None
    telegram_gm_prefix: str = ""

    # Indicator warmup start date. Technical indicators (SuperTrend, VWMA, SMA) need
    # historical data to "warm up" before producing valid signals. We fetch data from
    # this date to ensure indicators have enough history. The data is cached both in
    # memory (@ring.lru) and on disk (CSV files), so only the first run is slow.
    indicator_warmup_start: datetime = Field(default=datetime(2023, 1, 1))

    commission_ratio: Decimal = Decimal("0.001")
    investment_ratio: Decimal = Decimal("0.2")


@cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Module-level settings instance
settings = get_settings()


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
