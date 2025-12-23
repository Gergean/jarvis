"""Download command for fetching historical kline data from Binance."""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from binance.client import Client

from jarvis.logging import logger


def download(
    symbols: list[str],
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    data_dir: str = "data/binance",
) -> dict[str, int]:
    """Download historical kline data for multiple symbols.

    Uses Binance public API (no authentication required).

    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        interval: Kline interval (e.g., "1h", "4h", "1d")
        start_dt: Start date (defaults to 1 year ago)
        end_dt: End date (defaults to now)
        data_dir: Base directory for data storage

    Returns:
        Dictionary mapping symbol to number of klines downloaded
    """
    # Default dates
    if end_dt is None:
        end_dt = datetime.utcnow()
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365)

    logger.info("Downloading data for %d symbols", len(symbols))
    logger.info("Period: %s to %s", start_dt.date(), end_dt.date())
    logger.info("Interval: %s", interval)

    # Public client - no API key needed
    client = Client()

    results = {}

    for symbol in symbols:
        logger.info("Downloading %s...", symbol)

        all_klines = []
        current_start = start_dt

        while current_start < end_dt:
            start_ts = int(current_start.timestamp() * 1000)
            batch_end = min(current_start + timedelta(days=40), end_dt)
            end_ts = int(batch_end.timestamp() * 1000)

            try:
                klines = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_ts,
                    endTime=end_ts,
                    limit=1000,
                )
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                break

            if not klines:
                break

            all_klines.extend(klines)

            # Move to next batch
            last_time = klines[-1][0]
            current_start = datetime.fromtimestamp(last_time / 1000) + timedelta(hours=1)

        if not all_klines:
            logger.warning("No data fetched for %s", symbol)
            results[symbol] = 0
            continue

        # Save to CSV files (one per day)
        symbol_dir = Path(data_dir) / symbol / interval
        symbol_dir.mkdir(parents=True, exist_ok=True)

        by_day: dict[str, list] = defaultdict(list)
        for k in all_klines:
            dt = datetime.fromtimestamp(k[0] / 1000)
            by_day[dt.strftime("%Y%m%d")].append(k)

        for day_str, klines in by_day.items():
            file_path = symbol_dir / f"{day_str}.csv"
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                for k in klines:
                    writer.writerow(k)

        logger.info("  %s: %d klines, %d files -> %s", symbol, len(all_klines), len(by_day), symbol_dir)
        results[symbol] = len(all_klines)

    logger.info("Download complete")
    return results
