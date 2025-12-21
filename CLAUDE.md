# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jarvis is a cryptocurrency trading automation system that analyzes market data from Binance and executes trading decisions based on configurable signal generators.

## Common Commands

```bash
# Install dependencies
uv sync

# Lint and format
uv run ruff check src
uv run ruff format src

# Type checking
uv run pyright

# Run doctests (primary test method)
uv run python src/jarvis.py doctest -v

# Run a backtest
uv run python src/jarvis.py backtest -ba USDT -ta BTC ETH -i 1h -st 2020-01-01T00:00:00 -et 2020-12-01T00:00:00

# Execute live trade
uv run python src/jarvis.py trade -ba USDT -ta BTC ETH -i 1h -ir 0.2 -ep .env
```

## Architecture

### Core Engine (`src/jarvis/`)

Modular Python package containing:

**Signal Generators** (Strategy Pattern):
- `SignalGenerator` - Abstract base class with `get_signal(dt, symbol, interval)`
- `SuperTrendSignalGenerator` - Uses SuperTrend indicator with ATR (factor=3, atr_period=10)
- `VWMASignalGenerator` - Volume-Weighted Moving Average (signal_length=20)
- `SMASignalGenerator` - Simple Moving Average requiring position tracking

**Action Generators**:
- `ActionGenerator` - Abstract base with `get_action()`
- `AllInActionGenerator` - Combines signals via voting, calculates tradable amount from live balance, validates Binance order constraints (MARKET_LOT_SIZE, MIN_NOTIONAL, stepSize)

**Key Data Structures**:
```python
ActionType(Enum): BUY, SELL, STAY, ERR
Position(dataclass): symbol, spent (Decimal), amount (Decimal)
COMMISSION_RATIO = 0.001  # 0.1%
INVESTMENT_RATIO = 0.2    # 20% per trade
```

**Main Functions**:
- `backtest()` - Simulates trades on historical CSV data using FakeClient
- `trade()` - Live execution using real Binance client for orders, FakeClient for klines
- `FakeClient` - Mock Binance client reading from CSV files for deterministic testing

### Data Storage

- `data/binance/{SYMBOL}/{interval}/YYYYMMDD.csv` - Historical OHLCV klines
- `logs/backtest.log` - Rotating log files (5 backups)

## Environment Variables

Required in `.env`:
```
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
```

Optional:
```
TELEGRAM_BOT_TOKEN=...     # Trade notifications
TELEGRAM_DM_ID=...         # Private chat ID
TELEGRAM_GM_ID=...         # Group chat ID
DEBUG=True                 # Enables debug logs + Telegram
SENTRY_DSN=...             # Error tracking
```

## Key Dependencies

- `python-binance` - Binance API client
- `ta`, `pandas-ta` - Technical indicators (SMA, VWMA, SuperTrend)
- `mplfinance` - Chart visualization
- `ring` - LRU caching decorator

## Caching

Functions decorated with `@ring.lru()` cache expensive operations. Manual invalidation: `load_day_file.delete(...)`.
