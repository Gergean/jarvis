# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jarvis is a Binance Futures trading automation system using genetic algorithm-based strategies. It can train strategies that go long/short with configurable leverage.

## Common Commands

```bash
# Install dependencies
uv sync

# Lint and format
uv run ruff check src
uv run ruff format src

# Type checking
uv run mypy

# Run doctests
uv run python src/jarvis.py doctest -v

# Download data
uv run python src/jarvis.py download -s BTCUSDT -i 1h

# Train a strategy (walk-forward validation - default)
uv run python src/jarvis.py train -s BTCUSDT -i 1h -l 5

# Train with custom walk-forward periods
uv run python src/jarvis.py train -s BTCUSDT -i 1h --train-period 90d --test-period 2w --step-period 1w

# Train without walk-forward (not recommended)
uv run python src/jarvis.py train -s BTCUSDT -i 1h --no-walk-forward

# Test a strategy (out-of-sample)
uv run python src/jarvis.py test -s BTCUSDT_abc123 -i 1h -l 5

# Trade with strategies (dry run)
uv run python src/jarvis.py trade-ga -s BTCUSDT_abc123 --dry-run

# Paper trading
uv run python src/jarvis.py paper init test1 -b 1000 -c BTCUSDT:1h -s BTCUSDT_abc123
uv run python src/jarvis.py paper trade test1 -et 2025-10-15T00:00:00
uv run python src/jarvis.py paper info test1

# Pine Script export
uv run python src/jarvis.py pinescript -s BTCUSDT_abc123
```

## Architecture

### Core Engine (`src/jarvis/`)

Modular Python package containing:

**Genetics System** (`genetics/`):
- `Individual` - Trading strategy with weighted rules
- `Population` - Collection of individuals that evolve
- `Rule` - Single indicator condition with weight
- `Strategy` - Saved individual with metadata

**Commands** (`commands/`):
- `train` - Train GA strategies with walk-forward validation (default)
- `test` - Out-of-sample testing
- `trade` - Live futures trading (dry-run or real)
- `paper` - Paper trading simulation with elites system
- `download` - Fetch historical klines
- `pinescript` - Export strategy to TradingView Pine Script

**Walk-Forward Validation**:
Train uses rolling windows by default to prevent overfitting:
- `--train-period 3M` - Training period per window (default: 3 months)
- `--test-period 1M` - Test period per window (default: 1 month)
- `--step-period 1M` - Step size between windows (default: 1 month)
- Period formats: `Nd` (days), `Nw` (weeks), `NM` (months)

**Elites System** (for paper trading):
- Strategies evolve daily at 00:00 UTC
- Stored in `strategies/elites/{SYMBOL}/{interval}/YYYYMMDD_HHMMSS.json`
- Prevents "time travel" - paper trade uses only strategies available at that time
- Seed strategy required to start evolution chain

**Key Data Structures**:
```python
ActionType(Enum): LONG, SHORT, CLOSE, STAY, ERR
PositionSide(Enum): LONG, SHORT, NONE
FUTURES_TAKER_FEE = 0.0004  # 0.04%
FUNDING_FEE_RATE = 0.0001   # 0.01% per 8h
DEFAULT_LEVERAGE = 1
MAX_LEVERAGE = 10
```

### Data Storage

- `data/binance/{SYMBOL}/{interval}/YYYYMMDD.csv` - Historical OHLCV klines
- `strategies/*.json` - Saved GA strategies
- `strategies/elites/` - Daily evolved elite strategies
- `paper/*.json` - Paper trading wallets
- `results/*.json` - Test results

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
SENTRY_DSN=...             # Error tracking
```

## Key Dependencies

- `python-binance` - Binance API client
- `ta`, `pandas-ta`, `ta-lib` - Technical indicators
- `enlighten` - Progress bars
- `ring` - LRU caching

## Caching

Functions decorated with `@ring.lru()` cache expensive operations. Manual invalidation: `load_day_file.delete(...)`.

## Module Reference

### `jarvis/utils.py` - Utility Functions

**Time/Interval Conversion:**
- `interval_to_seconds(interval: str) -> int` - Convert '1h' to 3600
- `interval_to_timedelta(interval: str) -> timedelta` - Convert '1h' to timedelta
- `floor_dt(dt, delta) -> datetime` - Floor datetime to interval boundary
- `ceil_dt(dt, delta) -> datetime` - Ceiling datetime to interval boundary
- `timestamp_to_datetime(ts: int) -> datetime` - Millisecond timestamp to datetime (UTC naive)
- `datetime_to_timestamp(dt) -> int` - Datetime to millisecond timestamp
- `parse_period_to_days(period: str) -> int` - Parse '3M', '2w', '7d' to days

**Datetime Ranges:**
- `dt_range(start, end, delta) -> Iterator[datetime]` - Generate datetime range
- `num_of_intervals(start, end, delta) -> int` - Count intervals between dates

**Decimal/Number Formatting:**
- `decimal_as_str(value) -> str` - Format decimal with 8 decimal places
- `ratio_as_str(ratio: float) -> str` - Format as percentage (2 decimal places)
- `floor_to_step(number, step) -> Decimal` - Floor number to step size

**Other:**
- `flatten_list_of_lists(lst) -> list` - Flatten nested list
- `order_to_str(order: dict) -> str` - Format order for logging
- `assets_to_str(assets: dict) -> str` - Format asset holdings
- `calculate_avg_buy_price(...) -> float` - Weighted average buy price

### `jarvis/models.py` - Data Models & Constants

**Enums:**
- `ActionType` - LONG, SHORT, CLOSE, STAY, ERR
- `PositionSide` - LONG, SHORT, NONE
- `Color` - RED, GREEN

**Models:**
- `Kline` - Binance candlestick data (Pydantic model)
- `Position` - Spot position (symbol, spent, amount)
- `FuturesPosition` - Futures position with leverage, liquidation price
- `FakeResponse` - Mock HTTP response for testing

**Constants:**
```python
FUTURES_TAKER_FEE = Decimal("0.0004")   # 0.04%
FUTURES_MAKER_FEE = Decimal("0.0002")   # 0.02%
FUNDING_FEE_RATE = Decimal("0.0001")    # 0.01% per 8h
FUNDING_INTERVAL_HOURS = 8
DEFAULT_LEVERAGE = 1
MAX_LEVERAGE = 10
```

### `jarvis/client.py` - Binance Client

**Classes:**
- `CachedClient` - Caching proxy for Binance client, reads klines from CSV files
  - Supports offline mode (fake=True) with default symbol info
  - Caches klines locally in `data/binance/{SYMBOL}/{interval}/YYYYMMDD.csv`
  - Methods: `get_klines()`, `get_avg_price()`, `create_order()`, `get_symbol_info()`

**Functions:**
- `get_binance_client(fake=False, extra_params=None)` - Factory for client
- `get_day_file_path(symbol, interval, day)` - Path to CSV file
- `create_day_file(client, symbol, interval, day)` - Fetch and save klines
- `load_day_file(symbol, interval, day)` - Load klines from CSV (cached with @ring.lru)
- `get_klines_from_day_files(...)` - Load klines from multiple day files

### `jarvis/genetics/` - Genetic Algorithm System

**`indicators.py` - Technical Indicators:**
- `OHLCV` - NamedTuple with numpy arrays (fast calculation)
- `Indicator` - Abstract base class with calculate(), mutate(), random(), to_dict(), from_dict()
- Implementations: `RSI`, `SMA`, `EMA`, `MACD`, `MACD_HIST`, `VOLUME`, `PRICE`
- Uses TA-Lib (C) if available, falls back to `ta` (Python)

**`rule.py` - Trading Rules:**
- `Rule` - Combines indicator + target + weight
- Contribution formula: `(indicator_value - target) * weight / 100_000`
- Methods: `calculate_contribution()`, `mutate()`, `random()`, `to_dict()`, `from_dict()`

**`individual.py` - Trading Strategy:**
- `Individual` - Collection of rules with fitness score
- Signal calculation: sum all rule contributions, compare to thresholds
- Thresholds: LONG_THRESHOLD=1.0, SHORT_THRESHOLD=-1.0, CLOSE_THRESHOLD=0.5
- Methods: `get_signal()`, `mutate()`, `crossover()`, `to_pine_script()`

**`population.py` - Evolution:**
- `Population` - Collection of individuals that evolve
- `create_random()` - From seed or random individuals
- `evaluate_fitness()` - Backtest all individuals (fitness = return - buy&hold)
- `evolve()` - Tournament selection, crossover, mutation, elitism

**`strategy.py` - Persistence:**
- `TrainingConfig` - Training metadata (dates, generations, etc.)
- `Strategy` - Saved individual with ID, symbol, training config
- `TestResult` - Backtest/test results (return%, drawdown, trades)

### `jarvis/commands/` - CLI Commands

- `train.py` - GA training with walk-forward validation
- `test.py` - Out-of-sample strategy testing
- `trade.py` - Live futures trading (trade_with_strategies for GA)
- `paper.py` - Paper trading simulation (paper_init, paper_trade, paper_info, paper_list)
- `download.py` - Fetch historical klines from Binance
- `pinescript.py` - Export strategy to TradingView Pine Script
