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
- `download` - Fetch historical klines

**Walk-Forward Validation**:
Train uses rolling windows by default to prevent overfitting:
- `--train-period 3M` - Training period per window (default: 3 months)
- `--test-period 1M` - Test period per window (default: 1 month)
- `--step-period 1M` - Step size between windows (default: 1 month)
- Period formats: `Nd` (days), `Nw` (weeks), `NM` (months)

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
- `results/*.json` - Backtest/test results
- `logs/` - Rotating log files

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
