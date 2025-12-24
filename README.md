# Jarvis

Binance Futures trading automation system using genetic algorithm-based strategies.

## Features

- **GA Strategy Training**: Evolve trading strategies using genetic algorithms
- **Walk-Forward Validation**: Prevent overfitting with rolling train/test windows
- **Futures Support**: Long/short positions with configurable leverage (1-10x)
- **Paper Trading**: Simulate trading with daily strategy evolution (elites system)
- **Backtesting**: Realistic simulation with funding fees and liquidation
- **Pine Script Export**: Export strategies to TradingView

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

## Configuration

Create a `.env` file:

```env
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Optional
TELEGRAM_BOT_TOKEN=...
TELEGRAM_DM_ID=...
SENTRY_DSN=...
```

## Usage

### Download Historical Data

```bash
uv run python src/jarvis.py download -s BTCUSDT ETHUSDT -i 1h
```

### Train a Strategy

Training uses **walk-forward validation** by default to prevent overfitting.

```bash
# Default: 1 year data, 3M train / 1M test windows
uv run python src/jarvis.py train -s BTCUSDT -i 1h

# With 5x leverage
uv run python src/jarvis.py train -s BTCUSDT -i 1h -l 5

# Custom walk-forward periods (90 days train, 2 weeks test, 1 week step)
uv run python src/jarvis.py train -s BTCUSDT -i 1h --train-period 90d --test-period 2w --step-period 1w

# Disable walk-forward (not recommended - overfitting risk)
uv run python src/jarvis.py train -s BTCUSDT -i 1h --no-walk-forward
```

**Period formats**: `Nd` (days), `Nw` (weeks), `NM` (months)

### Test a Strategy

```bash
# Test on last 3 months (out-of-sample)
uv run python src/jarvis.py test -s BTCUSDT_abc123 -i 1h

# Test with leverage
uv run python src/jarvis.py test -s BTCUSDT_abc123 -i 1h -l 5
```

### Paper Trading

Paper trading simulates trading without real money. Strategies evolve daily at 00:00 UTC.

```bash
# Create wallet with seed strategy
uv run python src/jarvis.py paper init mywallet -b 1000 -c BTCUSDT:1h -s BTCUSDT_abc123

# Run paper trading to a specific date
uv run python src/jarvis.py paper trade mywallet -et 2025-10-15T00:00:00

# Check wallet status
uv run python src/jarvis.py paper info mywallet

# List all wallets
uv run python src/jarvis.py paper list
```

### Live Trading

```bash
# Dry run (simulation)
uv run python src/jarvis.py trade-ga -s BTCUSDT_abc123 --dry-run

# Live trading
uv run python src/jarvis.py trade-ga -s BTCUSDT_abc123
```

### Pine Script Export

```bash
uv run python src/jarvis.py pinescript -s BTCUSDT_abc123
```

## Development

```bash
# Lint
uv run ruff check src

# Format
uv run ruff format src

# Type check
uv run mypy

# Run doctests
uv run python src/jarvis.py doctest -v
```

## Project Structure

```
jarvis/
├── src/
│   ├── jarvis.py          # CLI entry point
│   └── jarvis/
│       ├── commands/      # CLI commands
│       │   ├── train.py   # GA training with walk-forward
│       │   ├── test.py    # Strategy testing
│       │   ├── trade.py   # Live trading
│       │   ├── paper.py   # Paper trading with elites
│       │   ├── download.py
│       │   └── pinescript.py
│       ├── genetics/      # Genetic algorithm
│       │   ├── individual.py
│       │   ├── population.py
│       │   ├── rule.py
│       │   ├── indicators.py
│       │   └── strategy.py
│       ├── client.py      # Binance API client
│       └── models.py      # Data models
├── data/                  # Historical OHLCV data
│   └── binance/{SYMBOL}/{interval}/YYYYMMDD.csv
├── strategies/            # Saved strategies
│   ├── *.json            # Trained strategies
│   ├── *.pine            # Pine Script exports
│   └── elites/           # Daily evolved elites
│       └── {SYMBOL}/{interval}/YYYYMMDD_HHMMSS.json
├── paper/                 # Paper trading wallets
└── results/               # Test results
```

## Documentation

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation (in Turkish).
