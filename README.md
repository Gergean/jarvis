# Jarvis

Binance Futures trading automation system using genetic algorithm-based strategies.

## Features

- **GA Strategy Training**: Evolve trading strategies using genetic algorithms
- **Walk-Forward Validation**: Prevent overfitting with rolling train/test windows
- **Futures Support**: Long/short positions with configurable leverage (1-10x)
- **Backtesting**: Realistic simulation with funding fees and liquidation
- **Out-of-Sample Testing**: Validate strategies on unseen data

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (optional)

## Installation

```bash
# Install dependencies
uv sync

# Or with just
just sync
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
just download -s BTCUSDT ETHUSDT -i 1h
```

### Train a Strategy

Training uses **walk-forward validation** by default to prevent overfitting.

```bash
# Default: 1 year data, 3M train / 1M test windows
just train -s BTCUSDT -i 1h

# With 5x leverage
just train -s BTCUSDT -i 1h -l 5

# Custom walk-forward periods (90 days train, 2 weeks test, 1 week step)
just train -s BTCUSDT -i 1h --train-period 90d --test-period 2w --step-period 1w

# Disable walk-forward (not recommended - overfitting risk)
just train -s BTCUSDT -i 1h --no-walk-forward

# Custom date range
just train -s BTCUSDT -i 1h -st 2024-01-01T00:00:00 -et 2024-12-01T00:00:00
```

**Period formats**: `Nd` (days), `Nw` (weeks), `NM` (months)

### Test a Strategy

```bash
# Test on last 3 months (out-of-sample)
just test -s BTCUSDT_abc123 -i 1h

# Test with leverage
just test -s BTCUSDT_abc123 -i 1h -l 5
```

### Live Trading

```bash
# Dry run (simulation)
just trade -s BTCUSDT_abc123 --dry-run

# Live trading (not yet implemented)
just trade -s BTCUSDT_abc123
```

## Development

```bash
# Run all checks
just check

# Individual checks
just lint
just fmt
just typecheck
just doctest

# Clean cache
just clean
```

## Project Structure

```
jarvis/
├── src/
│   ├── jarvis.py          # CLI entry point
│   └── jarvis/
│       ├── commands/      # CLI commands (train, test, trade, download)
│       ├── genetics/      # Genetic algorithm (population, individual, rules)
│       ├── signals/       # Signal generators (supertrend, vwma, sma)
│       ├── actions/       # Action generators
│       ├── client.py      # Binance API client
│       └── models.py      # Data models
├── data/                  # Historical OHLCV data
├── strategies/            # Saved strategies (JSON)
├── results/               # Backtest results
└── logs/                  # Log files
```
