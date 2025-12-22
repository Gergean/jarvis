# Default recipe
default:
    @just --list

# Run doctests
test *args:
    uv run python src/jarvis.py doctest {{ args }}

# Run doctests verbose
test-v:
    uv run python src/jarvis.py doctest -v

# Lint code
lint:
    uv run ruff check src

# Format code
fmt:
    uv run ruff format src

# Type check
typecheck:
    uv run mypy

# Run all checks (lint, format check, typecheck, test)
check: lint typecheck test

# Run backtest
backtest *args:
    uv run python src/jarvis.py backtest {{ args }}

# Run live trade
trade *args:
    uv run python src/jarvis.py trade {{ args }}

# Install dependencies
sync:
    uv sync
