# Default recipe
default:
    @just --list

# Install dependencies
sync:
    uv sync

# Lint code
lint:
    uv run ruff check src

# Format code
fmt:
    uv run ruff format src

# Type check
typecheck:
    uv run mypy

# Run doctests
doctest *args:
    uv run python src/jarvis.py doctest {{ args }}

# Run all checks (lint, typecheck, doctest)
check: lint typecheck doctest

# Download historical data
download *args:
    uv run python src/jarvis.py download {{ args }}

# Train a GA strategy
train *args:
    uv run python src/jarvis.py train {{ args }}

# Test a strategy (out-of-sample)
test *args:
    uv run python src/jarvis.py test {{ args }}

# Trade with GA strategies (use --dry-run for simulation)
trade *args:
    uv run python src/jarvis.py trade-ga {{ args }}

# Clean cache files
clean:
    rm -rf .mypy_cache .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
