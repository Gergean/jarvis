#!/bin/bash
cd /home/mirat/apps/jarvis
HOUR=$(date -u +%H)
if [[ "$HOUR" == "00" || "$HOUR" == "04" || "$HOUR" == "08" || "$HOUR" == "12" || "$HOUR" == "16" || "$HOUR" == "20" ]]; then
    echo "$(date): 4h candle close - REAL TRADE"
    uv run python src/jarvis.py trade -a mirat
else
    echo "$(date): Hourly signal - DRY RUN"
    uv run python src/jarvis.py trade -a mirat --dry-run
fi
