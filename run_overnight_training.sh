#!/bin/bash
cd /home/mirat/apps/jarvis

COINS="BTCUSDT ETHUSDT XRPUSDT LTCUSDT LINKUSDT ADAUSDT"
POP=600
GEN=120
INTERVAL=4h
LEVERAGE=1

echo "Starting overnight training at $(date)"
echo "Coins: $COINS"
echo "Population: $POP, Generations: $GEN, Interval: $INTERVAL"

for COIN in $COINS; do
    echo ""
    echo "=============================================="
    echo "Training $COIN at $(date)"
    echo "=============================================="

    /home/mirat/.local/bin/uv run python src/jarvis.py train $COIN -i $INTERVAL -p $POP -g $GEN -l $LEVERAGE

    echo "$COIN completed at $(date)"
done

echo ""
echo "=============================================="
echo "All trainings completed at $(date)"
echo "=============================================="

# Final summary notification
/home/mirat/.local/bin/uv run python src/jarvis.py message mirat "🌅 Gece eğitimi tamamlandı! 6 coin train edildi. Sabah raporu hazır."
