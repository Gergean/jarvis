#!/bin/bash
cd /home/mirat/apps/jarvis

echo "============================================"
echo "JARVIS TOPLU TRAINING"
echo "============================================"
echo "Coinler: AVAX, BNB, BTC, ETH, DOT, NEAR"
echo "Ayarlar: 100 nesil, 1000 populasyon, 4h"
echo "============================================"
echo ""

COINS="AVAXUSDT BNBUSDT BTCUSDT ETHUSDT DOTUSDT NEARUSDT"
POP=1000
GEN=100
INTERVAL=4h

for COIN in $COINS; do
    echo ""
    echo "============================================"
    echo ">>> $COIN TRAINING BASLADI - $(date)"
    echo "============================================"

    # Önce data indir
    uv run python src/jarvis.py download $COIN -i $INTERVAL

    # Training çalıştır
    uv run python src/jarvis.py train $COIN -i $INTERVAL -g $GEN -p $POP

    echo ""
    echo ">>> $COIN TAMAMLANDI - $(date)"
    echo "============================================"
done

echo ""
echo "============================================"
echo "TUM TRAININGLER TAMAMLANDI! - $(date)"
echo "============================================"
