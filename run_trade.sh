#!/bin/bash
cd /home/mirat/apps/jarvis

MAX_RETRIES=6
INITIAL_DELAY=15
MAX_DELAY=600  # 10dk cap

DELAY=$INITIAL_DELAY

for i in $(seq 1 $MAX_RETRIES); do
    # Run trade and capture output
    OUTPUT=$(/home/mirat/.local/bin/uv run python src/jarvis.py trade 2>&1)
    EXIT_CODE=$?
    
    # Log output
    echo "$(date): Attempt $i" >> /home/mirat/apps/jarvis/logs/trade.log
    echo "$OUTPUT" >> /home/mirat/apps/jarvis/logs/trade.log
    
    if [ $EXIT_CODE -eq 0 ]; then
        exit 0
    fi
    
    # Get last 8 lines of traceback
    TRACEBACK=$(echo "$OUTPUT" | tail -8)
    
    if [ $i -lt $MAX_RETRIES ]; then
        MINS=$((DELAY / 60))
        SECS=$((DELAY % 60))
        
        if [ $MINS -gt 0 ]; then
            TIME_STR="${MINS}dk ${SECS}sn"
        else
            TIME_STR="${SECS}sn"
        fi
        
        # Send Telegram notification
        /home/mirat/.local/bin/uv run python src/jarvis.py message mirat "⚠️ Deneme $i/$MAX_RETRIES başarısız!

$TRACEBACK

⏳ ${TIME_STR} sonra tekrar deneyeceğim..." 2>/dev/null
        
        sleep $DELAY
        
        # Double delay but cap at MAX_DELAY
        DELAY=$((DELAY * 2))
        if [ $DELAY -gt $MAX_DELAY ]; then
            DELAY=$MAX_DELAY
        fi
    fi
done

# All retries failed
/home/mirat/.local/bin/uv run python src/jarvis.py message mirat "☠️ ÖLÜMCÜL DARBE! $MAX_RETRIES deneme başarısız. Bir sonraki saatte tekrar denerim... 📜" 2>/dev/null
