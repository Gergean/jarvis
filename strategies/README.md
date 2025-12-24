# Jarvis GA Strategies

## Defansif Stratejiler (1x Leverage, 1D Interval)

Training: 2024-01-01 to 2025-12-01 | Walk-Forward Validation

| Coin | Strategy ID | Fitness | Aylik | Drawdown | Win Rate |
|------|-------------|---------|-------|----------|----------|
| DOGE | DOGEUSDT_d77db325 | 53.02 | %4.28 | %1.97 | 52% |
| ETH | ETHUSDT_5bdb12c7 | 34.70 | %4.03 | %2.52 | 83% |
| XRP | XRPUSDT_761b053e | 24.15 | %4.21 | %3.16 | 70% |
| SOL | SOLUSDT_739a3e8f | 20.57 | %4.32 | %3.43 | 78% |
| TRX | TRXUSDT_a4fe1282 | 16.84 | %2.04 | %1.31 | 57% |

### TradingView Backtest Sonuclari

| Coin | Donem | Getiri | Not |
|------|-------|--------|-----|
| SOL | 2020-2025 | +%1093 | 95.74% win rate, 47 islem |
| ETH | 2024-2025 | +%126 | Farkli timeframe'lerde de calisiyor |

### Dosyalar

```
strategies/
  DOGEUSDT_d77db325.json   # Strateji parametreleri
  DOGEUSDT_d77db325.pine   # TradingView Pine Script
  ETHUSDT_5bdb12c7.json
  ETHUSDT_5bdb12c7.pine
  XRPUSDT_761b053e.json
  XRPUSDT_761b053e.pine
  SOLUSDT_739a3e8f.json
  SOLUSDT_739a3e8f.pine
  TRXUSDT_a4fe1282.json
  TRXUSDT_a4fe1282.pine
```

### Kullanim

```bash
# Pine Script olustur
uv run python src/jarvis.py pinescript -s ETHUSDT_5bdb12c7

# Strateji test et
uv run python src/jarvis.py test -s ETHUSDT_5bdb12c7 -i 1d

# Yeni strateji train et
uv run python src/jarvis.py train -s BTCUSDT -i 1d -ps 100 -g 50
```
