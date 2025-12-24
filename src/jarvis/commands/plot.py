"""Plot command - Generate interactive TradingView-style charts."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from jarvis.client import get_binance_client
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.strategy import Strategy
from jarvis.models import ActionType, PositionSide
from jarvis.utils import datetime_to_timestamp, interval_to_timedelta

# HTML template with lightweight-charts
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #131722;
            color: #d1d4dc;
        }}
        #header {{
            padding: 12px 20px;
            background: #1e222d;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #header h1 {{ font-size: 18px; font-weight: 500; }}
        #stats {{ display: flex; gap: 24px; font-size: 13px; }}
        .stat {{ display: flex; gap: 6px; }}
        .stat-label {{ color: #787b86; }}
        .stat-value {{ font-weight: 500; }}
        .stat-value.positive {{ color: #26a69a; }}
        .stat-value.negative {{ color: #ef5350; }}
        #charts {{ display: flex; flex-direction: column; height: calc(100vh - 52px); }}
        #price-chart {{ flex: 5; position: relative; }}
        #signal-chart {{ flex: 1; border-top: 1px solid #2a2e39; }}
        #volume-chart {{ flex: 1; border-top: 1px solid #2a2e39; }}
        .legend {{
            position: absolute;
            top: 8px;
            left: 8px;
            z-index: 100;
            background: rgba(30, 34, 45, 0.9);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            display: flex;
            gap: 16px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 4px; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
        .legend-dot.long {{ background: rgba(38, 166, 154, 0.3); border: 1px solid #26a69a; }}
        .legend-dot.short {{ background: rgba(239, 83, 80, 0.3); border: 1px solid #ef5350; }}
        .chart-label {{
            position: absolute;
            top: 4px;
            left: 8px;
            z-index: 100;
            font-size: 11px;
            color: #787b86;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{symbol} - {strategy_id}</h1>
        <div id="stats">
            <div class="stat">
                <span class="stat-label">Period:</span>
                <span class="stat-value">{period}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Trades:</span>
                <span class="stat-value">{trade_count}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Win Rate:</span>
                <span class="stat-value">{win_rate}%</span>
            </div>
            <div class="stat">
                <span class="stat-label">Total PnL:</span>
                <span class="stat-value {pnl_class}">{total_pnl}</span>
            </div>
        </div>
    </div>
    <div id="charts">
        <div id="price-chart">
            <div class="legend">
                <div class="legend-item"><div class="legend-dot long"></div> Long</div>
                <div class="legend-item"><div class="legend-dot short"></div> Short</div>
            </div>
        </div>
        <div id="signal-chart" style="position: relative;">
            <div class="chart-label">Signal Strength</div>
        </div>
        <div id="volume-chart" style="position: relative;">
            <div class="chart-label">Volume</div>
        </div>
    </div>
    <script>
        const klineData = {kline_data};
        const volumeData = {volume_data};
        const trades = {trades_data};
        const signalData = {signal_data};

        // Common chart options
        const chartOptions = {{
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
            }},
            timeScale: {{
                borderColor: '#2a2e39',
                timeVisible: true,
            }},
        }};

        // Price chart
        const priceChart = LightweightCharts.createChart(
            document.getElementById('price-chart'),
            chartOptions
        );

        // Candlestick series
        const candleSeries = priceChart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        candleSeries.setData(klineData);

        // Find min/max prices for full height backgrounds
        let minPrice = Infinity;
        let maxPrice = -Infinity;
        klineData.forEach(k => {{
            minPrice = Math.min(minPrice, k.low);
            maxPrice = Math.max(maxPrice, k.high);
        }});
        const priceRange = maxPrice - minPrice;
        const bgTop = maxPrice + priceRange * 0.5;

        // Create separate area series for each trade (to avoid NaN connection issues)
        trades.forEach((trade, idx) => {{
            const tradeData = [];
            klineData.forEach(k => {{
                if (k.time >= trade.entry_time && k.time <= trade.exit_time) {{
                    tradeData.push({{ time: k.time, value: bgTop }});
                }}
            }});

            if (tradeData.length > 0) {{
                const bgSeries = priceChart.addAreaSeries({{
                    topColor: trade.side === 'long' ? 'rgba(38, 166, 154, 0.2)' : 'rgba(239, 83, 80, 0.2)',
                    bottomColor: trade.side === 'long' ? 'rgba(38, 166, 154, 0.2)' : 'rgba(239, 83, 80, 0.2)',
                    lineColor: 'transparent',
                    priceScaleId: 'bg',
                    lastValueVisible: false,
                    priceLineVisible: false,
                }});
                bgSeries.setData(tradeData);
            }}
        }});

        // Hide the background price scale
        priceChart.priceScale('bg').applyOptions({{
            visible: false,
        }});

        // Add trade markers - entry arrow only (PnL shown as styled label)
        const markers = [];
        trades.forEach(trade => {{
            // Entry marker (arrow)
            markers.push({{
                time: trade.entry_time,
                position: trade.side === 'long' ? 'belowBar' : 'aboveBar',
                color: trade.side === 'long' ? '#26a69a' : '#ef5350',
                shape: trade.side === 'long' ? 'arrowUp' : 'arrowDown',
                text: trade.side.toUpperCase(),
            }});
        }});
        // Sort markers by time (required by lightweight-charts)
        markers.sort((a, b) => a.time - b.time);
        candleSeries.setMarkers(markers);

        // Add styled PnL labels with background
        const priceChartContainer = document.getElementById('price-chart');
        trades.forEach(trade => {{
            const pnlText = (trade.pnl >= 0 ? '+' : '') + trade.pnl.toFixed(1) + '%';
            const label = document.createElement('div');
            label.className = 'pnl-label';
            label.textContent = pnlText;
            label.style.cssText = `
                position: absolute;
                background: black;
                padding: 4px 8px;
                border: 1px solid white;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 500;
                color: ${{trade.pnl >= 0 ? '#26a69a' : '#ef5350'}};
                z-index: 50;
                pointer-events: none;
                white-space: nowrap;
            `;
            priceChartContainer.appendChild(label);

            // Position the label
            const midTime = Math.floor((trade.entry_time + trade.exit_time) / 2);
            const midPrice = (trade.entry_price + trade.exit_price) / 2;

            function updateLabelPosition() {{
                const timeCoord = priceChart.timeScale().timeToCoordinate(midTime);
                const priceCoord = candleSeries.priceToCoordinate(midPrice);
                if (timeCoord !== null && priceCoord !== null) {{
                    label.style.left = (timeCoord - label.offsetWidth / 2) + 'px';
                    label.style.top = (priceCoord - label.offsetHeight / 2) + 'px';
                    label.style.display = 'block';
                }} else {{
                    label.style.display = 'none';
                }}
            }}

            // Update on scroll/zoom
            priceChart.timeScale().subscribeVisibleTimeRangeChange(updateLabelPosition);
            priceChart.subscribeCrosshairMove(updateLabelPosition);
            setTimeout(updateLabelPosition, 100);
        }});

        // Signal strength chart
        const signalChart = LightweightCharts.createChart(
            document.getElementById('signal-chart'),
            {{
                ...chartOptions,
                rightPriceScale: {{
                    ...chartOptions.rightPriceScale,
                    scaleMargins: {{ top: 0.1, bottom: 0.1 }},
                }},
            }}
        );

        const signalSeries = signalChart.addHistogramSeries({{
            priceFormat: {{ type: 'price', precision: 2 }},
            priceScaleId: '',
        }});
        signalSeries.setData(signalData);

        // Volume chart
        const volumeChart = LightweightCharts.createChart(
            document.getElementById('volume-chart'),
            chartOptions
        );

        const volumeSeries = volumeChart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: '',
        }});
        volumeSeries.setData(volumeData);

        // Sync all three time scales
        const charts = [priceChart, signalChart, volumeChart];

        charts.forEach((chart, i) => {{
            chart.timeScale().subscribeVisibleTimeRangeChange(timeRange => {{
                charts.forEach((otherChart, j) => {{
                    if (i !== j && timeRange) {{
                        otherChart.timeScale().setVisibleRange(timeRange);
                    }}
                }});
            }});
        }});

        // Sync crosshair across all charts
        function syncCrosshair(sourceChart, sourceSeries, targetCharts, targetSeries) {{
            sourceChart.subscribeCrosshairMove(param => {{
                if (param.time) {{
                    targetCharts.forEach((chart, i) => {{
                        const series = targetSeries[i];
                        const data = param.seriesData.get(sourceSeries);
                        if (data) {{
                            const price = data.value !== undefined ? data.value : data.close;
                            chart.setCrosshairPosition(price, param.time, series);
                        }}
                    }});
                }} else {{
                    targetCharts.forEach(chart => chart.clearCrosshairPosition());
                }}
            }});
        }}

        syncCrosshair(priceChart, candleSeries, [signalChart, volumeChart], [signalSeries, volumeSeries]);
        syncCrosshair(signalChart, signalSeries, [priceChart, volumeChart], [candleSeries, volumeSeries]);
        syncCrosshair(volumeChart, volumeSeries, [priceChart, signalChart], [candleSeries, signalSeries]);

        // Resize handler
        window.addEventListener('resize', () => {{
            const priceContainer = document.getElementById('price-chart');
            const signalContainer = document.getElementById('signal-chart');
            const volumeContainer = document.getElementById('volume-chart');
            priceChart.resize(priceContainer.clientWidth, priceContainer.clientHeight);
            signalChart.resize(signalContainer.clientWidth, signalContainer.clientHeight);
            volumeChart.resize(volumeContainer.clientWidth, volumeContainer.clientHeight);
        }});

        // Initial fit
        priceChart.timeScale().fitContent();
        signalChart.timeScale().fitContent();
        volumeChart.timeScale().fitContent();
    </script>
</body>
</html>
"""


def plot(
    strategy_id: str,
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    output_path: str | None = None,
) -> str:
    """Generate interactive TradingView-style chart showing strategy signals.

    Args:
        strategy_id: Strategy ID (e.g., BTCUSDT_abc123) or path to JSON
        interval: Kline interval
        start_dt: Chart start date. Default: 3 months ago
        end_dt: Chart end date. Default: now
        output_path: Output HTML file path. Default: charts/{strategy_id}.html

    Returns:
        Path to the generated HTML file
    """
    # Load strategy
    if strategy_id.endswith(".json"):
        strategy_path = Path(strategy_id)
    else:
        strategy_path = Path("strategies") / f"{strategy_id}.json"

    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy not found: {strategy_path}")

    with open(strategy_path) as f:
        strategy_data = json.load(f)

    strategy = Strategy.from_dict(strategy_data)
    individual = strategy.individual
    symbol = strategy.symbol

    # Default date range: 3 months
    if end_dt is None:
        end_dt = datetime.now()
    if start_dt is None:
        start_dt = end_dt - timedelta(days=90)

    # Load klines
    client = get_binance_client(fake=True)

    lookback = 200
    lookback_delta = interval_to_timedelta(interval) * lookback
    fetch_start = start_dt - lookback_delta

    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=datetime_to_timestamp(fetch_start),
        endTime=datetime_to_timestamp(end_dt),
        limit=50000,
    )

    if not klines:
        raise ValueError(f"No klines found for {symbol}")

    # Prepare data arrays
    n = len(klines)
    timestamps = []
    opens = np.zeros(n, dtype=np.float64)
    highs = np.zeros(n, dtype=np.float64)
    lows = np.zeros(n, dtype=np.float64)
    closes = np.zeros(n, dtype=np.float64)
    volumes = np.zeros(n, dtype=np.float64)

    for i, k in enumerate(klines):
        timestamps.append(k.open_time)
        opens[i] = float(k.open)
        highs[i] = float(k.high)
        lows[i] = float(k.low)
        closes[i] = float(k.close)
        volumes[i] = float(k.volume)

    # Generate trades and signal strength
    trades = []
    signal_strengths = []
    position_side = PositionSide.NONE
    entry_ts = None
    entry_price = None

    for i in range(lookback, n):
        start_idx = i - lookback + 1
        end_idx = i + 1
        ohlcv = OHLCV(
            open=opens[start_idx:end_idx],
            high=highs[start_idx:end_idx],
            low=lows[start_idx:end_idx],
            close=closes[start_idx:end_idx],
            volume=volumes[start_idx:end_idx],
        )

        # Get signal and signal strength
        signal = individual.get_signal(ohlcv, position_side)

        # Calculate signal strength (total score from all rules)
        strength = individual.get_total_score(ohlcv)
        signal_strengths.append({
            "ts": timestamps[i],
            "strength": strength,
        })

        price = closes[i]
        ts = timestamps[i]

        if signal == ActionType.LONG and position_side == PositionSide.NONE:
            entry_ts = ts
            entry_price = price
            position_side = PositionSide.LONG
        elif signal == ActionType.SHORT and position_side == PositionSide.NONE:
            entry_ts = ts
            entry_price = price
            position_side = PositionSide.SHORT
        elif signal == ActionType.CLOSE and position_side != PositionSide.NONE:
            if position_side == PositionSide.LONG:
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - price) / entry_price * 100
            trades.append({
                "entry_time": int(entry_ts.timestamp()),
                "entry_price": entry_price,
                "exit_time": int(ts.timestamp()),
                "exit_price": price,
                "side": "long" if position_side == PositionSide.LONG else "short",
                "pnl": round(pnl_pct, 2),
            })
            position_side = PositionSide.NONE
            entry_ts = None
            entry_price = None

    # Prepare kline data for JS (only data within display range)
    display_start_idx = 0
    for i, ts in enumerate(timestamps):
        if ts >= start_dt:
            display_start_idx = i
            break

    kline_data = []
    volume_data = []
    for i in range(display_start_idx, n):
        ts_unix = int(timestamps[i].timestamp())
        kline_data.append({
            "time": ts_unix,
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
        })
        volume_data.append({
            "time": ts_unix,
            "value": volumes[i],
            "color": "#26a69a80" if closes[i] >= opens[i] else "#ef535080",
        })

    # Signal strength data (aligned with display range)
    signal_data = []
    signal_start_idx = display_start_idx - lookback
    for i, ss in enumerate(signal_strengths):
        if i >= signal_start_idx:
            ts_unix = int(ss["ts"].timestamp())
            strength = ss["strength"]
            signal_data.append({
                "time": ts_unix,
                "value": strength,
                "color": "#26a69a" if strength >= 0 else "#ef5350",
            })

    # Calculate stats
    total_pnl = sum(t["pnl"] for t in trades)
    winning_trades = sum(1 for t in trades if t["pnl"] >= 0)
    win_rate = (winning_trades / len(trades) * 100) if trades else 0

    # Generate HTML
    html = HTML_TEMPLATE.format(
        title=f"{symbol} - {strategy.id}",
        symbol=symbol,
        strategy_id=strategy.id,
        period=f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
        trade_count=len(trades),
        win_rate=f"{win_rate:.0f}",
        total_pnl=f"{total_pnl:+.2f}%",
        pnl_class="positive" if total_pnl >= 0 else "negative",
        kline_data=json.dumps(kline_data),
        volume_data=json.dumps(volume_data),
        trades_data=json.dumps(trades),
        signal_data=json.dumps(signal_data),
    )

    # Save to file
    if output_path is None:
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)
        output_path = str(charts_dir / f"{strategy.id}.html")

    with open(output_path, "w") as f:
        f.write(html)

    # Summary
    print(f"Generated chart: {output_path}")
    print(f"Period: {start_dt.date()} to {end_dt.date()}")
    print(f"Trades: {len(trades)} ({winning_trades} wins, {len(trades) - winning_trades} losses)")
    print(f"Total PnL: {total_pnl:+.2f}%")

    return output_path
