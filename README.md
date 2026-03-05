<p align="center">
  <h1 align="center">OpenCrypto</h1>
  <p align="center">
    <strong>Modular Algorithmic Trading Framework for Crypto</strong>
  </p>
  <p align="center">
    <a href="https://github.com/kayrademirkan/opencrypto/actions/workflows/python-tests.yml"><img src="https://github.com/kayrademirkan/opencrypto/actions/workflows/python-tests.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue" alt="Python">
    <a href="https://github.com/kayrademirkan/opencrypto/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  </p>
  <p align="center">
    Build, backtest, and deploy your own trading strategies.<br>
    We handle the infrastructure. You bring the alpha.
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#modules">Modules</a> &bull;
    <a href="#write-your-strategy">Write Your Strategy</a> &bull;
    <a href="#backtesting">Backtesting</a>
  </p>
</p>

---

## What is OpenCrypto?

OpenCrypto is **not a trading bot**. It's the infrastructure layer that trading bots are built on.

It was extracted from a private trading system with an **85.5% win rate** across 400+ live trades. The proprietary strategy remains private, but everything else — the market data pipeline, risk management engine, position tracker, technical indicators, and backtesting framework — is now open source.

**You write a strategy. OpenCrypto handles everything else:**

- Real-time market data from Binance (Futures + Spot failover)
- 29 technical indicators computed automatically
- 9-check market manipulation detection
- Daily drawdown protection & BTC crash gate
- Position tracking with progressive trailing stop-loss
- Strategy-agnostic backtesting engine
- Optional: Telegram alerts, chart generation, LLM commentary

## Quickstart

### Install

```bash
pip install git+https://github.com/kayrademirkan/opencrypto.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/kayrademirkan/opencrypto.git
cd opencrypto
pip install -e .
```

### Your First Strategy in 30 Seconds

```python
from opencrypto import BaseStrategy, StrategySignal

class MyStrategy:
    name = "RSI_Bounce"
    version = "1.0"

    def generate_signal(self, symbol, df, context=None):
        rsi = float(df["rsi"].iloc[-1])
        close = float(df["close"].iloc[-1])
        atr = float(df["atr_14"].iloc[-1])

        if rsi < 30:
            return StrategySignal(
                symbol=symbol,
                direction="LONG",
                confidence=70.0,
                entry=close,
                sl=round(close - atr * 2.5, 6),
                tp=round(close + atr * 3.75, 6),
                reasons=["RSI oversold"],
            )
        return None
```

### Backtest It

```python
import asyncio
from opencrypto.backtest import run_backtest

async def main():
    report = await run_backtest(
        strategy=MyStrategy(),
        days=30,
        top_n=50,
    )
    stats = report.get("stats", {})
    if stats:
        print(f"Win Rate: {stats['win_rate']}%")
        print(f"Return: {stats['total_return']}%")
    else:
        print("No trades generated")

asyncio.run(main())
```

### Run the Example Bot

```bash
python examples/simple_ma_bot.py
```

## Architecture

```
opencrypto/
├── core/
│   ├── base_strategy.py      # Strategy Protocol — implement this
│   ├── data_bridge.py         # Market data (Binance Futures/Spot)
│   ├── shield_guard.py        # Risk management & manipulation detection
│   ├── position_manager.py    # Trade lifecycle & trailing SL
│   └── config.py              # Environment-based configuration
├── indicators/
│   ├── technical.py           # 17 standard indicators
│   └── smart_money.py         # 12 institutional/SMC detectors
├── backtest/
│   └── engine.py              # Strategy-agnostic backtester
└── plugins/
    ├── telegram.py            # Signal notifications (optional)
    ├── charts.py              # Candlestick charts (optional)
    └── llm.py                 # AI commentary via Groq (optional)
```

**Design principles:**
- **Strategy-agnostic**: The framework doesn't care what your strategy does. Implement `generate_signal()`, return a `StrategySignal`, done.
- **No secrets in code**: All API keys live in `.env`. The framework works with zero keys configured.
- **Optional everything**: Telegram, charts, LLM are plugins. Core works without them.
- **Battle-tested**: Every component was extracted from a system running live trades.

## Modules

### DataBridge — Market Data

Async market data fetcher with connection pooling and automatic Futures-to-Spot failover.

```python
from opencrypto import DataBridge

bridge = DataBridge()
coins = await bridge.fetch_top_coins(50)          # Top 50 by volume
df = await bridge.fetch_klines("BTCUSDT", "1h")   # OHLCV data
ob = await bridge.get_orderbook_depth("ETHUSDT")   # Bid/ask imbalance
```

### ShieldGuard — Risk Management

Consolidated risk engine combining 9 manipulation checks, daily drawdown limits, and BTC market condition monitoring.

```python
from opencrypto import ShieldGuard

guard = ShieldGuard(
    manipulation_threshold=15,    # Block signals with risk_score >= 15
    daily_max_drawdown=-6.0,      # Stop trading at -6% daily PnL
    daily_max_sl_count=3,         # Stop after 3 stop-losses per day
    max_open_long=6,              # Correlation risk cap
    max_open_short=3,
)

# Check for manipulation before opening a trade
manip = guard.detect_manipulation(df)
if manip.is_blocked:
    print(f"Blocked: {manip.warnings}")

# Monitor BTC conditions
btc = await guard.get_btc_gate()
if btc.is_crash:
    print("BTC crash detected — LONG signals blocked")
```

**Manipulation checks:** Volume spike, wick analysis, wash trading (4σ), pump & dump pattern, consecutive candles, taker imbalance, liquidation cascade, spread/gap spoofing, orderbook imbalance (OBI).

### PositionManager — Trade Lifecycle

Handles the full trade lifecycle: open, track, trail, close.

```python
from opencrypto import PositionManager

pm = PositionManager()
pm.save_signal(signal.to_dict())           # Open a position
trades = await pm.update_all_trades()       # Check TP/SL/timeout
stats = pm.get_trade_stats()                # Win rate, PnL, R-units
```

**Features:**
- Progressive trailing SL (30% progress → breakeven, tightens as profit grows)
- R-unit PnL tracking (risk-normalized returns)
- Position timeout (14h max hold, 10h for deep losses)
- BTC emergency protection (tighten SL during market crashes)

### Indicators — Technical Analysis

29 indicators computed automatically on any OHLCV DataFrame.

```python
from opencrypto.indicators import compute_all_indicators, detect_swing_points

df = await bridge.fetch_klines("BTCUSDT")
df = compute_all_indicators(df)  # Adds 33+ columns from 29 indicators

# Smart Money / ICT concepts
swings = detect_swing_points(df)
```

| Category | Indicators |
|----------|-----------|
| **Trend** | SMA (20/50/200), EMA (9/21), Supertrend, Ichimoku Cloud, ADX |
| **Momentum** | RSI, Stochastic RSI, MACD, Dynamic RSI Bands |
| **Volatility** | Bollinger Bands (%B, bandwidth), ATR |
| **Volume** | OBV, VWAP, Volume Profile (POC/VAH/VAL) |
| **Smart Money** | Order Blocks, FVG, Liquidity Sweep, BOS/CHoCH |
| **Structure** | Swing Points, Quasimodo, Fakeout, SR Flip, Compression, Wyckoff |

## Write Your Strategy

Implement the `BaseStrategy` protocol:

```python
from opencrypto import BaseStrategy, StrategySignal
import pandas as pd

class BollingerMeanReversion:
    name = "BB_MeanReversion"
    version = "1.0"

    def generate_signal(self, symbol: str, df: pd.DataFrame, context=None) -> StrategySignal | None:
        close = float(df["close"].iloc[-1])
        bb_lower = float(df["bb_lower"].iloc[-1])
        bb_upper = float(df["bb_upper"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
        atr = float(df["atr_14"].iloc[-1])

        # Buy when price touches lower Bollinger Band + RSI oversold
        if close <= bb_lower * 1.002 and rsi < 35:
            return StrategySignal(
                symbol=symbol,
                direction="LONG",
                confidence=65.0,
                entry=close,
                sl=round(close - atr * 2.5, 6),
                tp=round(close + atr * 3.75, 6),
                signal_type="bb_bounce",
                reasons=[
                    f"Price at lower BB: {close:.2f} <= {bb_lower:.2f}",
                    f"RSI oversold: {rsi:.0f}",
                ],
            )
        return None
```

**What your strategy gets for free:**
- `df` comes pre-loaded with 33+ columns from all 29 indicators
- `context` dict can include sentiment scores, orderbook data, MTF bias
- ShieldGuard blocks manipulated signals before your strategy even sees them
- PositionManager handles trailing SL, timeout, PnL tracking
- Backtest engine simulates your strategy with realistic fees and slippage

## Backtesting

The backtest engine accepts **any** strategy that implements `BaseStrategy`:

```python
import asyncio
from opencrypto.backtest import run_backtest

async def main():
    report = await run_backtest(
        strategy=MyStrategy(),
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],  # or top_n=100
        days=90,
        initial_capital=1000.0,
        risk_per_trade=0.02,       # 2% risk per trade
        max_drawdown=50.0,         # Stop at 50% drawdown
    )
    stats = report.get("stats", {})
    if stats:
        print(f"Win Rate: {stats['win_rate']}%")

asyncio.run(main())
```

**Backtest features:**
- H+1 execution delay (realistic entry at next bar's open)
- Slippage + fee modeling (5bp fee + 3bp slippage per side)
- Conservative SL-before-TP checking
- Progressive trailing stop-loss
- Equity curve and drawdown tracking
- Per-symbol and per-signal-type breakdown

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

```env
# Telegram (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Groq AI (optional)
GROQ_API_KEY=
USE_LLM=true
```

**Everything is optional.** The framework works with an empty `.env`.

## Plugins

| Plugin | Purpose | Requirement |
|--------|---------|-------------|
| `plugins.telegram` | Send signals to Telegram | `TELEGRAM_BOT_TOKEN` in .env |
| `plugins.charts` | Generate candlestick PNGs | `pip install -e ".[charts]"` |
| `plugins.llm` | AI trade commentary | `pip install -e ".[llm]"` + `GROQ_API_KEY` |

> Install all optional plugins at once: `pip install -e ".[all]"`

```python
# Telegram
from opencrypto.plugins.telegram import send_signal_message
await send_signal_message(signal.to_dict())

# Charts
from opencrypto.plugins.charts import generate_chart
path = await generate_chart(df, signal.to_dict())

# AI Commentary
from opencrypto.plugins.llm import ai_comment
result = ai_comment(signal.to_dict())
```

## Why Open Source This?

This framework was born from building a profitable trading bot. Along the way, I realized the infrastructure — data pipelines, risk management, position tracking, backtesting — is useful to any algorithmic trader, regardless of strategy.

The "secret sauce" (signal generation logic, scoring weights, self-learning optimizer) stays private. But the 90% of code that's infrastructure? That should be shared.

**If you find this useful:**
- Star the repo
- Open issues for bugs or feature requests
- PRs welcome for new indicators, exchange integrations, or plugins

## License

MIT — do whatever you want with it.
