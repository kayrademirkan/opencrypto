# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2025-03-04

Initial public release as a modular algorithmic trading framework.

### Added

**Core Framework**
- `BaseStrategy` protocol — implement `generate_signal()` to create any strategy.
- `StrategySignal` dataclass — universal contract between strategies and the engine.
- `DataBridge` — async Binance market data with Futures-to-Spot failover and connection pooling.
- `ShieldGuard` — 9-check manipulation detection, daily drawdown protection, BTC market gate.
- `PositionManager` — trade lifecycle management with progressive trailing SL and R-unit tracking.
- Custom exception hierarchy (`OpenCryptoError`, `DataFetchError`, `ManipulationDetectedError`, `StrategyImplementationError`, `BacktestError`).
- Configurable logging via `OPENCRYPTO_LOG_LEVEL` and `OPENCRYPTO_LOG_FORMAT` environment variables.

**Indicators (29 total)**
- 17 technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic RSI, ADX, OBV, VWAP, Ichimoku, Supertrend, Dynamic RSI Bands, Volume Profile, Support/Resistance, Kelly Criterion.
- 12 Smart Money / ICT detectors: Swing Points, BOS/CHoCH, Order Blocks, FVG, Liquidity Sweep, Wyckoff Phase, Quasimodo, Fakeout, SR/RS Flip, Compression, RSI Divergence, MTF Bias.
- `compute_all_indicators()` — single call to add all 33 indicator columns to any OHLCV DataFrame.

**Backtesting**
- Strategy-agnostic backtest engine accepting any `BaseStrategy` implementation.
- Realistic simulation: H+1 execution delay, slippage (3bp), fees (5bp), conservative SL-before-TP.
- Progressive trailing stop-loss matching live trading behaviour.
- Equity curve, drawdown tracking, and comprehensive statistics.

**Plugins (optional)**
- `plugins.telegram` — signal and trade notifications via Telegram Bot API.
- `plugins.charts` — candlestick chart generation with signal annotations (mplfinance).
- `plugins.llm` — AI-powered trade commentary via Groq API (fully optional).

**Examples**
- `simple_ma_bot.py` — EMA 9/21 crossover strategy.
- `bb_reversal_bot.py` — Bollinger Band + RSI + volume mean reversion strategy.

**Infrastructure**
- `pyproject.toml` with setuptools build, optional dependency groups, ruff/mypy/pytest config.
- GitHub Actions CI: ruff lint + format check, mypy, pytest across Python 3.11/3.12/3.13.
- 86 unit tests covering all modules.
- PEP 561 `py.typed` marker for downstream type checking.
- Community health files: CODE_OF_CONDUCT, CONTRIBUTING, SECURITY, issue/PR templates.

[1.0.0]: https://github.com/kayrademirkan/opencrypto/releases/tag/v1.0.0
