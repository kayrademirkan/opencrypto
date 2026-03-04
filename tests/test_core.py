"""Smoke tests for OpenCrypto core modules."""


import numpy as np
import pandas as pd

# ── Import smoke tests ──────────────────────────────────────────────

def test_top_level_import():
    import opencrypto
    assert hasattr(opencrypto, "__version__")
    assert hasattr(opencrypto, "BaseStrategy")
    assert hasattr(opencrypto, "ShieldGuard")
    assert hasattr(opencrypto, "DataBridge")
    assert hasattr(opencrypto, "PositionManager")


def test_exception_hierarchy():
    from opencrypto.core.exceptions import (
        BacktestError,
        DataFetchError,
        ManipulationDetectedError,
        OpenCryptoError,
        StrategyImplementationError,
    )
    assert issubclass(DataFetchError, OpenCryptoError)
    assert issubclass(ManipulationDetectedError, OpenCryptoError)
    assert issubclass(StrategyImplementationError, OpenCryptoError)
    assert issubclass(BacktestError, OpenCryptoError)

    err = DataFetchError("test", symbol="BTCUSDT", source="binance")
    assert err.symbol == "BTCUSDT"
    assert err.source == "binance"


def test_indicators_import():
    from opencrypto.indicators import (
        detect_order_blocks,
        sma,
    )
    assert callable(sma)
    assert callable(detect_order_blocks)


# ── StrategySignal ──────────────────────────────────────────────────

def test_strategy_signal_properties():
    from opencrypto import StrategySignal

    sig = StrategySignal(
        symbol="BTCUSDT",
        direction="LONG",
        confidence=75.0,
        entry=100.0,
        sl=95.0,
        tp=110.0,
    )
    assert sig.sl_pct == 5.0
    assert sig.tp_pct == 10.0
    assert sig.rr_ratio == 2.0
    assert "BTC/USDT" in sig.display_symbol

    d = sig.to_dict()
    assert d["symbol"] == "BTCUSDT"
    assert d["direction"] == "LONG"


# ── Indicators ──────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n),
        "high": close + abs(rng.normal(0, 0.5, n)),
        "low": close - abs(rng.normal(0, 0.5, n)),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
        "quote_volume": rng.uniform(100000, 500000, n),
        "trades": rng.integers(100, 1000, n),
        "taker_buy_base": rng.uniform(500, 2500, n),
        "taker_buy_quote": rng.uniform(50000, 250000, n),
    })


def test_compute_all_indicators():
    from opencrypto.indicators.technical import compute_all_indicators

    df = _make_ohlcv()
    result = compute_all_indicators(df)
    assert "rsi" in result.columns
    assert "ema_9" in result.columns
    assert "supertrend_dir" in result.columns
    assert len(result) == len(df)


def test_rsi_range():
    from opencrypto.indicators.technical import rsi

    series = pd.Series(np.random.default_rng(0).normal(100, 1, 200))
    vals = rsi(series, 14).dropna()
    assert vals.min() >= 0
    assert vals.max() <= 100


# ── ShieldGuard ─────────────────────────────────────────────────────

def test_shield_guard_manipulation_clean():
    from opencrypto import ShieldGuard

    guard = ShieldGuard()
    df = _make_ohlcv(60)
    result = guard.detect_manipulation(df)
    assert hasattr(result, "risk_score")
    assert hasattr(result, "is_blocked")
    assert 0 <= result.risk_score <= 100


def test_shield_guard_daily_tracker():
    from opencrypto import ShieldGuard

    guard = ShieldGuard(daily_max_drawdown=-5.0, daily_max_sl_count=2)
    assert not guard.is_daily_limit_hit()

    guard.record_trade_close(-3.0, is_sl=True)
    guard.record_trade_close(-3.0, is_sl=True)
    assert guard.is_daily_limit_hit()


def test_shield_guard_direction_cap():
    from opencrypto import ShieldGuard

    guard = ShieldGuard(max_open_long=2, max_open_short=1)
    assert guard.check_direction_cap("LONG", 1, 0) is True
    assert guard.check_direction_cap("LONG", 2, 0) is False
    assert guard.check_direction_cap("SHORT", 0, 1) is False


# ── BaseStrategy protocol ──────────────────────────────────────────

def test_base_strategy_protocol():
    from opencrypto import BaseStrategy

    class DummyStrategy:
        name = "Test"
        version = "0.1"

        def generate_signal(self, symbol, df, context=None):
            return None

    assert isinstance(DummyStrategy(), BaseStrategy)
