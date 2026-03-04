"""Tests for ShieldGuard risk management module."""

import numpy as np
import pandas as pd

from opencrypto import ShieldGuard


def _make_ohlcv(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10)
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n),
            "high": close + abs(rng.normal(0, 0.5, n)),
            "low": close - abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
            "quote_volume": rng.uniform(100000, 500000, n),
            "trades": rng.integers(100, 1000, n),
            "taker_buy_base": rng.uniform(500, 2500, n),
            "taker_buy_quote": rng.uniform(50000, 250000, n),
        }
    )


# ── Manipulation detection ──────────────────────────────────────────


def test_manipulation_clean_data():
    guard = ShieldGuard()
    df = _make_ohlcv(60)
    result = guard.detect_manipulation(df)
    assert hasattr(result, "risk_score")
    assert hasattr(result, "is_blocked")
    assert hasattr(result, "warnings")
    assert 0 <= result.risk_score <= 100


def test_manipulation_short_df_returns_clean():
    guard = ShieldGuard()
    df = _make_ohlcv(10)
    result = guard.detect_manipulation(df)
    assert result.risk_score == 0
    assert result.is_blocked is False


def test_manipulation_volume_spike():
    guard = ShieldGuard(manipulation_threshold=10)
    df = _make_ohlcv(60)
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 10
    result = guard.detect_manipulation(df)
    assert result.risk_score > 0
    assert result.vol_spike_ratio > 5


def test_manipulation_consecutive_candles():
    guard = ShieldGuard(manipulation_threshold=10)
    df = _make_ohlcv(60, seed=99)
    for i in range(-8, 0):
        df.loc[df.index[i], "close"] = df.loc[df.index[i], "open"] + 1.0
    result = guard.detect_manipulation(df)
    assert any("Consecutive" in w for w in result.warnings)


# ── Daily drawdown ──────────────────────────────────────────────────


def test_daily_tracker_reset():
    guard = ShieldGuard(daily_max_drawdown=-5.0, daily_max_sl_count=2)
    assert not guard.is_daily_limit_hit()
    stats = guard.daily_stats
    assert stats["realized_pnl"] == 0.0
    assert stats["sl_count"] == 0


def test_daily_drawdown_limit():
    guard = ShieldGuard(daily_max_drawdown=-5.0, daily_max_sl_count=10)
    guard.record_trade_close(-3.0, is_sl=True)
    assert not guard.is_daily_limit_hit()
    guard.record_trade_close(-3.0, is_sl=True)
    assert guard.is_daily_limit_hit()


def test_daily_sl_count_limit():
    guard = ShieldGuard(daily_max_drawdown=-100.0, daily_max_sl_count=2)
    guard.record_trade_close(-0.5, is_sl=True)
    guard.record_trade_close(-0.5, is_sl=True)
    assert guard.is_daily_limit_hit()


def test_daily_tracker_wins_dont_trigger():
    guard = ShieldGuard(daily_max_drawdown=-5.0, daily_max_sl_count=3)
    guard.record_trade_close(2.0, is_sl=False)
    guard.record_trade_close(3.0, is_sl=False)
    assert not guard.is_daily_limit_hit()


def test_daily_stopped_flag_persists():
    guard = ShieldGuard(daily_max_drawdown=-2.0, daily_max_sl_count=10)
    guard.record_trade_close(-3.0, is_sl=True)
    assert guard.is_daily_limit_hit()
    guard.record_trade_close(5.0, is_sl=False)
    assert guard.is_daily_limit_hit()


# ── Direction cap ───────────────────────────────────────────────────


def test_direction_cap_long():
    guard = ShieldGuard(max_open_long=3, max_open_short=2)
    assert guard.check_direction_cap("LONG", 2, 0) is True
    assert guard.check_direction_cap("LONG", 3, 0) is False


def test_direction_cap_short():
    guard = ShieldGuard(max_open_long=3, max_open_short=2)
    assert guard.check_direction_cap("SHORT", 0, 1) is True
    assert guard.check_direction_cap("SHORT", 0, 2) is False


def test_direction_cap_independent():
    guard = ShieldGuard(max_open_long=2, max_open_short=2)
    assert guard.check_direction_cap("LONG", 2, 0) is False
    assert guard.check_direction_cap("SHORT", 2, 0) is True
