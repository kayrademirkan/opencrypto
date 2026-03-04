"""Tests for backtest engine — simulate_trade, helpers, calc_stats."""

import numpy as np
import pandas as pd

from opencrypto.backtest.engine import (
    BacktestEngine,
    Trade,
    _apply_slip,
    _risk_per_unit,
    _roundtrip_fee_r,
    _rr,
    calc_stats,
    simulate_trade,
)
from opencrypto.core.base_strategy import StrategySignal


def _make_future_df(n: int = 80, base: float = 100.0, trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic future price data for trade simulation."""
    rng = np.random.default_rng(seed)
    close = base + trend * np.arange(n) + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, base * 0.5)
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, n),
            "high": close + abs(rng.normal(0, 0.3, n)),
            "low": close - abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.uniform(1000, 3000, n),
        }
    )


# ── Helper functions ────────────────────────────────────────────────


def test_risk_per_unit_long():
    assert _risk_per_unit(100, 95, "long") == 5.0


def test_risk_per_unit_short():
    assert _risk_per_unit(100, 105, "short") == 5.0


def test_risk_per_unit_minimum():
    result = _risk_per_unit(100, 100, "long")
    assert result == 100 * 0.001


def test_rr_long_profit():
    rr = _rr(100, 110, 95, "long")
    assert rr == 2.0


def test_rr_short_profit():
    rr = _rr(100, 90, 105, "short")
    assert rr == 2.0


def test_rr_loss():
    rr = _rr(100, 95, 95, "long")
    assert rr == -1.0


def test_roundtrip_fee():
    fee_r = _roundtrip_fee_r(100, 95, "long", fee_bps=5, slip_bps=3)
    assert fee_r > 0
    risk = 5.0
    expected = 2 * (5 + 3) / 10_000 * 100 / risk
    assert abs(fee_r - expected) < 1e-10


def test_apply_slip_long_entry():
    result = _apply_slip(100.0, direction="long", kind="entry", slip_bps=10)
    assert result > 100.0


def test_apply_slip_short_entry():
    result = _apply_slip(100.0, direction="short", kind="entry", slip_bps=10)
    assert result < 100.0


def test_apply_slip_long_exit():
    result = _apply_slip(100.0, direction="long", kind="sl", slip_bps=10)
    assert result < 100.0


# ── simulate_trade ──────────────────────────────────────────────────


def _long_signal(entry: float = 100.0) -> dict:
    return {
        "symbol": "TESTUSDT",
        "direction": "LONG",
        "entry": entry,
        "sl": entry * 0.97,
        "tp": entry * 1.045,
        "leverage": 3,
        "signal_type": "test",
        "confidence": 70,
        "rr_ratio": 1.5,
        "reasons": ["test"],
    }


def _short_signal(entry: float = 100.0) -> dict:
    return {
        "symbol": "TESTUSDT",
        "direction": "SHORT",
        "entry": entry,
        "sl": entry * 1.03,
        "tp": entry * 0.955,
        "leverage": 3,
        "signal_type": "test",
        "confidence": 70,
        "rr_ratio": 1.5,
        "reasons": ["test"],
    }


def test_simulate_trade_no_data():
    trade = simulate_trade(_long_signal(), pd.DataFrame())
    assert trade.exit_reason == "no_data"


def test_simulate_trade_long_tp():
    future = _make_future_df(80, base=100, trend=0.3, seed=10)
    trade = simulate_trade(_long_signal(), future, max_hold=72)
    assert trade.exit_reason in ("tp", "sl", "breakeven", "ttl")
    assert trade.entry_price > 0
    assert trade.exit_price is not None


def test_simulate_trade_short_tp():
    future = _make_future_df(80, base=100, trend=-0.3, seed=10)
    trade = simulate_trade(_short_signal(), future, max_hold=72)
    assert trade.exit_reason in ("tp", "sl", "breakeven", "ttl")


def test_simulate_trade_ttl_expiry():
    future = _make_future_df(10, base=100, trend=0.0, seed=42)
    signal = _long_signal()
    signal["sl"] = 50.0
    signal["tp"] = 200.0
    trade = simulate_trade(signal, future, max_hold=8)
    assert trade.exit_reason == "ttl"


def test_simulate_trade_invalid_levels():
    signal = _long_signal(entry=100.0)
    signal["sl"] = 200.0  # SL above entry for LONG = invalid
    signal["tp"] = 50.0
    future = _make_future_df(20)
    trade = simulate_trade(signal, future)
    assert trade.exit_reason == "invalid_levels"


def test_simulate_trade_pnl_sign():
    future_up = _make_future_df(80, base=100, trend=0.5, seed=7)
    trade = simulate_trade(_long_signal(), future_up, max_hold=72)
    if trade.exit_reason == "tp":
        assert trade.pnl_r > 0


def test_simulate_trade_fee_deducted():
    future = _make_future_df(80, base=100, trend=0.5, seed=7)
    trade = simulate_trade(_long_signal(), future, max_hold=72)
    assert trade.fee_r > 0


# ── calc_stats ──────────────────────────────────────────────────────


def test_calc_stats_empty():
    engine = BacktestEngine(strategy=_DummyStrat())
    result = calc_stats([], engine)
    assert "error" in result


def test_calc_stats_with_trades():
    engine = BacktestEngine(strategy=_DummyStrat(), initial_capital=1000.0)
    trades = [
        Trade(pnl_r=2.0, direction="long", exit_reason="tp"),
        Trade(pnl_r=-1.0, direction="short", exit_reason="sl"),
        Trade(pnl_r=1.5, direction="long", exit_reason="tp"),
    ]
    for t in trades:
        engine.results.append(t)
        engine._update_equity(t)

    stats = calc_stats(trades, engine)
    assert stats["total"] == 3
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["win_rate"] > 60
    assert stats["profit_factor"] > 1
    assert stats["total_r"] > 0


# ── BacktestEngine ──────────────────────────────────────────────────


class _DummyStrat:
    name = "DummyTest"
    version = "0.1"

    def generate_signal(self, symbol, df, context=None):
        close = float(df["close"].iloc[-1])
        ema9 = df["close"].ewm(span=9).mean().iloc[-1]
        ema21 = df["close"].ewm(span=21).mean().iloc[-1]
        if ema9 > ema21:
            return StrategySignal(
                symbol=symbol,
                direction="LONG",
                confidence=60.0,
                entry=close,
                sl=round(close * 0.97, 6),
                tp=round(close * 1.045, 6),
                reasons=["test"],
            )
        return None
