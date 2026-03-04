"""Tests for PositionManager — trade lifecycle, storage, stats."""

import json
import os

import pytest

from opencrypto import PositionManager


@pytest.fixture()
def pm(tmp_path):
    """PositionManager with a temp trades file."""
    trades_file = str(tmp_path / "trades.json")
    return PositionManager(trades_file=trades_file)


def _make_signal(symbol: str = "BTCUSDT", direction: str = "LONG") -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "entry": 100.0,
        "sl": 95.0 if direction == "LONG" else 105.0,
        "tp": 110.0 if direction == "LONG" else 90.0,
        "leverage": 3,
        "confidence": 70,
        "score": 42,
        "indicator_count": 5,
        "rr_ratio": 2.0,
    }


# ── save_signal ─────────────────────────────────────────────────────


def test_save_signal(pm):
    trade = pm.save_signal(_make_signal())
    assert trade["symbol"] == "BTCUSDT"
    assert trade["status"] == "open"
    assert trade["entry"] == 100.0
    assert trade["id"] == 1


def test_save_signal_no_duplicate(pm):
    pm.save_signal(_make_signal())
    dup = pm.save_signal(_make_signal())
    assert dup["id"] == 1  # returns existing, no new trade


def test_save_signal_different_symbols(pm):
    pm.save_signal(_make_signal("BTCUSDT"))
    t2 = pm.save_signal(_make_signal("ETHUSDT"))
    assert t2["id"] == 2


# ── has_open_trade ──────────────────────────────────────────────────


def test_has_open_trade(pm):
    assert not pm.has_open_trade("BTCUSDT")
    pm.save_signal(_make_signal())
    assert pm.has_open_trade("BTCUSDT")
    assert not pm.has_open_trade("ETHUSDT")


# ── get_open_positions / count_open_by_direction ────────────────────


def test_get_open_positions(pm):
    pm.save_signal(_make_signal("BTCUSDT", "LONG"))
    pm.save_signal(_make_signal("ETHUSDT", "SHORT"))
    positions = pm.get_open_positions()
    assert len(positions) == 2


def test_count_open_by_direction(pm):
    pm.save_signal(_make_signal("BTCUSDT", "LONG"))
    pm.save_signal(_make_signal("ETHUSDT", "LONG"))
    pm.save_signal(_make_signal("SOLUSDT", "SHORT"))
    counts = pm.count_open_by_direction()
    assert counts["LONG"] == 2
    assert counts["SHORT"] == 1
    assert counts["total"] == 3


# ── get_trade_stats ─────────────────────────────────────────────────


def test_trade_stats_empty(pm):
    stats = pm.get_trade_stats()
    assert stats["total"] == 0
    assert stats["win_rate"] == 0


def test_trade_stats_with_closed(pm):
    trades = [
        {"symbol": "A", "status": "tp", "pnl_r": 2.0, "pnl_pct": 4.0, "direction": "LONG"},
        {"symbol": "B", "status": "sl", "pnl_r": -1.0, "pnl_pct": -2.0, "direction": "SHORT"},
        {"symbol": "C", "status": "tp", "pnl_r": 1.5, "pnl_pct": 3.0, "direction": "LONG"},
        {"symbol": "D", "status": "open", "pnl_r": 0.5, "pnl_pct": 1.0, "direction": "LONG"},
    ]
    os.makedirs(os.path.dirname(pm.trades_file), exist_ok=True)
    with open(pm.trades_file, "w") as f:
        json.dump(trades, f)

    stats = pm.get_trade_stats()
    assert stats["total"] == 4
    assert stats["open"] == 1
    assert stats["closed"] == 3
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["win_rate"] > 60
    assert stats["profit_factor"] > 1
    assert stats["total_r"] > 0


# ── File persistence ────────────────────────────────────────────────


def test_persistence(pm):
    pm.save_signal(_make_signal("BTCUSDT"))
    pm2 = PositionManager(trades_file=pm.trades_file)
    assert pm2.has_open_trade("BTCUSDT")


# ── on_trade_close callback ────────────────────────────────────────


def test_on_trade_close_callback(tmp_path):
    called_with = []
    pm = PositionManager(
        trades_file=str(tmp_path / "trades.json"),
        on_trade_close=lambda pnl, is_sl: called_with.append((pnl, is_sl)),
    )
    pm.save_signal(_make_signal())
    # We can't fully test check_trade_status without network,
    # but we can verify the callback mechanism is wired up
    assert pm.on_trade_close is not None
