"""
OpenCrypto — PositionManager

Trade lifecycle management: open, track, trail, close.
- JSON file-based storage (upgradeable to DB)
- Trailing stop-loss with progressive tightening
- R-unit PnL tracking
- Position timeout management
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx

from opencrypto.core.config import DATA_DIR

logger = logging.getLogger(__name__)

FAPI_URL = "https://fapi.binance.com"


class PositionManager:
    """Manages open positions, trailing SL, and trade history.

    Usage:
        pm = PositionManager()
        pm.save_signal(signal_dict)
        trades = await pm.update_all_trades()
        stats = pm.get_trade_stats()
    """

    def __init__(
        self,
        trades_file: str | None = None,
        on_trade_close: Callable[[float, bool], None] | None = None,
        on_message: Callable[[str], Awaitable[None]] | None = None,
        max_position_age_hours: int = 14,
        deep_loss_timeout_hours: int = 10,
        deep_loss_threshold: float = -2.0,
    ):
        self.trades_file = trades_file or str(DATA_DIR / "trades_history.json")
        self.on_trade_close = on_trade_close
        self.on_message = on_message
        self.max_position_age_hours = max_position_age_hours
        self.deep_loss_timeout_hours = deep_loss_timeout_hours
        self.deep_loss_threshold = deep_loss_threshold

    def _load_trades(self) -> list[dict]:
        if not os.path.exists(self.trades_file):
            return []
        try:
            with open(self.trades_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _save_trades(self, trades: list[dict]):
        os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
        with open(self.trades_file, "w", encoding="utf-8") as f:
            json.dump(trades, f, ensure_ascii=False, indent=2, default=str)

    def has_open_trade(self, symbol: str) -> bool:
        """Check if symbol has an open trade or was closed within 2h."""
        trades = self._load_trades()
        now = datetime.now(UTC)
        for t in trades:
            if t["symbol"] != symbol:
                continue
            if t["status"] == "open":
                return True
            if t["status"] in ("tp", "sl") and t.get("closed_at"):
                try:
                    closed_at = datetime.fromisoformat(t["closed_at"])
                    hours_since = (now - closed_at).total_seconds() / 3600
                    if hours_since < 2:
                        return True
                except Exception:
                    pass
        return False

    def save_signal(self, signal: dict) -> dict:
        """Save a new signal as an open trade. Returns the trade dict."""
        trades = self._load_trades()

        for t in trades:
            if t["symbol"] == signal["symbol"] and t["status"] == "open":
                return t

        now = datetime.now(UTC)
        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal.get("tp1", signal.get("tp", entry))
        risk = abs(entry - sl)

        trade = {
            "id": len(trades) + 1,
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "signal_type": signal.get("signal_type", "unknown"),
            "entry": entry,
            "sl": sl,
            "sl_original": sl,
            "tp": tp,
            "leverage": signal.get("leverage", 3),
            "confidence": signal.get("confidence", 0),
            "score": signal.get("score", 0),
            "indicator_count": signal.get("indicator_count", 0),
            "rr_ratio": signal.get("rr_ratio", 0),
            "risk_1r": round(risk, 6),
            "pnl_r": 0.0,
            "pnl_pct": 0.0,
            "status": "open",
            "current_price": entry,
            "peak_price": entry,
            "opened_at": now.isoformat(),
            "closed_at": None,
            "last_checked": now.isoformat(),
        }

        trades.append(trade)
        self._save_trades(trades)
        return trade

    async def check_trade_status(self, trade: dict) -> dict:
        """Check a single trade: price update, trailing SL, TP/SL hit."""
        try:
            clean = trade["symbol"].replace(".P", "").upper()
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{FAPI_URL}/fapi/v1/ticker/price",
                    params={"symbol": clean},
                )
                if resp.status_code != 200:
                    return trade
                current_price = float(resp.json()["price"])
        except Exception:
            return trade

        trade["current_price"] = current_price
        trade["last_checked"] = datetime.now(UTC).isoformat()

        entry = trade["entry"]
        is_long = trade["direction"] == "LONG"

        # Position timeout
        if trade.get("opened_at"):
            try:
                opened_dt = datetime.fromisoformat(trade["opened_at"])
                age_hours = (datetime.now(UTC) - opened_dt).total_seconds() / 3600
                if is_long:
                    timeout_pnl = (current_price - entry) / entry * 100
                else:
                    timeout_pnl = (entry - current_price) / entry * 100

                should_timeout = False
                if (age_hours >= self.max_position_age_hours and timeout_pnl < 0.5) or (age_hours >= self.deep_loss_timeout_hours and timeout_pnl <= self.deep_loss_threshold):
                    should_timeout = True

                if should_timeout:
                    risk_1r = trade.get("risk_1r") or abs(entry - trade["sl"])
                    if not risk_1r or risk_1r <= 0:
                        risk_1r = entry * 0.02
                    pnl_r = timeout_pnl / 100 * entry / risk_1r
                    trade["pnl_pct"] = round(timeout_pnl, 2)
                    trade["pnl_r"] = round(pnl_r, 4)
                    trade["status"] = "sl"
                    trade["close_reason"] = f"timeout_{int(age_hours)}h"
                    trade["closed_at"] = datetime.now(UTC).isoformat()
                    if self.on_trade_close:
                        self.on_trade_close(timeout_pnl, True)
                    return trade
            except Exception:
                pass

        risk_1r = trade.get("risk_1r") or abs(entry - trade["sl"])
        if not risk_1r or risk_1r <= 0:
            risk_1r = entry * 0.02

        if is_long:
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100
        trade["pnl_pct"] = round(pnl_pct, 2)

        # Peak tracking
        peak = trade.get("peak_price", entry)
        if is_long:
            trade["peak_price"] = max(peak, current_price)
        else:
            trade["peak_price"] = min(peak, current_price)

        old_status = trade["status"]
        tp = trade.get("tp", trade.get("tp1", entry))
        sl = trade["sl"]

        # Progressive trailing SL
        tp_distance = abs(tp - entry)
        if tp_distance > 0:
            if is_long:
                progress = (current_price - entry) / tp_distance
            else:
                progress = (entry - current_price) / tp_distance

            if progress >= 0.30:
                offset = max(0.14, 0.36 - 0.20 * progress)
                trail_level = max(0, progress - offset)
                if is_long:
                    new_sl = entry + tp_distance * trail_level
                    if new_sl > trade["sl"]:
                        trade["sl"] = round(new_sl, 6)
                else:
                    new_sl = entry - tp_distance * trail_level
                    if new_sl < trade["sl"]:
                        trade["sl"] = round(new_sl, 6)

        sl = trade["sl"]

        # TP check
        tp_hit = (current_price >= tp) if is_long else (current_price <= tp)
        if tp_hit:
            if is_long:
                pnl_r = (tp - entry) / risk_1r
                pnl_pct = (tp - entry) / entry * 100
            else:
                pnl_r = (entry - tp) / risk_1r
                pnl_pct = (entry - tp) / entry * 100
            trade["pnl_r"] = round(pnl_r, 4)
            trade["pnl_pct"] = round(pnl_pct, 2)
            trade["current_price"] = tp
            trade["status"] = "tp"
            trade["close_reason"] = "tp_hit"
            trade["closed_at"] = datetime.now(UTC).isoformat()

        # SL check
        sl_hit = (current_price <= sl) if is_long else (current_price >= sl)
        if sl_hit and trade["status"] == "open":
            if is_long:
                pnl_r = (sl - entry) / risk_1r
                pnl_pct = (sl - entry) / entry * 100
            else:
                pnl_r = (entry - sl) / risk_1r
                pnl_pct = (entry - sl) / entry * 100
            trade["pnl_r"] = round(pnl_r, 4)
            trade["pnl_pct"] = round(pnl_pct, 2)
            trade["current_price"] = sl
            trade["status"] = "sl"
            trade["closed_at"] = datetime.now(UTC).isoformat()
            if pnl_r > 0.1:
                trade["close_reason"] = "trailing_profit"
            elif pnl_r >= -0.1:
                trade["close_reason"] = "breakeven"
            else:
                trade["close_reason"] = "stop_loss"

        if trade["status"] != old_status and self.on_trade_close:
            close_pnl = trade.get("pnl_pct", 0)
            is_sl = trade.get("close_reason") in ("stop_loss", "timeout")
            self.on_trade_close(close_pnl, is_sl)

        return trade

    async def update_all_trades(self) -> list[dict]:
        """Check all open trades for TP/SL/timeout."""
        trades = self._load_trades()
        open_trades = [t for t in trades if t["status"] == "open"]
        for trade in open_trades:
            await self.check_trade_status(trade)
            await asyncio.sleep(0.1)
        self._save_trades(trades)
        return trades

    async def btc_emergency_protection(self, btc_gate) -> int:
        """Tighten SL on open positions during BTC crash/dump/pump/rally."""
        is_crash = getattr(btc_gate, "is_crash", False)
        is_dump = getattr(btc_gate, "is_dump", False)
        is_pump = getattr(btc_gate, "is_pump", False)
        is_rally = getattr(btc_gate, "is_rally", False)

        if not (is_crash or is_dump or is_pump or is_rally):
            return 0

        trades = self._load_trades()
        open_trades = [t for t in trades if t["status"] == "open"]
        if not open_trades:
            return 0

        affected = 0
        changed = False

        for trade in open_trades:
            entry = trade["entry"]
            sl = trade["sl"]
            is_long = trade["direction"] == "LONG"
            current_price = trade.get("current_price", entry)

            if is_long:
                pnl_pct = (current_price - entry) / entry * 100
            else:
                pnl_pct = (entry - current_price) / entry * 100

            new_sl = None

            if is_long and (is_dump or is_crash):
                if is_crash:
                    if pnl_pct > 0:
                        new_sl = entry
                    elif pnl_pct > -2.0:
                        new_sl = (sl + entry) / 2
                else:
                    if pnl_pct > 0.3:
                        new_sl = entry
                if new_sl is not None and new_sl > sl:
                    trade["sl"] = round(new_sl, 6)
                    changed = True
                    affected += 1

            elif not is_long and (is_pump or is_rally):
                if is_rally:
                    if pnl_pct > 0:
                        new_sl = entry
                    elif pnl_pct > -2.0:
                        new_sl = (sl + entry) / 2
                else:
                    if pnl_pct > 0.3:
                        new_sl = entry
                if new_sl is not None and new_sl < sl:
                    trade["sl"] = round(new_sl, 6)
                    changed = True
                    affected += 1

        if changed:
            self._save_trades(trades)

        return affected

    def get_trade_stats(self, trades: list[dict] | None = None) -> dict:
        """Compute trade statistics (win rate, PnL, R-units, etc.)."""
        if trades is None:
            trades = self._load_trades()

        closed = [t for t in trades if t["status"] in ("tp", "sl")]
        open_trades = [t for t in trades if t["status"] == "open"]

        wins = [t for t in closed if t.get("pnl_r", 0) > 0.1]
        losses = [t for t in closed if t.get("pnl_r", 0) < -0.1]
        breakeven = [t for t in closed if -0.1 <= t.get("pnl_r", 0) <= 0.1]

        r_values = [t.get("pnl_r", 0) for t in closed]
        total_r = round(sum(r_values), 4) if r_values else 0
        avg_r = round(total_r / len(r_values), 4) if r_values else 0

        realized_pnl = round(sum(t.get("pnl_pct", 0) or 0 for t in closed), 2)
        unrealized_pnl = round(sum(t.get("pnl_pct", 0) or 0 for t in open_trades), 2)

        gp = sum(t.get("pnl_r", 0) for t in wins) if wins else 0
        gl = abs(sum(t.get("pnl_r", 0) for t in losses)) if losses else 0.01
        pf = round(gp / gl, 2) if gl > 0.01 else 0

        return {
            "total": len(trades),
            "open": len(open_trades),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate": round(len(wins) / (len(wins) + len(losses)) * 100, 1) if (wins or losses) else 0,
            "profit_factor": pf,
            "total_r": total_r,
            "avg_r": avg_r,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": round(realized_pnl + unrealized_pnl, 2),
        }

    def get_open_positions(self) -> list[dict]:
        """Return all currently open trades."""
        trades = self._load_trades()
        return [t for t in trades if t["status"] == "open"]

    def count_open_by_direction(self) -> dict[str, int]:
        """Count open positions per direction."""
        open_trades = self.get_open_positions()
        return {
            "LONG": sum(1 for t in open_trades if t["direction"] == "LONG"),
            "SHORT": sum(1 for t in open_trades if t["direction"] == "SHORT"),
            "total": len(open_trades),
        }
