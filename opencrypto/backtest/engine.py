"""
OpenCrypto — Strategy-Agnostic Backtest Engine

Backtest any strategy that implements BaseStrategy protocol.
Features: trailing SL, R-unit tracking, MCL, equity curve, drawdown.

Usage:
    from opencrypto.backtest import BacktestEngine, run_backtest
    from my_strategy import MyStrategy

    report = await run_backtest(MyStrategy(), days=30, top_n=50)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from opencrypto.core.base_strategy import BaseStrategy, StrategySignal
from opencrypto.core.config import DATA_DIR
from opencrypto.core.data_bridge import DataBridge
from opencrypto.core.shield_guard import ShieldGuard
from opencrypto.indicators.technical import compute_all_indicators

logger = logging.getLogger(__name__)

FAPI_URL = "https://fapi.binance.com"
BACKTEST_DIR = str(DATA_DIR / "backtest_results")

FEE_BPS = 5
SLIP_BPS = 3


@dataclass
class Trade:
    trade_id: int = 0
    symbol: str = ""
    direction: str = "long"
    signal_type: str = "unknown"
    entry_time: str = ""
    entry_price: float = 0.0
    sl: float = 0.0
    sl_original: float = 0.0
    tp: float = 0.0
    ttl_bars: int = 72
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    sl_moved_to_be: bool = False
    sl_locked_profit: bool = False
    peak_price: float = 0.0
    pnl_r: float = 0.0
    time_to_event_bars: int = 0
    pnl_pct: float = 0.0
    confidence: float = 0.0
    rr_ratio: float = 0.0
    leverage: int = 1
    manip_risk: float = 0.0
    reasons: list = field(default_factory=list)
    fee_r: float = 0.0


def _risk_per_unit(entry: float, sl: float, direction: str) -> float:
    if direction == "long":
        return max(abs(entry - sl), entry * 0.001)
    return max(abs(sl - entry), entry * 0.001)


def _rr(entry: float, price: float, sl: float, direction: str) -> float:
    risk = _risk_per_unit(entry, sl, direction)
    if direction == "long":
        return (price - entry) / risk
    return (entry - price) / risk


def _roundtrip_fee_r(entry: float, sl: float, direction: str, fee_bps: int, slip_bps: int) -> float:
    risk = _risk_per_unit(entry, sl, direction)
    total_fee = 2 * (fee_bps + slip_bps) / 10_000 * entry
    return total_fee / risk if risk > 0 else 0.0


def _apply_slip(px: float, *, direction: str, kind: str, slip_bps: int) -> float:
    frac = slip_bps / 10_000
    if kind == "entry":
        return px * (1 + frac) if direction == "long" else px * (1 - frac)
    else:
        return px * (1 - frac) if direction == "long" else px * (1 + frac)


def simulate_trade(signal_dict: dict, future_df: pd.DataFrame, *, max_hold: int = 72, exec_delay: int = 1) -> Trade:
    """Simulate a single trade with trailing SL."""
    direction = signal_dict["direction"].lower()
    is_long = direction == "long"
    leverage = signal_dict.get("leverage", 1)

    trade = Trade(
        symbol=signal_dict["symbol"],
        direction=direction,
        signal_type=signal_dict.get("signal_type", "unknown"),
        confidence=signal_dict.get("confidence", 0),
        rr_ratio=signal_dict.get("rr_ratio", 0),
        leverage=leverage,
        reasons=signal_dict.get("reasons", []),
    )

    if future_df.empty or len(future_df) < exec_delay + 1:
        trade.exit_reason = "no_data"
        return trade

    entry_raw = float(future_df["open"].iloc[exec_delay - 1]) if exec_delay > 0 else signal_dict["entry"]
    entry_px = _apply_slip(entry_raw, direction=direction, kind="entry", slip_bps=SLIP_BPS)
    trade.entry_price = entry_px

    sig_entry = signal_dict["entry"]
    tp_key = "tp1" if "tp1" in signal_dict else "tp"
    if sig_entry > 0 and abs(entry_px - sig_entry) / sig_entry < 0.05:
        ratio = entry_px / sig_entry
        trade.sl = signal_dict["sl"] * ratio
        trade.tp = signal_dict[tp_key] * ratio
    else:
        trade.sl = signal_dict["sl"]
        trade.tp = signal_dict[tp_key]

    trade.sl_original = trade.sl
    trade.peak_price = entry_px
    trade.ttl_bars = max_hold

    if direction == "long" and not (trade.sl < entry_px < trade.tp):
        trade.exit_reason = "invalid_levels"
        trade.exit_price = entry_px
        return trade
    if direction == "short" and not (trade.sl > entry_px > trade.tp):
        trade.exit_reason = "invalid_levels"
        trade.exit_price = entry_px
        return trade

    fee_r = _roundtrip_fee_r(entry_px, trade.sl, direction, FEE_BPS, SLIP_BPS)
    trade.fee_r = fee_r

    start_bar = exec_delay
    bars = min(len(future_df), max_hold + exec_delay)
    current_sl = trade.sl
    tp_target = trade.tp
    tp_distance = abs(tp_target - entry_px)

    for k_abs in range(start_bar, bars):
        k = k_abs - start_bar + 1
        hi = float(future_df["high"].iloc[k_abs])
        lo = float(future_df["low"].iloc[k_abs])

        if is_long:
            trade.peak_price = max(trade.peak_price, hi)
        else:
            trade.peak_price = min(trade.peak_price, lo)

        if tp_distance > 0:
            if is_long:
                progress = (trade.peak_price - entry_px) / tp_distance
            else:
                progress = (entry_px - trade.peak_price) / tp_distance

            if progress >= 0.40:
                if not trade.sl_moved_to_be:
                    trade.sl_moved_to_be = True
                if progress >= 1.3:
                    offset = 0.12
                elif progress >= 1.0:
                    offset = 0.18
                elif progress >= 0.80:
                    offset = 0.25
                elif progress >= 0.60:
                    offset = 0.30
                else:
                    offset = 0.38
                trail_level = max(0, progress - offset)
                if is_long:
                    new_sl = entry_px + tp_distance * trail_level
                    if new_sl > current_sl:
                        current_sl = new_sl
                        if trail_level >= 0.30:
                            trade.sl_locked_profit = True
                else:
                    new_sl = entry_px - tp_distance * trail_level
                    if new_sl < current_sl:
                        current_sl = new_sl
                        if trail_level >= 0.30:
                            trade.sl_locked_profit = True

        sl_hit = (lo <= current_sl) if is_long else (hi >= current_sl)
        if sl_hit:
            sl_exec = _apply_slip(current_sl, direction=direction, kind="sl", slip_bps=SLIP_BPS)
            trade.pnl_r = _rr(entry_px, sl_exec, trade.sl_original, direction) - fee_r
            trade.exit_price = sl_exec
            trade.time_to_event_bars = k
            if abs(current_sl - entry_px) / entry_px < 0.002:
                trade.exit_reason = "breakeven"
            else:
                trade.exit_reason = "sl"
            if "timestamp" in future_df.columns:
                trade.exit_time = str(future_df["timestamp"].iloc[k_abs])
            break

        tp_hit = (hi >= tp_target) if is_long else (lo <= tp_target)
        if tp_hit:
            tp_exec = _apply_slip(tp_target, direction=direction, kind="tp", slip_bps=SLIP_BPS)
            trade.pnl_r = _rr(entry_px, tp_exec, trade.sl_original, direction) - fee_r
            trade.exit_price = tp_exec
            trade.exit_reason = "tp"
            trade.time_to_event_bars = k
            if "timestamp" in future_df.columns:
                trade.exit_time = str(future_df["timestamp"].iloc[k_abs])
            break

    if trade.exit_price is None:
        last_idx = min(bars - 1, len(future_df) - 1)
        exit_raw = float(future_df["close"].iloc[last_idx])
        exit_px = _apply_slip(exit_raw, direction=direction, kind="ttl", slip_bps=SLIP_BPS)
        trade.time_to_event_bars = bars - start_bar
        trade.pnl_r = _rr(entry_px, exit_px, trade.sl_original, direction) - fee_r
        trade.exit_price = exit_px
        trade.exit_reason = "ttl"
        if "timestamp" in future_df.columns and last_idx < len(future_df):
            trade.exit_time = str(future_df["timestamp"].iloc[last_idx])

    if trade.exit_price and trade.entry_price > 0:
        if is_long:
            trade.pnl_pct = round((trade.exit_price - entry_px) / entry_px * 100, 4)
        else:
            trade.pnl_pct = round((entry_px - trade.exit_price) / entry_px * 100, 4)

    trade.sl = current_sl
    return trade


class BacktestEngine:
    """Walk-forward backtest engine that accepts any BaseStrategy."""

    def __init__(
        self,
        strategy: BaseStrategy,
        shield_guard: ShieldGuard | None = None,
        lookback: int = 200,
        step: int = 6,
        max_hold: int = 72,
        initial_capital: float = 1000.0,
        risk_per_trade: float = 0.02,
        max_drawdown_pct: float = 50.0,
    ):
        self.strategy = strategy
        self.shield = shield_guard or ShieldGuard()
        self.lookback = lookback
        self.step = step
        self.max_hold = max_hold
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct

        self.equity_curve: list[float] = [initial_capital]
        self.drawdown_curve: list[float] = [0.0]
        self.max_dd_reached = 0.0
        self.results: list[Trade] = []
        self.skipped_manip = 0
        self.total_checked = 0
        self.trade_counter = 0
        self.stopped = False

    def _update_equity(self, trade: Trade):
        risk_amount = self.capital * self.risk_per_trade
        pnl_usd = trade.pnl_r * risk_amount
        self.capital += pnl_usd
        self.peak_capital = max(self.peak_capital, self.capital)
        self.equity_curve.append(round(self.capital, 2))
        dd = (self.peak_capital - self.capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0
        self.drawdown_curve.append(round(dd, 2))
        self.max_dd_reached = max(self.max_dd_reached, dd)

    async def run_coin(self, symbol: str, df: pd.DataFrame) -> list[Trade]:
        found: list[Trade] = []
        if len(df) < self.lookback + 50:
            return found

        for idx in range(self.lookback, len(df) - self.max_hold - 1, self.step):
            self.total_checked += 1

            if self.stopped or (
                self.peak_capital > 0
                and (self.peak_capital - self.capital) / self.peak_capital * 100 >= self.max_drawdown_pct
            ):
                self.stopped = True
                continue

            window = df.iloc[idx - self.lookback : idx + 1].copy().reset_index(drop=True)
            try:
                window = compute_all_indicators(window)
            except Exception:
                continue

            manip = self.shield.detect_manipulation(window)
            if manip.is_blocked:
                self.skipped_manip += 1
                continue

            try:
                signal = self.strategy.generate_signal(symbol, window)
            except Exception:
                continue
            if signal is None:
                continue

            signal_dict = signal.to_dict() if isinstance(signal, StrategySignal) else signal
            future = df.iloc[idx + 1 : idx + 1 + self.max_hold + 1].copy().reset_index(drop=True)

            trade = simulate_trade(signal_dict, future, max_hold=self.max_hold)

            self.trade_counter += 1
            trade.trade_id = self.trade_counter
            if "timestamp" in df.columns:
                trade.entry_time = str(df["timestamp"].iloc[idx])

            self._update_equity(trade)
            found.append(trade)
            self.results.append(trade)

        return found


def calc_stats(trades: list[Trade], engine: BacktestEngine) -> dict:
    if not trades:
        return {"error": "No trades"}

    n = len(trades)
    wins = [t for t in trades if t.pnl_r > 0]
    losses = [t for t in trades if t.pnl_r <= 0]
    nw, nl = len(wins), len(losses)

    r_values = [t.pnl_r for t in trades]
    total_r = round(sum(r_values), 4)
    avg_r = round(np.mean(r_values), 4)
    avg_win_r = round(np.mean([t.pnl_r for t in wins]), 4) if wins else 0
    avg_loss_r = round(np.mean([t.pnl_r for t in losses]), 4) if losses else 0

    gp = sum(t.pnl_r for t in wins) if wins else 0
    gl = abs(sum(t.pnl_r for t in losses)) if losses else 0.01
    pf = round(gp / gl, 2) if gl > 0.01 else float("inf")

    max_dd = round(engine.max_dd_reached, 2)
    final_capital = round(engine.capital, 2)
    total_return = round((engine.capital - engine.initial_capital) / engine.initial_capital * 100, 2)

    win_rate = nw / n if n > 0 else 0
    ev = round(win_rate * avg_win_r + (1 - win_rate) * avg_loss_r, 4)

    tp_h = sum(1 for t in trades if t.exit_reason == "tp")
    sl_h = sum(1 for t in trades if t.exit_reason == "sl")
    be_h = sum(1 for t in trades if t.exit_reason == "breakeven")
    ttlh = sum(1 for t in trades if t.exit_reason == "ttl")

    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    lwr = round(sum(1 for t in longs if t.pnl_r > 0) / len(longs) * 100, 1) if longs else 0
    swr = round(sum(1 for t in shorts if t.pnl_r > 0) / len(shorts) * 100, 1) if shorts else 0

    return {
        "total": n,
        "wins": nw,
        "losses": nl,
        "win_rate": round(nw / n * 100, 2),
        "profit_factor": pf,
        "ev": ev,
        "total_r": total_r,
        "avg_r": avg_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "initial_capital": engine.initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "avg_hold_h": round(np.mean([t.time_to_event_bars for t in trades]), 1),
        "tp_rate": round(tp_h / n * 100, 1),
        "sl_rate": round(sl_h / n * 100, 1),
        "be_rate": round(be_h / n * 100, 1),
        "ttl_rate": round(ttlh / n * 100, 1),
        "trailing_be_count": sum(1 for t in trades if t.sl_moved_to_be),
        "trailing_lock_count": sum(1 for t in trades if t.sl_locked_profit),
        "longs": len(longs),
        "shorts": len(shorts),
        "long_wr": lwr,
        "short_wr": swr,
        "manip_filtered": engine.skipped_manip,
    }


async def fetch_historical(symbol: str, interval: str = "1h", days: int = 30) -> pd.DataFrame:
    all_data = []
    end_ms = int(datetime.now(UTC).timestamp() * 1000)
    start_ms = int((datetime.now(UTC) - timedelta(days=days)).timestamp() * 1000)
    clean = symbol.replace(".P", "").upper()

    async with httpx.AsyncClient(timeout=30) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": clean,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
            try:
                resp = await client.get(f"{FAPI_URL}/fapi/v1/klines", params=params)
                if resp.status_code != 200:
                    resp = await client.get("https://api.binance.com/api/v3/klines", params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                all_data.extend(data)
                cursor = data[-1][6] + 1
                if len(data) < 1000:
                    break
                await asyncio.sleep(0.2)
            except Exception as exc:
                logger.debug("Historical fetch failed for %s: %s", symbol, exc)
                break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
    ].copy()
    return df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


async def run_backtest(
    strategy: BaseStrategy,
    symbols: list[str] | None = None,
    top_n: int = 100,
    days: int = 30,
    step: int = 6,
    max_hold: int = 72,
    initial_capital: float = 1000.0,
    risk_per_trade: float = 0.02,
    max_drawdown: float = 50.0,
    save: bool = True,
) -> dict:
    """Run full backtest with any strategy."""
    t0 = datetime.now(UTC)

    logger.info("Backtest started — %s v%s", strategy.name, strategy.version)
    logger.info(
        "Period: %d days | Step: %dh | Max hold: %dh | Capital: $%.0f | Risk/trade: %.1f%%",
        days,
        step,
        max_hold,
        initial_capital,
        risk_per_trade * 100,
    )

    bridge = DataBridge()
    if symbols is None:
        symbols = await bridge.fetch_top_coins(top_n)

    engine = BacktestEngine(
        strategy=strategy,
        lookback=200,
        step=step,
        max_hold=max_hold,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        max_drawdown_pct=max_drawdown,
    )

    for i, sym in enumerate(symbols):
        df = await fetch_historical(sym, "1h", days)
        if df.empty or len(df) < 250:
            logger.debug("[%d/%d] %s — skip (%d bars)", i + 1, len(symbols), sym, len(df))
            continue
        sigs = await engine.run_coin(sym, df)
        if sigs:
            w = sum(1 for s in sigs if s.pnl_r > 0)
            pnl_r = sum(s.pnl_r for s in sigs)
            logger.info(
                "[%d/%d] %s — %d trades | %dW/%dL | R: %+.2f",
                i + 1,
                len(symbols),
                sym,
                len(sigs),
                w,
                len(sigs) - w,
                pnl_r,
            )
        else:
            logger.debug("[%d/%d] %s — no signals", i + 1, len(symbols), sym)
        if engine.stopped:
            logger.warning("Backtest stopped — max drawdown %.1f%% reached", max_drawdown)
            break
        await asyncio.sleep(0.3)

    t1 = datetime.now(UTC)
    elapsed = (t1 - t0).total_seconds()

    if not engine.results:
        logger.info("Backtest finished — no trades found (%.1fs)", elapsed)
        return {"status": "done", "signals": 0, "elapsed": round(elapsed, 1)}

    stats = calc_stats(engine.results, engine)

    logger.info(
        "Results: %d trades | WR: %.1f%% | PF: %.2f | R: %+.2fR (avg %.2fR)",
        stats["total"],
        stats["win_rate"],
        stats["profit_factor"],
        stats["total_r"],
        stats["avg_r"],
    )
    logger.info(
        "Capital: $%.0f → $%.0f (%+.2f%%) | Max DD: %.1f%% | Time: %.0fs",
        stats["initial_capital"],
        stats["final_capital"],
        stats["total_return"],
        stats["max_drawdown"],
        elapsed,
    )

    report = {
        "meta": {
            "strategy": strategy.name,
            "version": strategy.version,
            "start": t0.isoformat(),
            "end": t1.isoformat(),
            "elapsed": round(elapsed, 1),
            "days": days,
            "coins": len(symbols),
        },
        "stats": stats,
        "equity_curve": engine.equity_curve,
        "drawdown_curve": engine.drawdown_curve,
    }

    if save:
        os.makedirs(BACKTEST_DIR, exist_ok=True)
        fname = f"bt_{strategy.name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(BACKTEST_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Report saved: %s", path)

    return report
