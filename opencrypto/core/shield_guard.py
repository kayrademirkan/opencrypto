"""
OpenCrypto — ShieldGuard Risk Management

Consolidated risk management module:
- Market manipulation detection (9 checks)
- Daily drawdown protection
- BTC market gate (crash/dump/pump/rally detection)
- Position direction caps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import pandas as pd

from opencrypto.core.exceptions import DataFetchError
from opencrypto.indicators.technical import compute_all_indicators

logger = logging.getLogger(__name__)


@dataclass
class ManipulationResult:
    risk_score: int = 0
    warnings: list[str] = field(default_factory=list)
    vol_spike_ratio: float = 0.0
    is_blocked: bool = False


@dataclass
class BTCGate:
    trend: str = "neutral"
    allow_long: bool = True
    allow_short: bool = True
    reason: str = "BTC neutral"
    btc_1h_change: float = 0.0
    btc_4h_change: float = 0.0
    btc_24h_change: float = 0.0
    btc_momentum: float = 0.0
    btc_rsi: float = 50.0
    is_crash: bool = False
    is_dump: bool = False
    is_pump: bool = False
    is_rally: bool = False
    is_transition_dump: bool = False
    is_transition_pump: bool = False
    is_pump_exhausted: bool = False


class ShieldGuard:
    """Unified risk management for the trading framework.

    Usage:
        guard = ShieldGuard(daily_max_drawdown=-6.0, daily_max_sl=3)
        manip = guard.detect_manipulation(df)
        if manip.is_blocked:
            skip this signal

        btc = await guard.get_btc_gate()
        if not btc.allow_long:
            skip LONG signals
    """

    def __init__(
        self,
        manipulation_threshold: int = 15,
        daily_max_drawdown: float = -6.0,
        daily_max_sl_count: int = 3,
        max_open_long: int = 6,
        max_open_short: int = 3,
    ):
        self.manipulation_threshold = manipulation_threshold
        self.daily_max_drawdown = daily_max_drawdown
        self.daily_max_sl_count = daily_max_sl_count
        self.max_open_long = max_open_long
        self.max_open_short = max_open_short

        self._daily_tracker = {
            "date": None,
            "realized_pnl": 0.0,
            "sl_count": 0,
            "day_stopped": False,
        }
        self._btc_cache: BTCGate | None = None
        self._prev_btc_momentum: float = 0.0
        self._pump_exhausted_alerted: bool = False

    # ─── MANIPULATION DETECTION ───

    def detect_manipulation(self, df: pd.DataFrame) -> ManipulationResult:
        """9-check manipulation detection system.
        Returns ManipulationResult with risk_score 0-100.
        """
        result = ManipulationResult()

        if len(df) < 30:
            return result

        recent = df.iloc[-5:]
        hist = df.iloc[-50:]
        vol_mean = hist["volume"].mean()
        vol_std = hist["volume"].std()
        if pd.isna(vol_std):
            vol_std = 0.0
        spike = 0

        # 1. Volume spike
        last_vol = float(recent["volume"].iloc[-1])
        if vol_mean > 0:
            spike = last_vol / vol_mean
            if spike > 5:
                result.risk_score += 25
                result.warnings.append(f"Volume spike: {spike:.1f}x")
            elif spike > 3:
                result.risk_score += 15
                result.warnings.append(f"High volume: {spike:.1f}x")

        # 2. Wick analysis
        for i in range(-3, 0):
            if i >= -len(recent):
                bar = recent.iloc[i]
                body = abs(bar["close"] - bar["open"])
                upper_wick = bar["high"] - max(bar["close"], bar["open"])
                lower_wick = min(bar["close"], bar["open"]) - bar["low"]
                total_wick = upper_wick + lower_wick
                if body > 0 and total_wick > body * 3:
                    result.risk_score += 15
                    result.warnings.append("Excessive wick — manipulation sign")
                    break

        # 3. Wash trading (4 sigma)
        if vol_std > 0 and last_vol > vol_mean + 4 * vol_std:
            result.risk_score += 15
            result.warnings.append("Wash trading suspected")

        # 4. Pump & Dump pattern
        for hours, label in [(6, "6h"), (12, "12h"), (24, "24h")]:
            if len(df) >= hours:
                segment = df.iloc[-hours:]
                seg_high = segment["close"].max()
                seg_low = segment["close"].min()
                mid = (seg_high + seg_low) / 2
                if mid > 0:
                    volatility = (seg_high - seg_low) / mid * 100
                    if volatility > 15:
                        result.risk_score += 20
                        result.warnings.append(f"Pump&Dump ({label}): {volatility:.1f}%")
                        break
                    elif volatility > 10:
                        result.risk_score += 10
                        result.warnings.append(f"High volatility ({label}): {volatility:.1f}%")
                        break

        # 5. Consecutive candles — 7+ same direction = overextension
        if len(df) >= 8:
            last_8 = df.iloc[-8:]
            green = sum(1 for _, row in last_8.iterrows() if row["close"] > row["open"])
            red = 8 - green
            if green >= 7 or red >= 7:
                result.risk_score += 15
                result.warnings.append(f"Consecutive candles: {green}G/{red}R — overextended")

        # 6. Taker buy/sell imbalance
        if "taker_buy_base" in df.columns and len(df) >= 5:
            taker_buy = float(df["taker_buy_base"].iloc[-5:].sum())
            total_vol = float(df["volume"].iloc[-5:].sum())
            if total_vol > 0:
                taker_ratio = taker_buy / total_vol
                if taker_ratio > 0.85 or taker_ratio < 0.15:
                    result.risk_score += 10
                    result.warnings.append(f"Taker imbalance: {taker_ratio * 100:.0f}%")

        # 7. Liquidation cascade
        if len(df) >= 3:
            for i in range(-3, 0):
                bar = df.iloc[i]
                move = abs(bar["close"] - bar["open"]) / (bar["open"] + 1e-10) * 100
                vol_ratio = bar["volume"] / (vol_mean + 1e-10)
                if move > 3 and vol_ratio > 3:
                    result.risk_score += 10
                    result.warnings.append("Liquidation cascade suspected")
                    break

        # 8. Spread / gap analysis (spoofing sign)
        if len(df) >= 5:
            avg_body = (hist["close"] - hist["open"]).abs().mean()
            for i in range(-4, 0):
                if i + 1 <= -1:
                    gap = abs(df["open"].iloc[i + 1] - df["close"].iloc[i])
                    if avg_body > 0 and gap > avg_body * 3:
                        result.risk_score += 10
                        result.warnings.append("Large gap — spoofing sign")
                        break

        # 9. OBI (Orderbook Imbalance from taker data)
        if "taker_buy_base" in df.columns and len(df) >= 10:
            recent_10 = df.iloc[-10:]
            taker_buy_10 = float(recent_10["taker_buy_base"].sum())
            total_vol_10 = float(recent_10["volume"].sum())
            if total_vol_10 > 0:
                taker_sell_10 = total_vol_10 - taker_buy_10
                obi = (taker_buy_10 - taker_sell_10) / total_vol_10
                if abs(obi) > 0.7:
                    result.risk_score += 15
                    side = "BUY" if obi > 0 else "SELL"
                    result.warnings.append(f"OBI manipulation: {side} pressure ({obi:+.2f})")
                elif abs(obi) > 0.5:
                    result.risk_score += 8
                    side = "buy" if obi > 0 else "sell"
                    result.warnings.append(f"OBI imbalanced: {side} ({obi:+.2f})")

        result.risk_score = min(result.risk_score, 100)
        result.vol_spike_ratio = round(spike, 2)
        result.is_blocked = result.risk_score >= self.manipulation_threshold
        return result

    # ─── DAILY DRAWDOWN PROTECTION ───

    def _reset_daily_tracker(self):
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._daily_tracker["date"] != today:
            self._daily_tracker = {
                "date": today,
                "realized_pnl": 0.0,
                "sl_count": 0,
                "day_stopped": False,
            }

    def record_trade_close(self, pnl_pct: float, is_sl: bool = False):
        """Record a closed trade for daily drawdown tracking."""
        self._reset_daily_tracker()
        self._daily_tracker["realized_pnl"] += pnl_pct
        if is_sl:
            self._daily_tracker["sl_count"] += 1
        if self.is_daily_limit_hit():
            self._daily_tracker["day_stopped"] = True

    def is_daily_limit_hit(self) -> bool:
        """Check if daily drawdown or SL limit is reached."""
        self._reset_daily_tracker()
        if self._daily_tracker["day_stopped"]:
            return True
        if self._daily_tracker["realized_pnl"] <= self.daily_max_drawdown:
            return True
        if self._daily_tracker["sl_count"] >= self.daily_max_sl_count:
            return True
        return False

    @property
    def daily_stats(self) -> dict:
        self._reset_daily_tracker()
        return dict(self._daily_tracker)

    # ─── BTC MARKET GATE ───

    async def _fetch_btc_1h_change(self) -> float:
        endpoints = [
            "https://fapi.binance.com/fapi/v1/klines",
            "https://api.binance.com/api/v3/klines",
        ]
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 4}
        for url in endpoints:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if len(data) < 3:
                        continue
                    closes = [float(d[4]) for d in data]
                    return round((closes[-2] - closes[-3]) / closes[-3] * 100, 3)
            except Exception:
                continue
        return 0.0

    async def _fetch_btc_4h_df(self) -> pd.DataFrame:
        endpoints = [
            "https://fapi.binance.com/fapi/v1/klines",
            "https://api.binance.com/api/v3/klines",
        ]
        params = {"symbol": "BTCUSDT", "interval": "4h", "limit": 100}
        for url in endpoints:
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if not data or len(data) < 50:
                        continue
                    df = pd.DataFrame(
                        data,
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
                    for col in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "quote_volume",
                        "taker_buy_base",
                        "taker_buy_quote",
                    ]:
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
                    return df.reset_index(drop=True)
            except Exception:
                continue
        raise DataFetchError("All BTC 4h endpoints failed", symbol="BTCUSDT", source="binance")

    async def get_btc_gate(self) -> BTCGate:
        """Fetch BTC data and determine market conditions."""
        try:
            btc_1h_change = await self._fetch_btc_1h_change()
            df = await self._fetch_btc_4h_df()
            df = compute_all_indicators(df)
            close = float(df["close"].iloc[-1])
            ema9 = float(df["ema_9"].iloc[-1])
            ema21 = float(df["ema_21"].iloc[-1])
            rsi_val = float(df["rsi"].iloc[-1])

            btc_4h_change = 0.0
            btc_24h_change = 0.0
            if len(df) >= 2:
                btc_4h_change = (close - float(df["close"].iloc[-2])) / float(df["close"].iloc[-2]) * 100
            if len(df) >= 7:
                btc_24h_change = (close - float(df["close"].iloc[-7])) / float(df["close"].iloc[-7]) * 100

            btc_momentum = round(btc_1h_change * 3 + btc_4h_change * 2 + btc_24h_change, 2)

            is_crash = btc_momentum <= -10
            is_dump = btc_momentum <= -4
            is_pump = btc_momentum >= 4
            is_rally = btc_momentum >= 10

            momentum_delta = btc_momentum - self._prev_btc_momentum
            is_transition_dump = (momentum_delta <= -6) and (self._prev_btc_momentum >= 2)
            is_transition_pump = (momentum_delta >= 6) and (self._prev_btc_momentum <= -2)

            if is_transition_dump and not is_dump:
                is_dump = True

            self._prev_btc_momentum = btc_momentum

            is_pump_exhausted = is_pump and rsi_val >= 72

            if ema9 > ema21 and close > ema21:
                if rsi_val > 70:
                    trend, allow_long, allow_short = "bullish", True, True
                    reason = f"BTC bullish but RSI high ({rsi_val:.0f})"
                else:
                    trend, allow_long, allow_short = "bullish", True, False
                    reason = f"BTC uptrend (EMA9={ema9:.0f})"
            elif ema9 < ema21 and close < ema21:
                if rsi_val < 30:
                    trend, allow_long, allow_short = "bearish", True, True
                    reason = f"BTC bearish but RSI low ({rsi_val:.0f})"
                else:
                    trend, allow_long, allow_short = "bearish", False, True
                    reason = f"BTC downtrend (EMA9={ema9:.0f})"
            else:
                trend, allow_long, allow_short = "neutral", True, True
                reason = "BTC neutral zone"

            gate = BTCGate(
                trend=trend,
                allow_long=allow_long,
                allow_short=allow_short,
                reason=reason,
                btc_1h_change=btc_1h_change,
                btc_4h_change=round(btc_4h_change, 3),
                btc_24h_change=round(btc_24h_change, 3),
                btc_momentum=btc_momentum,
                btc_rsi=round(rsi_val, 1),
                is_crash=is_crash,
                is_dump=is_dump,
                is_pump=is_pump,
                is_rally=is_rally,
                is_transition_dump=is_transition_dump,
                is_transition_pump=is_transition_pump,
                is_pump_exhausted=is_pump_exhausted,
            )
            self._btc_cache = gate
            return gate

        except Exception as exc:
            logger.warning("BTC gate fetch failed, using cache: %s", exc)
            if self._btc_cache is not None:
                return self._btc_cache
            return BTCGate()

    def check_direction_cap(self, direction: str, open_long: int, open_short: int) -> bool:
        """Return True if allowed to open this direction, False if cap reached."""
        if direction == "LONG" and open_long >= self.max_open_long:
            return False
        if direction == "SHORT" and open_short >= self.max_open_short:
            return False
        return True
