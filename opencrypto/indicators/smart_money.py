"""
OpenCrypto — Smart Money / ICT Indicators

Institutional price action detection tools:
- Swing Points, Break of Structure (BOS), Change of Character (CHoCH)
- Quasimodo (QML), Fakeout Detection, SR/RS Flip
- Order Blocks, Fair Value Gaps (FVG), Liquidity Sweeps
- Wyckoff Phase Detection, Compression, RSI Divergence
- Multi-Timeframe Bias (MTF)
"""

import numpy as np
import pandas as pd

from opencrypto.indicators.technical import (
    ema,
    rsi,
    supertrend,
)


def _find_pivots(series: np.ndarray, left: int = 5, right: int = 5) -> list[tuple]:
    pivots = []
    for i in range(left, len(series) - right):
        is_high = all(series[i] >= series[i - j] for j in range(1, left + 1)) and all(
            series[i] >= series[i + j] for j in range(1, right + 1)
        )
        is_low = all(series[i] <= series[i - j] for j in range(1, left + 1)) and all(
            series[i] <= series[i + j] for j in range(1, right + 1)
        )
        if is_high:
            pivots.append((i, series[i], "high"))
        if is_low:
            pivots.append((i, series[i], "low"))
    return pivots


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Detect regular and hidden RSI divergence from price pivots."""
    result = {"bullish": False, "bearish": False, "hidden_bull": False, "hidden_bear": False, "detail": ""}
    if len(df) < lookback:
        return result
    window = df.iloc[-lookback:]
    price = window["close"].values
    rsi_vals = window["rsi"].values if "rsi" in window.columns else rsi(window["close"], 14).values
    price_pivots = _find_pivots(price, left=3, right=3)
    rsi_pivots = _find_pivots(rsi_vals, left=3, right=3)
    price_lows = [(i, v) for i, v, t in price_pivots if t == "low"]
    price_highs = [(i, v) for i, v, t in price_pivots if t == "high"]
    rsi_lows = [(i, v) for i, v, t in rsi_pivots if t == "low"]
    rsi_highs = [(i, v) for i, v, t in rsi_pivots if t == "high"]
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        pl1, pl2 = price_lows[-2], price_lows[-1]
        rl1, rl2 = rsi_lows[-2], rsi_lows[-1]
        if pl2[1] < pl1[1] and rl2[1] > rl1[1]:
            result["bullish"] = True
            result["detail"] = f"Bullish Div: Price {pl1[1]:.2f}->{pl2[1]:.2f}, RSI {rl1[1]:.0f}->{rl2[1]:.0f}"
        elif pl2[1] > pl1[1] and rl2[1] < rl1[1]:
            result["hidden_bull"] = True
            result["detail"] = "Hidden Bull"
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        ph1, ph2 = price_highs[-2], price_highs[-1]
        rh1, rh2 = rsi_highs[-2], rsi_highs[-1]
        if ph2[1] > ph1[1] and rh2[1] < rh1[1]:
            result["bearish"] = True
            result["detail"] = f"Bearish Div: Price {ph1[1]:.2f}->{ph2[1]:.2f}, RSI {rh1[1]:.0f}->{rh2[1]:.0f}"
        elif ph2[1] < ph1[1] and rh2[1] > rh1[1]:
            result["hidden_bear"] = True
            result["detail"] = "Hidden Bear"
    return result


def detect_order_blocks(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """Identify institutional order blocks from volume-impulse candle patterns."""
    blocks = []
    if len(df) < lookback + 5:
        return blocks
    window = df.iloc[-lookback:]
    vol_mean = window["volume"].mean()
    for i in range(2, len(window) - 2):
        body = abs(window["close"].iloc[i] - window["open"].iloc[i])
        prev_body = abs(window["close"].iloc[i - 1] - window["open"].iloc[i - 1])
        vol = window["volume"].iloc[i]
        if vol > vol_mean * 1.5 and body > prev_body * 1.5:
            ob_low = float(window["low"].iloc[i - 1])
            ob_high = float(window["high"].iloc[i - 1])
            ob_type = "bullish" if window["close"].iloc[i] > window["open"].iloc[i] else "bearish"
            invalidated = False
            mitigate_count = 0
            future_bars = window.iloc[i + 1 :]
            for j in range(len(future_bars)):
                bar_close = float(future_bars["close"].iloc[j])
                bar_low = float(future_bars["low"].iloc[j])
                bar_high = float(future_bars["high"].iloc[j])
                if ob_type == "bullish":
                    if bar_close < ob_low:
                        invalidated = True
                        break
                    if bar_low <= ob_high and bar_close > ob_low:
                        mitigate_count += 1
                else:
                    if bar_close > ob_high:
                        invalidated = True
                        break
                    if bar_high >= ob_low and bar_close < ob_high:
                        mitigate_count += 1
            if not invalidated:
                strength = max(0.3, 1.0 - mitigate_count * 0.25)
                blocks.append(
                    {
                        "type": ob_type,
                        "low": ob_low,
                        "high": ob_high,
                        "index": int(window.index[i]),
                        "mitigate_count": mitigate_count,
                        "strength": round(strength, 2),
                        "fresh": mitigate_count == 0,
                    }
                )
    return blocks[-5:]


def detect_fvg(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """Detect Fair Value Gaps (imbalance zones) in recent price action."""
    gaps = []
    if len(df) < lookback + 3:
        return gaps
    window = df.iloc[-lookback:]
    for i in range(2, len(window)):
        if window["low"].iloc[i] > window["high"].iloc[i - 2]:
            gaps.append(
                {"type": "bullish", "top": float(window["low"].iloc[i]), "bottom": float(window["high"].iloc[i - 2])}
            )
        elif window["high"].iloc[i] < window["low"].iloc[i - 2]:
            gaps.append(
                {"type": "bearish", "top": float(window["low"].iloc[i - 2]), "bottom": float(window["high"].iloc[i])}
            )
    return gaps[-5:]


def detect_liquidity_sweep(df: pd.DataFrame) -> dict:
    """Detect liquidity sweeps, equal highs/lows, and stop hunts."""
    result = {
        "bullish_sweep": False,
        "bearish_sweep": False,
        "equal_highs": False,
        "equal_lows": False,
        "stop_hunt_bull": False,
        "stop_hunt_bear": False,
        "sweep_strength": 0,
        "detail": "",
    }
    if len(df) < 20:
        return result
    recent = df.iloc[-5:]
    prev = df.iloc[-20:-5]
    prev_low = prev["low"].min()
    prev_high = prev["high"].max()
    last_low = recent["low"].min()
    last_high = recent["high"].max()
    last_close = float(recent["close"].iloc[-1])
    last_vol = float(recent["volume"].iloc[-1])
    avg_vol = float(prev["volume"].mean())
    tolerance = 0.002
    highs = prev["high"].values
    lows = prev["low"].values
    eq_h = sum(
        1
        for i in range(len(highs))
        for j in range(i + 2, len(highs))
        if abs(highs[i] - highs[j]) / (highs[i] + 1e-10) < tolerance
    )
    eq_l = sum(
        1
        for i in range(len(lows))
        for j in range(i + 2, len(lows))
        if abs(lows[i] - lows[j]) / (lows[i] + 1e-10) < tolerance
    )
    if eq_h >= 2:
        result["equal_highs"] = True
    if eq_l >= 2:
        result["equal_lows"] = True
    sweep_str = 0
    if last_low < prev_low and last_close > prev_low:
        result["bullish_sweep"] = True
        sweep_str = 1
        parts = [f"Bullish sweep: {prev_low:.4f}"]
        sw = min(recent["close"].min(), recent["open"].min()) - last_low
        sb = abs(float(recent["close"].iloc[-1]) - float(recent["open"].iloc[-1]))
        if sw > sb * 2:
            sweep_str += 1
            parts.append("wick reject")
        if last_vol > avg_vol * 1.5:
            sweep_str += 1
            parts.append("vol spike")
        if result["equal_lows"]:
            sweep_str += 1
            parts.append("EQL swept")
            result["stop_hunt_bull"] = True
        result["detail"] = " + ".join(parts)
    if last_high > prev_high and last_close < prev_high:
        result["bearish_sweep"] = True
        ss = 1
        parts = [f"Bearish sweep: {prev_high:.4f}"]
        sw = last_high - max(float(recent["close"].iloc[-1]), float(recent["open"].iloc[-1]))
        sb = abs(float(recent["close"].iloc[-1]) - float(recent["open"].iloc[-1]))
        if sw > sb * 2:
            ss += 1
            parts.append("wick reject")
        if last_vol > avg_vol * 1.5:
            ss += 1
            parts.append("vol spike")
        if result["equal_highs"]:
            ss += 1
            parts.append("EQH swept")
            result["stop_hunt_bear"] = True
        sweep_str = max(sweep_str, ss)
        result["detail"] = " + ".join(parts)
    result["sweep_strength"] = min(sweep_str, 3)
    return result


def detect_wyckoff_phase(df: pd.DataFrame) -> dict:
    """Classify current market phase using Wyckoff methodology."""
    result = {"phase": "unknown", "detail": ""}
    if len(df) < 50:
        return result
    recent_20 = df.iloc[-20:]
    recent_50 = df.iloc[-50:]
    vol_trend = recent_20["volume"].mean() / (recent_50["volume"].mean() + 1e-10)
    price_range_20 = (recent_20["high"].max() - recent_20["low"].min()) / (recent_20["close"].mean() + 1e-10)
    price_change = (float(recent_20["close"].iloc[-1]) - float(recent_20["close"].iloc[0])) / (
        float(recent_20["close"].iloc[0]) + 1e-10
    )
    if price_range_20 < 0.05 and vol_trend > 1.2:
        result["phase"] = "accumulation"
        result["detail"] = "Tight range + rising volume"
    elif price_range_20 < 0.05 and vol_trend < 0.8:
        result["phase"] = "distribution"
        result["detail"] = "Tight range + declining volume"
    elif price_change > 0.03 and vol_trend > 1.0:
        result["phase"] = "markup"
        result["detail"] = "Upward move + volume"
    elif price_change < -0.03 and vol_trend > 1.0:
        result["phase"] = "markdown"
        result["detail"] = "Downward move + volume"
    elif price_range_20 < 0.03:
        recent_3 = df.iloc[-3:]
        if float(recent_3["low"].min()) < recent_20["low"].quantile(0.1):
            result["phase"] = "spring"
            result["detail"] = "Wyckoff Spring"
    return result


def detect_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3, lookback: int = 60) -> dict:
    result = {
        "swing_highs": [],
        "swing_lows": [],
        "structure": "unknown",
        "last_hh": None,
        "last_ll": None,
        "last_hl": None,
        "last_lh": None,
        "trend_shifts": 0,
    }
    if len(df) < lookback:
        return result
    window = df.iloc[-lookback:]
    highs = window["high"].values
    lows = window["low"].values
    swing_highs = []
    swing_lows = []
    for i in range(left, len(window) - right):
        is_sh = all(highs[i] >= highs[i - j] for j in range(1, left + 1)) and all(
            highs[i] >= highs[i + j] for j in range(1, right + 1)
        )
        is_sl = all(lows[i] <= lows[i - j] for j in range(1, left + 1)) and all(
            lows[i] <= lows[i + j] for j in range(1, right + 1)
        )
        if is_sh:
            swing_highs.append((int(window.index[i]), float(highs[i])))
        if is_sl:
            swing_lows.append((int(window.index[i]), float(lows[i])))
    result["swing_highs"] = swing_highs[-10:]
    result["swing_lows"] = swing_lows[-10:]
    if len(swing_highs) >= 2:
        if swing_highs[-1][1] > swing_highs[-2][1]:
            result["last_hh"] = swing_highs[-1]
        else:
            result["last_lh"] = swing_highs[-1]
    if len(swing_lows) >= 2:
        if swing_lows[-1][1] > swing_lows[-2][1]:
            result["last_hl"] = swing_lows[-1]
        else:
            result["last_ll"] = swing_lows[-1]
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh_c = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] > swing_highs[i - 1][1])
        ll_c = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] < swing_lows[i - 1][1])
        hl_c = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] > swing_lows[i - 1][1])
        lh_c = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] < swing_highs[i - 1][1])
        bull = hh_c + hl_c
        bear = ll_c + lh_c
        if bull > bear + 1:
            result["structure"] = "bullish"
        elif bear > bull + 1:
            result["structure"] = "bearish"
        else:
            result["structure"] = "range"
        result["trend_shifts"] = min(hh_c, 1) + min(ll_c, 1)
    return result


def detect_bos(df: pd.DataFrame, swings: dict | None = None) -> dict:
    """Detect Break of Structure (BOS) and Change of Character (CHoCH)."""
    result = {
        "bullish_bos": False,
        "bearish_bos": False,
        "choch_bull": False,
        "choch_bear": False,
        "bos_level": 0.0,
        "detail": "",
    }
    if swings is None:
        swings = detect_swing_points(df)
    current_close = float(df["close"].iloc[-1])
    sh = swings["swing_highs"]
    sl = swings["swing_lows"]
    if len(sh) < 2 or len(sl) < 2:
        return result
    last_sh = sh[-1][1]
    last_sl = sl[-1][1]
    if current_close > last_sh:
        result["bullish_bos"] = True
        result["bos_level"] = last_sh
        if swings["structure"] == "bearish":
            result["choch_bull"] = True
            result["detail"] = f"CHoCH Bull: {last_sh:.4f} broken"
        else:
            result["detail"] = f"BOS Bull: {last_sh:.4f}"
    if current_close < last_sl:
        result["bearish_bos"] = True
        result["bos_level"] = last_sl
        if swings["structure"] == "bullish":
            result["choch_bear"] = True
            result["detail"] = f"CHoCH Bear: {last_sl:.4f} broken"
        else:
            result["detail"] = f"BOS Bear: {last_sl:.4f}"
    return result


def detect_qml(df: pd.DataFrame, swings: dict | None = None) -> dict:
    """Detect Quasimodo (QML) reversal patterns."""
    result = {"bullish_qml": False, "bearish_qml": False, "qml_level": 0.0, "detail": ""}
    if swings is None:
        swings = detect_swing_points(df)
    sh = swings["swing_highs"]
    sl = swings["swing_lows"]
    current_close = float(df["close"].iloc[-1])
    if len(sh) >= 3 and len(sl) >= 3:
        sh1, sh2 = sh[-3][1], sh[-2][1]
        sl2, sl3 = sl[-2][1], sl[-1][1]
        if sh2 > sh1 and sl3 < sl2:
            qml_level = sh1
            dist = abs(current_close - qml_level) / (qml_level + 1e-10) * 100
            if dist < 3.0 and current_close > sl3:
                result["bullish_qml"] = True
                result["qml_level"] = qml_level
                result["detail"] = f"Bull QML: HH({sh2:.2f})->LL({sl3:.2f})"
        sl1 = sl[-3][1]
        sh3 = sh[-1][1]
        if sl2 < sl1 and sh3 > sh2:
            qml_level = sl1
            dist = abs(current_close - qml_level) / (qml_level + 1e-10) * 100
            if dist < 3.0 and current_close < sh3:
                result["bearish_qml"] = True
                result["qml_level"] = qml_level
                result["detail"] = f"Bear QML: LL({sl2:.2f})->HH({sh3:.2f})"
    return result


def detect_fakeout(df: pd.DataFrame) -> dict:
    """Detect false breakout / fakeout patterns (wick, volume, trap variants)."""
    result = {"bullish_fakeout": False, "bearish_fakeout": False, "fakeout_type": "", "is_trap": False, "detail": ""}
    if len(df) < 25:
        return result
    prev = df.iloc[-25:-3]
    recent_3 = df.iloc[-3:]
    last = df.iloc[-1]
    prev_high = float(prev["high"].max())
    prev_low = float(prev["low"].min())
    avg_vol = float(prev["volume"].mean())
    lc = float(last["close"])
    lh = float(last["high"])
    ll = float(last["low"])
    if lh > prev_high and lc < prev_high:
        result["bearish_fakeout"] = True
        result["fakeout_type"] = "v1_wick"
        result["detail"] = f"Bear Fakeout: {prev_high:.4f} wick above"
    if not result["bearish_fakeout"]:
        r3h = float(recent_3["high"].max())
        r3c = float(recent_3["close"].iloc[-1])
        r3v = float(recent_3["volume"].mean())
        if r3h > prev_high and r3c < prev_high and r3v < avg_vol * 0.8:
            result["bearish_fakeout"] = True
            result["fakeout_type"] = "v2_volume"
            result["detail"] = "Bear Fakeout V2: low vol breakout"
    if not result["bearish_fakeout"] and len(df) >= 3:
        pb = df.iloc[-2]
        if float(pb["close"]) > prev_high and lc < float(pb["open"]) and lc < prev_high:
            result["bearish_fakeout"] = True
            result["fakeout_type"] = "v3_trap"
            result["is_trap"] = True
            result["detail"] = "Bear Trap V3"
    if ll < prev_low and lc > prev_low:
        result["bullish_fakeout"] = True
        result["fakeout_type"] = "v1_wick"
        result["detail"] = f"Bull Fakeout: {prev_low:.4f} wick below"
    if not result["bullish_fakeout"]:
        r3l = float(recent_3["low"].min())
        r3c = float(recent_3["close"].iloc[-1])
        r3v = float(recent_3["volume"].mean())
        if r3l < prev_low and r3c > prev_low and r3v < avg_vol * 0.8:
            result["bullish_fakeout"] = True
            result["fakeout_type"] = "v2_volume"
            result["detail"] = "Bull Fakeout V2: low vol breakdown"
    if not result["bullish_fakeout"] and len(df) >= 3:
        pb = df.iloc[-2]
        if float(pb["close"]) < prev_low and lc > float(pb["open"]) and lc > prev_low:
            result["bullish_fakeout"] = True
            result["fakeout_type"] = "v3_trap"
            result["is_trap"] = True
            result["detail"] = "Bull Trap V3"
    return result


def detect_sr_flip(df: pd.DataFrame, swings: dict | None = None) -> dict:
    """Detect Support-Resistance and Resistance-Support flips with retest quality."""
    result = {"sr_flip": False, "rs_flip": False, "flip_level": 0.0, "retest_quality": 0, "detail": ""}
    if swings is None:
        swings = detect_swing_points(df)
    if len(df) < 30:
        return result
    cc = float(df["close"].iloc[-1])
    cl = float(df["low"].iloc[-1])
    ch = float(df["high"].iloc[-1])
    sh = swings["swing_highs"]
    sl = swings["swing_lows"]
    for i in range(len(sh) - 2, -1, -1):
        level = sh[i][1]
        idx = sh[i][0]
        broken = any(
            j < len(df) and float(df["close"].iloc[j]) > level * 1.003 for j in range(idx + 1, min(idx + 20, len(df)))
        )
        if broken and cc > level:
            dist = (cl - level) / (level + 1e-10) * 100
            if -0.5 <= dist <= 1.5:
                result["sr_flip"] = True
                result["flip_level"] = level
                q = 1 + (1 if dist >= 0 else 0) + (1 if cc > cl else 0)
                result["retest_quality"] = q
                result["detail"] = f"SR Flip: {level:.4f} (Q={q})"
                break
    if not result["sr_flip"]:
        for i in range(len(sl) - 2, -1, -1):
            level = sl[i][1]
            idx = sl[i][0]
            broken = any(
                j < len(df) and float(df["close"].iloc[j]) < level * 0.997
                for j in range(idx + 1, min(idx + 20, len(df)))
            )
            if broken and cc < level:
                dist = (level - ch) / (level + 1e-10) * 100
                if -0.5 <= dist <= 1.5:
                    result["rs_flip"] = True
                    result["flip_level"] = level
                    q = 1 + (1 if dist >= 0 else 0) + (1 if cc < ch else 0)
                    result["retest_quality"] = q
                    result["detail"] = f"RS Flip: {level:.4f} (Q={q})"
                    break
    return result


def detect_compression(df: pd.DataFrame) -> dict:
    """Detect price compression (narrowing range + declining volume) before breakout."""
    result = {"compression": False, "compression_strength": 0, "bias": "neutral", "detail": ""}
    if len(df) < 25:
        return result
    r5 = df.iloc[-5:]
    r20 = df.iloc[-20:]
    range_5 = float(r5["high"].max() - r5["low"].min())
    range_20 = float(r20["high"].max() - r20["low"].min())
    rr = range_5 / (range_20 + 1e-10)
    vr = float(r5["volume"].mean()) / (float(r20["volume"].mean()) + 1e-10)
    inside_c = sum(
        1
        for i in range(1, len(r5))
        if float(r5["high"].iloc[i]) <= float(r5["high"].iloc[i - 1])
        and float(r5["low"].iloc[i]) >= float(r5["low"].iloc[i - 1])
    )
    s = (1 if rr < 0.35 else 0) + (1 if rr < 0.25 else 0) + (1 if vr < 0.7 else 0) + (1 if inside_c >= 2 else 0)
    if s >= 2:
        result["compression"] = True
        result["compression_strength"] = min(s, 3)
        mid = (float(r5["high"].max()) + float(r5["low"].min())) / 2
        lc = float(r5["close"].iloc[-1])
        result["bias"] = "bullish" if lc > mid else ("bearish" if lc < mid else "neutral")
        result["detail"] = f"Compression: range={rr:.0%} vol={vr:.0%} inside={inside_c}"
    return result


def compute_mtf_bias(df_4h: pd.DataFrame) -> dict:
    """Compute multi-timeframe bias from 4h candle data.
    Used to confirm/reject signals from shorter timeframes.
    """
    result = {
        "bias": "neutral",
        "ema_trend": "flat",
        "rsi_zone": "neutral",
        "supertrend_dir": 0,
        "confirms_long": True,
        "confirms_short": True,
        "detail": "MTF: Insufficient data",
    }
    if len(df_4h) < 60:
        return result
    close = df_4h["close"]
    ema9_vals = ema(close, 9)
    ema21_vals = ema(close, 21)
    rsi_vals = rsi(close, 14)
    _, st_dir = supertrend(df_4h, period=10, multiplier=3.0)
    last = len(df_4h) - 1
    ema9_now = float(ema9_vals.iloc[last])
    ema21_now = float(ema21_vals.iloc[last])
    rsi_now = float(rsi_vals.iloc[last])
    close_now = float(close.iloc[last])
    st_dir_now = int(st_dir.iloc[last])
    if ema9_now > ema21_now and close_now > ema21_now:
        ema_trend = "up"
    elif ema9_now < ema21_now and close_now < ema21_now:
        ema_trend = "down"
    else:
        ema_trend = "flat"
    if rsi_now < 35:
        rsi_zone = "oversold"
    elif rsi_now > 65:
        rsi_zone = "overbought"
    else:
        rsi_zone = "neutral"
    bull_count = 0
    bear_count = 0
    if ema_trend == "up":
        bull_count += 1
    elif ema_trend == "down":
        bear_count += 1
    if st_dir_now == 1:
        bull_count += 1
    elif st_dir_now == -1:
        bear_count += 1
    if rsi_zone == "overbought":
        bear_count += 0.5
    elif rsi_zone == "oversold":
        bull_count += 0.5
    if bull_count >= 2:
        bias = "bullish"
    elif bear_count >= 2:
        bias = "bearish"
    else:
        bias = "neutral"
    confirms_long = bias != "bearish"
    confirms_short = bias != "bullish"
    detail = (
        f"4h: EMA={ema_trend}, RSI={rsi_now:.0f}({rsi_zone}), "
        f"ST={'up' if st_dir_now == 1 else 'down'} -> {bias.upper()}"
    )
    return {
        "bias": bias,
        "ema_trend": ema_trend,
        "rsi_zone": rsi_zone,
        "supertrend_dir": st_dir_now,
        "confirms_long": confirms_long,
        "confirms_short": confirms_short,
        "detail": detail,
    }
