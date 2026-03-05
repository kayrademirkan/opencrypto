"""
OpenCrypto — Technical Analysis Indicators

Standard technical indicators for algorithmic trading.
All functions accept pandas Series/DataFrames and return computed values.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average over *period* bars."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average over *period* bars."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands: upper, middle, lower, %B, bandwidth."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-10)
    bandwidth = ((upper - lower) / (middle + 1e-10)) * 100
    return upper, middle, lower, pct_b, bandwidth


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def stochastic_rsi(
    series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI (%K and %D lines)."""
    rsi_vals = rsi(series, rsi_period)
    rsi_min = rsi_vals.rolling(window=stoch_period).min()
    rsi_max = rsi_vals.rolling(window=stoch_period).max()
    stoch_rsi = (rsi_vals - rsi_min) / (rsi_max - rsi_min + 1e-10)
    k = stoch_rsi.rolling(window=k_smooth).mean() * 100
    d = k.rolling(window=d_smooth).mean()
    return k, d


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index with DI+ and DI-."""
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_vals = atr(df, period)
    di_plus = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr_vals + 1e-10))
    di_minus = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr_vals + 1e-10))
    dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10))
    adx_vals = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx_vals, di_plus, di_minus


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.where(df["close"] > df["close"].shift(1), 1, np.where(df["close"] < df["close"].shift(1), -1, 0))
    return (df["volume"] * direction).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (session-cumulative)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (tp * df["volume"]).cumsum()
    return cum_tp_vol / (cum_vol + 1e-10)


def ichimoku(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Ichimoku Cloud components: tenkan, kijun, senkou A/B, chikou."""
    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou}


def volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """Volume Profile: Point of Control, Value Area High/Low."""
    if len(df) < 20:
        mid = float(df["close"].iloc[-1]) if len(df) > 0 else 0
        return {"poc": mid, "vah": mid * 1.01, "val": mid * 0.99, "profile": []}
    price_range = np.linspace(float(df["low"].min()), float(df["high"].max()), bins + 1)
    vol_at_price = np.zeros(bins)
    for i in range(bins):
        mask = (df["close"] >= price_range[i]) & (df["close"] < price_range[i + 1])
        vol_at_price[i] = float(df.loc[mask, "volume"].sum())
    poc_idx = int(np.argmax(vol_at_price))
    poc = (price_range[poc_idx] + price_range[poc_idx + 1]) / 2
    total_vol = vol_at_price.sum()
    if total_vol == 0:
        return {
            "poc": round(poc, 6),
            "vah": round(poc * 1.01, 6),
            "val": round(poc * 0.99, 6),
            "profile": vol_at_price.tolist(),
        }
    sorted_idx = np.argsort(vol_at_price)[::-1]
    cum = 0.0
    va_indices = []
    for idx in sorted_idx:
        cum += vol_at_price[idx]
        va_indices.append(idx)
        if cum / total_vol >= 0.70:
            break
    va_lo = min(va_indices)
    va_hi = max(va_indices)
    val_price = (price_range[va_lo] + price_range[va_lo + 1]) / 2
    vah_price = (price_range[va_hi] + price_range[va_hi + 1]) / 2
    return {
        "poc": round(poc, 6),
        "vah": round(vah_price, 6),
        "val": round(val_price, 6),
        "profile": vol_at_price.tolist(),
    }


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator. Returns (supertrend_line, direction).
    direction: 1 = uptrend (bullish), -1 = downtrend (bearish)
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr_vals = atr(df, period)
    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals
    st_line = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            pass
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            pass
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]
        if st_line.iloc[i - 1] == upper_band.iloc[i - 1]:
            if df["close"].iloc[i] > upper_band.iloc[i]:
                st_line.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                st_line.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        else:
            if df["close"].iloc[i] < lower_band.iloc[i]:
                st_line.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                st_line.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
    return st_line, direction


def dynamic_rsi_bands(series: pd.Series, rsi_period: int = 14, bb_period: int = 20, bb_std: float = 2.0) -> dict:
    """Dynamic oversold/overbought thresholds using Bollinger Bands on RSI."""
    rsi_vals = rsi(series, rsi_period)
    rsi_sma = rsi_vals.rolling(window=bb_period).mean()
    rsi_std = rsi_vals.rolling(window=bb_period).std()
    upper = rsi_sma + bb_std * rsi_std
    lower = rsi_sma - bb_std * rsi_std
    last_idx = len(series) - 1
    if last_idx < bb_period + rsi_period:
        return {
            "rsi": 50.0,
            "upper": 70.0,
            "lower": 30.0,
            "rsi_sma": 50.0,
            "is_oversold": False,
            "is_overbought": False,
        }
    rsi_now = float(rsi_vals.iloc[last_idx])
    upper_now = float(np.clip(upper.iloc[last_idx], 55, 85))
    lower_now = float(np.clip(lower.iloc[last_idx], 15, 45))
    rsi_sma_now = float(rsi_sma.iloc[last_idx])
    return {
        "rsi": round(rsi_now, 1),
        "upper": round(upper_now, 1),
        "lower": round(lower_now, 1),
        "rsi_sma": round(rsi_sma_now, 1),
        "is_oversold": rsi_now < lower_now,
        "is_overbought": rsi_now > upper_now,
    }


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Find nearest support and resistance from recent highs/lows."""
    if len(df) < window:
        close = float(df["close"].iloc[-1]) if len(df) > 0 else 0
        return {"support": close * 0.98, "resistance": close * 1.02}
    recent = df.tail(window)
    return {
        "support": round(float(recent["low"].min()), 6),
        "resistance": round(float(recent["high"].max()), 6),
    }


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly Criterion optimal position size fraction (clamped 0-1)."""
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    f = (win_rate * b - q) / b
    return max(0.0, min(f, 1.0))


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard technical indicators on an OHLCV DataFrame.
    Adds indicator columns in-place and returns the DataFrame.
    """
    close = df["close"]
    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)
    df["sma_200"] = sma(close, 200)
    df["ema_9"] = ema(close, 9)
    df["ema_21"] = ema(close, 21)
    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)
    df["rsi"] = rsi(close, 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(close)
    df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_pct_b"], df["bb_bandwidth"] = bollinger_bands(close)
    df["atr_14"] = atr(df, 14)
    df["vol_sma_20"] = sma(df["volume"], 20)
    df["stoch_rsi_k"], df["stoch_rsi_d"] = stochastic_rsi(close)
    adx_vals, di_p, di_m = adx(df)
    df["adx"] = adx_vals
    df["di_plus"] = di_p
    df["di_minus"] = di_m
    df["obv"] = obv(df)
    df["vwap"] = vwap(df)
    ichi = ichimoku(df)
    df["ichi_tenkan"] = ichi["tenkan"]
    df["ichi_kijun"] = ichi["kijun"]
    df["ichi_senkou_a"] = ichi["senkou_a"]
    df["ichi_senkou_b"] = ichi["senkou_b"]
    st_line, st_dir = supertrend(df, period=10, multiplier=3.0)
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    dyn_rsi = dynamic_rsi_bands(close)
    df["dyn_rsi_upper"] = dyn_rsi["upper"]
    df["dyn_rsi_lower"] = dyn_rsi["lower"]
    return df
