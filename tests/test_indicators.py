"""Tests for technical and smart money indicators."""

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
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


# ── Technical indicators ────────────────────────────────────────────


def test_sma():
    from opencrypto.indicators.technical import sma

    s = pd.Series(range(20), dtype=float)
    result = sma(s, 5)
    assert result.iloc[4] == 2.0  # mean of 0,1,2,3,4
    assert pd.isna(result.iloc[3])


def test_ema():
    from opencrypto.indicators.technical import ema

    s = pd.Series([1.0] * 20)
    result = ema(s, 10)
    assert abs(result.iloc[-1] - 1.0) < 1e-10


def test_rsi_range():
    from opencrypto.indicators.technical import rsi

    series = pd.Series(np.random.default_rng(0).normal(100, 1, 200))
    vals = rsi(series, 14).dropna()
    assert vals.min() >= 0
    assert vals.max() <= 100


def test_rsi_extreme_up():
    from opencrypto.indicators.technical import rsi

    rising = pd.Series(np.arange(1, 51, dtype=float))
    vals = rsi(rising, 14).dropna()
    assert vals.iloc[-1] > 90


def test_macd():
    from opencrypto.indicators.technical import macd

    series = pd.Series(np.random.default_rng(1).normal(100, 1, 100))
    line, signal, hist = macd(series)
    assert len(line) == 100
    assert len(signal) == 100
    assert abs(hist.iloc[-1] - (line.iloc[-1] - signal.iloc[-1])) < 1e-10


def test_bollinger_bands():
    from opencrypto.indicators.technical import bollinger_bands

    series = pd.Series(np.random.default_rng(2).normal(100, 2, 100))
    upper, mid, lower, _pct_b, bw = bollinger_bands(series, 20, 2.0)
    valid_idx = 25
    assert upper.iloc[valid_idx] > mid.iloc[valid_idx] > lower.iloc[valid_idx]
    assert bw.iloc[valid_idx] > 0


def test_atr():
    from opencrypto.indicators.technical import atr

    df = _make_ohlcv(50)
    result = atr(df, 14)
    assert len(result) == 50
    assert result.dropna().iloc[-1] > 0


def test_stochastic_rsi():
    from opencrypto.indicators.technical import stochastic_rsi

    series = pd.Series(np.random.default_rng(3).normal(100, 1, 200))
    k, _d = stochastic_rsi(series)
    k_vals = k.dropna()
    assert k_vals.min() >= -1
    assert k_vals.max() <= 101


def test_adx():
    from opencrypto.indicators.technical import adx

    df = _make_ohlcv(100)
    adx_vals, _di_p, _di_m = adx(df, 14)
    assert len(adx_vals) == 100
    assert adx_vals.dropna().iloc[-1] >= 0


def test_obv():
    from opencrypto.indicators.technical import obv

    df = _make_ohlcv(50)
    result = obv(df)
    assert len(result) == 50


def test_vwap():
    from opencrypto.indicators.technical import vwap

    df = _make_ohlcv(50)
    result = vwap(df)
    assert result.iloc[-1] > 0


def test_ichimoku():
    from opencrypto.indicators.technical import ichimoku

    df = _make_ohlcv(100)
    ichi = ichimoku(df)
    assert "tenkan" in ichi
    assert "kijun" in ichi
    assert "senkou_a" in ichi
    assert "senkou_b" in ichi
    assert "chikou" in ichi


def test_supertrend():
    from opencrypto.indicators.technical import supertrend

    df = _make_ohlcv(100)
    _st_line, direction = supertrend(df, 10, 3.0)
    assert set(direction.unique()).issubset({-1, 1, 0})


def test_volume_profile():
    from opencrypto.indicators.technical import volume_profile

    df = _make_ohlcv(100)
    vp = volume_profile(df)
    assert "poc" in vp
    assert "vah" in vp
    assert "val" in vp
    assert vp["val"] <= vp["poc"] <= vp["vah"]


def test_dynamic_rsi_bands():
    from opencrypto.indicators.technical import dynamic_rsi_bands

    series = pd.Series(np.random.default_rng(4).normal(100, 1, 200))
    result = dynamic_rsi_bands(series)
    assert "rsi" in result
    assert "upper" in result
    assert "lower" in result
    assert result["lower"] < result["upper"]


def test_find_support_resistance():
    from opencrypto.indicators.technical import find_support_resistance

    df = _make_ohlcv(50)
    sr = find_support_resistance(df)
    assert sr["support"] < sr["resistance"]


def test_kelly_criterion():
    from opencrypto.indicators.technical import kelly_criterion

    assert kelly_criterion(0.6, 2.0, 1.0) > 0
    assert kelly_criterion(0.0, 2.0, 1.0) == 0.0
    assert kelly_criterion(0.5, 1.0, 0.0) == 0.0


def test_compute_all_indicators():
    from opencrypto.indicators.technical import compute_all_indicators

    df = _make_ohlcv()
    result = compute_all_indicators(df)
    expected_cols = [
        "rsi",
        "ema_9",
        "ema_21",
        "sma_20",
        "sma_50",
        "sma_200",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "adx",
        "stoch_rsi_k",
        "obv",
        "vwap",
        "supertrend_dir",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing: {col}"
    assert len(result) == len(df)


# ── Smart Money indicators ──────────────────────────────────────────


def test_detect_swing_points():
    from opencrypto.indicators.smart_money import detect_swing_points

    df = _make_ohlcv(100)
    result = detect_swing_points(df)
    assert "swing_highs" in result
    assert "swing_lows" in result
    assert result["structure"] in ("bullish", "bearish", "range", "unknown")


def test_detect_bos():
    from opencrypto.indicators.smart_money import detect_bos

    df = _make_ohlcv(100)
    result = detect_bos(df)
    assert "bullish_bos" in result
    assert "bearish_bos" in result
    assert isinstance(result["bullish_bos"], bool)


def test_detect_fvg():
    from opencrypto.indicators.smart_money import detect_fvg

    df = _make_ohlcv(100)
    gaps = detect_fvg(df)
    assert isinstance(gaps, list)
    for gap in gaps:
        assert gap["type"] in ("bullish", "bearish")
        assert "top" in gap and "bottom" in gap


def test_detect_order_blocks():
    from opencrypto.indicators.smart_money import detect_order_blocks

    df = _make_ohlcv(100)
    blocks = detect_order_blocks(df)
    assert isinstance(blocks, list)
    for b in blocks:
        assert b["type"] in ("bullish", "bearish")
        assert 0 < b["strength"] <= 1.0


def test_detect_liquidity_sweep():
    from opencrypto.indicators.smart_money import detect_liquidity_sweep

    df = _make_ohlcv(60)
    result = detect_liquidity_sweep(df)
    assert "bullish_sweep" in result
    assert "bearish_sweep" in result
    assert 0 <= result["sweep_strength"] <= 3


def test_detect_wyckoff_phase():
    from opencrypto.indicators.smart_money import detect_wyckoff_phase

    df = _make_ohlcv(100)
    result = detect_wyckoff_phase(df)
    assert result["phase"] in ("accumulation", "distribution", "markup", "markdown", "spring", "unknown")


def test_detect_rsi_divergence():
    from opencrypto.indicators.smart_money import detect_rsi_divergence
    from opencrypto.indicators.technical import compute_all_indicators

    df = _make_ohlcv(200)
    df = compute_all_indicators(df)
    result = detect_rsi_divergence(df)
    assert "bullish" in result
    assert "bearish" in result
    assert isinstance(result["bullish"], bool)


def test_detect_qml():
    from opencrypto.indicators.smart_money import detect_qml

    df = _make_ohlcv(100)
    result = detect_qml(df)
    assert "bullish_qml" in result
    assert "bearish_qml" in result


def test_detect_fakeout():
    from opencrypto.indicators.smart_money import detect_fakeout

    df = _make_ohlcv(60)
    result = detect_fakeout(df)
    assert "bullish_fakeout" in result
    assert "bearish_fakeout" in result
    assert "fakeout_type" in result


def test_detect_sr_flip():
    from opencrypto.indicators.smart_money import detect_sr_flip

    df = _make_ohlcv(100)
    result = detect_sr_flip(df)
    assert "sr_flip" in result
    assert "rs_flip" in result


def test_detect_compression():
    from opencrypto.indicators.smart_money import detect_compression

    df = _make_ohlcv(60)
    result = detect_compression(df)
    assert "compression" in result
    assert "bias" in result
    assert result["bias"] in ("bullish", "bearish", "neutral")


def test_compute_mtf_bias_insufficient():
    from opencrypto.indicators.smart_money import compute_mtf_bias

    df = _make_ohlcv(30)
    result = compute_mtf_bias(df)
    assert result["bias"] == "neutral"


def test_compute_mtf_bias_full():
    from opencrypto.indicators.smart_money import compute_mtf_bias

    df = _make_ohlcv(100)
    result = compute_mtf_bias(df)
    assert result["bias"] in ("bullish", "bearish", "neutral")
    assert isinstance(result["confirms_long"], bool)
    assert isinstance(result["confirms_short"], bool)
