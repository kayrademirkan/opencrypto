"""
Bollinger Band Mean Reversion Bot — OpenCrypto Example

A multi-indicator strategy that combines Bollinger Bands, RSI, and volume
to detect oversold/overbought conditions for mean reversion trades.

This demonstrates:
- Using multiple indicators together
- Bi-directional trading (LONG + SHORT)
- Dynamic confidence scoring based on confluence
- Using the `context` parameter for ShieldGuard integration

Usage:
    python examples/bb_reversal_bot.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from opencrypto import StrategySignal

logger = logging.getLogger(__name__)


class BBReversalStrategy:
    """Bollinger Band mean reversion with RSI + volume confirmation.

    Entry criteria (LONG):
        1. Price touches or pierces the lower Bollinger Band
        2. RSI is below the dynamic oversold threshold
        3. Volume is above 20-period average (institutional interest)

    Entry criteria (SHORT):
        1. Price touches or pierces the upper Bollinger Band
        2. RSI is above the dynamic overbought threshold
        3. Volume is above 20-period average

    Each confirmed condition adds to the confidence score.
    """

    name = "BB_MeanReversion"
    version = "1.0"

    def __init__(self, bb_touch_tolerance: float = 0.002, min_confidence: float = 55.0):
        self.bb_touch_tolerance = bb_touch_tolerance
        self.min_confidence = min_confidence

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        context: dict | None = None,
    ) -> StrategySignal | None:
        if len(df) < 60:
            return None

        close = float(df["close"].iloc[-1])
        bb_upper = float(df["bb_upper"].iloc[-1])
        bb_lower = float(df["bb_lower"].iloc[-1])
        rsi_val = float(df["rsi"].iloc[-1])
        atr_val = float(df["atr_14"].iloc[-1])
        vol = float(df["volume"].iloc[-1])
        vol_avg = float(df["vol_sma_20"].iloc[-1]) if "vol_sma_20" in df.columns else vol

        dyn_upper = float(df["dyn_rsi_upper"].iloc[-1]) if "dyn_rsi_upper" in df.columns else 70
        dyn_lower = float(df["dyn_rsi_lower"].iloc[-1]) if "dyn_rsi_lower" in df.columns else 30

        stoch_k = float(df["stoch_rsi_k"].iloc[-1]) if "stoch_rsi_k" in df.columns else 50

        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return None

        long_reasons = []
        short_reasons = []
        long_score = 0.0
        short_score = 0.0

        # --- LONG checks ---
        if close <= bb_lower * (1 + self.bb_touch_tolerance):
            long_score += 20
            long_reasons.append(f"Price at lower BB: {close:.2f} <= {bb_lower:.2f}")

        if rsi_val < dyn_lower:
            long_score += 20
            long_reasons.append(f"RSI oversold: {rsi_val:.0f} < {dyn_lower:.0f}")
        elif rsi_val < 40:
            long_score += 10
            long_reasons.append(f"RSI low zone: {rsi_val:.0f}")

        if stoch_k < 20:
            long_score += 10
            long_reasons.append(f"StochRSI oversold: {stoch_k:.0f}")

        if vol > vol_avg * 1.2:
            long_score += 10
            long_reasons.append(f"Volume confirmation: {vol / vol_avg:.1f}x avg")

        # --- SHORT checks ---
        if close >= bb_upper * (1 - self.bb_touch_tolerance):
            short_score += 20
            short_reasons.append(f"Price at upper BB: {close:.2f} >= {bb_upper:.2f}")

        if rsi_val > dyn_upper:
            short_score += 20
            short_reasons.append(f"RSI overbought: {rsi_val:.0f} > {dyn_upper:.0f}")
        elif rsi_val > 60:
            short_score += 10
            short_reasons.append(f"RSI high zone: {rsi_val:.0f}")

        if stoch_k > 80:
            short_score += 10
            short_reasons.append(f"StochRSI overbought: {stoch_k:.0f}")

        if vol > vol_avg * 1.2:
            short_score += 10
            short_reasons.append(f"Volume confirmation: {vol / vol_avg:.1f}x avg")

        # --- Decision ---
        base_confidence = 45.0

        if long_score >= 30 and long_score > short_score:
            confidence = base_confidence + long_score
            if confidence < self.min_confidence:
                return None
            return StrategySignal(
                symbol=symbol,
                direction="LONG",
                confidence=min(confidence, 95.0),
                entry=close,
                sl=round(close - atr_val * 2.5, 6),
                tp=round(close + atr_val * 3.0, 6),
                leverage=3,
                signal_type="bb_oversold_reversal",
                reasons=long_reasons,
                score=long_score,
                indicator_count=len(long_reasons),
            )

        if short_score >= 30 and short_score > long_score:
            confidence = base_confidence + short_score
            if confidence < self.min_confidence:
                return None
            return StrategySignal(
                symbol=symbol,
                direction="SHORT",
                confidence=min(confidence, 95.0),
                entry=close,
                sl=round(close + atr_val * 2.5, 6),
                tp=round(close - atr_val * 3.0, 6),
                leverage=3,
                signal_type="bb_overbought_reversal",
                reasons=short_reasons,
                score=short_score,
                indicator_count=len(short_reasons),
            )

        return None


async def main():
    from opencrypto.backtest import run_backtest

    strategy = BBReversalStrategy()
    logger.info("Running backtest for: %s v%s", strategy.name, strategy.version)

    report = await run_backtest(
        strategy=strategy,
        symbols=[
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "AVAXUSDT",
            "DOTUSDT",
            "LINKUSDT",
        ],
        days=60,
        step=4,
        max_hold=48,
        initial_capital=1000.0,
        risk_per_trade=0.02,
    )

    if report.get("stats"):
        s = report["stats"]
        logger.info(
            "Final — WR: %s%% | PF: %s | R: %sR | Return: %s%%",
            s["win_rate"],
            s["profit_factor"],
            s["total_r"],
            s["total_return"],
        )


if __name__ == "__main__":
    asyncio.run(main())
