"""
Simple Moving Average Crossover Bot — OpenCrypto Example

A minimal strategy that demonstrates the OpenCrypto framework.
Buys when EMA9 crosses above EMA21, sells when it crosses below.

Usage:
    # Run as a backtest
    python examples/simple_ma_bot.py

    # Or import and use with the engine
    from examples.simple_ma_bot import SimpleMAStrategy
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from opencrypto.core.base_strategy import StrategySignal

logger = logging.getLogger(__name__)


class SimpleMAStrategy:
    """EMA 9/21 crossover strategy — the simplest possible example.

    This strategy:
    - Goes LONG when EMA9 crosses above EMA21
    - Goes SHORT when EMA9 crosses below EMA21
    - Uses ATR for stop-loss and take-profit calculation
    - Requires ADX > 20 for trend confirmation
    """

    name = "SimpleMA_Crossover"
    version = "1.0"

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        context: dict | None = None,
    ) -> StrategySignal | None:
        if len(df) < 50:
            return None

        idx = len(df) - 1
        close = float(df["close"].iloc[idx])
        ema9 = float(df["ema_9"].iloc[idx])
        ema21 = float(df["ema_21"].iloc[idx])
        atr_val = float(df["atr_14"].iloc[idx]) if "atr_14" in df.columns else close * 0.02
        adx_val = float(df["adx"].iloc[idx]) if "adx" in df.columns else 25

        prev_ema9 = float(df["ema_9"].iloc[idx - 1]) if idx > 0 else ema9
        prev_ema21 = float(df["ema_21"].iloc[idx - 1]) if idx > 0 else ema21

        if adx_val < 20:
            return None

        # Golden cross: EMA9 crosses above EMA21
        if prev_ema9 <= prev_ema21 and ema9 > ema21:
            sl_price = close - atr_val * 2.5
            tp_price = close + atr_val * 3.75
            return StrategySignal(
                symbol=symbol,
                direction="LONG",
                confidence=60 + min(adx_val - 20, 15),
                entry=close,
                sl=round(sl_price, 6),
                tp=round(tp_price, 6),
                leverage=3,
                signal_type="ema_golden_cross",
                reasons=[
                    "EMA9 crossed above EMA21",
                    f"ADX: {adx_val:.0f} (trend strength)",
                    f"ATR: {atr_val:.4f}",
                ],
                score=adx_val,
                indicator_count=3,
            )

        # Death cross: EMA9 crosses below EMA21
        if prev_ema9 >= prev_ema21 and ema9 < ema21:
            sl_price = close + atr_val * 2.5
            tp_price = close - atr_val * 3.75
            return StrategySignal(
                symbol=symbol,
                direction="SHORT",
                confidence=60 + min(adx_val - 20, 15),
                entry=close,
                sl=round(sl_price, 6),
                tp=round(tp_price, 6),
                leverage=3,
                signal_type="ema_death_cross",
                reasons=[
                    "EMA9 crossed below EMA21",
                    f"ADX: {adx_val:.0f} (trend strength)",
                    f"ATR: {atr_val:.4f}",
                ],
                score=adx_val,
                indicator_count=3,
            )

        return None


async def main():
    from opencrypto.backtest import run_backtest

    strategy = SimpleMAStrategy()
    logger.info("Running backtest for: %s v%s", strategy.name, strategy.version)

    report = await run_backtest(
        strategy=strategy,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        days=30,
        step=6,
        max_hold=72,
        initial_capital=1000.0,
    )

    if report.get("stats"):
        s = report["stats"]
        logger.info(
            "Final — WR: %s%% | R: %sR | Return: %s%%",
            s["win_rate"],
            s["total_r"],
            s["total_return"],
        )


if __name__ == "__main__":
    asyncio.run(main())
