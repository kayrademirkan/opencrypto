"""
OpenCrypto — Chart Plugin (Optional)

Generate candlestick charts with signal annotations.
Requires matplotlib and mplfinance.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from opencrypto.core.config import DATA_DIR

logger = logging.getLogger(__name__)

CHARTS_DIR = str(DATA_DIR / "charts")


def generate_chart(
    df,
    signal: dict,
    trade_id: int | None = None,
) -> str | None:
    """Generate a candlestick chart PNG. Returns file path or None on failure."""
    try:
        import mplfinance as mpf
        import pandas as pd

        os.makedirs(CHARTS_DIR, exist_ok=True)

        chart_df = df.tail(60).copy()
        if "timestamp" in chart_df.columns:
            chart_df.index = pd.DatetimeIndex(chart_df["timestamp"])
        chart_df = chart_df[["open", "high", "low", "close", "volume"]].copy()
        chart_df.columns = ["Open", "High", "Low", "Close", "Volume"]

        sym = signal.get("symbol", "UNKNOWN")
        direction = signal.get("direction", "?")

        fname = f"chart_{sym}_{trade_id or 'x'}_{datetime.now(UTC).strftime('%H%M%S')}.png"
        fpath = os.path.join(CHARTS_DIR, fname)

        entry = signal.get("entry", 0)
        hlines = [entry] if entry > 0 else []
        sl = signal.get("sl", 0)
        tp = signal.get("tp1", signal.get("tp", 0))
        if sl > 0:
            hlines.append(sl)
        if tp > 0:
            hlines.append(tp)

        colors = []
        for h in hlines:
            if h == entry:
                colors.append("blue")
            elif h == sl:
                colors.append("red")
            else:
                colors.append("green")

        kwargs = {}
        if hlines:
            kwargs["hlines"] = dict(hlines=hlines, colors=colors, linestyle="--", linewidths=0.8)

        mpf.plot(
            chart_df,
            type="candle",
            volume=True,
            style="charles",
            title=f"{sym} {direction}",
            savefig=fpath,
            figsize=(12, 7),
            **kwargs,
        )
        return fpath

    except Exception as exc:
        logger.error("Chart generation failed: %s", exc)
        return None
