"""
BaseStrategy — The contract every trading strategy must fulfill.

Developers implement `generate_signal()` with their own logic.
The framework handles everything else: data fetching, risk management,
position tracking, notifications, and backtesting.

Example:
    from opencrypto import BaseStrategy, StrategySignal
    from opencrypto.indicators import rsi, ema

    class MyStrategy(BaseStrategy):
        name = "SimpleRSI"
        version = "1.0"

        def generate_signal(self, symbol, df, context=None):
            if df["rsi"].iloc[-1] < 30:
                return StrategySignal(
                    symbol=symbol,
                    direction="LONG",
                    confidence=65.0,
                    entry=float(df["close"].iloc[-1]),
                    sl=float(df["close"].iloc[-1] * 0.97),
                    tp=float(df["close"].iloc[-1] * 1.045),
                    reasons=["RSI oversold"],
                )
            return None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class StrategySignal:
    """A trading signal produced by a strategy.

    This is the universal contract between your strategy and the framework.
    The framework uses this to manage positions, send notifications,
    run backtests, and track performance.
    """

    symbol: str
    direction: str  # "LONG" or "SHORT"
    confidence: float  # 0-100, higher = more confident
    entry: float  # Entry price
    sl: float  # Stop-loss price
    tp: float  # Take-profit price

    leverage: int = 3
    signal_type: str = "custom"  # Free-form label for your signal category
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0  # Internal scoring metric
    indicator_count: int = 0  # How many indicators confirmed

    # Optional metadata your strategy can attach
    metadata: dict = field(default_factory=dict)

    @property
    def sl_pct(self) -> float:
        if self.entry <= 0:
            return 0.0
        return round(abs(self.entry - self.sl) / self.entry * 100, 2)

    @property
    def tp_pct(self) -> float:
        if self.entry <= 0:
            return 0.0
        return round(abs(self.tp - self.entry) / self.entry * 100, 2)

    @property
    def rr_ratio(self) -> float:
        sl_d = abs(self.entry - self.sl)
        if sl_d <= 0:
            return 0.0
        return round(abs(self.tp - self.entry) / sl_d, 2)

    @property
    def display_symbol(self) -> str:
        sym = self.symbol
        if "USDT" in sym:
            idx = sym.rfind("USDT")
            base = sym[:idx]
            sym = f"{base}/USDT" if base else sym
        if not sym.endswith(".P"):
            sym += ".P"
        elif sym.endswith(".P.P"):
            sym = sym[:-2]
        return sym

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage and Telegram messages."""
        return {
            "symbol": self.symbol,
            "display_symbol": self.display_symbol,
            "direction": self.direction,
            "signal_type": self.signal_type,
            "entry": self.entry,
            "sl": round(self.sl, 6),
            "sl_pct": self.sl_pct,
            "tp": round(self.tp, 6),
            "tp1": round(self.tp, 6),
            "tp_pct": self.tp_pct,
            "tp1_pct": self.tp_pct,
            "leverage": self.leverage,
            "confidence": round(self.confidence, 1),
            "rr_ratio": self.rr_ratio,
            "score": round(self.score, 1),
            "indicator_count": self.indicator_count,
            "reasons": self.reasons,
            "signal_type_label": self.signal_type,
            "metadata": self.metadata,
        }


@runtime_checkable
class BaseStrategy(Protocol):
    """Protocol that every strategy must implement.

    Attributes:
        name: Human-readable strategy name (e.g. "MeanReversion_v2")
        version: Strategy version string (e.g. "1.0")
    """

    name: str
    version: str

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        context: dict | None = None,
    ) -> StrategySignal | None:
        """Analyze a symbol's OHLCV data and optionally produce a signal.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            df: OHLCV DataFrame with indicators already computed.
                Guaranteed columns: open, high, low, close, volume.
                Framework adds: sma_20, sma_50, sma_200, ema_9, ema_21,
                rsi, macd_line, macd_signal, macd_hist, bb_upper, bb_lower,
                atr_14, adx, stoch_rsi_k, obv, vwap, supertrend_dir, etc.
            context: Optional dict with extra info:
                - "sentiment_score": int (-100 to 100)
                - "orderbook": dict with "imbalance" key
                - "mtf_data": dict from compute_mtf_bias()
                - "btc_gate": dict with BTC trend info
                - "shield_guard": ShieldGuard instance

        Returns:
            StrategySignal if a trade should be opened, None otherwise.
        """
        ...
