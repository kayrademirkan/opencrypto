"""Core module tests — imports, exceptions, strategy protocol, config."""


# ── Import smoke tests ──────────────────────────────────────────────


def test_top_level_import():
    import opencrypto

    assert hasattr(opencrypto, "__version__")
    assert hasattr(opencrypto, "BaseStrategy")
    assert hasattr(opencrypto, "ShieldGuard")
    assert hasattr(opencrypto, "DataBridge")
    assert hasattr(opencrypto, "PositionManager")


def test_exception_hierarchy():
    from opencrypto.core.exceptions import (
        BacktestError,
        DataFetchError,
        ManipulationDetectedError,
        OpenCryptoError,
        StrategyImplementationError,
    )

    assert issubclass(DataFetchError, OpenCryptoError)
    assert issubclass(ManipulationDetectedError, OpenCryptoError)
    assert issubclass(StrategyImplementationError, OpenCryptoError)
    assert issubclass(BacktestError, OpenCryptoError)

    err = DataFetchError("test", symbol="BTCUSDT", source="binance")
    assert err.symbol == "BTCUSDT"
    assert err.source == "binance"

    merr = ManipulationDetectedError("blocked", risk_score=85, warnings=["spike"])
    assert merr.risk_score == 85
    assert merr.warnings == ["spike"]


def test_indicators_import():
    from opencrypto.indicators import (
        detect_order_blocks,
        sma,
    )

    assert callable(sma)
    assert callable(detect_order_blocks)


# ── StrategySignal ──────────────────────────────────────────────────


def test_strategy_signal_properties():
    from opencrypto import StrategySignal

    sig = StrategySignal(
        symbol="BTCUSDT",
        direction="LONG",
        confidence=75.0,
        entry=100.0,
        sl=95.0,
        tp=110.0,
    )
    assert sig.sl_pct == 5.0
    assert sig.tp_pct == 10.0
    assert sig.rr_ratio == 2.0
    assert "BTC/USDT" in sig.display_symbol

    d = sig.to_dict()
    assert d["symbol"] == "BTCUSDT"
    assert d["direction"] == "LONG"
    assert d["tp1"] == 110.0
    assert d["leverage"] == 3


def test_strategy_signal_edge_cases():
    from opencrypto import StrategySignal

    sig = StrategySignal(symbol="ETHUSDT", direction="SHORT", confidence=50.0, entry=0.0, sl=0.0, tp=0.0)
    assert sig.sl_pct == 0.0
    assert sig.tp_pct == 0.0
    assert sig.rr_ratio == 0.0


def test_strategy_signal_metadata():
    from opencrypto import StrategySignal

    sig = StrategySignal(
        symbol="SOLUSDT",
        direction="LONG",
        confidence=80.0,
        entry=150.0,
        sl=140.0,
        tp=170.0,
        metadata={"timeframe": "1h", "custom_score": 42},
    )
    assert sig.metadata["timeframe"] == "1h"
    d = sig.to_dict()
    assert d["metadata"]["custom_score"] == 42


# ── BaseStrategy protocol ──────────────────────────────────────────


def test_base_strategy_protocol():
    from opencrypto import BaseStrategy

    class DummyStrategy:
        name = "Test"
        version = "0.1"

        def generate_signal(self, symbol, df, context=None):
            return None

    assert isinstance(DummyStrategy(), BaseStrategy)


def test_non_strategy_fails_protocol():
    from opencrypto import BaseStrategy

    class NotAStrategy:
        pass

    assert not isinstance(NotAStrategy(), BaseStrategy)


# ── Config ──────────────────────────────────────────────────────────


def test_config_defaults():
    from opencrypto.core.config import BINANCE_FUTURES_URL, BINANCE_SPOT_URL, DATA_DIR

    assert "fapi.binance.com" in BINANCE_FUTURES_URL
    assert "api.binance.com" in BINANCE_SPOT_URL
    assert DATA_DIR.name == "data"
