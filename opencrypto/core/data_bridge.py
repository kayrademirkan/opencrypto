"""
OpenCrypto — DataBridge

Unified async market data fetcher. Supports Binance Futures + Spot
with automatic failover. Includes sentiment data aggregation.
"""

from __future__ import annotations

import logging
import re

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


FAPI_URL = "https://fapi.binance.com"
SPOT_URL = "https://api.binance.com"

_BLACKLIST_PATTERNS = re.compile(
    r"(BUSDUSDT|TUSDUSDT|USDCUSDT|DAIUSDT|FDUSDUSDT|USDPUSDT|EURUSDT|"
    r"GBPUSDT|AUDUSDT|JPYUSDT|XAGUSDT|XAUUSDT|"
    r"UPUSDT|DOWNUSDT|BULLUSDT|BEARUSDT|2LUSDT|2SUSDT|3LUSDT|3SUSDT|5LUSDT|5SUSDT)"
)
_JUNK_COINS: set[str] = set()
_VALID_SYMBOL = re.compile(r"^[A-Z0-9]+USDT$")
_MIN_QUOTE_VOLUME = 10_000_000

_shared_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=15,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            http2=False,
        )
    return _shared_client


def set_junk_coins(coins: set[str]):
    """Configure coins to exclude from scanning."""
    global _JUNK_COINS
    _JUNK_COINS = coins


def set_min_volume(volume: float):
    """Set minimum 24h quote volume for coin filtering."""
    global _MIN_QUOTE_VOLUME
    _MIN_QUOTE_VOLUME = volume


class DataBridge:
    """Async market data provider with connection pooling and failover."""

    async def fetch_top_coins(self, limit: int = 50) -> list[str]:
        """Fetch top coins by 24h USDT volume from Binance Futures."""
        client = await _get_client()
        try:
            resp = await client.get(f"{FAPI_URL}/fapi/v1/ticker/24hr")
            if resp.status_code != 200:
                resp = await client.get(f"{SPOT_URL}/api/v3/ticker/24hr")
            resp.raise_for_status()
            tickers = resp.json()
        except Exception as exc:
            logger.warning("Failed to fetch top coins, using fallback list: %s", exc)
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

        filtered = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not _VALID_SYMBOL.match(sym):
                continue
            if _BLACKLIST_PATTERNS.search(sym):
                continue
            if sym in _JUNK_COINS:
                continue
            quote_vol = float(t.get("quoteVolume", 0))
            if quote_vol < _MIN_QUOTE_VOLUME:
                continue
            filtered.append((sym, quote_vol))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in filtered[:limit]]

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 720,
    ) -> pd.DataFrame:
        """Fetch OHLCV klines from Binance (Futures -> Spot failover)."""
        client = await _get_client()
        clean = symbol.replace(".P", "").upper()
        params = {"symbol": clean, "interval": interval, "limit": limit}

        for url in [f"{FAPI_URL}/fapi/v1/klines", f"{SPOT_URL}/api/v3/klines"]:
            try:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if not data:
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
                return (
                    df[
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
                    ]
                    .copy()
                    .reset_index(drop=True)
                )
            except Exception as exc:
                logger.debug("Kline fetch failed for %s from %s: %s", symbol, url, exc)
                continue
        logger.warning("All kline endpoints failed for %s", symbol)
        return pd.DataFrame()

    async def fetch_klines_4h(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch 4-hour klines for multi-timeframe analysis."""
        return await self.fetch_klines(symbol, "4h", limit)

    async def get_current_price(self, symbol: str) -> float | None:
        """Get latest price for a symbol (Futures -> Spot failover)."""
        client = await _get_client()
        clean = symbol.replace(".P", "").upper()
        for base_url, path in [
            (FAPI_URL, "/fapi/v1/ticker/price"),
            (SPOT_URL, "/api/v3/ticker/price"),
        ]:
            try:
                resp = await client.get(f"{base_url}{path}", params={"symbol": clean})
                if resp.status_code == 200:
                    return float(resp.json()["price"])
            except Exception as exc:
                logger.debug("Price fetch failed for %s from %s: %s", symbol, base_url, exc)
        return None

    async def get_orderbook_depth(self, symbol: str, depth: int = 20) -> dict:
        """Fetch orderbook and compute bid/ask imbalance (Futures -> Spot failover)."""
        client = await _get_client()
        clean = symbol.replace(".P", "").upper()
        for base_url, path in [
            (FAPI_URL, "/fapi/v1/depth"),
            (SPOT_URL, "/api/v3/depth"),
        ]:
            try:
                resp = await client.get(f"{base_url}{path}", params={"symbol": clean, "limit": depth})
                if resp.status_code != 200:
                    continue
                data = resp.json()
                bid_vol = sum(float(b[1]) for b in data.get("bids", []))
                ask_vol = sum(float(a[1]) for a in data.get("asks", []))
                total = bid_vol + ask_vol
                imbalance = (bid_vol - ask_vol) / total if total > 0 else 0
                return {
                    "bid_volume": round(bid_vol, 2),
                    "ask_volume": round(ask_vol, 2),
                    "imbalance": round(imbalance, 4),
                }
            except Exception as exc:
                logger.debug("Orderbook fetch failed for %s from %s: %s", symbol, base_url, exc)
        return {"imbalance": 0}

    async def get_24h_stats(self, symbol: str) -> dict:
        """Get 24h ticker stats."""
        client = await _get_client()
        clean = symbol.replace(".P", "").upper()
        try:
            resp = await client.get(
                f"{FAPI_URL}/fapi/v1/ticker/24hr",
                params={"symbol": clean},
            )
            if resp.status_code == 200:
                d = resp.json()
                return {
                    "price_change_pct": float(d.get("priceChangePercent", 0)),
                    "volume": float(d.get("volume", 0)),
                    "quote_volume": float(d.get("quoteVolume", 0)),
                    "high": float(d.get("highPrice", 0)),
                    "low": float(d.get("lowPrice", 0)),
                }
        except Exception as exc:
            logger.debug("24h stats fetch failed for %s: %s", symbol, exc)
        return {}
