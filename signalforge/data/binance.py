"""Binance spot market data.

Uses the public `data-api.binance.vision` mirror, which serves klines without
an API key and without the geographic restrictions of the main endpoint.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from signalforge.data.base import OHLCV_COLUMNS, get_timeframe, validate_ohlcv

log = logging.getLogger(__name__)

BASE_URL = "https://data-api.binance.vision"
MAX_LIMIT = 1000  # Binance caps a single klines response at 1000 bars


class BinanceProvider:
    name = "binance"

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: dict) -> list | dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    f"{BASE_URL}{path}", params=params, timeout=self.timeout
                )
                if resp.status_code == 429:
                    # Respect the documented back-off rather than hammering.
                    wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                    log.warning("Binance rate limited, sleeping %ss", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Binance request failed: {last_error}") from last_error

    def fetch(self, provider_symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fetch up to `bars` klines, paginating backwards as needed."""
        tf = get_timeframe(timeframe)
        interval_ms = tf.minutes * 60_000
        frames: list[pd.DataFrame] = []
        remaining = bars
        end_time: int | None = None

        while remaining > 0:
            limit = min(MAX_LIMIT, remaining)
            params: dict[str, object] = {
                "symbol": provider_symbol,
                "interval": tf.binance,
                "limit": limit,
            }
            if end_time is not None:
                params["endTime"] = end_time

            raw = self._get("/api/v3/klines", params)
            if not raw:
                break

            chunk = self._to_frame(raw, interval_ms)
            frames.append(chunk)
            remaining -= len(chunk)

            if len(raw) < limit:
                break  # provider has no more history
            # Step back one interval before the earliest bar we just received.
            end_time = int(raw[0][0]) - 1

        if not frames:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return validate_ohlcv(df.tail(bars), symbol=provider_symbol)

    @staticmethod
    def _to_frame(raw: list, interval_ms: int) -> pd.DataFrame:
        """Convert Binance's array-of-arrays into our OHLCV contract.

        Binance stamps a kline with its *open* time; we index by close time so
        that a bar's timestamp is the moment its information became known.
        """
        df = pd.DataFrame(
            raw,
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
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        close_time = pd.to_datetime(
            df["open_time"].astype("int64") + interval_ms, unit="ms", utc=True
        )
        out = df[OHLCV_COLUMNS].astype("float64")
        out.index = close_time
        out.index.name = "timestamp"
        return out

    def fetch_ticker(self, provider_symbol: str) -> dict[str, float]:
        """Current best bid/ask, used to sanity-check a signal's entry price."""
        raw = self._get("/api/v3/ticker/bookTicker", {"symbol": provider_symbol})
        return {
            "bid": float(raw["bidPrice"]),
            "ask": float(raw["askPrice"]),
            "spread": float(raw["askPrice"]) - float(raw["bidPrice"]),
        }

    def fetch_24h_stats(self, provider_symbol: str) -> dict[str, float]:
        """24-hour change and volume — the crude 'is this thing moving' check."""
        raw = self._get("/api/v3/ticker/24hr", {"symbol": provider_symbol})
        return {
            "price_change_pct": float(raw["priceChangePercent"]),
            "volume": float(raw["volume"]),
            "quote_volume": float(raw["quoteVolume"]),
            "high": float(raw["highPrice"]),
            "low": float(raw["lowPrice"]),
            "trades": int(raw["count"]),
        }

    def top_movers(self, quote: str = "USDT", limit: int = 15) -> list[dict]:
        """Rank the whole spot market by 24h move.

        This is what powers the 'something is exploding right now' alert: it
        scans every pair, not just the configured watchlist.
        """
        raw = self._get("/api/v3/ticker/24hr", {})
        rows = []
        for entry in raw:
            symbol = entry.get("symbol", "")
            if not symbol.endswith(quote):
                continue
            try:
                quote_volume = float(entry["quoteVolume"])
                change = float(entry["priceChangePercent"])
            except (KeyError, ValueError):
                continue
            # Ignore illiquid pairs; a 400% move on $50k of volume is noise.
            if quote_volume < 5_000_000:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "price_change_pct": change,
                    "quote_volume": quote_volume,
                    "last": float(entry["lastPrice"]),
                }
            )
        rows.sort(key=lambda r: abs(r["price_change_pct"]), reverse=True)
        return rows[:limit]
