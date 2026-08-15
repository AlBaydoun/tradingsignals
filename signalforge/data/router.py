"""Routes a symbol+timeframe request to the right provider, through the cache."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from signalforge.config import DataConfig
from signalforge.data.base import (
    OHLCV_COLUMNS,
    drop_incomplete_bar,
    summarize,
    validate_ohlcv,
)
from signalforge.data.binance import BinanceProvider
from signalforge.data.cache import PriceCache
from signalforge.data.yahoo import YahooProvider
from signalforge.universe import Instrument, get_instrument

log = logging.getLogger(__name__)


class DataRouter:
    """Single entry point for price data.

    Handles provider selection, caching, incremental history accumulation and
    parallel fetching. Callers just ask for bars and get a clean frame back.
    """

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
        self.cache = PriceCache(self.config.cache_dir, self.config.cache_ttl_seconds)
        self.providers = {
            "binance": BinanceProvider(
                timeout=self.config.request_timeout, max_retries=self.config.max_retries
            ),
            "yahoo": YahooProvider(
                timeout=self.config.request_timeout, max_retries=self.config.max_retries
            ),
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: int | None = None,
        *,
        use_cache: bool = True,
        drop_forming: bool = True,
    ) -> pd.DataFrame:
        """Return the most recent `bars` completed candles for a symbol."""
        instrument = get_instrument(symbol)
        want = bars or self.config.history_bars.get(timeframe.upper(), 5000)

        cached = self.cache.read(symbol, timeframe) if use_cache else None
        if (
            cached is not None
            and len(cached) >= want
            and self.cache.is_fresh(symbol, timeframe)
        ):
            out = cached.tail(want)
            return drop_incomplete_bar(out, timeframe) if drop_forming else out

        try:
            fresh = self.providers[instrument.provider].fetch(
                instrument.provider_symbol, timeframe, want
            )
        except Exception as exc:
            log.warning("Fetch failed for %s %s: %s", symbol, timeframe, exc)
            fresh = pd.DataFrame(columns=OHLCV_COLUMNS)

        if fresh.empty:
            # A provider outage should degrade to stale data, not to nothing.
            if cached is not None and not cached.empty:
                log.warning("Serving stale cache for %s %s", symbol, timeframe)
                out = cached.tail(want)
                return drop_incomplete_bar(out, timeframe) if drop_forming else out
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        merged = self.cache.merge_and_write(symbol, timeframe, fresh)
        out = merged.tail(want)
        out.attrs["symbol"] = symbol
        out.attrs["timeframe"] = timeframe
        return drop_incomplete_bar(out, timeframe) if drop_forming else out

    def get_many(
        self,
        symbols: list[str],
        timeframe: str,
        bars: int | None = None,
        *,
        max_workers: int = 4,
    ) -> dict[str, pd.DataFrame]:
        """Fetch several symbols concurrently.

        Kept deliberately modest: providers throttle hard, and a burst of 20
        parallel requests gets the whole run rate-limited.
        """
        out: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.get_bars, sym, timeframe, bars): sym for sym in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    out[symbol] = future.result()
                except Exception as exc:
                    log.warning("Could not load %s: %s", symbol, exc)
                    out[symbol] = pd.DataFrame(columns=OHLCV_COLUMNS)
        return out

    def get_multi_timeframe(
        self, symbol: str, timeframes: list[str], bars: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """All requested timeframes for one symbol."""
        return {tf: self.get_bars(symbol, tf, bars) for tf in timeframes}

    def live_quote(self, symbol: str) -> dict[str, float]:
        """Best available current price information for a symbol."""
        instrument = get_instrument(symbol)
        try:
            if instrument.provider == "binance":
                ticker = self.providers["binance"].fetch_ticker(
                    instrument.provider_symbol
                )
                stats = self.providers["binance"].fetch_24h_stats(
                    instrument.provider_symbol
                )
                return {**ticker, **stats, "price": ticker["ask"]}
            return self.providers["yahoo"].fetch_quote(instrument.provider_symbol)
        except Exception as exc:
            log.warning("Live quote failed for %s: %s", symbol, exc)
            return {}

    def effective_spread_pips(self, symbol: str) -> float:
        """Real spread where we can measure it, configured estimate otherwise.

        Backtests that assume a fixed spread flatter themselves. Where the
        provider exposes a live book, use it.
        """
        instrument = get_instrument(symbol)
        if instrument.provider == "binance":
            try:
                ticker = self.providers["binance"].fetch_ticker(
                    instrument.provider_symbol
                )
                measured = ticker["spread"] / instrument.pip_size
                # Binance spot is tighter than any CFD broker will give you, so
                # never report better than the configured estimate.
                return max(measured, instrument.typical_spread_pips)
            except Exception:
                pass
        return instrument.typical_spread_pips

    def health(self, symbols: list[str], timeframe: str = "H1") -> dict[str, dict]:
        """Per-symbol data availability, for the `doctor` command."""
        report: dict[str, dict] = {}
        for symbol in symbols:
            try:
                df = self.get_bars(symbol, timeframe, 300)
                report[symbol] = {
                    "ok": not df.empty,
                    "provider": get_instrument(symbol).provider,
                    **summarize(df),
                }
            except Exception as exc:
                report[symbol] = {"ok": False, "error": str(exc)}
        return report

    def scan_crypto_movers(self, limit: int = 15) -> list[dict]:
        """Market-wide crypto movers, beyond the configured watchlist."""
        try:
            return self.providers["binance"].top_movers(limit=limit)
        except Exception as exc:
            log.warning("Mover scan failed: %s", exc)
            return []
