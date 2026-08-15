"""On-disk cache for price data.

Providers rate-limit, and refetching 20,000 bars on every backtest wastes both
time and goodwill. The cache stores gzipped CSV (no binary dependencies) keyed
by symbol and timeframe, and serves stale-but-recent data during a provider
outage rather than failing the run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from signalforge.data.base import OHLCV_COLUMNS, validate_ohlcv

log = logging.getLogger(__name__)


class PriceCache:
    def __init__(self, cache_dir: str | Path, ttl_seconds: int = 300):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_").replace("=", "_").replace("^", "_")
        return self.dir / f"{safe}__{timeframe}.csv.gz"

    def age_seconds(self, symbol: str, timeframe: str) -> float:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return float("inf")
        return time.time() - path.stat().st_mtime

    def is_fresh(self, symbol: str, timeframe: str) -> bool:
        return self.age_seconds(symbol, timeframe) < self.ttl

    def read(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return validate_ohlcv(df, symbol=symbol)
        except Exception as exc:  # a corrupt cache must never break a run
            log.warning("Discarding unreadable cache %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return None

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = self._path(symbol, timeframe)
        tmp = path.with_suffix(".tmp.gz")
        try:
            df[OHLCV_COLUMNS].to_csv(tmp, compression="gzip")
            tmp.replace(path)
        except Exception as exc:  # pragma: no cover - disk failure
            log.warning("Could not write cache %s: %s", path.name, exc)
            tmp.unlink(missing_ok=True)

    def merge_and_write(
        self, symbol: str, timeframe: str, fresh: pd.DataFrame
    ) -> pd.DataFrame:
        """Append new bars to whatever we already had.

        This is what lets the engine accumulate deep history over time on
        providers that only serve a short window per request.
        """
        existing = self.read(symbol, timeframe)
        if existing is None or existing.empty:
            merged = fresh
        else:
            merged = pd.concat([existing, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged = validate_ohlcv(merged, symbol=symbol)
        self.write(symbol, timeframe, merged)
        return merged

    def clear(self, symbol: str | None = None) -> int:
        pattern = f"{symbol}__*.csv.gz" if symbol else "*.csv.gz"
        removed = 0
        for path in self.dir.glob(pattern):
            path.unlink()
            removed += 1
        return removed

    def stats(self) -> dict[str, object]:
        files = list(self.dir.glob("*.csv.gz"))
        return {
            "entries": len(files),
            "bytes": sum(f.stat().st_size for f in files),
            "dir": str(self.dir),
        }
