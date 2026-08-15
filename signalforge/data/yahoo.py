"""Yahoo Finance chart data for FX, metals, energy and index futures.

Yahoo is unofficial and rate-limits aggressively, so requests are chunked,
retried and cached. It is the only free source that covers spot FX, gold and
index futures on intraday timeframes without an API key.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from signalforge.data.base import (
    OHLCV_COLUMNS,
    get_timeframe,
    resample_ohlcv,
    validate_ohlcv,
)

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# How far back Yahoo will serve each interval, and how much it will return in a
# single request. Exceeding either returns an empty result rather than an error.
INTERVAL_LIMITS: dict[str, tuple[int, int]] = {
    # interval: (max total history days, max days per request)
    "1m": (30, 7),
    "2m": (60, 60),
    "5m": (60, 60),
    "15m": (60, 60),
    "30m": (60, 60),
    "60m": (730, 730),
    "1d": (36500, 36500),
}


class YahooProvider:
    name = "yahoo"

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

    def _get_chart(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        params = {
            "interval": interval,
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "includePrePost": "false",
            "events": "div,splits",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    CHART_URL.format(symbol=symbol), params=params, timeout=self.timeout
                )
                # Yahoo signals throttling with 429, 999 and — confusingly —
                # 422 on requests that succeed when retried. All three are
                # transient, so back off rather than giving up on the symbol.
                if resp.status_code in (422, 429, 999):
                    wait = 2 ** (attempt + 1)
                    log.warning(
                        "Yahoo throttled %s (HTTP %d), sleeping %ss",
                        symbol,
                        resp.status_code,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return self._parse(resp.json(), interval)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        log.warning("Yahoo request failed for %s: %s", symbol, last_error)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    @staticmethod
    def _parse(payload: dict, interval: str) -> pd.DataFrame:
        """Pull the OHLCV arrays out of Yahoo's nested chart response."""
        chart = (payload or {}).get("chart") or {}
        results = chart.get("result")
        if not results:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        result = results[0]
        timestamps = result.get("timestamp")
        quotes = (result.get("indicators") or {}).get("quote")
        if not timestamps or not quotes:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        quote = quotes[0]
        df = pd.DataFrame(
            {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )
        # Yahoo stamps bars with their open time; shift to close time so the
        # index means "everything known as of here".
        minutes = {
            "1m": 1,
            "2m": 2,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "60m": 60,
            "1d": 1440,
        }[interval]
        idx = pd.to_datetime(timestamps, unit="s", utc=True) + pd.Timedelta(
            minutes=minutes
        )
        df.index = idx
        df.index.name = "timestamp"
        return df.dropna(subset=["close"])

    def fetch(self, provider_symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fetch bars, resampling from 60m when Yahoo has no native H4."""
        tf = get_timeframe(timeframe)

        # Yahoo has no 4-hour interval, so build it from hourly bars.
        if timeframe.upper() == "H4":
            hourly = self.fetch(provider_symbol, "H1", bars * 4 + 50)
            if hourly.empty:
                return hourly
            out = resample_ohlcv(hourly, "H4").tail(bars)
            return validate_ohlcv(out, symbol=provider_symbol)

        interval = tf.yahoo
        total_days, chunk_days = INTERVAL_LIMITS.get(interval, (60, 60))

        # How far back we need to reach to collect `bars` bars, generously
        # padded for weekends and holidays when the market is shut.
        span_days = (bars * tf.minutes) / (60 * 24)
        if tf.minutes < 1440:
            span_days *= 1.6  # ~24/5 market plus holidays
        span_days = min(span_days + 5, total_days)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=span_days)

        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            chunk = self._get_chart(provider_symbol, interval, cursor, chunk_end)
            if not chunk.empty:
                frames.append(chunk)
            cursor = chunk_end
            if cursor < end:
                time.sleep(0.35)  # be a polite client

        if not frames:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return validate_ohlcv(df.tail(bars), symbol=provider_symbol)

    def fetch_quote(self, provider_symbol: str) -> dict[str, float]:
        """Latest price and day range, straight from the chart metadata."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=5)
        params = {
            "interval": "1d",
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
        }
        try:
            resp = self.session.get(
                CHART_URL.format(symbol=provider_symbol),
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            meta = resp.json()["chart"]["result"][0]["meta"]
            return {
                "price": float(meta.get("regularMarketPrice", 0.0)),
                "day_high": float(meta.get("regularMarketDayHigh", 0.0)),
                "day_low": float(meta.get("regularMarketDayLow", 0.0)),
                "previous_close": float(
                    meta.get("chartPreviousClose", meta.get("previousClose", 0.0))
                ),
                "fifty_two_week_high": float(meta.get("fiftyTwoWeekHigh", 0.0)),
                "fifty_two_week_low": float(meta.get("fiftyTwoWeekLow", 0.0)),
            }
        except Exception as exc:
            log.warning("Yahoo quote failed for %s: %s", provider_symbol, exc)
            return {}
