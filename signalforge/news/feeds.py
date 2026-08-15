"""News feed aggregation.

Pulls RSS from sources that serve without an API key or a Cloudflare challenge,
maps each headline to the instruments it plausibly concerns, and scores it.

Headline flow is used here for context and risk, not as a trade trigger. By the
time a story reaches a public RSS feed, the move is usually over.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
import requests

from signalforge.news.sentiment import (
    SentimentScore,
    aggregate,
    relevant_symbols,
    score_text,
)

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Verified to respond without an API key. Sources behind Cloudflare (FXStreet,
# CoinDesk) are deliberately excluded rather than retried forever.
FEEDS: dict[str, dict[str, str]] = {
    "yahoo_finance": {
        "url": "https://finance.yahoo.com/news/rssindex",
        "market": "general",
    },
    "investing_stocks": {
        "url": "https://www.investing.com/rss/news_25.rss",
        "market": "equities",
    },
    "investing_forex": {
        "url": "https://www.investing.com/rss/news_1.rss",
        "market": "forex",
    },
    "investing_commodities": {
        "url": "https://www.investing.com/rss/news_11.rss",
        "market": "commodities",
    },
    "wsj_markets": {
        "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "market": "general",
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "market": "crypto",
    },
    "google_forex": {
        "url": (
            "https://news.google.com/rss/search?q=forex+OR+currency+market"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        "market": "forex",
    },
    "google_crypto": {
        "url": (
            "https://news.google.com/rss/search?q=bitcoin+OR+crypto+market"
            "&hl=en-US&gl=US&ceid=US:en"
        ),
        "market": "crypto",
    },
}


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published: datetime
    market: str
    symbols: list[str]
    sentiment: SentimentScore

    @property
    def age_hours(self) -> float:
        return (
            datetime.now(timezone.utc) - self.published
        ).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat(),
            "age_hours": round(self.age_hours, 2),
            "symbols": self.symbols,
            "sentiment": self.sentiment.score,
            "sentiment_label": self.sentiment.label(),
        }


class NewsAggregator:
    """Fetches, parses and scores headlines across all configured feeds."""

    def __init__(self, timeout: int = 15, max_age_hours: float = 48.0):
        self.timeout = timeout
        self.max_age_hours = max_age_hours
        self._cache: list[NewsItem] = []
        self._fetched_at: datetime | None = None

    def _fetch_one(self, name: str, spec: dict[str, str]) -> list[NewsItem]:
        try:
            resp = requests.get(
                spec["url"],
                timeout=self.timeout,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception as exc:
            log.warning("Feed %s failed: %s", name, exc)
            return []

        items: list[NewsItem] = []
        now = datetime.now(timezone.utc)

        for entry in parsed.entries[:60]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue

            published = now
            for field in ("published_parsed", "updated_parsed"):
                value = entry.get(field)
                if value:
                    try:
                        published = datetime(*value[:6], tzinfo=timezone.utc)
                        break
                    except (TypeError, ValueError):
                        continue

            age = (now - published).total_seconds() / 3600.0
            if age > self.max_age_hours or age < -1:
                continue

            items.append(
                NewsItem(
                    title=title,
                    source=name,
                    url=entry.get("link", ""),
                    published=published,
                    market=spec.get("market", "general"),
                    symbols=relevant_symbols(title),
                    sentiment=score_text(title),
                )
            )
        return items

    def fetch(self, max_workers: int = 6) -> list[NewsItem]:
        """Pull every feed in parallel, keeping whatever succeeds."""
        items: list[NewsItem] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._fetch_one, name, spec): name
                for name, spec in FEEDS.items()
            }
            for future in as_completed(futures):
                try:
                    items.extend(future.result())
                except Exception as exc:
                    log.warning("Feed %s raised: %s", futures[future], exc)

        # De-duplicate: the same story reaches several aggregators.
        seen: set[str] = set()
        unique: list[NewsItem] = []
        for item in sorted(items, key=lambda i: i.published, reverse=True):
            key = item.title.lower()[:90]
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        self._cache = unique
        self._fetched_at = datetime.now(timezone.utc)
        return unique

    def items(self, refresh_if_stale_minutes: float = 20.0) -> list[NewsItem]:
        if self._fetched_at is None:
            return self.fetch()
        age = (datetime.now(timezone.utc) - self._fetched_at).total_seconds() / 60.0
        if age > refresh_if_stale_minutes:
            return self.fetch()
        return self._cache

    def for_symbol(self, symbol: str, max_age_hours: float = 24.0) -> list[NewsItem]:
        return [
            item
            for item in self.items()
            if symbol.upper() in item.symbols and item.age_hours <= max_age_hours
        ]

    def sentiment_for(
        self, symbol: str, max_age_hours: float = 24.0
    ) -> SentimentScore:
        """Time-decayed sentiment across all headlines mentioning `symbol`."""
        relevant = self.for_symbol(symbol, max_age_hours)
        return aggregate([(i.title, i.age_hours) for i in relevant])

    def market_sentiment(self, market: str) -> SentimentScore:
        """Sentiment for a whole asset class, as a risk-on/risk-off proxy."""
        relevant = [i for i in self.items() if i.market == market]
        return aggregate([(i.title, i.age_hours) for i in relevant])

    def headlines_for_briefing(
        self, symbol: str, limit: int = 5
    ) -> list[dict[str, object]]:
        """The strongest recent headlines, for the signal explanation."""
        relevant = self.for_symbol(symbol, max_age_hours=24.0)
        relevant.sort(key=lambda i: abs(i.sentiment.score) * i.sentiment.confidence, reverse=True)
        return [
            {
                "title": i.title,
                "source": i.source,
                "age_hours": round(i.age_hours, 1),
                "sentiment": i.sentiment.label(),
                "url": i.url,
            }
            for i in relevant[:limit]
        ]
