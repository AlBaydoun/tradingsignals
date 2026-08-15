"""Economic calendar, news feeds and headline sentiment."""

from signalforge.news.calendar import CalendarEvent, EconomicCalendar, EventRisk
from signalforge.news.feeds import FEEDS, NewsAggregator, NewsItem
from signalforge.news.sentiment import (
    SentimentScore,
    aggregate,
    relevant_symbols,
    score_text,
)

__all__ = [
    "EconomicCalendar",
    "CalendarEvent",
    "EventRisk",
    "NewsAggregator",
    "NewsItem",
    "FEEDS",
    "SentimentScore",
    "score_text",
    "aggregate",
    "relevant_symbols",
]
