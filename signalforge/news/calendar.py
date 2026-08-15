"""Economic calendar ingestion and event-risk blackouts.

A model trained on price alone has no idea that Non-Farm Payrolls lands in four
minutes. It will happily emit a confident EURUSD signal into an event that
routinely moves the pair 80 pips in a second and blows through stops.

This module pulls the week's scheduled releases, maps them onto the instruments
they actually affect, and produces blackout windows the signal generator
respects. Blocking a trade you would have won is cheap. Taking a trade into a
central bank decision is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from signalforge.universe import INSTRUMENT_CURRENCIES

log = logging.getLogger(__name__)

# Free weekly calendar feed, no API key required. Only the current week is
# published — the next-week endpoint that used to exist now returns 404, so the
# engine plans around a rolling seven-day horizon rather than a fortnight.
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

IMPACT_WEIGHT = {"High": 1.0, "Medium": 0.5, "Low": 0.15, "Holiday": 0.3}

# Releases that reliably produce violent, stop-hunting moves.
TIER_ONE_EVENTS = (
    "non-farm employment change",
    "nonfarm payrolls",
    "fomc statement",
    "federal funds rate",
    "cpi m/m",
    "core cpi",
    "ecb press conference",
    "main refinancing rate",
    "interest rate decision",
    "gdp q/q",
    "unemployment rate",
    "retail sales m/m",
    "boe official bank rate",
    "monetary policy statement",
)


@dataclass
class CalendarEvent:
    """One scheduled economic release."""

    title: str
    currency: str
    time: datetime
    impact: str
    forecast: str = ""
    previous: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact == "High"

    @property
    def is_tier_one(self) -> bool:
        """Whether this is one of the handful of releases that reprice a market."""
        lowered = self.title.lower()
        return any(key in lowered for key in TIER_ONE_EVENTS)

    def minutes_until(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (self.time - now).total_seconds() / 60.0

    def affects(self, symbol: str) -> bool:
        return self.currency in INSTRUMENT_CURRENCIES.get(symbol.upper(), ())

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "currency": self.currency,
            "time": self.time.isoformat(),
            "impact": self.impact,
            "forecast": self.forecast,
            "previous": self.previous,
            "tier_one": self.is_tier_one,
        }


@dataclass
class EventRisk:
    """The calendar's verdict on trading a symbol right now."""

    symbol: str
    blocked: bool
    reason: str
    # 0..1. Feeds the model and scales position size down near events.
    risk_score: float
    next_event: CalendarEvent | None = None
    minutes_to_next: float | None = None
    events_in_window: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "blocked": self.blocked,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 3),
            "next_event": self.next_event.to_dict() if self.next_event else None,
            "minutes_to_next": round(self.minutes_to_next, 1)
            if self.minutes_to_next is not None
            else None,
            "events_in_window": self.events_in_window,
        }


class EconomicCalendar:
    """Fetches and queries the week's scheduled economic events."""

    def __init__(self, timeout: int = 20, cache_minutes: int = 60):
        self.timeout = timeout
        self.cache_minutes = cache_minutes
        self._events: list[CalendarEvent] = []
        self._fetched_at: datetime | None = None

    def _is_stale(self) -> bool:
        if self._fetched_at is None:
            return True
        age = (datetime.now(timezone.utc) - self._fetched_at).total_seconds() / 60.0
        return age > self.cache_minutes

    def refresh(self) -> list[CalendarEvent]:
        """Pull the calendar, keeping the previous copy if the fetch fails."""
        events: list[CalendarEvent] = []
        try:
            resp = requests.get(
                CALENDAR_URL,
                timeout=self.timeout,
                headers={"User-Agent": "SignalForge/0.1"},
            )
            resp.raise_for_status()
            events.extend(self._parse(resp.json()))
        except Exception as exc:
            log.warning("Calendar fetch failed: %s", exc)

        if events:
            self._events = sorted(events, key=lambda e: e.time)
            self._fetched_at = datetime.now(timezone.utc)
        elif not self._events:
            log.warning("No calendar data available; event filtering is disabled")
        return self._events

    @staticmethod
    def _parse(payload: list) -> list[CalendarEvent]:
        events: list[CalendarEvent] = []
        for entry in payload or []:
            try:
                raw_time = entry.get("date")
                if not raw_time:
                    continue
                # Feed uses ISO-8601 with a UTC offset.
                event_time = datetime.fromisoformat(raw_time)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                events.append(
                    CalendarEvent(
                        title=str(entry.get("title", "")).strip(),
                        currency=str(entry.get("country", "")).strip().upper(),
                        time=event_time.astimezone(timezone.utc),
                        impact=str(entry.get("impact", "Low")).strip(),
                        forecast=str(entry.get("forecast", "") or ""),
                        previous=str(entry.get("previous", "") or ""),
                    )
                )
            except Exception:
                continue
        return events

    def events(self, refresh_if_stale: bool = True) -> list[CalendarEvent]:
        if refresh_if_stale and self._is_stale():
            self.refresh()
        return self._events

    def upcoming(
        self, symbol: str, within_minutes: float = 240.0
    ) -> list[CalendarEvent]:
        """Events affecting `symbol` in the next `within_minutes`."""
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(minutes=within_minutes)
        return [
            e
            for e in self.events()
            if e.affects(symbol) and now <= e.time <= horizon
        ]

    def recent(self, symbol: str, within_minutes: float = 60.0) -> list[CalendarEvent]:
        """Events that have just fired — the market is still digesting them."""
        now = datetime.now(timezone.utc)
        floor = now - timedelta(minutes=within_minutes)
        return [
            e for e in self.events() if e.affects(symbol) and floor <= e.time <= now
        ]

    def assess(
        self,
        symbol: str,
        *,
        blackout_minutes: float = 30.0,
        allow_medium_impact: bool = True,
        lookahead_minutes: float = 240.0,
    ) -> EventRisk:
        """Decide whether `symbol` is safe to trade right now.

        Blocks a window on **both sides** of a release: before, because the move
        is unpredictable; after, because the initial spike frequently reverses
        and spreads widen to several times normal.
        """
        now = datetime.now(timezone.utc)
        upcoming = self.upcoming(symbol, lookahead_minutes)
        just_passed = self.recent(symbol, blackout_minutes)

        if not upcoming and not just_passed:
            return EventRisk(symbol, False, "no scheduled events nearby", 0.0)

        # --- immediate blocks --------------------------------------------
        for event in just_passed:
            if event.is_high_impact or event.is_tier_one:
                elapsed = abs(event.minutes_until(now))
                return EventRisk(
                    symbol,
                    True,
                    f"{event.title} ({event.currency}) released {elapsed:.0f} min ago — "
                    "spreads are still wide and the initial move often reverses",
                    1.0,
                    event,
                    -elapsed,
                    len(just_passed),
                )

        for event in upcoming:
            minutes = event.minutes_until(now)
            if minutes > blackout_minutes:
                continue
            blocking = event.is_high_impact or (
                event.impact == "Medium" and not allow_medium_impact
            )
            if blocking:
                return EventRisk(
                    symbol,
                    True,
                    f"{event.title} ({event.currency}) in {minutes:.0f} min — "
                    f"{event.impact.lower()} impact",
                    1.0,
                    event,
                    minutes,
                    len(upcoming),
                )

        # --- graded risk, no block ---------------------------------------
        score = 0.0
        for event in upcoming:
            minutes = max(event.minutes_until(now), 1.0)
            weight = IMPACT_WEIGHT.get(event.impact, 0.1)
            if event.is_tier_one:
                weight = min(1.0, weight * 1.5)
            # Risk decays with distance: an event in 4 hours barely registers.
            score += weight * max(0.0, 1.0 - minutes / lookahead_minutes)

        score = min(1.0, score)
        next_event = upcoming[0] if upcoming else None
        return EventRisk(
            symbol=symbol,
            blocked=False,
            reason=(
                f"{len(upcoming)} event(s) within {lookahead_minutes / 60:.0f}h"
                if upcoming
                else "clear"
            ),
            risk_score=score,
            next_event=next_event,
            minutes_to_next=next_event.minutes_until(now) if next_event else None,
            events_in_window=len(upcoming),
        )

    def week_ahead(self, symbols: list[str]) -> list[dict]:
        """High-impact events for a watchlist, for the weekly briefing."""
        currencies = set()
        for symbol in symbols:
            currencies.update(INSTRUMENT_CURRENCIES.get(symbol.upper(), ()))

        return [
            e.to_dict()
            for e in self.events()
            if e.currency in currencies
            and (e.is_high_impact or e.is_tier_one)
            and e.time >= datetime.now(timezone.utc)
        ]
