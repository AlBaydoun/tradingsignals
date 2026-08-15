"""The trade journal — how the engine finds out whether it was right.

Every signal is recorded when it is issued and updated when it resolves. This
is the feedback loop: without it the engine can only ever quote backtest
numbers, and backtest numbers are the thing most likely to be wrong.

Stored as JSON Lines so the file is append-only, survives a crash mid-write,
and can be inspected with ordinary text tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """One issued signal and, once known, what became of it."""

    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    issued_at: str
    entry: float
    stop_loss: float
    take_profits: list[float]
    lots: float
    risk_amount: float
    model_confidence: float
    measured_accuracy: float | None
    reward_risk: float
    regime: str
    quality: str

    # Filled in when the trade resolves.
    status: str = "open"  # open | won | lost | expired | cancelled
    closed_at: str | None = None
    exit_price: float | None = None
    pnl: float | None = None
    r_multiple: float | None = None
    exit_reason: str | None = None
    max_favourable_excursion: float | None = None
    max_adverse_excursion: float | None = None
    notes: str = ""
    features_snapshot: dict = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        return self.status in ("won", "lost", "expired")

    @property
    def won(self) -> bool:
        return self.status == "won"

    def to_dict(self) -> dict:
        return asdict(self)


class TradeJournal:
    """Append-only record of signals and their outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: JournalEntry) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def all_entries(self) -> list[JournalEntry]:
        """Read the journal, collapsing updates so the latest state wins."""
        if not self.path.exists():
            return []

        by_id: dict[str, JournalEntry] = {}
        with open(self.path) as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    entry = JournalEntry(**payload)
                    by_id[entry.signal_id] = entry
                except Exception as exc:
                    log.warning("Skipping malformed journal line %d: %s", line_number, exc)
        return list(by_id.values())

    def update_outcome(
        self,
        signal_id: str,
        *,
        status: str,
        exit_price: float,
        pnl: float,
        r_multiple: float,
        exit_reason: str,
        mfe: float | None = None,
        mae: float | None = None,
    ) -> bool:
        """Close out a signal by appending its updated record."""
        entries = {e.signal_id: e for e in self.all_entries()}
        entry = entries.get(signal_id)
        if entry is None:
            log.warning("No journal entry for signal %s", signal_id)
            return False

        entry.status = status
        entry.closed_at = datetime.now(timezone.utc).isoformat()
        entry.exit_price = exit_price
        entry.pnl = pnl
        entry.r_multiple = r_multiple
        entry.exit_reason = exit_reason
        entry.max_favourable_excursion = mfe
        entry.max_adverse_excursion = mae
        self.record(entry)
        return True

    def open_signals(self) -> list[JournalEntry]:
        return [e for e in self.all_entries() if e.status == "open"]

    def closed_signals(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> list[JournalEntry]:
        entries = [e for e in self.all_entries() if e.is_closed]
        if symbol:
            entries = [e for e in entries if e.symbol == symbol.upper()]
        if timeframe:
            entries = [e for e in entries if e.timeframe == timeframe.upper()]
        return sorted(entries, key=lambda e: e.issued_at)

    def outcomes(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> list[bool]:
        """Chronological win/loss sequence, for drift detection."""
        return [
            e.won
            for e in self.closed_signals(symbol, timeframe)
            if e.status in ("won", "lost")
        ]

    def live_statistics(
        self, symbol: str | None = None, timeframe: str | None = None
    ) -> dict[str, object]:
        """What actually happened, as opposed to what the backtest promised."""
        closed = [
            e
            for e in self.closed_signals(symbol, timeframe)
            if e.status in ("won", "lost")
        ]
        if not closed:
            return {"trades": 0, "message": "no closed trades yet"}

        wins = [e for e in closed if e.won]
        r_multiples = [e.r_multiple for e in closed if e.r_multiple is not None]
        pnls = [e.pnl for e in closed if e.pnl is not None]

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))

        # The comparison that matters: promised versus delivered.
        promised = [
            e.measured_accuracy for e in closed if e.measured_accuracy is not None
        ]

        return {
            "trades": len(closed),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(closed), 4),
            "expectancy_r": round(float(np.mean(r_multiples)), 4) if r_multiples else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "profit_factor": round(gross_profit / gross_loss, 3)
            if gross_loss > 0
            else float("inf"),
            "promised_accuracy": round(float(np.mean(promised)), 4) if promised else None,
            "delivered_minus_promised": (
                round(len(wins) / len(closed) - float(np.mean(promised)), 4)
                if promised
                else None
            ),
        }

    def by_confidence_bucket(self) -> list[dict]:
        """Live calibration check: does a 65% signal really win 65% of the time?"""
        closed = [e for e in self.all_entries() if e.status in ("won", "lost")]
        if not closed:
            return []

        buckets: list[dict] = []
        for lower in np.arange(0.45, 1.0, 0.05):
            upper = lower + 0.05
            in_bin = [
                e for e in closed if lower <= e.model_confidence < upper
            ]
            if not in_bin:
                continue
            wins = sum(1 for e in in_bin if e.won)
            buckets.append(
                {
                    "range": f"{lower:.2f}-{upper:.2f}",
                    "trades": len(in_bin),
                    "win_rate": round(wins / len(in_bin), 4),
                    "mean_confidence": round(
                        float(np.mean([e.model_confidence for e in in_bin])), 4
                    ),
                }
            )
        return buckets

    def summary(self) -> dict[str, object]:
        entries = self.all_entries()
        closed = [e for e in entries if e.is_closed]
        return {
            "total_signals": len(entries),
            "open": len([e for e in entries if e.status == "open"]),
            "closed": len(closed),
            "path": str(self.path),
            **self.live_statistics(),
        }
