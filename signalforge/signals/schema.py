"""The signal data model — what the engine actually delivers.

A signal carries not just the trade, but the evidence behind it and the honest
statement of how often signals like it have worked. The accuracy figure is
deliberately a *measured out-of-sample frequency*, never the model's own
confidence, and it is allowed to be `None` when there is not enough history to
make a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum


class Direction(str, Enum):
    LONG = "BUY"
    SHORT = "SELL"

    @classmethod
    def from_int(cls, value: int) -> "Direction":
        return cls.LONG if value > 0 else cls.SHORT

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class SignalQuality(str, Enum):
    """A blunt, human-readable grade."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    WATCH_ONLY = "watch_only"  # interesting, but not tradable as it stands


@dataclass
class Evidence:
    """Why the engine believes this trade has an edge."""

    regime: str = ""
    regime_scores: dict[str, float] = field(default_factory=dict)
    top_features: list[tuple[str, float]] = field(default_factory=list)
    multi_timeframe_agreement: float = 0.0  # -1..1
    anomaly: dict[str, object] = field(default_factory=dict)
    news_sentiment: float = 0.0
    news_headlines: list[dict] = field(default_factory=list)
    event_risk: dict[str, object] = field(default_factory=dict)
    backtest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Signal:
    """One actionable trade idea."""

    # --- identity ------------------------------------------------------
    symbol: str
    mt5_symbol: str
    market: str
    timeframe: str
    direction: Direction
    generated_at: datetime
    valid_until: datetime

    # --- the trade -----------------------------------------------------
    entry: float
    stop_loss: float
    take_profits: list[float]
    lots: float
    risk_amount: float
    risk_percent: float
    stop_distance_pips: float
    reward_risk: float

    # --- confidence ----------------------------------------------------
    # The model's calibrated probability for its chosen direction.
    model_confidence: float
    # The realised out-of-sample hit rate of past signals at this confidence.
    # None means: not enough history to make an honest claim.
    measured_accuracy: float | None
    directional_edge: float
    quality: SignalQuality

    # --- context -------------------------------------------------------
    evidence: Evidence = field(default_factory=Evidence)
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    stop_basis: str = ""
    atr: float = 0.0
    spread_pips: float = 0.0

    @property
    def is_actionable(self) -> bool:
        return self.quality is not SignalQuality.WATCH_ONLY and self.lots > 0

    @property
    def expectancy_r(self) -> float:
        """Expected R per trade using the *measured* hit rate, not the model's."""
        probability = self.measured_accuracy
        if probability is None:
            return 0.0
        return probability * self.reward_risk - (1.0 - probability)

    def minutes_remaining(self) -> float:
        return (
            self.valid_until - datetime.now(timezone.utc)
        ).total_seconds() / 60.0

    @property
    def is_expired(self) -> bool:
        return self.minutes_remaining() <= 0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["direction"] = self.direction.value
        out["quality"] = self.quality.value
        out["generated_at"] = self.generated_at.isoformat()
        out["valid_until"] = self.valid_until.isoformat()
        out["expectancy_r"] = round(self.expectancy_r, 3)
        out["is_actionable"] = self.is_actionable
        return out


@dataclass
class WatchItem:
    """A market worth watching that is not yet a trade.

    Covers the "something is about to happen" case: a coiling market, or one
    already exploding but without a clean entry.
    """

    symbol: str
    mt5_symbol: str
    timeframe: str
    reason: str
    ignition_score: float
    coiling_score: float
    direction_hint: int
    price: float
    price_change_pct: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalBundle:
    """Everything produced by one run of the engine."""

    generated_at: datetime
    signals: list[Signal] = field(default_factory=list)
    watchlist: list[WatchItem] = field(default_factory=list)
    # Per (symbol, timeframe) ranking of where the edge currently is.
    rankings: list[dict] = field(default_factory=list)
    market_summary: str = ""
    blocked: list[dict] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def actionable(self) -> list[Signal]:
        return [s for s in self.signals if s.is_actionable]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
            "watchlist": [w.to_dict() for w in self.watchlist],
            "rankings": self.rankings,
            "market_summary": self.market_summary,
            "blocked": self.blocked,
            "diagnostics": self.diagnostics,
        }


def grade(
    model_confidence: float,
    measured_accuracy: float | None,
    reward_risk: float,
    edge: float,
    *,
    min_confidence: float = 0.58,
) -> SignalQuality:
    """Grade a signal, weighting measured history above model confidence.

    A signal whose confidence bucket has no measured track record can never be
    graded STRONG, no matter how certain the model claims to be.
    """
    if measured_accuracy is None:
        # Unproven confidence band: tradable at best, never strong.
        return (
            SignalQuality.WEAK
            if model_confidence >= min_confidence
            else SignalQuality.WATCH_ONLY
        )

    breakeven = 1.0 / (1.0 + reward_risk)
    margin = measured_accuracy - breakeven

    if margin <= 0:
        # Loses money at this reward:risk regardless of how confident it looks.
        return SignalQuality.WATCH_ONLY
    if margin >= 0.10 and model_confidence >= min_confidence and abs(edge) >= 0.15:
        return SignalQuality.STRONG
    if margin >= 0.05:
        return SignalQuality.MODERATE
    return SignalQuality.WEAK
