"""Conditional edge maps — learning *where* a model works, not just whether.

The backtest that motivated this module found, on BTCUSDT H4:

    strong uptrend     PF 1.76
    strong downtrend   PF 1.29
    weak uptrend       PF 1.05
    range              PF 0.97
    weak downtrend     PF 0.95

The blended profit factor was 1.07 — marginal, barely worth trading. But the
blend is an average of a real edge in trending markets and a steady bleed
everywhere else. A model reported as one number hides that completely.

So training measures performance per regime and per session, stores the map,
and signal generation consults it: if this model has historically lost money in
the conditions holding right now, the signal is vetoed regardless of how
confident the model is. The engine learns its own competence boundary instead
of assuming it is uniform.

The guard against over-fitting this is a minimum trade count. A regime with
eleven trades tells you nothing, and slicing a backtest finely enough will
always produce a flattering subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# Four-hour blocks rather than 24 separate hours: hourly buckets shatter the
# sample into slices too thin to mean anything.
SESSION_BUCKETS: dict[str, tuple[int, int]] = {
    "asia_early": (0, 4),
    "asia_late": (4, 8),
    "london_open": (8, 12),
    "london_ny_overlap": (12, 16),
    "ny_afternoon": (16, 20),
    "ny_close": (20, 24),
}


def session_bucket(hour_utc: int) -> str:
    for name, (start, end) in SESSION_BUCKETS.items():
        if start <= hour_utc < end:
            return name
    return "unknown"


@dataclass
class ConditionSlice:
    """Measured performance under one condition."""

    condition: str
    trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    total_pnl: float

    @property
    def is_reliable(self) -> bool:
        """Enough trades for the number to carry information."""
        return self.trades >= 25

    @property
    def is_losing(self) -> bool:
        return self.profit_factor < 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConditionalEdge:
    """Where a model's edge actually lives."""

    by_regime: dict[str, ConditionSlice] = field(default_factory=dict)
    by_session: dict[str, ConditionSlice] = field(default_factory=dict)
    total_trades: int = 0
    # Regimes/sessions must clear this profit factor to stay enabled.
    min_profit_factor: float = 1.0
    min_trades: int = 25

    def to_dict(self) -> dict:
        return {
            "by_regime": {k: v.to_dict() for k, v in self.by_regime.items()},
            "by_session": {k: v.to_dict() for k, v in self.by_session.items()},
            "total_trades": self.total_trades,
            "min_profit_factor": self.min_profit_factor,
            "min_trades": self.min_trades,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConditionalEdge":
        if not payload:
            return cls()
        return cls(
            by_regime={
                k: ConditionSlice(**v) for k, v in payload.get("by_regime", {}).items()
            },
            by_session={
                k: ConditionSlice(**v) for k, v in payload.get("by_session", {}).items()
            },
            total_trades=payload.get("total_trades", 0),
            min_profit_factor=payload.get("min_profit_factor", 1.0),
            min_trades=payload.get("min_trades", 25),
        )

    # -- the gates -------------------------------------------------------

    def regime_verdict(self, regime: str) -> tuple[bool, str]:
        """Whether to allow a trade in this regime, and why.

        An unmeasured regime is allowed through — absence of evidence is not
        evidence of failure, and blocking everything unseen would freeze a
        freshly trained model solid.
        """
        entry = self.by_regime.get(regime)
        if entry is None:
            return True, ""
        if not entry.is_reliable:
            return True, f"{regime}: only {entry.trades} past trades, unmeasured"
        if entry.profit_factor < self.min_profit_factor:
            return False, (
                f"this model has lost money in {regime.replace('_', ' ')} "
                f"conditions (profit factor {entry.profit_factor:.2f} over "
                f"{entry.trades} trades)"
            )
        return True, (
            f"{regime.replace('_', ' ')} is a profitable regime for this model "
            f"(profit factor {entry.profit_factor:.2f})"
        )

    def session_verdict(self, hour_utc: int) -> tuple[bool, str]:
        bucket = session_bucket(hour_utc)
        entry = self.by_session.get(bucket)
        if entry is None or not entry.is_reliable:
            return True, ""
        if entry.profit_factor < self.min_profit_factor:
            return False, (
                f"this model has lost money during {bucket.replace('_', ' ')} "
                f"(profit factor {entry.profit_factor:.2f} over "
                f"{entry.trades} trades)"
            )
        return True, ""

    def best_conditions(self, limit: int = 3) -> list[ConditionSlice]:
        reliable = [s for s in self.by_regime.values() if s.is_reliable]
        return sorted(reliable, key=lambda s: -s.profit_factor)[:limit]

    def worst_conditions(self, limit: int = 3) -> list[ConditionSlice]:
        reliable = [s for s in self.by_regime.values() if s.is_reliable]
        return sorted(reliable, key=lambda s: s.profit_factor)[:limit]

    def describe(self) -> str:
        """A sentence about where this model does and does not work."""
        if self.total_trades < self.min_trades:
            return "Too few backtest trades to map where this model works."

        good = [s for s in self.by_regime.values() if s.is_reliable and not s.is_losing]
        bad = [s for s in self.by_regime.values() if s.is_reliable and s.is_losing]

        if not good and not bad:
            return "No regime has enough trades to judge."
        if not good:
            return "This model lost money in every measurable regime."

        parts = [
            "Profitable in "
            + ", ".join(s.condition.replace("_", " ") for s in good)
            + "."
        ]
        if bad:
            parts.append(
                "Blocked in "
                + ", ".join(s.condition.replace("_", " ") for s in bad)
                + "."
            )
        return " ".join(parts)


def _slice_stats(condition: str, group: pd.DataFrame) -> ConditionSlice:
    pnl = group["pnl"]
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl <= 0].sum()))
    r_multiples = group.get("r_multiple", pd.Series(dtype="float64"))

    return ConditionSlice(
        condition=condition,
        trades=int(len(group)),
        win_rate=round(float((pnl > 0).mean()), 4),
        profit_factor=round(
            gross_profit / gross_loss if gross_loss > 0 else float("inf"), 3
        ),
        expectancy_r=round(float(r_multiples.mean()), 4) if len(r_multiples) else 0.0,
        total_pnl=round(float(pnl.sum()), 2),
    )


def build_conditional_edge(
    trades: pd.DataFrame,
    *,
    min_trades: int = 25,
    min_profit_factor: float = 1.0,
) -> ConditionalEdge:
    """Measure per-regime and per-session performance from a backtest trade log."""
    edge = ConditionalEdge(
        min_trades=min_trades, min_profit_factor=min_profit_factor
    )
    if trades is None or trades.empty:
        return edge

    edge.total_trades = int(len(trades))

    if "regime" in trades.columns:
        for regime, group in trades.groupby("regime"):
            if not regime:
                continue
            edge.by_regime[str(regime)] = _slice_stats(str(regime), group)

    if "entry_time" in trades.columns:
        frame = trades.copy()
        hours = pd.DatetimeIndex(frame["entry_time"]).hour
        frame["_bucket"] = [session_bucket(int(h)) for h in hours]
        for bucket, group in frame.groupby("_bucket"):
            edge.by_session[str(bucket)] = _slice_stats(str(bucket), group)

    return edge
