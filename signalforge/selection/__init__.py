"""Instrument and timeframe selection."""

from signalforge.selection.hunter import (
    HuntResult,
    default_universe,
    describe,
    hunt,
)
from signalforge.selection.ranker import (
    Ranking,
    best_timeframe,
    cost_ratio,
    liquidity_score,
    rank_candidate,
    summarise,
    top_opportunities,
)

__all__ = [
    "Ranking",
    "rank_candidate",
    "best_timeframe",
    "top_opportunities",
    "cost_ratio",
    "liquidity_score",
    "summarise",
    "HuntResult",
    "hunt",
    "describe",
    "default_universe",
]
