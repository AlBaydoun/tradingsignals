"""Market regime classification."""

from signalforge.regime.detector import (
    RegimeDetector,
    RegimeState,
    TREND_REGIMES,
    VOL_REGIMES,
    classify_trend,
    classify_volatility,
)

__all__ = [
    "RegimeDetector",
    "RegimeState",
    "VOL_REGIMES",
    "TREND_REGIMES",
    "classify_volatility",
    "classify_trend",
]
