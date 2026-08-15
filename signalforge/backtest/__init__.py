"""Cost-aware backtesting and performance measurement."""

from signalforge.backtest.engine import (
    Backtester,
    BacktestConfig,
    Position,
    signals_from_model,
    walk_forward_backtest,
)
from signalforge.backtest.metrics import PerformanceReport, by_hour, by_regime, compute

__all__ = [
    "Backtester",
    "BacktestConfig",
    "Position",
    "walk_forward_backtest",
    "signals_from_model",
    "PerformanceReport",
    "compute",
    "by_regime",
    "by_hour",
]
