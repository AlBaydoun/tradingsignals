"""
SignalForge — an adaptive, self-learning market signal engine for MetaTrader 5.

The engine reads price history, live volatility, cross-market structure, the
economic calendar and the news flow, then emits ranked, risk-sized trade
signals that can be executed by hand on the MT5 mobile app.

Nothing here is investment advice, and no configuration of this software can
make trading profitable on its own. Every probability the engine reports is an
estimate measured out-of-sample on historical data; the future is allowed to
disagree with it. Read `docs/HONEST_LIMITATIONS.md` before risking money.
"""

__version__ = "0.1.0"

from typing import Any

from signalforge.config import Config, load_config

__all__ = ["Config", "load_config", "Signal", "SignalBundle", "__version__"]

# Signal types are exposed lazily so that importing `signalforge.data` does not
# drag in LightGBM and the whole model stack.
_LAZY = {"Signal": "signalforge.signals.schema", "SignalBundle": "signalforge.signals.schema"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
