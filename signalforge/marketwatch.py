"""Sweeping the wider market for movement, without pretending to trade it.

The engine draws a hard line between two activities that are easy to confuse:

* **Trading** needs a trained, walk-forward-validated model. That is the
  `watchlist`, and it is deliberately small — every extra model trained makes
  the multiple-comparison correction harsher for all of them, so "train
  everything" is not a free upgrade, it is a way of guaranteeing that something
  looks good by chance.
* **Watching** needs only a price feed. Any instrument the engine can price can
  be surveyed for unusual movement, and told about.

This module is the second one. Everything it reports is explicitly *not* a
signal: it can say a market is moving abnormally, and it cannot say which way
it will go next or whether that movement is exploitable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from signalforge.config import MarketWatchConfig
from signalforge.selection import default_universe, hunt
from signalforge.universe import INSTRUMENTS

log = logging.getLogger(__name__)


@dataclass
class Alert:
    """One instrument doing something worth a human glance."""

    symbol: str
    mt5_symbol: str
    market: str
    timeframe: str
    score: float
    change_pct: float
    vol_expansion: float
    efficiency_ratio: float
    cost_ratio: float
    has_model: bool
    verdict: str
    # Why it surfaced, in plain words.
    headline: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sweep:
    """The result of one pass over the wider market."""

    at: datetime
    timeframe: str
    surveyed: int
    open_markets: int
    alerts: list[Alert]
    crypto_movers: list[dict]

    def to_dict(self) -> dict:
        return {
            "at": self.at.isoformat(),
            "timeframe": self.timeframe,
            "surveyed": self.surveyed,
            "open_markets": self.open_markets,
            "alerts": [a.to_dict() for a in self.alerts],
            "crypto_movers": self.crypto_movers,
        }

    def describe(self) -> str:
        if not self.surveyed:
            return "Market sweep found nothing to survey."
        if not self.alerts:
            return (
                f"Swept {self.surveyed} instruments ({self.open_markets} open) "
                f"on {self.timeframe}. Nothing moving unusually."
            )
        names = ", ".join(a.symbol for a in self.alerts[:4])
        return (
            f"Swept {self.surveyed} instruments ({self.open_markets} open) on "
            f"{self.timeframe}. {len(self.alerts)} worth a look: {names}"
            f"{' …' if len(self.alerts) > 4 else ''}"
        )


def symbols_for(config: MarketWatchConfig) -> list[str]:
    """Which instruments this configuration wants swept."""
    if config.mode == "none":
        return []
    if config.mode == "markets" and config.markets:
        return default_universe(config.markets)
    return sorted(INSTRUMENTS)


def _headline(result) -> str:
    """One sentence on why this instrument surfaced, in a person's words."""
    bits: list[str] = []
    if result.vol_expansion > 1.35:
        bits.append(f"volatility {result.vol_expansion:.1f}x its own median")
    elif result.vol_percentile > 80:
        bits.append(f"at the {result.vol_percentile:.0f}th percentile of its range")

    if abs(result.displacement_atr) > 2.0:
        way = "up" if result.displacement_atr > 0 else "down"
        bits.append(f"{abs(result.displacement_atr):.1f} ATR {way} over 20 bars")

    if result.efficiency_ratio >= 0.4:
        bits.append("moving directionally, not chopping")
    elif result.efficiency_ratio < 0.15:
        bits.append("but the movement is chop, not travel")

    if not bits:
        bits.append(f"scoring {result.score:.0f} on movement per unit of cost")
    return "; ".join(bits)


def sweep(
    router,
    config: MarketWatchConfig,
    *,
    trained: set[str] | None = None,
    now: datetime | None = None,
) -> Sweep:
    """Survey the wider market once.

    Failures degrade to a smaller sweep rather than raising: this runs inside a
    long-lived watch loop, and a throttled provider must not stop the signals.
    """
    now = now or datetime.now(timezone.utc)
    trained = trained or set()
    symbols = symbols_for(config)

    if not symbols:
        return Sweep(now, config.timeframe, 0, 0, [], [])

    try:
        frames = router.get_many(symbols, config.timeframe, 500)
    except Exception as exc:
        log.warning("Market sweep could not load data: %s", exc)
        return Sweep(now, config.timeframe, 0, 0, [], [])

    results = hunt(
        frames,
        config.timeframe,
        hour_utc=now.hour,
        is_weekend=now.weekday() >= 5,
        models=trained,
    )
    open_markets = sum(1 for r in results if r.market_open)

    alerts = [
        Alert(
            symbol=r.symbol,
            mt5_symbol=r.mt5_symbol,
            market=r.market,
            timeframe=r.timeframe,
            score=r.score,
            change_pct=r.change_pct,
            vol_expansion=r.vol_expansion,
            efficiency_ratio=r.efficiency_ratio,
            cost_ratio=r.cost_ratio,
            has_model=r.has_model,
            verdict=r.verdict,
            headline=_headline(r),
        )
        for r in results
        if r.market_open and r.score >= config.min_score
    ][: config.max_reported]

    movers: list[dict] = []
    if config.crypto_movers:
        try:
            movers = router.scan_crypto_movers(8)
        except Exception as exc:
            log.debug("Crypto mover scan failed: %s", exc)

    return Sweep(
        at=now,
        timeframe=config.timeframe,
        surveyed=len(results),
        open_markets=open_markets,
        alerts=alerts,
        crypto_movers=movers,
    )


def format_sweep(result: Sweep, *, watchlist: list[str] | None = None) -> str:
    """A terminal readout of one sweep."""
    watchlist = watchlist or []
    lines = [f"  {result.describe()}"]

    for alert in result.alerts:
        mark = "*" if alert.symbol in watchlist else " "
        lines.append(
            f"  {mark} {alert.symbol:10s} {alert.score:5.1f}  "
            f"{alert.change_pct:+6.2f}%  {alert.headline}"
        )
        if not alert.has_model:
            lines.append(
                "               no model — this is a market to look at, "
                "never a trade to take"
            )

    if result.crypto_movers:
        lines.append("")
        lines.append("  Market-wide crypto movers (24h, beyond the universe):")
        for mover in result.crypto_movers[:6]:
            lines.append(
                f"    {mover['symbol']:12s} {mover['price_change_pct']:+7.2f}%  "
                f"vol ${mover['quote_volume'] / 1e6:.0f}M"
            )
    return "\n".join(lines)
