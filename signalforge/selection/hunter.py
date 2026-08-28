"""The hunt: which markets are worth *looking* at right now.

This answers a different question from `ranker.py`. The ranker scores a trained
model's expected value on an instrument it already knows. The hunter runs
before any of that, on price data alone, across a universe far wider than the
watchlist, and answers: **where is there enough movement, cheaply enough, to be
worth the cost of training and watching a model at all?**

Three things decide that, and only three:

* **Heat** — is this market moving unusually for *itself*? Compared against its
  own ATR history, never against another instrument's. Gold moving 1% and
  EURUSD moving 1% are not the same event.
* **Cost efficiency** — how many round-trip spreads does one ATR pay for? This
  is a multiplier on everything else, not one term among several, because a
  market whose typical move does not clear its own spread is untradable at any
  level of excitement.
* **Directionality** — is the movement going somewhere, or is it chop? A market
  with a huge ATR and an efficiency ratio of 0.05 is expensive noise.

**What this does not tell you.** A high score means the market is *worth
studying*, not that its direction is predictable. Volatility is a necessary
condition for profit, never a sufficient one. Only `train` and `backtest` can
say whether there is an edge, and most of the time they say there is not.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from signalforge.features import indicators as ta
from signalforge.selection.ranker import liquidity_score
from signalforge.universe import INSTRUMENTS, Instrument, get_instrument

# Below this, one ATR does not pay for the round trip twice over and the
# instrument is dropped outright rather than ranked low.
MIN_COST_RATIO = 1.5
# The ratio at which cost stops holding the score back at all.
GOOD_COST_RATIO = 6.0


@dataclass
class HuntResult:
    """One instrument, surveyed."""

    symbol: str
    mt5_symbol: str
    market: str
    timeframe: str
    score: float  # 0..100
    atr_pct: float  # ATR as a percentage of price — raw movement
    atr_pips: float
    cost_ratio: float  # one ATR divided by the round-trip cost
    spread_pips: float
    vol_percentile: float  # where current ATR sits in its own history, 0..100
    vol_expansion: float  # current ATR over its median — >1 means waking up
    efficiency_ratio: float  # 0..1, directional travel over total travel
    displacement_atr: float  # signed 20-bar move in ATR units
    change_pct: float  # 20-bar percentage move
    liquidity: float
    has_model: bool
    tradable_cost: bool
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def direction_label(self) -> str:
        if self.displacement_atr > 1.0:
            return "up"
        if self.displacement_atr < -1.0:
            return "down"
        return "flat"


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    value = series.iloc[-1]
    return default if pd.isna(value) else float(value)


def cost_multiplier(ratio: float) -> float:
    """0 when costs eat the move, 1 when they are irrelevant.

    Deliberately a multiplier rather than a weighted term: no amount of
    volatility rescues an instrument whose spread is the size of its range.
    """
    if ratio <= MIN_COST_RATIO:
        return 0.0
    span = GOOD_COST_RATIO - MIN_COST_RATIO
    return float(np.clip((ratio - MIN_COST_RATIO) / span, 0.0, 1.0))


def evaluate(
    symbol: str,
    frame: pd.DataFrame,
    timeframe: str,
    *,
    instrument: Instrument | None = None,
    spread_pips: float | None = None,
    hour_utc: int = 12,
    is_weekend: bool = False,
    has_model: bool = False,
    lookback: int = 250,
) -> HuntResult | None:
    """Score one instrument. Returns None if there is not enough data."""
    if frame is None or len(frame) < 60:
        return None

    inst = instrument or get_instrument(symbol)
    spread = float(spread_pips if spread_pips is not None else inst.typical_spread_pips)

    close = frame["close"].astype(float)
    atr_series = ta.atr(frame["high"], frame["low"], close, 14).dropna()
    if atr_series.empty:
        return None

    atr_now = float(atr_series.iloc[-1])
    price = float(close.iloc[-1])
    if atr_now <= 0 or price <= 0:
        return None

    reasons: list[str] = []

    # --- movement, measured against this instrument's own history ---------
    window = atr_series.tail(lookback)
    median_atr = float(window.median())
    expansion = atr_now / median_atr if median_atr > 0 else 1.0
    percentile = float((window <= atr_now).mean() * 100.0)
    atr_pct = atr_now / price * 100.0
    atr_pips = atr_now / inst.pip_size

    # --- cost -------------------------------------------------------------
    round_trip = spread * inst.pip_size * 2.0
    ratio = atr_now / round_trip if round_trip > 0 else 99.0
    cost_factor = cost_multiplier(ratio)
    if ratio < MIN_COST_RATIO:
        reasons.append(
            f"one ATR ({atr_pips:.0f} pips) does not clear the {spread:.1f}-pip "
            "round trip — untradable at this spread whatever it does"
        )
    elif ratio < 3.0:
        reasons.append(f"thin cost margin: one ATR is only {ratio:.1f} round trips")

    # --- is the movement going anywhere? ----------------------------------
    er = _safe_last(ta.efficiency_ratio(close, 20))
    displacement = float((close.iloc[-1] - close.iloc[-21]) / atr_now) if len(close) > 21 else 0.0
    change_pct = (
        float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0) if len(close) > 21 else 0.0
    )
    if er < 0.2:
        reasons.append(f"movement is choppy (efficiency {er:.2f}) — range, not trend")

    liquidity = liquidity_score(inst.market, hour_utc, inst.trades_weekends, is_weekend)
    if liquidity < 0.4:
        reasons.append("closed or illiquid at this hour — spreads will be worse")

    # --- combine ----------------------------------------------------------
    heat = 0.5 * (percentile / 100.0) + 0.5 * float(
        np.clip((expansion - 0.8) / 0.8, 0.0, 1.0)
    )
    direction = 0.6 * float(np.clip(er / 0.5, 0.0, 1.0)) + 0.4 * float(
        np.clip(abs(displacement) / 3.0, 0.0, 1.0)
    )
    blend = 0.40 * heat + 0.35 * direction + 0.25 * liquidity
    score = float(np.clip(100.0 * cost_factor * blend, 0.0, 100.0))

    if expansion > 1.4:
        reasons.append(f"volatility is {expansion:.1f}x its own median — expanding")
    elif expansion < 0.7:
        reasons.append(f"volatility is {expansion:.1f}x its median — compressed and quiet")

    if not has_model:
        reasons.append("no trained model yet — this is a candidate, not a signal")

    return HuntResult(
        symbol=symbol,
        mt5_symbol=inst.mt5_symbol,
        market=inst.market,
        timeframe=timeframe,
        score=round(score, 1),
        atr_pct=round(atr_pct, 3),
        atr_pips=round(atr_pips, 1),
        cost_ratio=round(ratio, 2),
        spread_pips=spread,
        vol_percentile=round(percentile, 1),
        vol_expansion=round(expansion, 2),
        efficiency_ratio=round(er, 3),
        displacement_atr=round(displacement, 2),
        change_pct=round(change_pct, 2),
        liquidity=round(liquidity, 2),
        has_model=has_model,
        tradable_cost=ratio >= MIN_COST_RATIO,
        verdict=_verdict(score, ratio, expansion, er),
        reasons=reasons,
    )


def _verdict(score: float, ratio: float, expansion: float, er: float) -> str:
    if ratio < MIN_COST_RATIO:
        return "costs dominate — skip"
    if score >= 55.0 and expansion > 1.2 and er >= 0.3:
        return "hot and directional — train and watch this"
    if score >= 55.0:
        return "active and affordable — worth training"
    if score >= 35.0:
        return "moderate — usable, not exciting"
    if expansion < 0.75:
        return "coiled and quiet — watch for expansion"
    return "little on offer right now"


def hunt(
    frames: dict[str, pd.DataFrame],
    timeframe: str,
    *,
    hour_utc: int = 12,
    is_weekend: bool = False,
    spreads: dict[str, float] | None = None,
    models: set[str] | None = None,
    limit: int | None = None,
) -> list[HuntResult]:
    """Survey a universe of instruments and rank them, best first."""
    spreads = spreads or {}
    models = models or set()

    results: list[HuntResult] = []
    for symbol, frame in frames.items():
        try:
            result = evaluate(
                symbol,
                frame,
                timeframe,
                spread_pips=spreads.get(symbol),
                hour_utc=hour_utc,
                is_weekend=is_weekend,
                has_model=symbol in models,
            )
        except (KeyError, ValueError, IndexError):
            continue
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit] if limit else results


def default_universe(markets: list[str] | None = None) -> list[str]:
    """Everything the engine knows how to price, optionally by market."""
    if not markets:
        return sorted(INSTRUMENTS)
    wanted = {m.lower() for m in markets}
    return sorted(s for s, i in INSTRUMENTS.items() if i.market in wanted)


def describe(results: list[HuntResult]) -> str:
    """A short honest readout of what the survey found."""
    if not results:
        return "No instruments could be surveyed — check data availability."

    affordable = [r for r in results if r.tradable_cost]
    if not affordable:
        return (
            f"All {len(results)} instruments surveyed are currently priced out: "
            "their typical move does not clear their own spread on this "
            "timeframe. Try a higher timeframe, where the move is larger and "
            "the spread is unchanged."
        )

    hot = [r for r in affordable if r.vol_expansion > 1.2]
    best = results[0]
    parts = [
        f"{len(affordable)} of {len(results)} instruments clear their own costs "
        f"on {best.timeframe}.",
    ]
    if hot:
        names = ", ".join(r.symbol for r in hot[:4])
        parts.append(f"{len(hot)} are in expanding volatility ({names}).")
    else:
        parts.append("None are in expanding volatility — the whole survey is quiet.")
    parts.append(
        f"Top candidate: {best.symbol} on {best.timeframe} — {best.verdict}."
    )
    parts.append(
        "This ranks where movement is worth its cost. It does not claim the "
        "direction is predictable; only training and an out-of-sample backtest "
        "can say that."
    )
    return " ".join(parts)
