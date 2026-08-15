"""Deterministic signal explanation.

Builds the "why" for a signal from the evidence, with no LLM involved. This is
the default path: it is free, instant, reproducible, and cannot hallucinate a
justification that the numbers do not support.

The LLM layer in `llm.py` sits on top of this and is strictly optional.
"""

from __future__ import annotations

import numpy as np

from signalforge.regime import RegimeState

# Deliberately not importing from `signalforge.signals`: that package's __init__
# pulls in the generator, which imports this module. Direction is a one-word
# lookup, so the dependency is not worth a circular import.
_SIDE = {1: "buy", -1: "sell"}


def multi_timeframe_agreement(
    features: dict[str, float], direction: int
) -> tuple[float, list[str]]:
    """How much the higher timeframes agree with the trade direction.

    Returns a score in -1..1 and the human-readable observations behind it.
    """
    notes: list[str] = []
    votes: list[float] = []

    for prefix in ("h1", "h4", "d1", "m30", "m15"):
        trend = features.get(f"{prefix}_trend_dir")
        if trend is None or not np.isfinite(trend):
            continue
        agrees = np.sign(trend) == np.sign(direction)
        votes.append(1.0 if agrees else -1.0)
        notes.append(
            f"{prefix.upper()} trend {'agrees' if agrees else 'disagrees'}"
        )

        adx = features.get(f"{prefix}_adx")
        if adx is not None and np.isfinite(adx) and adx > 0.25:
            votes.append(0.5 if agrees else -0.5)
            notes.append(f"{prefix.upper()} trend is strong (ADX {adx * 100:.0f})")

    if not votes:
        return 0.0, ["no higher-timeframe context available"]

    return float(np.clip(np.mean(votes), -1.0, 1.0)), notes


def describe_evidence(
    *,
    symbol: str,
    timeframe: str,
    direction: int,
    regime: RegimeState,
    features: dict[str, float],
    measured_accuracy: float | None,
    model_confidence: float,
    reward_risk: float,
    news_sentiment: float,
    anomaly_note: str = "",
    event_note: str = "",
) -> str:
    """Compose the explanation paragraph."""
    side = _SIDE.get(int(np.sign(direction)), "flat")
    parts: list[str] = []

    parts.append(
        f"{side.capitalize()} setup on {symbol} {timeframe} in a market that is "
        f"{regime.describe()}."
    )

    # --- what the model is reacting to -----------------------------------
    drivers: list[str] = []

    rsi = features.get("rsi_14")
    if rsi is not None and np.isfinite(rsi):
        if rsi > 70:
            drivers.append(f"RSI is stretched at {rsi:.0f}")
        elif rsi < 30:
            drivers.append(f"RSI is washed out at {rsi:.0f}")

    di_spread = features.get("di_spread")
    if di_spread is not None and np.isfinite(di_spread) and abs(di_spread) > 0.05:
        drivers.append(
            f"directional pressure is {'upward' if di_spread > 0 else 'downward'}"
        )

    dist = features.get("dist_ema_21_atr")
    if dist is not None and np.isfinite(dist) and abs(dist) > 1.0:
        drivers.append(
            f"price is {abs(dist):.1f} ATR "
            f"{'above' if dist > 0 else 'below'} its 21-period mean"
        )

    squeeze_bars = features.get("squeeze_duration")
    if squeeze_bars is not None and np.isfinite(squeeze_bars) and squeeze_bars > 4:
        drivers.append(f"volatility has been compressing for {squeeze_bars:.0f} bars")

    volume_z = features.get("volume_zscore")
    if volume_z is not None and np.isfinite(volume_z) and volume_z > 1.5:
        drivers.append(f"volume is {volume_z:.1f} standard deviations above normal")

    if drivers:
        parts.append("Drivers: " + "; ".join(drivers) + ".")

    # --- multi-timeframe -------------------------------------------------
    agreement, notes = multi_timeframe_agreement(features, direction)
    if notes and notes[0] != "no higher-timeframe context available":
        if agreement > 0.3:
            parts.append("Higher timeframes support the direction.")
        elif agreement < -0.3:
            parts.append(
                "Higher timeframes lean the other way — this is a counter-trend "
                "trade and should be sized accordingly."
            )
        else:
            parts.append("Higher timeframes are mixed.")

    # --- regime fit ------------------------------------------------------
    if regime.trend_following_score > 0.6 and "trend" in regime.trend:
        parts.append("Conditions favour trend continuation.")
    elif regime.mean_reversion_score > 0.6:
        parts.append("Conditions favour mean reversion rather than breakout.")
    if regime.breakout_score > 0.7:
        parts.append("A volatility expansion looks overdue.")

    # --- honesty about the odds ------------------------------------------
    if measured_accuracy is not None:
        breakeven = 1.0 / (1.0 + reward_risk)
        parts.append(
            f"Signals at this confidence have historically resolved correctly "
            f"{measured_accuracy:.0%} of the time out-of-sample, against a "
            f"{breakeven:.0%} break-even requirement at 1:{reward_risk:.1f}."
        )
    else:
        parts.append(
            "There is no measured track record at this confidence level, so the "
            "edge here is asserted by the model rather than demonstrated."
        )

    if news_sentiment > 0.3:
        parts.append("Recent headlines skew positive.")
    elif news_sentiment < -0.3:
        parts.append("Recent headlines skew negative.")

    if anomaly_note:
        parts.append(anomaly_note)
    if event_note:
        parts.append(event_note)

    return " ".join(parts)


def build_warnings(
    *,
    regime: RegimeState,
    reward_risk: float,
    measured_accuracy: float | None,
    cost_ratio: float,
    agreement: float,
    direction: int,
    event_risk_score: float,
    conversion_approximated: bool,
) -> list[str]:
    """Collect everything the user should know before clicking buy."""
    warnings: list[str] = []

    if measured_accuracy is None:
        warnings.append(
            "No measured accuracy at this confidence level — the probability is "
            "the model's own estimate, not a verified frequency."
        )
    else:
        breakeven = 1.0 / (1.0 + reward_risk)
        if measured_accuracy < breakeven + 0.03:
            warnings.append(
                f"Historical accuracy ({measured_accuracy:.0%}) is barely above the "
                f"{breakeven:.0%} break-even. The edge is within noise."
            )

    if cost_ratio < 3.0:
        warnings.append(
            f"Expected move is only {cost_ratio:.1f}x the round-trip cost. "
            "A wider-than-usual spread erases this trade."
        )

    if regime.volatility == "extreme":
        warnings.append(
            "Volatility is extreme. Stops are more likely to be gapped through "
            "than filled at the requested price."
        )
    if regime.volatility == "compressed" and regime.squeeze:
        warnings.append(
            "The market is coiled. Breaks from a squeeze often whip both ways "
            "before trending."
        )

    if agreement < -0.3:
        warnings.append("This trades against the higher-timeframe trend.")

    if event_risk_score > 0.5:
        warnings.append(
            "A scheduled economic release is approaching — expect wider spreads."
        )

    if conversion_approximated:
        warnings.append(
            "Lot size uses an approximated currency conversion. Verify the size "
            "in MT5 before executing."
        )

    return warnings


def market_summary(
    regimes: dict[str, RegimeState], tradable_count: int, total_count: int
) -> str:
    """A paragraph describing overall conditions across the watchlist."""
    if not regimes:
        return "No market data available."

    vol_counts: dict[str, int] = {}
    trend_counts: dict[str, int] = {}
    for state in regimes.values():
        vol_counts[state.volatility] = vol_counts.get(state.volatility, 0) + 1
        key = (
            "trending"
            if "trend" in state.trend
            else "ranging"
        )
        trend_counts[key] = trend_counts.get(key, 0) + 1

    dominant_vol = max(vol_counts.items(), key=lambda kv: kv[1])[0]
    trending = trend_counts.get("trending", 0)
    total = len(regimes)

    lines = [
        f"Across {total} instruments, volatility is predominantly {dominant_vol} "
        f"and {trending} of {total} are trending."
    ]

    if tradable_count == 0:
        lines.append(
            "Nothing currently clears the cost and confidence thresholds. "
            "The correct action is to wait."
        )
    else:
        lines.append(
            f"{tradable_count} of {total_count} symbol/timeframe combinations "
            "are tradable right now."
        )

    return " ".join(lines)
