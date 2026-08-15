"""Which pairs, and on which timeframe, are worth trading right now.

This answers two of the questions the engine exists to answer: *which market*
and *which timeframe*. Both are ranked by the same logic — expected edge after
costs, discounted by how reliable the evidence is.

The most important term is the cost ratio. A 55%-accurate model on M1 EURUSD is
worthless because the spread eats the entire expected move; the same model on
H4 is a business. Ranking without costs would put M1 at the top every time,
because that is where the most signals are.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Ranking:
    """One (symbol, timeframe) candidate, scored."""

    symbol: str
    timeframe: str
    score: float  # 0..100
    expected_r: float
    measured_accuracy: float | None
    model_accuracy: float
    cost_ratio: float  # expected move divided by round-trip cost
    regime_fit: float
    liquidity_score: float
    data_quality: float
    sample_size: int
    tradable: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def cost_ratio(atr: float, spread_pips: float, pip_size: float, tp_mult: float = 2.0) -> float:
    """How many times the round-trip cost the expected move is worth.

    Below ~3 the strategy is fighting its own broker. Below 2 it is hopeless
    regardless of predictive accuracy.
    """
    cost = spread_pips * pip_size * 2.0  # in and out
    if cost <= 0:
        return 99.0
    return float((atr * tp_mult) / cost)


def liquidity_score(market: str, hour_utc: int, trades_weekends: bool, is_weekend: bool) -> float:
    """0..1 estimate of whether this market is liquid at this moment.

    Trading FX during the Asian session on a Friday evening means wide spreads
    and thin books — the model has no idea, so this is applied on top.
    """
    if is_weekend and not trades_weekends:
        return 0.0

    if market == "crypto":
        # 24/7, but genuinely quieter outside US/EU hours.
        return 0.75 if 0 <= hour_utc < 6 else 1.0

    if market == "forex":
        if 12 <= hour_utc < 16:  # London/New York overlap
            return 1.0
        if 7 <= hour_utc < 21:
            return 0.85
        if 0 <= hour_utc < 7:  # Tokyo
            return 0.55
        return 0.35  # the illiquid gap between New York and Sydney

    if market in ("indices", "energy", "metals"):
        if 13 <= hour_utc < 21:  # US cash session
            return 1.0
        if 7 <= hour_utc < 13:
            return 0.7
        return 0.3

    return 0.6


def rank_candidate(
    *,
    symbol: str,
    timeframe: str,
    model_accuracy: float,
    measured_accuracy: float | None,
    sample_size: int,
    reward_risk: float,
    atr: float,
    spread_pips: float,
    pip_size: float,
    market: str,
    hour_utc: int,
    trades_weekends: bool,
    is_weekend: bool,
    regime_fit: float,
    data_rows: int,
    expected_rows: int,
) -> Ranking:
    """Score one symbol/timeframe combination."""
    reasons: list[str] = []

    # --- cost viability --------------------------------------------------
    ratio = cost_ratio(atr, spread_pips, pip_size, reward_risk)
    if ratio < 2.0:
        reasons.append(
            f"expected move is only {ratio:.1f}x the round-trip cost — "
            "costs dominate on this timeframe"
        )

    # --- expected value --------------------------------------------------
    # Prefer the measured hit rate; fall back to the model's, heavily discounted
    # since an uncalibrated accuracy claim is worth much less.
    if measured_accuracy is not None:
        probability = measured_accuracy
    else:
        probability = 0.5 + (model_accuracy - 0.5) * 0.5
        reasons.append("no measured track record at this confidence — discounted")

    # Costs expressed in R, so they can be subtracted from expectancy directly.
    cost_r = 1.0 / max(ratio, 0.1)
    expected_r = probability * reward_risk - (1.0 - probability) - cost_r

    # --- confidence in the estimate --------------------------------------
    # A 60% hit rate on 30 trades is a rumour; on 800 it is a fact.
    sample_confidence = float(np.clip(sample_size / 500.0, 0.0, 1.0))
    if sample_size < 100:
        reasons.append(f"only {sample_size} out-of-sample observations")

    liquidity = liquidity_score(market, hour_utc, trades_weekends, is_weekend)
    if liquidity < 0.5:
        reasons.append("thin liquidity at this hour — spreads will be wider")

    data_quality = float(np.clip(data_rows / max(expected_rows, 1), 0.0, 1.0))
    if data_quality < 0.7:
        reasons.append(f"incomplete history ({data_rows} of ~{expected_rows} bars)")

    # --- combine ---------------------------------------------------------
    # Expected value is the core; everything else scales it down. Nothing here
    # can turn a negative expectancy positive.
    base = float(np.clip(expected_r, -1.0, 2.0))
    score = 50.0 * (base + 1.0) / 3.0 * 100.0 / 50.0  # map -1..2 onto 0..100
    score *= 0.4 + 0.6 * sample_confidence
    score *= 0.5 + 0.5 * liquidity
    score *= 0.6 + 0.4 * regime_fit
    score *= 0.5 + 0.5 * data_quality
    score = float(np.clip(score, 0.0, 100.0))

    tradable = (
        expected_r > 0.0
        and ratio >= 2.0
        and liquidity >= 0.4
        and data_quality >= 0.5
        and sample_size >= 50
    )
    if not tradable and not reasons:
        reasons.append("negative expected value after costs")

    return Ranking(
        symbol=symbol,
        timeframe=timeframe,
        score=round(score, 1),
        expected_r=round(expected_r, 4),
        measured_accuracy=measured_accuracy,
        model_accuracy=round(model_accuracy, 4),
        cost_ratio=round(ratio, 2),
        regime_fit=round(regime_fit, 3),
        liquidity_score=round(liquidity, 3),
        data_quality=round(data_quality, 3),
        sample_size=sample_size,
        tradable=tradable,
        reasons=reasons,
    )


def best_timeframe(rankings: list[Ranking], symbol: str) -> Ranking | None:
    """The highest-scoring timeframe for one symbol."""
    candidates = [r for r in rankings if r.symbol == symbol and r.tradable]
    return max(candidates, key=lambda r: r.score) if candidates else None


def top_opportunities(rankings: list[Ranking], limit: int = 8) -> list[Ranking]:
    """Best tradable candidates overall, one entry per symbol.

    Deduplicating by symbol stops the list filling up with five timeframes of
    the same instrument, which is one position, not five.
    """
    tradable = sorted(
        [r for r in rankings if r.tradable], key=lambda r: r.score, reverse=True
    )
    seen: set[str] = set()
    out: list[Ranking] = []
    for ranking in tradable:
        if ranking.symbol in seen:
            continue
        seen.add(ranking.symbol)
        out.append(ranking)
        if len(out) >= limit:
            break
    return out


def summarise(rankings: list[Ranking]) -> str:
    """A one-paragraph readout of where the opportunity is."""
    if not rankings:
        return "No instruments evaluated."

    tradable = [r for r in rankings if r.tradable]
    if not tradable:
        best = max(rankings, key=lambda r: r.expected_r)
        return (
            f"Nothing currently clears the cost hurdle across "
            f"{len({r.symbol for r in rankings})} instruments. The closest is "
            f"{best.symbol} on {best.timeframe} at {best.expected_r:+.3f}R expected. "
            "Sitting out is the correct trade."
        )

    timeframes: dict[str, int] = {}
    for ranking in tradable:
        timeframes[ranking.timeframe] = timeframes.get(ranking.timeframe, 0) + 1
    favoured = max(timeframes.items(), key=lambda kv: kv[1])[0]

    best = tradable[0] if tradable else None
    return (
        f"{len(tradable)} of {len(rankings)} symbol/timeframe combinations clear "
        f"costs. {favoured} is currently the most productive timeframe. "
        f"Best candidate: {best.symbol} on {best.timeframe} "
        f"({best.expected_r:+.3f}R expected, cost ratio {best.cost_ratio:.1f}x)."
    )
