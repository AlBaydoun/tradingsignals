"""Stop-loss and take-profit placement.

A stop is not a risk-tolerance setting, it is a statement about where the trade
idea is wrong. Two candidate stops are considered and the more defensible one
wins:

* **Volatility stop** — a multiple of ATR. Adapts to conditions and keeps the
  stop out of the market's ordinary noise.
* **Structure stop** — just beyond the most recent confirmed swing. If price
  trades through it, the pattern that justified the trade has broken.

The structure stop is preferred when it sits *further* away, because a stop
inside the recent swing range is a stop the market reaches by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signalforge.features import indicators as ta
from signalforge.universe import Instrument


@dataclass
class TradeLevels:
    """Complete price levels for one trade."""

    entry: float
    stop_loss: float
    take_profits: list[float]
    stop_distance_pips: float
    reward_risk: float
    # Which method set the stop, for the explanation.
    stop_basis: str
    atr: float
    warnings: list[str]

    @property
    def primary_target(self) -> float:
        return self.take_profits[0] if self.take_profits else self.entry

    @property
    def final_target(self) -> float:
        return self.take_profits[-1] if self.take_profits else self.entry

    def to_dict(self) -> dict:
        return {
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "stop_distance_pips": round(self.stop_distance_pips, 1),
            "reward_risk": round(self.reward_risk, 2),
            "stop_basis": self.stop_basis,
            "atr": round(self.atr, 6),
            "warnings": self.warnings,
        }


def compute_levels(
    df: pd.DataFrame,
    instrument: Instrument,
    direction: int,
    *,
    entry_price: float | None = None,
    sl_atr_mult: float = 1.5,
    tp_atr_mults: list[float] | None = None,
    atr_period: int = 14,
    structure_lookback: int = 20,
    structure_buffer_atr: float = 0.25,
    min_reward_risk: float = 1.2,
    use_structure: bool = True,
) -> TradeLevels:
    """Build entry, stop and targets for a trade in `direction` (+1 / -1)."""
    tp_atr_mults = tp_atr_mults or [1.0, 2.0, 3.0]
    warnings: list[str] = []

    high, low, close = df["high"], df["low"], df["close"]
    atr_series = ta.atr(high, low, close, atr_period)
    atr_value = float(atr_series.iloc[-1])
    entry = float(entry_price if entry_price is not None else close.iloc[-1])

    if not np.isfinite(atr_value) or atr_value <= 0:
        warnings.append("ATR unavailable; falling back to a 1% stop")
        atr_value = entry * 0.01

    # --- candidate 1: volatility stop -----------------------------------
    volatility_stop = entry - direction * sl_atr_mult * atr_value
    stop = volatility_stop
    basis = f"{sl_atr_mult:.1f}x ATR"

    # --- candidate 2: structure stop -------------------------------------
    if use_structure:
        window = df.tail(structure_lookback)
        buffer = structure_buffer_atr * atr_value
        if direction > 0:
            swing = float(window["low"].min())
            structure_stop = swing - buffer
            # Only adopt it if it is further away than the volatility stop.
            if structure_stop < volatility_stop and structure_stop < entry:
                stop, basis = structure_stop, "below recent swing low"
        else:
            swing = float(window["high"].max())
            structure_stop = swing + buffer
            if structure_stop > volatility_stop and structure_stop > entry:
                stop, basis = structure_stop, "above recent swing high"

    stop_distance = abs(entry - stop)
    stop_pips = stop_distance / instrument.pip_size

    # A stop inside the spread is not a stop, it is a donation.
    spread_price = instrument.typical_spread_pips * instrument.pip_size
    if stop_distance < spread_price * 3.0:
        stop = entry - direction * spread_price * 3.0
        stop_distance = abs(entry - stop)
        stop_pips = stop_distance / instrument.pip_size
        basis = "widened to clear the spread"
        warnings.append(
            "Stop was inside three spreads of entry and has been widened. "
            "This timeframe may be too fast for this instrument's costs."
        )

    # --- targets ----------------------------------------------------------
    targets = [
        instrument.round_price(entry + direction * mult * atr_value)
        for mult in tp_atr_mults
    ]

    first_reward = abs(targets[0] - entry)
    reward_risk = first_reward / stop_distance if stop_distance > 0 else 0.0

    if reward_risk < min_reward_risk:
        # Push the first target out to make the trade worth taking at all.
        needed = stop_distance * min_reward_risk
        targets[0] = instrument.round_price(entry + direction * needed)
        reward_risk = min_reward_risk
        warnings.append(
            f"First target moved out to reach the {min_reward_risk:.1f}:1 minimum."
        )
        # Keep the ladder monotonic after adjusting the first rung.
        for i in range(1, len(targets)):
            if direction > 0 and targets[i] <= targets[i - 1]:
                targets[i] = instrument.round_price(
                    targets[i - 1] + 0.5 * atr_value
                )
            elif direction < 0 and targets[i] >= targets[i - 1]:
                targets[i] = instrument.round_price(
                    targets[i - 1] - 0.5 * atr_value
                )

    return TradeLevels(
        entry=instrument.round_price(entry),
        stop_loss=instrument.round_price(stop),
        take_profits=targets,
        stop_distance_pips=stop_pips,
        reward_risk=reward_risk,
        stop_basis=basis,
        atr=atr_value,
        warnings=warnings,
    )


def breakeven_trigger(levels: TradeLevels, direction: int, fraction: float = 0.6) -> float:
    """Price at which to move the stop to entry.

    Defaults to 60% of the way to the first target — far enough that the move
    has proven itself, close enough to protect the position.
    """
    distance = abs(levels.take_profits[0] - levels.entry)
    return levels.entry + direction * distance * fraction


def trailing_stop(
    df: pd.DataFrame, direction: int, atr_mult: float = 2.0, atr_period: int = 14
) -> float:
    """A Chandelier-style trailing stop from the highest close since entry."""
    high, low, close = df["high"], df["low"], df["close"]
    atr_value = float(ta.atr(high, low, close, atr_period).iloc[-1])
    if direction > 0:
        return float(high.tail(atr_period).max()) - atr_mult * atr_value
    return float(low.tail(atr_period).min()) + atr_mult * atr_value


def expected_value(
    win_probability: float, reward_risk: float, cost_r: float = 0.0
) -> float:
    """Expectancy in R multiples.

    Positive expectancy is necessary but not sufficient — a 0.02R edge is
    indistinguishable from noise once slippage varies.
    """
    return win_probability * reward_risk - (1.0 - win_probability) * 1.0 - cost_r


def minimum_win_rate(reward_risk: float, cost_r: float = 0.0) -> float:
    """The break-even hit rate for a given reward:risk.

    At 2:1 you need 33%. At 1:1 you need 50%. Quoting a strategy's win rate
    without its reward:risk is meaningless, and this is why.
    """
    if reward_risk <= 0:
        return 1.0
    return (1.0 + cost_r) / (1.0 + reward_risk)
