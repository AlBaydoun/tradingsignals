"""Triple-barrier labelling.

Naively labelling "did price go up over the next N bars" produces a model that
predicts a number nobody can trade: it ignores the stop you would really have
been carrying, and it treats a +0.1 pip drift as a win.

The triple-barrier method instead asks the question a trader actually faces:
starting here, with this take-profit and this stop-loss, which one gets hit
first? Three barriers bound every trade — an upper (profit), a lower (stop) and
a vertical one (time). The label is which barrier was touched first.

Two refinements matter enough to implement carefully:

* **Cost gating.** A barrier narrower than the round-trip spread is unreachable
  in practice. Those bars are excluded rather than labelled, so the model is
  never rewarded for predicting a move it could not have captured.
* **Sample weighting.** Overlapping label windows are not independent
  observations. Weighting each sample by how uniquely it occupies its window
  stops the model from over-counting a single big move that touched fifty
  consecutive bars' barriers.

Reference: Marcos López de Prado, *Advances in Financial Machine Learning*, ch. 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signalforge.features import indicators as ta


@dataclass
class LabelResult:
    """Labels plus everything needed to train and validate honestly."""

    labels: pd.Series  # -1, 0, +1
    returns: pd.Series  # realised return at the barrier touch
    event_end: pd.Series  # bar *position* where the event resolved
    # The same thing as a timestamp. Positions are only valid against the frame
    # they were computed on; once rows are filtered out for training they are
    # meaningless. Timestamps survive filtering, so this is what purging uses.
    event_end_time: pd.Series
    weights: pd.Series  # sample weights (uniqueness x magnitude)
    holding_bars: pd.Series  # how many bars the event took to resolve
    tradable: pd.Series  # False where the barrier was inside the spread
    ambiguous: pd.Series  # True where both barriers fell in the same bar

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "label": self.labels,
                "label_return": self.returns,
                "event_end": self.event_end,
                "event_end_time": self.event_end_time,
                "sample_weight": self.weights,
                "holding_bars": self.holding_bars,
                "tradable": self.tradable,
                "ambiguous": self.ambiguous,
            }
        )

    def summary(self) -> dict[str, float]:
        valid = self.tradable & ~self.ambiguous
        counts = self.labels[valid].value_counts()
        total = max(1, int(valid.sum()))
        return {
            "total_bars": int(len(self.labels)),
            "tradable": int(valid.sum()),
            "excluded_by_cost": int((~self.tradable).sum()),
            "ambiguous": int(self.ambiguous.sum()),
            "pct_long": round(100.0 * counts.get(1, 0) / total, 2),
            "pct_short": round(100.0 * counts.get(-1, 0) / total, 2),
            "pct_timeout": round(100.0 * counts.get(0, 0) / total, 2),
            "mean_holding_bars": round(float(self.holding_bars[valid].mean()), 2)
            if valid.any()
            else 0.0,
        }


def _scan_barriers(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk forward from each bar until a barrier is touched.

    Returns (label, realised return, end index, ambiguous flag).

    Intrabar ambiguity is handled honestly: when a single bar's range spans both
    barriers, we cannot know from OHLC data which was touched first, so the bar
    is flagged and excluded rather than guessed. Guessing in the optimistic
    direction is the single most effective way to build a backtest that cannot
    be reproduced live.
    """
    n = len(close)
    labels = np.zeros(n, dtype="int8")
    returns = np.zeros(n, dtype="float64")
    ends = np.full(n, -1, dtype="int64")
    ambiguous = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            ends[i] = -1
            continue

        stop = i + horizon
        # Refuse to label a bar whose full horizon has not happened yet. A
        # truncated window would record "neither barrier was hit" after
        # checking only a handful of bars, biasing the tail of every training
        # set toward the timeout class — and the final bar would resolve to
        # itself with a fabricated zero return.
        if stop > n - 1:
            ends[i] = -1
            continue

        entry = close[i]
        resolved = False

        for j in range(i + 1, stop + 1):
            hit_upper = high[j] >= upper[i]
            hit_lower = low[j] <= lower[i]

            if hit_upper and hit_lower:
                ambiguous[i] = True
                labels[i] = 0
                returns[i] = 0.0
                ends[i] = j
                resolved = True
                break
            if hit_upper:
                labels[i] = 1
                returns[i] = (upper[i] - entry) / entry
                ends[i] = j
                resolved = True
                break
            if hit_lower:
                labels[i] = -1
                returns[i] = (lower[i] - entry) / entry
                ends[i] = j
                resolved = True
                break

        if not resolved:
            # The vertical barrier: time ran out, so book the drift.
            labels[i] = 0
            returns[i] = (close[stop] - entry) / entry
            ends[i] = stop

    return labels, returns, ends, ambiguous


def _uniqueness_weights(ends: np.ndarray, n: int) -> np.ndarray:
    """Down-weight samples whose label windows overlap.

    If one violent move resolves the barriers of eighty consecutive bars, those
    eighty rows are not eighty independent observations of the market — they are
    roughly one. Without this correction the model becomes wildly overconfident
    about whatever happened during the largest moves in the training set.
    """
    concurrency = np.zeros(n, dtype="float64")
    for i in range(n):
        if ends[i] < 0:
            continue
        concurrency[i : ends[i] + 1] += 1.0

    weights = np.zeros(n, dtype="float64")
    for i in range(n):
        if ends[i] < 0:
            continue
        window = concurrency[i : ends[i] + 1]
        active = window[window > 0]
        weights[i] = float(np.mean(1.0 / active)) if len(active) else 0.0
    return weights


def apply_triple_barrier(
    df: pd.DataFrame,
    *,
    upper_atr_mult: float = 1.5,
    lower_atr_mult: float = 1.5,
    horizon: int = 24,
    atr_period: int = 14,
    cost_price_units: float = 0.0,
    min_cost_multiple: float = 1.5,
    weight_by_return: bool = True,
) -> LabelResult:
    """Label every bar in `df` by which barrier its trade would have hit first.

    `cost_price_units` is the round-trip cost (spread plus commission) expressed
    in the instrument's price units. Bars whose profit barrier does not clear
    `min_cost_multiple` times that cost are marked untradable.
    """
    if df.empty or len(df) < atr_period + horizon + 5:
        empty_f = pd.Series(dtype="float64", index=df.index)
        empty_i = pd.Series(dtype="int64", index=df.index)
        empty_b = pd.Series(dtype="bool", index=df.index)
        empty_t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        return LabelResult(
            empty_i.astype("int8"), empty_f, empty_i, empty_t, empty_f,
            empty_i, empty_b, empty_b,
        )

    high, low, close = df["high"], df["low"], df["close"]
    atr_ = ta.atr(high, low, close, atr_period)

    upper = close + upper_atr_mult * atr_
    lower = close - lower_atr_mult * atr_

    labels, returns, ends, ambiguous = _scan_barriers(
        high.to_numpy(),
        low.to_numpy(),
        close.to_numpy(),
        upper.to_numpy(),
        lower.to_numpy(),
        horizon,
    )

    n = len(df)
    # A barrier the spread swallows is not a barrier.
    barrier_distance = (upper_atr_mult * atr_).to_numpy()
    tradable = barrier_distance >= (min_cost_multiple * cost_price_units)
    tradable &= ends >= 0
    tradable &= ~np.isnan(barrier_distance)

    weights = _uniqueness_weights(ends, n)
    if weight_by_return:
        # Larger realised moves carry more information than marginal ones.
        magnitude = np.abs(returns)
        scale = np.nanmean(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1.0
        weights = weights * (1.0 + magnitude / max(scale, 1e-12))

    weights[~tradable] = 0.0
    weights[ambiguous] = 0.0

    holding = np.where(ends >= 0, ends - np.arange(n), 0)

    # Timestamp form, so purging still works after rows are filtered out.
    end_times = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    resolved = ends >= 0
    end_times.loc[df.index[resolved]] = df.index[ends[resolved]]

    return LabelResult(
        labels=pd.Series(labels, index=df.index, name="label"),
        returns=pd.Series(returns, index=df.index, name="label_return"),
        event_end=pd.Series(ends, index=df.index, name="event_end"),
        event_end_time=end_times.rename("event_end_time"),
        weights=pd.Series(weights, index=df.index, name="sample_weight"),
        holding_bars=pd.Series(holding, index=df.index, name="holding_bars"),
        tradable=pd.Series(tradable, index=df.index, name="tradable"),
        ambiguous=pd.Series(ambiguous, index=df.index, name="ambiguous"),
    )


def cost_in_price_units(
    spread_pips: float, pip_size: float, slippage_pips: float = 0.5
) -> float:
    """Round-trip trading cost as a price distance.

    Spread is paid once on entry (you buy the ask, you sell the bid), and
    slippage is assumed on both sides. This is the number that decides whether a
    scalping timeframe is viable at all.
    """
    return (spread_pips + 2.0 * slippage_pips) * pip_size


def label_distribution_warning(result: LabelResult) -> str | None:
    """Flag label sets that will train a useless model.

    A wildly imbalanced or mostly-timeout label set means the barriers are
    mis-sized for the timeframe — usually too wide, so nothing resolves.
    """
    summary = result.summary()
    if summary["tradable"] < 200:
        return (
            f"Only {summary['tradable']} tradable labels — too few to train on. "
            "The spread is probably too wide for this timeframe."
        )
    if summary["pct_timeout"] > 75.0:
        return (
            f"{summary['pct_timeout']}% of labels hit the time barrier. "
            "Barriers are too wide or the horizon too short."
        )
    if summary["pct_long"] < 10.0 or summary["pct_short"] < 10.0:
        return (
            "Severe class imbalance "
            f"({summary['pct_long']}% long / {summary['pct_short']}% short)."
        )
    return None
