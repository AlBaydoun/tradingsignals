"""Market regime classification.

A model trained across every market condition learns the average of several
incompatible behaviours. Trend-following works in expansion and bleeds in
chop; mean reversion does the opposite. So before deciding *what* to trade, the
engine decides *what kind of market it is looking at*, and lets that gate the
signal.

Two views are produced:

* A **rule-based** classification (volatility state + trend state) that a human
  can read and argue with.
* An **unsupervised** cluster from a Gaussian mixture, which catches structure
  the hand-written rules miss and feeds the model as a feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from signalforge.features import indicators as ta

log = logging.getLogger(__name__)

VOL_REGIMES = ("compressed", "normal", "elevated", "extreme")
TREND_REGIMES = (
    "strong_downtrend",
    "weak_downtrend",
    "range",
    "weak_uptrend",
    "strong_uptrend",
)


@dataclass
class RegimeState:
    """The market's condition at a point in time."""

    volatility: str
    trend: str
    cluster: int
    # 0..1 scores for how well each strategy family suits current conditions.
    trend_following_score: float
    mean_reversion_score: float
    breakout_score: float
    # Raw drivers, kept for explanation and debugging.
    atr_percentile: float
    adx: float
    efficiency_ratio: float
    hurst: float
    vol_expansion: float
    squeeze: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """A one-line human summary, used in signal explanations."""
        vol_text = {
            "compressed": "volatility compressed",
            "normal": "normal volatility",
            "elevated": "elevated volatility",
            "extreme": "extreme volatility",
        }[self.volatility]
        trend_text = self.trend.replace("_", " ")
        extra = " with a squeeze building" if self.squeeze else ""
        return f"{trend_text}, {vol_text}{extra}"


def classify_volatility(
    atr_percentile: float, vol_expansion: float, squeeze: bool
) -> str:
    """Bucket the volatility state.

    Percentile is used rather than an absolute level so a "quiet" BTC and a
    "quiet" EURUSD both land in the same bucket.
    """
    if np.isnan(atr_percentile):
        return "normal"
    if squeeze or atr_percentile < 0.20:
        return "compressed"
    if atr_percentile > 0.90 or vol_expansion > 1.8:
        return "extreme"
    if atr_percentile > 0.70 or vol_expansion > 1.3:
        return "elevated"
    return "normal"


def classify_trend(adx: float, di_spread: float, efficiency: float) -> str:
    """Bucket the trend state.

    ADX gives strength, the DI spread gives direction, and the efficiency ratio
    guards against ADX's habit of reading "strong" during a wide, choppy range.
    """
    if np.isnan(adx) or np.isnan(di_spread):
        return "range"

    strong = adx > 0.25 and abs(efficiency) > 0.35
    weak = adx > 0.18

    if di_spread > 0.05:
        if strong:
            return "strong_uptrend"
        if weak:
            return "weak_uptrend"
    elif di_spread < -0.05:
        if strong:
            return "strong_downtrend"
        if weak:
            return "weak_downtrend"
    return "range"


def _strategy_scores(
    vol_regime: str, trend_regime: str, hurst: float, squeeze: bool, efficiency: float
) -> tuple[float, float, float]:
    """Score trend-following, mean-reversion and breakout suitability.

    These are priors, not predictions. They bias which signals survive the
    filter, and the model's own probability still has to clear its threshold.
    """
    trend_score = 0.0
    reversion_score = 0.0
    breakout_score = 0.0

    if "strong" in trend_regime:
        trend_score += 0.5
    elif "weak" in trend_regime:
        trend_score += 0.25
    else:
        reversion_score += 0.35

    if vol_regime in ("elevated", "extreme"):
        trend_score += 0.2
        breakout_score += 0.25
        reversion_score -= 0.15
    elif vol_regime == "compressed":
        breakout_score += 0.35
        reversion_score += 0.2
        trend_score -= 0.15

    if squeeze:
        breakout_score += 0.3

    # Hurst above 0.5 means the series persists; below means it reverts.
    if not np.isnan(hurst):
        trend_score += float(np.clip((hurst - 0.5) * 1.5, -0.25, 0.25))
        reversion_score += float(np.clip((0.5 - hurst) * 1.5, -0.25, 0.25))

    if not np.isnan(efficiency):
        trend_score += float(np.clip((efficiency - 0.3) * 0.6, -0.2, 0.3))
        reversion_score += float(np.clip((0.3 - efficiency) * 0.6, -0.2, 0.3))

    clip = lambda v: float(np.clip(v, 0.0, 1.0))  # noqa: E731
    return clip(trend_score), clip(reversion_score), clip(breakout_score)


class RegimeDetector:
    """Rule-based regime classification plus an unsupervised cluster label."""

    def __init__(self, n_clusters: int = 4, random_state: int = 7):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gmm: GaussianMixture | None = None
        self.scaler: StandardScaler | None = None
        self._cluster_profiles: dict[int, dict[str, float]] = {}

    # -- the unsupervised half ------------------------------------------

    @staticmethod
    def _cluster_inputs(df: pd.DataFrame) -> pd.DataFrame:
        """The small, meaningful set of drivers the mixture model clusters on."""
        high, low, close = df["high"], df["low"], df["close"]
        atr_ = ta.atr(high, low, close, 14)
        adx_, plus_di, minus_di = ta.adx(high, low, close, 14)

        out = pd.DataFrame(index=df.index)
        out["log_vol"] = np.log(ta.realized_volatility(close, 20).replace(0.0, np.nan))
        out["atr_pct"] = atr_ / close
        out["adx"] = adx_
        out["efficiency"] = ta.efficiency_ratio(close, 20)
        out["vol_expansion"] = ta.realized_volatility(
            close, 10
        ) / ta.realized_volatility(close, 50).replace(0.0, np.nan)
        out["abs_di_spread"] = (plus_di - minus_di).abs()
        return out.replace([np.inf, -np.inf], np.nan)

    def fit(self, df: pd.DataFrame) -> RegimeDetector:
        """Learn the cluster structure from a symbol's own history."""
        inputs = self._cluster_inputs(df).dropna()
        if len(inputs) < 200:
            log.warning("Too little history to fit regime clusters (%d rows)", len(inputs))
            return self

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(inputs)
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=self.random_state,
            n_init=3,
            reg_covar=1e-4,
        )
        labels = self.gmm.fit_predict(scaled)

        # Record what each cluster looks like so the labels mean something.
        inputs = inputs.assign(_cluster=labels)
        self._cluster_profiles = {
            int(c): grp.drop(columns="_cluster").mean().to_dict()
            for c, grp in inputs.groupby("_cluster")
        }
        return self

    def predict_clusters(self, df: pd.DataFrame) -> pd.Series:
        """Cluster id per bar; -1 where the inputs were not yet warmed up."""
        if self.gmm is None or self.scaler is None:
            return pd.Series(-1, index=df.index, dtype="int64")

        inputs = self._cluster_inputs(df)
        valid = inputs.dropna()
        out = pd.Series(-1, index=df.index, dtype="int64")
        if valid.empty:
            return out
        labels = self.gmm.predict(self.scaler.transform(valid))
        out.loc[valid.index] = labels
        return out

    @property
    def cluster_profiles(self) -> dict[int, dict[str, float]]:
        return self._cluster_profiles

    # -- the rule-based half --------------------------------------------

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime labels for every bar — used to slice backtest results."""
        high, low, close = df["high"], df["low"], df["close"]
        atr_ = ta.atr(high, low, close, 14)
        adx_, plus_di, minus_di = ta.adx(high, low, close, 14)

        frame = pd.DataFrame(index=df.index)
        frame["atr_percentile"] = ta.percentile_rank(atr_, 252)
        frame["adx"] = adx_ / 100.0
        frame["di_spread"] = (plus_di - minus_di) / 100.0
        frame["efficiency"] = ta.efficiency_ratio(close, 20)
        frame["hurst"] = ta.hurst_exponent(close, 100)
        frame["vol_expansion"] = ta.realized_volatility(
            close, 10
        ) / ta.realized_volatility(close, 50).replace(0.0, np.nan)
        frame["squeeze"] = ta.squeeze_on(high, low, close, 20)
        frame["cluster"] = self.predict_clusters(df)

        frame["vol_regime"] = [
            classify_volatility(a, v, bool(s))
            for a, v, s in zip(
                frame["atr_percentile"], frame["vol_expansion"], frame["squeeze"]
            )
        ]
        frame["trend_regime"] = [
            classify_trend(a, d, e)
            for a, d, e in zip(frame["adx"], frame["di_spread"], frame["efficiency"])
        ]
        return frame

    def current(self, df: pd.DataFrame) -> RegimeState:
        """The regime as of the most recent completed bar."""
        frame = self.classify_series(df)
        if frame.empty:
            return RegimeState(
                "normal", "range", -1, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan,
                np.nan, False,
            )

        row = frame.iloc[-1]
        vol_regime = str(row["vol_regime"])
        trend_regime = str(row["trend_regime"])
        squeeze = bool(row["squeeze"])
        hurst = float(row["hurst"]) if pd.notna(row["hurst"]) else np.nan
        efficiency = float(row["efficiency"]) if pd.notna(row["efficiency"]) else np.nan

        trend_score, reversion_score, breakout_score = _strategy_scores(
            vol_regime, trend_regime, hurst, squeeze, efficiency
        )

        return RegimeState(
            volatility=vol_regime,
            trend=trend_regime,
            cluster=int(row["cluster"]),
            trend_following_score=trend_score,
            mean_reversion_score=reversion_score,
            breakout_score=breakout_score,
            atr_percentile=float(row["atr_percentile"])
            if pd.notna(row["atr_percentile"])
            else np.nan,
            adx=float(row["adx"]) if pd.notna(row["adx"]) else np.nan,
            efficiency_ratio=efficiency,
            hurst=hurst,
            vol_expansion=float(row["vol_expansion"])
            if pd.notna(row["vol_expansion"])
            else np.nan,
            squeeze=squeeze,
        )

    def regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime information as numeric model features."""
        frame = self.classify_series(df)
        out = pd.DataFrame(index=df.index)
        out["regime_cluster"] = frame["cluster"]
        for name in VOL_REGIMES:
            out[f"vol_regime_{name}"] = (frame["vol_regime"] == name).astype("float64")
        for name in TREND_REGIMES:
            out[f"trend_regime_{name}"] = (frame["trend_regime"] == name).astype(
                "float64"
            )
        return out
