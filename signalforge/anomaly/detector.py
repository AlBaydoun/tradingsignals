"""Explosive-move detection — the "something is happening right now" alert.

Two different questions, deliberately kept apart:

* **Ignition**: is this market moving abnormally *right now*? Detected from
  volume spikes, price velocity in ATR units, and range expansion.
* **Coiling**: is this market compressing in a way that historically precedes a
  violent move? Detected from squeezes, narrowing range, and volume drying up.

Coiling tells you *that* a break is coming, never *which way*. Any code that
claims otherwise is fitting noise. The engine treats a coil as a reason to
watch and to size down, not as a directional signal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from signalforge.features import indicators as ta


@dataclass
class AnomalyReport:
    """What is unusual about a market at this moment."""

    symbol: str
    timeframe: str
    # 0-100. Above ~70 means a genuine outlier versus the symbol's own history.
    ignition_score: float
    coiling_score: float
    direction: int  # +1 up, -1 down, 0 undetermined
    # The individual drivers, so a human can see why the score fired.
    volume_zscore: float
    price_velocity_atr: float
    range_expansion: float
    consecutive_run: float
    squeeze_bars: float
    atr_percentile: float
    price_change_pct: float
    triggers: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_igniting(self) -> bool:
        return self.ignition_score >= 65.0

    @property
    def is_coiling(self) -> bool:
        return self.coiling_score >= 65.0

    def describe(self) -> str:
        if self.is_igniting:
            way = "upward" if self.direction > 0 else "downward" if self.direction < 0 else ""
            return (
                f"{self.symbol} is moving abnormally {way} on {self.timeframe} "
                f"({', '.join(self.triggers) or 'multiple factors'})"
            )
        if self.is_coiling:
            return (
                f"{self.symbol} is compressing on {self.timeframe} — "
                f"{int(self.squeeze_bars)} bars of squeeze, expansion likely "
                f"(direction unknown)"
            )
        return f"{self.symbol} is behaving normally on {self.timeframe}"


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if value is not None and np.isfinite(value) else default


def ignition_series(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Per-bar ignition drivers, for backtesting the detector itself."""
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    atr_ = ta.atr(high, low, close, atr_period)
    atr_safe = atr_.replace(0.0, np.nan)

    out = pd.DataFrame(index=df.index)
    # How far price moved in the last 3 bars, measured in ATRs.
    out["velocity_atr"] = (close - close.shift(3)) / atr_safe
    out["abs_velocity_atr"] = out["velocity_atr"].abs()
    out["volume_z"] = ta.volume_zscore(volume, 50) if volume.sum() > 0 else 0.0
    rng = high - low
    out["range_expansion"] = rng / rng.rolling(20, min_periods=10).mean().replace(
        0.0, np.nan
    )
    out["atr_percentile"] = ta.percentile_rank(atr_, 252)
    out["bar_return_atr"] = (close - df["open"]) / atr_safe
    return out


def coiling_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar compression drivers."""
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    out = pd.DataFrame(index=df.index)
    squeeze = ta.squeeze_on(high, low, close, 20)
    out["squeeze"] = squeeze
    # Consecutive bars inside the squeeze — longer coils release harder.
    out["squeeze_bars"] = squeeze.groupby((squeeze == 0).cumsum()).cumsum()
    out["bb_width_pct"] = ta.percentile_rank(ta.bollinger_width(close, 20), 252)
    out["atr_percentile"] = ta.percentile_rank(ta.atr(high, low, close, 14), 252)
    out["range_contraction"] = 1.0 / (
        (high - low) / (high - low).rolling(20, min_periods=10).mean()
    ).replace(0.0, np.nan)
    if volume.sum() > 0:
        out["volume_dryup"] = 1.0 / ta.relative_volume(volume, 50).replace(0.0, np.nan)
    else:
        out["volume_dryup"] = 1.0
    return out


def detect(
    df: pd.DataFrame, symbol: str, timeframe: str, atr_period: int = 14
) -> AnomalyReport:
    """Score the most recent bar for ignition and coiling."""
    if df.empty or len(df) < 60:
        return AnomalyReport(
            symbol, timeframe, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, ["insufficient_data"]
        )

    ign = ignition_series(df, atr_period).iloc[-1]
    coil = coiling_series(df).iloc[-1]

    volume_z = _safe(ign["volume_z"])
    velocity = _safe(ign["velocity_atr"])
    abs_velocity = abs(velocity)
    range_exp = _safe(ign["range_expansion"], 1.0)
    atr_pct = _safe(ign["atr_percentile"], 0.5)

    close = df["close"]
    change_pct = float((close.iloc[-1] / close.iloc[-min(20, len(close))] - 1.0) * 100.0)

    # Consecutive same-direction bars.
    direction_signs = np.sign(close.diff().tail(10).to_numpy())
    run = 0.0
    for sign in reversed(direction_signs):
        if sign == 0 or (run != 0 and np.sign(run) != sign):
            break
        run += sign

    triggers: list[str] = []

    # --- ignition score ------------------------------------------------
    # Each component is capped so that no single extreme reading can carry the
    # score on its own; a real ignition shows up in several at once.
    score = 0.0
    if volume_z > 2.0:
        score += min(30.0, 10.0 * (volume_z - 1.0))
        triggers.append(f"volume {volume_z:.1f}σ above normal")
    if abs_velocity > 1.5:
        score += min(30.0, 12.0 * (abs_velocity - 0.5))
        triggers.append(f"{abs_velocity:.1f} ATR move in 3 bars")
    if range_exp > 1.8:
        score += min(20.0, 10.0 * (range_exp - 1.0))
        triggers.append(f"range {range_exp:.1f}x normal")
    if atr_pct > 0.85:
        score += 10.0
        triggers.append("volatility in top 15% of its range")
    if abs(run) >= 4:
        score += min(10.0, 2.5 * abs(run))
        triggers.append(f"{int(abs(run))} consecutive bars in one direction")

    ignition_score = float(np.clip(score, 0.0, 100.0))

    # --- coiling score -------------------------------------------------
    coil_score = 0.0
    squeeze_bars = _safe(coil["squeeze_bars"])
    if squeeze_bars > 0:
        coil_score += min(40.0, 4.0 * squeeze_bars)
    bb_pct = _safe(coil["bb_width_pct"], 0.5)
    if bb_pct < 0.2:
        coil_score += 25.0 * (1.0 - bb_pct / 0.2)
    coil_atr_pct = _safe(coil["atr_percentile"], 0.5)
    if coil_atr_pct < 0.25:
        coil_score += 20.0 * (1.0 - coil_atr_pct / 0.25)
    if _safe(coil["volume_dryup"], 1.0) > 1.4:
        coil_score += 15.0

    coiling_score = float(np.clip(coil_score, 0.0, 100.0))

    # An ignition in progress is by definition no longer a coil.
    if ignition_score > 50.0:
        coiling_score *= 0.3

    direction = int(np.sign(velocity)) if abs_velocity > 0.5 else 0

    return AnomalyReport(
        symbol=symbol,
        timeframe=timeframe,
        ignition_score=ignition_score,
        coiling_score=coiling_score,
        direction=direction,
        volume_zscore=volume_z,
        price_velocity_atr=velocity,
        range_expansion=range_exp,
        consecutive_run=run,
        squeeze_bars=squeeze_bars,
        atr_percentile=atr_pct,
        price_change_pct=change_pct,
        triggers=triggers,
    )


def scan(
    frames: dict[str, pd.DataFrame], timeframe: str, min_score: float = 60.0
) -> list[AnomalyReport]:
    """Score a whole watchlist and return only what is genuinely unusual."""
    reports = [detect(df, sym, timeframe) for sym, df in frames.items() if not df.empty]
    interesting = [
        r for r in reports if r.ignition_score >= min_score or r.coiling_score >= min_score
    ]
    interesting.sort(
        key=lambda r: max(r.ignition_score, r.coiling_score), reverse=True
    )
    return interesting


def anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly drivers as model features.

    The model gets to decide for itself whether a volume spike is bullish; the
    scores above are for the human-facing alert, these are for training.
    """
    ign = ignition_series(df)
    coil = coiling_series(df)
    out = pd.DataFrame(index=df.index)
    out["ign_velocity_atr"] = ign["velocity_atr"]
    out["ign_volume_z"] = ign["volume_z"]
    out["ign_range_expansion"] = ign["range_expansion"]
    out["coil_squeeze_bars"] = coil["squeeze_bars"]
    out["coil_bb_width_pct"] = coil["bb_width_pct"]
    out["coil_volume_dryup"] = coil["volume_dryup"]
    return out
