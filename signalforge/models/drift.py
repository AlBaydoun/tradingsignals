"""Drift detection — noticing when the model's world has changed.

A model trained on 2024's market will keep emitting confident signals in a 2026
market that behaves nothing like it. Nothing in the model itself will complain.
Two independent alarms are wired up here:

* **Feature drift** (Population Stability Index): the *inputs* have moved away
  from the training distribution. This fires early, before losses accumulate.
* **Performance drift**: the realised hit rate of recent closed trades has
  fallen below what the model's own reliability table promised. This fires
  late, but it is the one that actually matters.

Feature drift is a warning. Performance drift disables the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    psi_overall: float
    psi_by_feature: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)
    live_hit_rate: float | None = None
    expected_hit_rate: float | None = None
    performance_gap: float | None = None
    consecutive_losses: int = 0
    action: str = "none"  # none | warn | retrain | disable
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference and a current distribution.

    Rules of thumb: below 0.1 is stable, 0.1-0.25 is a moderate shift worth
    watching, above 0.25 means the feature is behaving differently enough that
    the model's learned relationship may no longer hold.
    """
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if len(baseline) < 50 or len(current) < 20:
        return 0.0

    # Quantile edges from the baseline so each reference bucket is equally full.
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(baseline, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    # A tiny floor avoids a divide-by-zero turning into an infinite PSI when a
    # bucket empties out entirely.
    base_pct = np.clip(base_counts / max(base_counts.sum(), 1), 1e-6, None)
    curr_pct = np.clip(curr_counts / max(curr_counts.sum(), 1), 1e-6, None)

    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    threshold: float = 0.25,
    max_features: int = 60,
) -> tuple[float, dict[str, float], list[str]]:
    """PSI across the feature matrix.

    Only the first `max_features` shared columns are checked — PSI on 150
    columns is slow and the signal saturates well before that.
    """
    shared = [c for c in baseline.columns if c in current.columns][:max_features]
    scores: dict[str, float] = {}

    for col in shared:
        try:
            scores[col] = population_stability_index(
                baseline[col].to_numpy(dtype="float64"),
                current[col].to_numpy(dtype="float64"),
            )
        except Exception:
            continue

    if not scores:
        return 0.0, {}, []

    drifted = [c for c, v in scores.items() if v > threshold]
    overall = float(np.mean(list(scores.values())))
    return overall, scores, drifted


def performance_drift(
    outcomes: list[bool],
    expected_hit_rate: float,
    *,
    window: int = 50,
    min_hit_rate: float = 0.40,
    max_consecutive_losses: int = 12,
) -> tuple[float | None, int, list[str]]:
    """Compare realised results against what the model promised.

    `outcomes` is a chronological list of closed trades, True for a win.
    """
    reasons: list[str] = []
    if not outcomes:
        return None, 0, reasons

    recent = outcomes[-window:]
    hit_rate = float(np.mean(recent))

    consecutive = 0
    for won in reversed(outcomes):
        if won:
            break
        consecutive += 1

    if len(recent) >= 20:
        if hit_rate < min_hit_rate:
            reasons.append(
                f"live hit rate {hit_rate:.1%} over the last {len(recent)} trades "
                f"is below the {min_hit_rate:.0%} floor"
            )
        # A gap this wide means the calibration is no longer describing reality.
        if expected_hit_rate and hit_rate < expected_hit_rate - 0.15:
            reasons.append(
                f"live hit rate {hit_rate:.1%} is far below the calibrated "
                f"expectation of {expected_hit_rate:.1%}"
            )

    if consecutive >= max_consecutive_losses:
        reasons.append(f"{consecutive} consecutive losing trades")

    return hit_rate, consecutive, reasons


def assess(
    baseline_features: pd.DataFrame | None,
    current_features: pd.DataFrame | None,
    outcomes: list[bool],
    expected_hit_rate: float | None,
    *,
    psi_threshold: float = 0.25,
    window: int = 50,
    min_hit_rate: float = 0.40,
    max_consecutive_losses: int = 12,
) -> DriftReport:
    """Combine both alarms into a single recommended action."""
    psi_overall, psi_by_feature, drifted = 0.0, {}, []
    if baseline_features is not None and current_features is not None:
        psi_overall, psi_by_feature, drifted = feature_drift(
            baseline_features, current_features, threshold=psi_threshold
        )

    hit_rate, consecutive, reasons = performance_drift(
        outcomes,
        expected_hit_rate or 0.0,
        window=window,
        min_hit_rate=min_hit_rate,
        max_consecutive_losses=max_consecutive_losses,
    )

    action = "none"
    if reasons:
        # Live losses outrank any statistical comfort from the feature check.
        action = "disable"
    elif psi_overall > psi_threshold or len(drifted) > 8:
        action = "retrain"
        reasons.append(
            f"feature distribution has shifted (PSI {psi_overall:.3f}, "
            f"{len(drifted)} features beyond threshold)"
        )
    elif psi_overall > psi_threshold * 0.6:
        action = "warn"
        reasons.append(f"feature drift building (PSI {psi_overall:.3f})")

    gap = None
    if hit_rate is not None and expected_hit_rate:
        gap = round(hit_rate - expected_hit_rate, 4)

    return DriftReport(
        psi_overall=round(psi_overall, 4),
        psi_by_feature={k: round(v, 4) for k, v in psi_by_feature.items()},
        drifted_features=drifted,
        live_hit_rate=round(hit_rate, 4) if hit_rate is not None else None,
        expected_hit_rate=expected_hit_rate,
        performance_gap=gap,
        consecutive_losses=consecutive,
        action=action,
        reasons=reasons,
    )
