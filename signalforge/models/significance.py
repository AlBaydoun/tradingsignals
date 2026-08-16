"""Guarding against being fooled by the number of models you trained.

Train ten symbols across four timeframes and you have fitted forty models. At
the conventional 95% level you expect around two of them to look significant
purely by chance, even if every single one is worthless. Pick "the best three"
from that run and you have selected noise and given it a confidence interval.

This module makes that failure mode visible. Each model gets a p-value for the
hypothesis that its accuracy beats a coin flip, then the whole batch is
corrected together with Benjamini-Hochberg, which controls the expected
*proportion* of false discoveries rather than the probability of any single
one.

Benjamini-Hochberg is used rather than Bonferroni deliberately: Bonferroni
controls the family-wise error rate and is brutally conservative on forty
correlated tests, which would throw away genuine edges along with the noise.
FDR keeps the discovery rate usable while still pricing in the search.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

try:
    from scipy.stats import binomtest

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    HAS_SCIPY = False


@dataclass
class SignificanceResult:
    """One model's standing after the batch has been accounted for."""

    key: str
    accuracy: float
    effective_n: int
    p_value: float
    # p-value adjusted for how many models were tested alongside it.
    q_value: float
    survives_correction: bool
    naive_significant: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def demoted(self) -> bool:
        """Looked significant alone, but not once the search is priced in."""
        return self.naive_significant and not self.survives_correction


def binomial_p_value(successes: int, trials: int) -> float:
    """One-sided p-value for "this model beats a coin flip".

    Falls back to a normal approximation if scipy is unavailable; the
    difference is immaterial above ~30 trials and both are conservative below
    it.
    """
    if trials <= 0:
        return 1.0
    successes = int(np.clip(successes, 0, trials))

    if HAS_SCIPY:
        return float(binomtest(successes, trials, 0.5, alternative="greater").pvalue)

    # Normal approximation with a continuity correction.
    mean = 0.5 * trials
    sd = np.sqrt(0.25 * trials)
    if sd == 0:
        return 1.0
    z = (successes - mean - 0.5) / sd
    return float(0.5 * np.math.erfc(z / np.sqrt(2.0)))


def benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR correction.

    Returns (survives, q_values) in the original input order.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    order = np.argsort(p_values)
    ordered = np.asarray(p_values, dtype="float64")[order]

    # q_i = p_i * n / rank, made monotone from the largest p downwards so a
    # borderline test cannot be rescued by a worse one ranked below it.
    ranks = np.arange(1, n + 1)
    raw_q = ordered * n / ranks
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotone_q = np.clip(monotone_q, 0.0, 1.0)

    survives_ordered = monotone_q <= alpha

    survives = [False] * n
    q_values = [1.0] * n
    for position, original_index in enumerate(order):
        survives[original_index] = bool(survives_ordered[position])
        q_values[original_index] = float(monotone_q[position])

    return survives, q_values


def assess_batch(
    models: list[dict], alpha: float = 0.05
) -> list[SignificanceResult]:
    """Score a whole training run together.

    Each entry needs `key`, `accuracy` and `effective_samples`.
    """
    if not models:
        return []

    p_values: list[float] = []
    for entry in models:
        effective_n = max(0, int(entry.get("effective_samples", 0)))
        accuracy = float(entry.get("accuracy", 0.5))
        successes = int(round(accuracy * effective_n))
        p_values.append(binomial_p_value(successes, effective_n))

    survives, q_values = benjamini_hochberg(p_values, alpha)

    return [
        SignificanceResult(
            key=entry.get("key", "?"),
            accuracy=float(entry.get("accuracy", 0.5)),
            effective_n=int(entry.get("effective_samples", 0)),
            p_value=round(p_values[i], 5),
            q_value=round(q_values[i], 5),
            survives_correction=survives[i],
            naive_significant=p_values[i] <= alpha,
        )
        for i, entry in enumerate(models)
    ]


def describe_batch(results: list[SignificanceResult], alpha: float = 0.05) -> str:
    """A short readout of what surviving the correction means for this run."""
    if not results:
        return "No models to assess."

    total = len(results)
    naive = sum(1 for r in results if r.naive_significant)
    survivors = sum(1 for r in results if r.survives_correction)
    demoted = [r for r in results if r.demoted]

    lines = [
        f"Tested {total} models. {naive} clear the naive 5% bar; "
        f"{survivors} survive Benjamini-Hochberg at alpha={alpha}."
    ]

    if demoted:
        names = ", ".join(r.key for r in demoted[:5])
        lines.append(
            f"{len(demoted)} model(s) looked significant alone but not once the "
            f"size of the search is priced in: {names}."
        )

    if survivors == 0:
        expected_false = total * alpha
        lines.append(
            f"Nothing survives. With {total} models tested you would expect "
            f"about {expected_false:.1f} to look good by chance alone, so any "
            "apparent winner here is most likely noise."
        )
    return " ".join(lines)


def minimum_accuracy_for_significance(
    effective_n: int, alpha: float = 0.05, n_tests: int = 1
) -> float:
    """The accuracy a model needs to clear the bar at this sample size.

    Useful before training: at 90 effective observations against 40 tested
    models, a model needs roughly 62% accuracy to mean anything. That is worth
    knowing in advance rather than discovering afterwards.
    """
    if effective_n <= 0:
        return 1.0
    # Šidák-adjusted per-test level, a reasonable stand-in for the FDR
    # threshold when planning rather than evaluating.
    adjusted = 1.0 - (1.0 - alpha) ** (1.0 / max(n_tests, 1))
    z = _normal_quantile(1.0 - adjusted)
    return float(min(1.0, 0.5 + z * np.sqrt(0.25 / effective_n)))


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > high:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
