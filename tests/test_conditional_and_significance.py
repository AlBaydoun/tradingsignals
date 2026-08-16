"""Tests for the two enhancements that guard against self-deception.

`conditional.py` stops a model trading in conditions where it has historically
lost money. `significance.py` stops a batch of models producing a winner by
sheer number of attempts. Both are defences against the same failure: believing
a number that arose from the search rather than from the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.models.conditional import (
    ConditionalEdge,
    build_conditional_edge,
    session_bucket,
)
from signalforge.models.significance import (
    assess_batch,
    benjamini_hochberg,
    binomial_p_value,
    describe_batch,
    minimum_accuracy_for_significance,
)


def make_trades(spec: dict[str, tuple[int, float]], hour: int = 13) -> pd.DataFrame:
    """Build a trade log: regime -> (n_trades, win_rate)."""
    rows = []
    timestamp = pd.Timestamp("2025-01-01 00:00", tz="UTC").replace(hour=hour)
    for regime, (count, win_rate) in spec.items():
        wins = int(round(count * win_rate))
        for i in range(count):
            won = i < wins
            rows.append(
                {
                    # Minutes, not hours: stepping by hours would scatter the
                    # trades across every session bucket and defeat the point.
                    "entry_time": timestamp + pd.Timedelta(minutes=i),
                    "regime": regime,
                    # 2R wins against 1R losses.
                    "pnl": 200.0 if won else -100.0,
                    "r_multiple": 2.0 if won else -1.0,
                }
            )
    return pd.DataFrame(rows)


class TestConditionalEdge:
    def test_measures_per_regime_performance(self):
        trades = make_trades({"strong_uptrend": (60, 0.7), "range": (60, 0.2)})
        edge = build_conditional_edge(trades)

        assert edge.by_regime["strong_uptrend"].profit_factor > 1.0
        assert edge.by_regime["range"].profit_factor < 1.0
        assert edge.total_trades == 120

    def test_blocks_a_losing_regime(self):
        trades = make_trades({"strong_uptrend": (60, 0.7), "range": (60, 0.2)})
        edge = build_conditional_edge(trades)

        allowed, reason = edge.regime_verdict("range")
        assert not allowed
        assert "lost money" in reason
        assert "range" in reason

    def test_allows_a_profitable_regime(self):
        trades = make_trades({"strong_uptrend": (60, 0.7), "range": (60, 0.2)})
        edge = build_conditional_edge(trades)

        allowed, reason = edge.regime_verdict("strong_uptrend")
        assert allowed
        assert "profitable" in reason

    def test_thin_slices_cannot_veto(self):
        """A losing regime with too few trades must not start making rules."""
        trades = make_trades({"range": (8, 0.0)})  # 8 straight losses
        edge = build_conditional_edge(trades, min_trades=25)

        allowed, reason = edge.regime_verdict("range")
        assert allowed, "8 trades should not be enough to block a regime"
        assert "unmeasured" in reason

    def test_unseen_regime_is_allowed(self):
        """Absence of evidence is not evidence of failure."""
        edge = build_conditional_edge(make_trades({"strong_uptrend": (60, 0.7)}))
        allowed, reason = edge.regime_verdict("weak_downtrend")
        assert allowed
        assert reason == ""

    def test_empty_trade_log_gates_nothing(self):
        edge = build_conditional_edge(pd.DataFrame())
        assert edge.regime_verdict("range")[0]
        assert edge.session_verdict(13)[0]
        assert edge.total_trades == 0

    def test_session_gate_blocks_a_losing_block(self):
        trades = make_trades({"range": (60, 0.15)}, hour=17)
        edge = build_conditional_edge(trades)

        allowed, reason = edge.session_verdict(17)
        assert not allowed
        assert "ny afternoon" in reason

        # A different session has no data and must stay open.
        assert edge.session_verdict(2)[0]

    def test_session_buckets_cover_the_clock(self):
        assert session_bucket(0) == "asia_early"
        assert session_bucket(13) == "london_ny_overlap"
        assert session_bucket(23) == "ny_close"
        for hour in range(24):
            assert session_bucket(hour) != "unknown"

    def test_survives_a_round_trip_through_json(self):
        """The map is persisted with the model, so it must serialise."""
        original = build_conditional_edge(
            make_trades({"strong_uptrend": (60, 0.7), "range": (60, 0.2)})
        )
        restored = ConditionalEdge.from_dict(original.to_dict())

        assert restored.total_trades == original.total_trades
        assert not restored.regime_verdict("range")[0]
        assert restored.regime_verdict("strong_uptrend")[0]

    def test_describe_names_both_sides(self):
        edge = build_conditional_edge(
            make_trades({"strong_uptrend": (60, 0.7), "range": (60, 0.2)})
        )
        text = edge.describe()
        assert "strong uptrend" in text
        assert "range" in text


class TestSignificance:
    def test_coin_flip_is_not_significant(self):
        assert binomial_p_value(50, 100) > 0.4

    def test_clear_edge_is_significant(self):
        assert binomial_p_value(65, 100) < 0.01

    def test_small_sample_cannot_prove_much(self):
        """60% on 10 trades is not evidence; 60% on 1000 is."""
        assert binomial_p_value(6, 10) > 0.05
        assert binomial_p_value(600, 1000) < 1e-9

    def test_benjamini_hochberg_controls_discoveries(self):
        # One genuine effect buried among nineteen null results.
        p_values = [0.001] + [0.4 + i * 0.03 for i in range(19)]
        survives, q_values = benjamini_hochberg(p_values, alpha=0.05)

        assert survives[0], "the genuine effect should survive"
        assert sum(survives) == 1, "no null result should survive"
        assert all(q >= p for q, p in zip(q_values, p_values))

    def test_q_values_are_monotone(self):
        p_values = [0.01, 0.02, 0.03, 0.2, 0.5]
        _, q_values = benjamini_hochberg(p_values)
        ordered = [q for _, q in sorted(zip(p_values, q_values))]
        assert ordered == sorted(ordered), "q-values must not decrease with p"

    def test_batch_of_pure_noise_yields_no_survivors(self):
        """Forty worthless models must not produce a winner."""
        rng = np.random.default_rng(0)
        models = []
        for i in range(40):
            n = 300
            # True accuracy is exactly 0.5; any excess is sampling noise.
            successes = int(rng.binomial(n, 0.5))
            models.append(
                {
                    "key": f"SYM{i}/H1",
                    "accuracy": successes / n,
                    "effective_samples": n,
                }
            )

        assessed = assess_batch(models, alpha=0.05)
        assert not any(a.survives_correction for a in assessed), (
            "a batch of coin flips produced a 'significant' model — the "
            "multiple-comparison correction is not working"
        )

    def test_correction_demotes_a_lone_lucky_model(self):
        """The whole point: looking good alone is not looking good in a batch."""
        models = [
            {"key": "LUCKY/H1", "accuracy": 0.56, "effective_samples": 300}
        ] + [
            {"key": f"NULL{i}/H1", "accuracy": 0.50, "effective_samples": 300}
            for i in range(39)
        ]

        assessed = assess_batch(models, alpha=0.05)
        lucky = next(a for a in assessed if a.key == "LUCKY/H1")

        assert lucky.naive_significant, "0.56 on 300 should clear a naive test"
        assert lucky.demoted, "it should not survive being one of 40 attempts"

    def test_genuine_strong_edge_survives_the_batch(self):
        """The correction must not throw away real findings."""
        models = [
            {"key": "REAL/H4", "accuracy": 0.62, "effective_samples": 800}
        ] + [
            {"key": f"NULL{i}/H1", "accuracy": 0.50, "effective_samples": 400}
            for i in range(30)
        ]

        assessed = assess_batch(models, alpha=0.05)
        real = next(a for a in assessed if a.key == "REAL/H4")
        assert real.survives_correction

    def test_required_accuracy_rises_as_sample_shrinks(self):
        small = minimum_accuracy_for_significance(90, n_tests=40)
        large = minimum_accuracy_for_significance(900, n_tests=40)
        assert small > large > 0.5
        # At 90 observations against 40 tests you need a large edge.
        assert small > 0.60

    def test_describe_batch_is_informative(self):
        models = [
            {"key": f"S{i}/H1", "accuracy": 0.50, "effective_samples": 200}
            for i in range(20)
        ]
        text = describe_batch(assess_batch(models))
        assert "20 models" in text
        assert "chance" in text.lower()

    def test_empty_batch_is_handled(self):
        assert assess_batch([]) == []
        assert benjamini_hochberg([]) == ([], [])
