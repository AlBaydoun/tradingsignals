"""Integration tests: the pieces must fit together and degrade gracefully."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.anomaly import detect
from signalforge.config import Config, load_config
from signalforge.features import build_feature_matrix, clean_for_model
from signalforge.labeling import apply_triple_barrier
from signalforge.learning import JournalEntry, TradeJournal
from signalforge.models import SignalModel, population_stability_index
from signalforge.news import score_text, relevant_symbols
from signalforge.regime import RegimeDetector, classify_trend, classify_volatility
from signalforge.signals.schema import SignalQuality, grade

from tests.test_no_lookahead import make_ohlcv


class TestConfig:
    def test_defaults_load(self):
        config = Config()
        assert config.risk.risk_percent_per_trade > 0
        assert config.watchlist

    def test_yaml_overrides_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("risk:\n  risk_percent_per_trade: 0.25\n")
        config = load_config(path)
        assert config.risk.risk_percent_per_trade == 0.25
        # Untouched values must keep their defaults.
        assert config.risk.sl_atr_mult == 1.5


class TestFeaturePipeline:
    def test_matrix_has_no_constant_or_empty_columns(self):
        df = make_ohlcv(900)
        features = build_feature_matrix(df, "M15")
        cleaned, dropped = clean_for_model(features)

        assert cleaned.shape[1] > 80, "expected a rich feature set"
        assert cleaned.iloc[-1].isna().sum() == 0, "latest row must be complete"
        variance = cleaned.std(numeric_only=True)
        assert (variance > 0).all(), "a constant column survived cleaning"

    def test_short_series_returns_empty_rather_than_raising(self):
        df = make_ohlcv(30)
        features = build_feature_matrix(df, "M15")
        assert features.empty or features.shape[1] == 0

    def test_zero_volume_instrument_still_builds(self):
        """Spot FX from Yahoo reports no volume; the matrix must still form."""
        df = make_ohlcv(500)
        df["volume"] = 0.0
        features = build_feature_matrix(df, "M15")
        assert not features.empty
        assert "has_volume" in features.columns
        assert features["has_volume"].iloc[-1] == 0.0


class TestRegime:
    def test_volatility_buckets(self):
        assert classify_volatility(0.05, 1.0, False) == "compressed"
        assert classify_volatility(0.50, 1.0, False) == "normal"
        assert classify_volatility(0.95, 1.0, False) == "extreme"
        assert classify_volatility(0.50, 1.0, True) == "compressed"

    def test_trend_buckets(self):
        assert classify_trend(0.35, 0.15, 0.5) == "strong_uptrend"
        assert classify_trend(0.35, -0.15, 0.5) == "strong_downtrend"
        assert classify_trend(0.10, 0.01, 0.1) == "range"

    def test_detector_produces_a_state(self):
        df = make_ohlcv(900)
        detector = RegimeDetector(n_clusters=3).fit(df)
        state = detector.current(df)
        assert state.volatility in (
            "compressed", "normal", "elevated", "extreme",
        )
        assert 0.0 <= state.trend_following_score <= 1.0
        assert state.describe()


class TestAnomaly:
    def test_quiet_market_does_not_fire(self):
        rng = np.random.default_rng(3)
        n = 400
        close = 100 + np.cumsum(rng.normal(0, 0.01, n))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        )
        report = detect(df, "TEST", "H1")
        assert not report.is_igniting

    def test_violent_move_fires(self):
        df = make_ohlcv(400)
        # Inject a genuine shock into the final bars.
        df.iloc[-3:, df.columns.get_loc("close")] *= [1.05, 1.09, 1.14]
        df.iloc[-3:, df.columns.get_loc("high")] = df["close"].iloc[-3:] * 1.01
        df.iloc[-3:, df.columns.get_loc("volume")] *= 25
        report = detect(df, "TEST", "H1")
        assert report.ignition_score > 40, f"score was {report.ignition_score}"
        assert report.triggers


class TestSignalGrading:
    def test_no_measured_accuracy_can_never_be_strong(self):
        quality = grade(
            model_confidence=0.95, measured_accuracy=None, reward_risk=2.0, edge=0.5
        )
        assert quality is not SignalQuality.STRONG

    def test_losing_edge_is_watch_only(self):
        # 30% at 2:1 loses money; break-even is 33%.
        quality = grade(0.80, measured_accuracy=0.30, reward_risk=2.0, edge=0.4)
        assert quality is SignalQuality.WATCH_ONLY

    def test_genuine_edge_grades_strong(self):
        quality = grade(0.70, measured_accuracy=0.50, reward_risk=2.0, edge=0.3)
        assert quality is SignalQuality.STRONG


class TestSentiment:
    def test_direction_is_detected(self):
        assert score_text("Bitcoin surges to record high").score > 0.3
        assert score_text("Gold plunges after crash").score < -0.3

    def test_negation_flips_polarity(self):
        plain = score_text("stocks rally").score
        negated = score_text("stocks did not rally").score
        assert negated < plain

    def test_symbol_extraction(self):
        assert "BTCUSDT" in relevant_symbols("Bitcoin hits new high")
        assert "XAUUSD" in relevant_symbols("Gold prices climb")
        # Word boundaries: "solution" must not match "sol".
        assert "SOLUSDT" not in relevant_symbols("A solution was found")

    def test_empty_text_is_neutral(self):
        result = score_text("")
        assert result.score == 0.0
        assert result.confidence == 0.0


class TestDrift:
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(0, 1, 2000)
        assert population_stability_index(sample, sample.copy()) < 0.01

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(1)
        baseline = rng.normal(0, 1, 2000)
        shifted = rng.normal(2.5, 1, 2000)
        assert population_stability_index(baseline, shifted) > 0.25


class TestJournal:
    def test_record_and_resolve(self, tmp_path):
        journal = TradeJournal(tmp_path / "trades.jsonl")
        journal.record(
            JournalEntry(
                signal_id="abc",
                symbol="EURUSD",
                timeframe="H1",
                direction="BUY",
                issued_at="2026-01-01T00:00:00+00:00",
                entry=1.08,
                stop_loss=1.075,
                take_profits=[1.09],
                lots=0.1,
                risk_amount=50.0,
                model_confidence=0.62,
                measured_accuracy=0.55,
                reward_risk=2.0,
                regime="uptrend",
                quality="moderate",
            )
        )
        assert len(journal.open_signals()) == 1

        journal.update_outcome(
            "abc",
            status="won",
            exit_price=1.09,
            pnl=100.0,
            r_multiple=2.0,
            exit_reason="target",
        )
        assert len(journal.open_signals()) == 0
        stats = journal.live_statistics()
        assert stats["trades"] == 1
        assert stats["win_rate"] == 1.0

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "trades.jsonl"
        path.write_text('{"broken": true}\nnot json at all\n')
        journal = TradeJournal(path)
        assert journal.all_entries() == []


class TestModelTraining:
    def test_reports_honest_accuracy_on_noise(self):
        """On pure noise the model must land near 50%, not above it.

        A model that scores well here is leaking, not learning.
        """
        df = make_ohlcv(4000, seed=42)
        features = build_feature_matrix(df, "M15")
        X, _ = clean_for_model(features)
        labels = apply_triple_barrier(df, horizon=16, cost_price_units=0.0)
        frame = labels.to_frame()

        usable = frame["tradable"] & ~frame["ambiguous"] & X.notna().all(axis=1)
        if usable.sum() < 1200:
            pytest.skip("not enough usable synthetic rows")

        from signalforge.config import ModelConfig

        model = SignalModel(
            ModelConfig(min_train_bars=800, min_test_bars=250, n_splits=3, n_estimators=80)
        )
        report = model.fit(
            X[usable],
            frame["label"][usable],
            sample_weight=frame["sample_weight"][usable],
            event_end_time=frame["event_end_time"][usable],
            symbol="SYNTHETIC",
            timeframe="M15",
        )

        assert 0.35 < report.directional_accuracy < 0.62, (
            f"accuracy {report.directional_accuracy:.3f} on random data suggests "
            "information is leaking into the model"
        )

    def test_oos_predictions_are_exposed_for_backtesting(self):
        df = make_ohlcv(3000, seed=7)
        features = build_feature_matrix(df, "M15")
        X, _ = clean_for_model(features)
        labels = apply_triple_barrier(df, horizon=16, cost_price_units=0.0)
        frame = labels.to_frame()
        usable = frame["tradable"] & ~frame["ambiguous"] & X.notna().all(axis=1)

        from signalforge.config import ModelConfig

        model = SignalModel(
            ModelConfig(min_train_bars=700, min_test_bars=200, n_splits=3, n_estimators=60)
        )
        model.fit(
            X[usable],
            frame["label"][usable],
            sample_weight=frame["sample_weight"][usable],
            event_end_time=frame["event_end_time"][usable],
        )

        oos = model.oos_signals()
        assert not oos.empty
        # Walk-forward predictions must cover fewer rows than the training set,
        # since the first fold's training block is never predicted.
        assert len(oos) < int(usable.sum())
