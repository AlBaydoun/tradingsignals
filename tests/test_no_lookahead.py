"""Tests for the property the whole system depends on: no lookahead.

If a feature at bar *t* can see bar *t+1*, every accuracy number the engine
reports is fiction. These tests are the guard against that, and they are the
most important tests in the repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.data.base import align_to_signal_frame, resample_ohlcv, validate_ohlcv
from signalforge.features import build_feature_matrix, indicators as ta
from signalforge.labeling import apply_triple_barrier
from signalforge.models.validation import PurgedWalkForward, check_no_leakage


def make_ohlcv(n: int = 800, seed: int = 0, freq: str = "15min") -> pd.DataFrame:
    """A synthetic but realistic price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0, 0.002, n)) * close

    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.0005, n)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.lognormal(10, 1, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
    )
    return validate_ohlcv(df, symbol="TEST")


class TestFeatureCausality:
    """Truncating the data must not change features on the remaining bars."""

    def test_features_unchanged_by_future_data(self):
        df = make_ohlcv(800)
        full = build_feature_matrix(df, "M15")
        truncated = build_feature_matrix(df.iloc[:-200], "M15")

        shared_index = truncated.index[-100:]
        shared_columns = [c for c in full.columns if c in truncated.columns]

        a = full.loc[shared_index, shared_columns].to_numpy(dtype=float)
        b = truncated.loc[shared_index, shared_columns].to_numpy(dtype=float)

        both_nan = np.isnan(a) & np.isnan(b)
        difference = np.abs(np.where(both_nan, 0.0, a - b))
        assert np.nanmax(difference) < 1e-9, (
            "Features changed when future bars were removed — a feature is "
            "reading data from the future."
        )

    @pytest.mark.parametrize(
        "name,fn",
        [
            ("rsi", lambda d: ta.rsi(d["close"], 14)),
            ("atr", lambda d: ta.atr(d["high"], d["low"], d["close"], 14)),
            ("adx", lambda d: ta.adx(d["high"], d["low"], d["close"], 14)[0]),
            ("supertrend", lambda d: ta.supertrend(d["high"], d["low"], d["close"])[1]),
            ("sar", lambda d: ta.parabolic_sar(d["high"], d["low"])),
            ("efficiency", lambda d: ta.efficiency_ratio(d["close"], 20)),
            ("hurst", lambda d: ta.hurst_exponent(d["close"], 100)),
            ("bb_percent_b", lambda d: ta.bollinger_percent_b(d["close"], 20)),
            ("donchian_upper", lambda d: ta.donchian(d["high"], d["low"], 20)[1]),
            ("swing_highs", lambda d: ta.swing_highs(d["high"])),
            ("obv", lambda d: ta.obv(d["close"], d["volume"])),
        ],
    )
    def test_indicator_is_causal(self, name, fn):
        df = make_ohlcv(600)
        full = fn(df)
        truncated = fn(df.iloc[:-150])

        overlap = truncated.index[-80:]
        a = full.loc[overlap].to_numpy(dtype=float)
        b = truncated.loc[overlap].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        difference = np.abs(np.where(both_nan, 0.0, a - b))

        assert np.nanmax(difference) < 1e-9, f"{name} is not causal"

    def test_swing_highs_are_confirmation_shifted(self):
        """A swing high must be flagged only once its right bars exist."""
        df = make_ohlcv(300)
        pivots = ta.swing_highs(df["high"], left=3, right=3)

        # Every flagged bar must have a genuine pivot 3 bars earlier.
        flagged = np.flatnonzero(pivots.to_numpy() > 0)
        for position in flagged:
            pivot_position = position - 3
            if pivot_position < 3 or pivot_position >= len(df) - 3:
                continue
            window = df["high"].iloc[pivot_position - 3 : pivot_position + 4]
            assert df["high"].iloc[pivot_position] == pytest.approx(window.max()), (
                "Swing high flagged at a bar that is not the local maximum "
                "three bars back — the confirmation shift is wrong."
            )


class TestMultiTimeframeAlignment:
    """Higher timeframe context must never leak backwards."""

    def test_asof_merge_uses_only_past_bars(self):
        base = make_ohlcv(400, freq="15min")
        higher = resample_ohlcv(base, "H1")

        aligned = align_to_signal_frame(higher[["close"]], base.index, prefix="h1")

        for timestamp in base.index[50:150]:
            value = aligned.loc[timestamp, "h1_close"]
            if pd.isna(value):
                continue
            # The value must come from an H1 bar that had already closed.
            available = higher[higher.index <= timestamp]
            assert not available.empty
            assert value == pytest.approx(float(available["close"].iloc[-1])), (
                "Higher-timeframe value at this bar does not match the last "
                "completed higher-timeframe bar."
            )

    def test_resample_does_not_leak(self):
        base = make_ohlcv(400, freq="15min")
        full = resample_ohlcv(base, "H1")
        partial = resample_ohlcv(base.iloc[:-50], "H1")

        shared = partial.index[:-1]  # the final bar of a truncated set is partial
        pd.testing.assert_frame_equal(
            full.loc[shared], partial.loc[shared], check_exact=False, rtol=1e-9
        )


class TestLabelling:
    def test_labels_only_use_future_within_horizon(self):
        df = make_ohlcv(500)
        result = apply_triple_barrier(df, horizon=20, cost_price_units=0.0)
        frame = result.to_frame()

        resolved = frame[frame["event_end"] >= 0]
        positions = np.arange(len(frame))[frame["event_end"].to_numpy() >= 0]
        ends = resolved["event_end"].to_numpy()

        assert np.all(ends > positions), "A label resolved at or before its own bar"
        assert np.all(ends - positions <= 20), "A label resolved beyond its horizon"

    def test_ambiguous_bars_are_excluded_not_guessed(self):
        df = make_ohlcv(500)
        result = apply_triple_barrier(df, horizon=20, cost_price_units=0.0)
        frame = result.to_frame()

        ambiguous = frame[frame["ambiguous"]]
        if not ambiguous.empty:
            assert (ambiguous["sample_weight"] == 0).all(), (
                "Ambiguous bars must carry zero weight rather than a guessed label"
            )

    def test_cost_filter_excludes_unreachable_barriers(self):
        df = make_ohlcv(500)
        # A cost larger than any plausible ATR move must exclude everything.
        result = apply_triple_barrier(
            df, horizon=20, cost_price_units=1000.0, min_cost_multiple=1.5
        )
        assert not result.tradable.any(), (
            "Bars remained tradable despite costs exceeding the barrier width"
        )


class TestPurgedWalkForward:
    def test_train_always_precedes_test(self):
        splitter = PurgedWalkForward(
            n_splits=4, embargo_bars=10, min_train_bars=200, min_test_bars=100
        )
        for fold in splitter.split(1200):
            assert fold.train.max() < fold.test.min()

    def test_overlapping_labels_are_purged(self):
        n = 1200
        # Every label takes 30 bars to resolve, so folds must purge the tail.
        event_end = np.arange(n) + 30
        splitter = PurgedWalkForward(
            n_splits=4, embargo_bars=10, min_train_bars=200, min_test_bars=100
        )

        folds = list(splitter.split(n, event_end))
        assert folds, "splitter produced no folds"

        for fold in folds:
            problems = check_no_leakage(fold.train, fold.test, event_end)
            assert not problems, f"leakage in fold {fold.index}: {problems}"
            assert fold.purged > 0, "expected samples to be purged with 30-bar labels"

    def test_purging_survives_row_filtering(self):
        """Event ends must be tracked by timestamp, not by row position.

        Training drops rows (cost filtering, NaN features), which shifts every
        position. A purge index computed against the unfiltered frame then
        points at the wrong row, silently disabling purging. Regression test
        for exactly that bug.
        """
        df = make_ohlcv(2000)
        result = apply_triple_barrier(df, horizon=24, cost_price_units=0.0)
        frame = result.to_frame()

        # Drop a third of the rows, as real cost/NaN filtering would.
        rng = np.random.default_rng(0)
        keep = pd.Series(rng.random(len(df)) > 0.33, index=df.index)
        kept = frame[keep]

        end_times = pd.to_datetime(kept["event_end_time"], utc=True)
        # Int64 nanoseconds, matching the production path. Searching over the
        # object arrays that `.to_numpy()` produces for tz-aware datetimes
        # returns wrong positions.
        index_ns = np.asarray(kept.index.asi8, dtype="int64")
        missing = end_times.isna().to_numpy()
        end_ns = np.where(
            missing, np.iinfo("int64").max, end_times.astype("int64").to_numpy()
        )
        positions = np.searchsorted(index_ns, end_ns, side="left")
        positions = np.where(missing, -1, positions)
        positions = np.where(positions >= len(kept), len(kept) - 1, positions)

        resolved = positions >= 0
        row_positions = np.arange(len(kept))
        # Every resolved label must end at or after its own row, and the gap
        # must stay bounded by the horizon once re-indexed.
        assert np.all(positions[resolved] >= row_positions[resolved])
        spans = positions[resolved] - row_positions[resolved]
        assert spans.max() <= 24, (
            f"a label spans {spans.max()} filtered rows, which exceeds the "
            "24-bar horizon — positions were not remapped correctly"
        )
        assert spans.mean() < 15, "mean label span is implausibly large"

    def test_embargo_is_applied_between_folds(self):
        splitter = PurgedWalkForward(
            n_splits=3, embargo_bars=25, min_train_bars=200, min_test_bars=100
        )
        folds = list(splitter.split(1500))
        for earlier, later in zip(folds, folds[1:]):
            gap = later.test.min() - earlier.test.max()
            assert gap >= 25, f"embargo not honoured: gap was {gap}"
