"""Tests for broker-name handling and the pre-training instrument survey.

Two separate failure modes are guarded here. The first is cosmetic but costly:
a signal that names a symbol the user cannot find in MT5 is a signal they
cannot trade. The second is the reason `hunt` exists at all — ranking markets
by volatility while ignoring what they cost to trade would recommend exactly
the instruments that lose money fastest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.selection.hunter import (
    GOOD_COST_RATIO,
    MIN_COST_RATIO,
    cost_multiplier,
    default_universe,
    describe,
    evaluate,
    hunt,
)
from signalforge.universe import (
    INSTRUMENTS,
    apply_overrides,
    get_instrument,
    mt5_name,
    resolve_symbol,
)


# The synthetic series below is priced around 100, so the real spreads of a
# 29,000-point index would swamp it. These keep the cost gate open so the
# volatility and direction terms are what the assertions actually measure.
CHEAP = {"XAUUSD": 0.5, "EURUSD": 0.1, "US500": 0.5}


@pytest.fixture(autouse=True)
def restore_universe():
    """Overrides mutate module-level state, so put it back afterwards."""
    saved = dict(INSTRUMENTS)
    yield
    INSTRUMENTS.clear()
    INSTRUMENTS.update(saved)


def synthetic_bars(
    n: int = 400,
    *,
    start: float = 100.0,
    drift: float = 0.0,
    noise: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """A price series with controllable trendiness and volatility."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, noise, n)
    close = start + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, noise * 0.5, n))
    low = close - np.abs(rng.normal(0, noise * 0.5, n))
    index = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.lognormal(10, 0.3, n),
        },
        index=index,
    )


class TestBrokerNames:
    def test_canonical_names_pass_through(self):
        assert resolve_symbol("EURUSD") == "EURUSD"
        assert resolve_symbol("eurusd") == "EURUSD"

    @pytest.mark.parametrize(
        "broker_name,canonical",
        [
            ("US100", "NAS100"),
            ("US100.std", "NAS100"),
            ("USTEC", "NAS100"),
            ("WTI", "USOIL"),
            ("WTI.m", "USOIL"),
            ("BRENT.m", "BRENT"),
            ("BTCUSD", "BTCUSDT"),
            ("btcusd", "BTCUSDT"),
            ("GOLD", "XAUUSD"),
            ("EURUSD_ecn", "EURUSD"),
            ("EURUSD#", "EURUSD"),
            ("US30.std", "US30"),
        ],
    )
    def test_broker_names_resolve(self, broker_name, canonical):
        assert resolve_symbol(broker_name) == canonical
        assert get_instrument(broker_name).symbol == canonical

    def test_unknown_symbol_still_errors_usefully(self):
        with pytest.raises(KeyError, match="Unknown instrument"):
            get_instrument("NOTATHING")


class TestPerInstrumentOverrides:
    def test_override_sets_the_exact_mt5_name(self):
        apply_overrides({"NAS100": {"mt5_symbol": "US100.std"}})
        assert mt5_name("NAS100") == "US100.std"

    def test_explicit_name_ignores_the_global_suffix(self):
        """The whole point: a named symbol is already exact.

        A broker with `US100.std` and `EURUSD.m` on the same account cannot be
        described by one suffix. Naming a symbol must therefore switch the
        global suffix off for that symbol only.
        """
        apply_overrides({"NAS100": {"mt5_symbol": "US100.std"}})
        assert mt5_name("NAS100", ".m") == "US100.std"
        assert mt5_name("EURUSD", ".m") == "EURUSD.m"

    def test_mixed_suffixes_coexist(self):
        apply_overrides(
            {
                "XAUUSD": {"mt5_symbol": "XAUUSD"},
                "NAS100": {"mt5_symbol": "US100.std"},
                "USOIL": {"mt5_symbol": "WTI.m"},
                "BTCUSDT": {"mt5_symbol": "BTCUSD"},
            }
        )
        assert mt5_name("XAUUSD") == "XAUUSD"
        assert mt5_name("NAS100") == "US100.std"
        assert mt5_name("USOIL") == "WTI.m"
        assert mt5_name("BTCUSDT") == "BTCUSD"

    def test_override_accepts_a_broker_name_as_its_key(self):
        apply_overrides({"WTI": {"typical_spread_pips": 5.0}})
        assert get_instrument("USOIL").typical_spread_pips == 5.0

    def test_cost_settings_can_be_overridden(self):
        apply_overrides({"XAUUSD": {"typical_spread_pips": 4.5, "commission_per_lot": 7.0}})
        gold = get_instrument("XAUUSD")
        assert gold.typical_spread_pips == 4.5
        assert gold.commission_per_lot == 7.0

    def test_unknown_instrument_is_rejected(self):
        with pytest.raises(KeyError, match="unknown instrument"):
            apply_overrides({"FTSE250": {"mt5_symbol": "UK250"}})

    def test_typo_in_a_field_name_is_rejected_not_ignored(self):
        """A misspelled spread override that silently does nothing is worse
        than one that fails loudly — the user would trust a backtest built on
        the default cost assumption while believing they had changed it."""
        with pytest.raises(KeyError, match="Cannot override"):
            apply_overrides({"XAUUSD": {"typical_spread": 4.5}})

    def test_empty_overrides_change_nothing(self):
        before = get_instrument("EURUSD")
        assert apply_overrides({}) == []
        assert get_instrument("EURUSD") == before


class TestCostGate:
    def test_cost_multiplier_is_zero_when_costs_dominate(self):
        assert cost_multiplier(MIN_COST_RATIO) == 0.0
        assert cost_multiplier(0.5) == 0.0

    def test_cost_multiplier_saturates(self):
        assert cost_multiplier(GOOD_COST_RATIO) == pytest.approx(1.0)
        assert cost_multiplier(50.0) == pytest.approx(1.0)

    def test_cost_multiplier_is_monotone(self):
        values = [cost_multiplier(r) for r in np.linspace(0.0, 12.0, 40)]
        assert values == sorted(values)

    def test_a_wide_spread_cannot_be_outvolatilised(self):
        """The central claim of the module: no amount of movement rescues an
        instrument whose spread is the size of its range."""
        bars = synthetic_bars(drift=0.3, noise=1.0, seed=3)

        cheap = evaluate("XAUUSD", bars, "H1", spread_pips=1.0, hour_utc=14)
        pricey = evaluate("XAUUSD", bars, "H1", spread_pips=400.0, hour_utc=14)

        assert cheap is not None and pricey is not None
        assert pricey.score == 0.0
        assert not pricey.tradable_cost
        assert cheap.score > pricey.score
        assert "untradable" in " ".join(pricey.reasons)


class TestHunt:
    def test_trending_market_outranks_chop_at_equal_cost(self):
        trending = synthetic_bars(drift=0.35, noise=0.4, seed=1)
        choppy = synthetic_bars(drift=0.0, noise=0.4, seed=2)

        a = evaluate("XAUUSD", trending, "H1", spread_pips=1.0, hour_utc=14)
        b = evaluate("XAUUSD", choppy, "H1", spread_pips=1.0, hour_utc=14)

        assert a.efficiency_ratio > b.efficiency_ratio
        assert a.score > b.score

    def test_volatility_is_measured_against_the_instruments_own_history(self):
        """Gold moving 1% and EURUSD moving 1% are not the same event, so the
        comparison has to be internal, not cross-sectional."""
        quiet = synthetic_bars(noise=0.4, seed=5)
        waking = quiet.copy()
        # Triple the range of the last 30 bars only.
        tail = waking.index[-30:]
        mid = waking.loc[tail, "close"]
        waking.loc[tail, "high"] = mid + 3.0
        waking.loc[tail, "low"] = mid - 3.0

        base = evaluate("XAUUSD", quiet, "H1", spread_pips=1.0, hour_utc=14)
        hot = evaluate("XAUUSD", waking, "H1", spread_pips=1.0, hour_utc=14)

        assert hot.vol_expansion > base.vol_expansion
        assert hot.vol_percentile >= base.vol_percentile

    def test_results_are_ranked_and_capped(self):
        frames = {
            "XAUUSD": synthetic_bars(drift=0.3, seed=1),
            "EURUSD": synthetic_bars(drift=0.0, seed=2),
            "US500": synthetic_bars(drift=0.1, seed=3),
        }
        results = hunt(frames, "H1", hour_utc=14, limit=2, spreads=CHEAP)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_short_history_is_skipped_not_guessed(self):
        assert evaluate("EURUSD", synthetic_bars(20), "H1") is None
        assert hunt({"EURUSD": synthetic_bars(20)}, "H1") == []

    def test_a_broken_frame_does_not_kill_the_survey(self):
        frames = {
            "XAUUSD": synthetic_bars(seed=1),
            "EURUSD": pd.DataFrame(),
            "NOTREAL": synthetic_bars(seed=2),
        }
        results = hunt(frames, "H1", hour_utc=14)
        assert [r.symbol for r in results] == ["XAUUSD"]

    def test_closed_market_scores_below_an_open_one(self):
        bars = synthetic_bars(drift=0.3, seed=1)
        open_now = evaluate("US500", bars, "H1", spread_pips=0.5, hour_utc=15)
        weekend = evaluate(
            "US500", bars, "H1", spread_pips=0.5, hour_utc=15, is_weekend=True
        )
        assert weekend.score < open_now.score
        assert weekend.liquidity == 0.0

    def test_model_presence_is_reported_not_scored(self):
        """Having a model must not flatter an instrument's ranking — the hunt
        is about the market, not about what we happen to have trained."""
        bars = synthetic_bars(drift=0.3, seed=1)
        frames = {"XAUUSD": bars}
        with_model = hunt(frames, "H1", hour_utc=14, spreads=CHEAP, models={"XAUUSD"})[0]
        without = hunt(frames, "H1", hour_utc=14, spreads=CHEAP)[0]

        assert with_model.score > 0.0, "the fixture must produce a real score"

        assert with_model.has_model and not without.has_model
        assert with_model.score == without.score

    def test_describe_refuses_to_promise_direction(self):
        frames = {"XAUUSD": synthetic_bars(drift=0.3, seed=1)}
        text = describe(hunt(frames, "H1", hour_utc=14, spreads=CHEAP))
        assert "does not claim the direction is predictable" in text

    def test_describe_handles_an_empty_survey(self):
        assert "No instruments" in describe([])

    def test_default_universe_can_be_filtered_by_market(self):
        energy = default_universe(["energy"])
        assert "USOIL" in energy and "BRENT" in energy
        assert "EURUSD" not in energy
        assert set(default_universe()) == set(INSTRUMENTS)
