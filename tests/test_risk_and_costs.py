"""Tests for the arithmetic that decides how much money is at stake.

A lookahead bug produces a fictional backtest. A sizing bug produces a real
loss. These check the pip-value conversions, the lot maths, and the cost model
that gates every signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.backtest import BacktestConfig, Backtester
from signalforge.labeling import cost_in_price_units
from signalforge.risk import (
    calculate_lots,
    compute_levels,
    correlation_adjusted_risk,
    expected_value,
    minimum_win_rate,
    pip_value_per_lot,
    portfolio_heat,
)
from signalforge.selection import cost_ratio, liquidity_score
from signalforge.universe import get_instrument

from tests.test_no_lookahead import make_ohlcv


class TestPipValue:
    def test_quote_currency_matches_account(self):
        """EURUSD on a USD account: one pip on one lot is exactly $10."""
        value, approximated, _ = pip_value_per_lot(
            get_instrument("EURUSD"), 1.0850, "USD"
        )
        assert value == pytest.approx(10.0)
        assert not approximated

    def test_base_currency_matches_account(self):
        """USDJPY on a USD account: pip value depends on the rate."""
        instrument = get_instrument("USDJPY")
        value, approximated, _ = pip_value_per_lot(instrument, 150.00, "USD")
        # 0.01 * 100_000 / 150 = 6.667
        assert value == pytest.approx(1000.0 / 150.0, rel=1e-6)
        assert not approximated

    def test_cross_pair_without_rate_is_flagged(self):
        """EURGBP on a USD account cannot be converted without a GBPUSD rate."""
        value, approximated, warnings = pip_value_per_lot(
            get_instrument("EURGBP"), 0.8600, "USD"
        )
        assert approximated, "cross-pair conversion should be flagged"
        assert warnings, "an explanatory warning should be attached"
        assert "Verify" in warnings[0]

    def test_cross_pair_with_rate_converts(self):
        value, approximated, _ = pip_value_per_lot(
            get_instrument("EURGBP"), 0.8600, "USD", conversion_rate=1.27
        )
        assert value == pytest.approx(10.0 * 1.27)
        assert not approximated

    def test_gold_pip_value(self):
        """XAUUSD: 0.1 pip on a 100oz contract is $10."""
        value, _, _ = pip_value_per_lot(get_instrument("XAUUSD"), 2400.0, "USD")
        assert value == pytest.approx(10.0)


class TestPositionSizing:
    def test_risk_is_honoured_exactly(self):
        instrument = get_instrument("EURUSD")
        size = calculate_lots(
            instrument,
            entry_price=1.0850,
            stop_price=1.0800,  # 50 pips
            account_balance=10_000.0,
            risk_percent=1.0,  # $100
        )
        # $100 risk / (50 pips * $10 per pip) = 0.2 lots
        assert size.lots == pytest.approx(0.2, abs=0.01)
        assert size.risk_amount <= 100.0 + 1e-6

    def test_rounding_never_increases_risk(self):
        """Lot rounding must go down, so realised risk never exceeds the target."""
        instrument = get_instrument("EURUSD")
        for balance in (523.0, 1237.0, 3891.0, 7777.0):
            size = calculate_lots(
                instrument,
                entry_price=1.0850,
                stop_price=1.0793,
                account_balance=balance,
                risk_percent=0.5,
            )
            target = balance * 0.005
            assert size.risk_amount <= target + 1e-6, (
                f"rounding increased risk above target at balance {balance}"
            )

    def test_account_too_small_returns_zero_lots(self):
        size = calculate_lots(
            get_instrument("EURUSD"),
            entry_price=1.0850,
            stop_price=1.0000,  # 850 pips
            account_balance=100.0,
            risk_percent=0.5,
        )
        assert size.lots == 0.0
        assert not size.is_tradable
        assert size.warnings

    def test_zero_stop_distance_is_rejected(self):
        size = calculate_lots(
            get_instrument("EURUSD"),
            entry_price=1.0850,
            stop_price=1.0850,
            account_balance=10_000.0,
            risk_percent=1.0,
        )
        assert size.lots == 0.0

    def test_correlated_exposure_reduces_size(self):
        risk, note = correlation_adjusted_risk("GBPUSD", ["EURUSD"], 1.0)
        assert risk < 1.0
        assert note and "EURUSD" in note

    def test_uncorrelated_exposure_is_unchanged(self):
        risk, note = correlation_adjusted_risk("BTCUSDT", ["EURUSD"], 1.0)
        assert risk == 1.0
        assert note is None

    def test_portfolio_heat_flags_overexposure(self):
        percent, too_hot = portfolio_heat([100.0] * 8, 10_000.0)
        assert percent == pytest.approx(8.0)
        assert too_hot


class TestLevels:
    def test_long_levels_are_ordered(self):
        df = make_ohlcv(300)
        levels = compute_levels(df, get_instrument("EURUSD"), direction=1)
        assert levels.stop_loss < levels.entry < levels.take_profits[0]
        assert levels.take_profits == sorted(levels.take_profits)

    def test_short_levels_are_ordered(self):
        df = make_ohlcv(300)
        levels = compute_levels(df, get_instrument("EURUSD"), direction=-1)
        assert levels.take_profits[0] < levels.entry < levels.stop_loss
        assert levels.take_profits == sorted(levels.take_profits, reverse=True)

    def test_stop_is_never_inside_the_spread(self):
        df = make_ohlcv(300)
        instrument = get_instrument("EURUSD")
        levels = compute_levels(
            df, instrument, direction=1, sl_atr_mult=0.0001  # absurdly tight
        )
        spread_price = instrument.typical_spread_pips * instrument.pip_size
        assert abs(levels.entry - levels.stop_loss) >= spread_price * 2.9

    def test_minimum_reward_risk_is_enforced(self):
        df = make_ohlcv(300)
        levels = compute_levels(
            df,
            get_instrument("EURUSD"),
            direction=1,
            tp_atr_mults=[0.1, 0.2, 0.3],  # far too close
            sl_atr_mult=2.0,
            min_reward_risk=1.5,
        )
        assert levels.reward_risk >= 1.5 - 1e-9


class TestCostModel:
    def test_break_even_win_rates(self):
        assert minimum_win_rate(1.0) == pytest.approx(0.5)
        assert minimum_win_rate(2.0) == pytest.approx(1 / 3)
        assert minimum_win_rate(3.0) == pytest.approx(0.25)

    def test_expectancy_sign(self):
        # 55% at 2:1 is clearly profitable.
        assert expected_value(0.55, 2.0) > 0
        # 30% at 2:1 is not.
        assert expected_value(0.30, 2.0) < 0

    def test_cost_ratio_penalises_wide_spreads(self):
        tight = cost_ratio(atr=0.0010, spread_pips=0.8, pip_size=0.0001)
        wide = cost_ratio(atr=0.0010, spread_pips=8.0, pip_size=0.0001)
        assert tight > wide
        assert wide < 2.0, "an 8-pip spread against a 10-pip ATR should be rejected"

    def test_round_trip_cost_includes_slippage(self):
        cost = cost_in_price_units(spread_pips=1.0, pip_size=0.0001, slippage_pips=0.5)
        # 1.0 spread + 2 x 0.5 slippage = 2.0 pips
        assert cost == pytest.approx(0.0002)

    def test_liquidity_is_zero_for_closed_markets(self):
        assert liquidity_score("forex", 12, trades_weekends=False, is_weekend=True) == 0.0
        assert liquidity_score("crypto", 12, trades_weekends=True, is_weekend=True) > 0.0

    def test_london_newyork_overlap_scores_highest(self):
        overlap = liquidity_score("forex", 14, False, False)
        asian = liquidity_score("forex", 3, False, False)
        dead = liquidity_score("forex", 22, False, False)
        assert overlap > asian > dead


class TestBacktesterPessimism:
    def test_ambiguous_bar_resolves_as_a_loss(self):
        """When one bar spans stop and target, the stop must win."""
        instrument = get_instrument("EURUSD")

        index = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.1000] * 5,
                # Bar 2 sweeps far in both directions.
                "high": [1.1005, 1.1005, 1.1200, 1.1005, 1.1005],
                "low": [1.0995, 1.0995, 1.0800, 1.0995, 1.0995],
                "close": [1.1000] * 5,
                "volume": [1000.0] * 5,
            },
            index=index,
        )

        signals = pd.DataFrame(
            {
                "direction": [1, 0, 0, 0, 0],
                "stop_loss": [1.0900, np.nan, np.nan, np.nan, np.nan],
                "take_profit": [1.1100, np.nan, np.nan, np.nan, np.nan],
                "confidence": [0.7, 0.0, 0.0, 0.0, 0.0],
            },
            index=index,
        )

        trades, _, _ = Backtester(instrument, BacktestConfig()).run(df, signals)
        assert not trades.empty, "expected the trade to be taken"
        assert trades.iloc[0]["exit_reason"] == "stop_ambiguous"
        assert trades.iloc[0]["pnl"] < 0, (
            "an ambiguous bar must be booked as a loss, not a win"
        )

    def test_costs_are_always_charged(self):
        df = make_ohlcv(400)
        instrument = get_instrument("EURUSD")
        signals = pd.DataFrame(
            {
                "direction": [1] * len(df),
                "stop_loss": df["close"] * 0.99,
                "take_profit": df["close"] * 1.01,
                "confidence": [0.7] * len(df),
            },
            index=df.index,
        )
        trades, _, report = Backtester(instrument, BacktestConfig()).run(df, signals)
        if not trades.empty:
            assert (trades["cost"] > 0).all(), "every trade must pay a cost"
            assert report.total_costs > 0
