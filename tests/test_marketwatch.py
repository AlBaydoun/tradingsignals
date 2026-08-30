"""Tests for the wider-market sweep.

The sweep exists to answer "what is moving" across everything the engine can
price, while the watchlist answers "what can I trade". The tests here are
mostly about keeping that line intact: a sweep must never imply a trade, and
it must not fall over and take the signal loop down with it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from signalforge.config import MarketWatchConfig
from signalforge.marketwatch import Sweep, format_sweep, sweep, symbols_for
from signalforge.universe import INSTRUMENTS

from tests.test_universe_and_hunt import synthetic_bars


class FakeRouter:
    """A router that returns the frames it was given, or fails on demand."""

    def __init__(self, frames=None, *, fail=False, movers=None):
        self.frames = frames or {}
        self.fail = fail
        self.movers = movers or []

    def get_many(self, symbols, timeframe, bars):
        if self.fail:
            raise RuntimeError("provider throttled")
        return {s: self.frames.get(s, pd.DataFrame()) for s in symbols}

    def scan_crypto_movers(self, limit=15):
        if self.fail:
            raise RuntimeError("binance unreachable")
        return self.movers[:limit]


WEEKDAY = datetime(2026, 8, 26, 14, 0, tzinfo=timezone.utc)  # a Wednesday
WEEKEND = datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc)  # a Saturday


class TestUniverseSelection:
    def test_all_means_every_instrument(self):
        assert set(symbols_for(MarketWatchConfig(mode="all"))) == set(INSTRUMENTS)

    def test_markets_restricts_to_those_types(self):
        chosen = symbols_for(MarketWatchConfig(mode="markets", markets=["energy"]))
        assert "USOIL" in chosen and "BRENT" in chosen
        assert "EURUSD" not in chosen

    def test_none_disables_the_sweep(self):
        assert symbols_for(MarketWatchConfig(mode="none")) == []

    def test_markets_mode_with_no_markets_falls_back_to_all(self):
        """An empty list is a configuration mistake, not a request for silence."""
        assert len(symbols_for(MarketWatchConfig(mode="markets", markets=[]))) == len(
            INSTRUMENTS
        )


class TestSweep:
    def _config(self, **kw):
        base = dict(mode="markets", markets=["crypto"], min_score=0.0,
                    crypto_movers=False)
        base.update(kw)
        return MarketWatchConfig(**base)

    def test_a_quiet_market_produces_no_alerts(self):
        frames = {"BTCUSDT": synthetic_bars(noise=0.4, seed=9)}
        result = sweep(FakeRouter(frames), self._config(min_score=95.0), now=WEEKDAY)
        assert result.alerts == []
        assert "Nothing moving unusually" in result.describe()

    def test_alerts_carry_a_plain_language_reason(self):
        frames = {"BTCUSDT": synthetic_bars(drift=0.4, noise=0.4, seed=1)}
        result = sweep(FakeRouter(frames), self._config(), now=WEEKDAY)
        assert result.alerts, "a strongly trending market should surface"
        assert result.alerts[0].headline

    def test_closed_markets_never_alert(self):
        """A shut exchange is not news, whatever Friday looked like."""
        frames = {"EURUSD": synthetic_bars(drift=0.4, seed=1)}
        config = MarketWatchConfig(mode="markets", markets=["forex"],
                                   min_score=0.0, crypto_movers=False)
        result = sweep(FakeRouter(frames), config, now=WEEKEND)
        assert result.alerts == []
        assert result.open_markets == 0

    def test_a_dead_provider_degrades_instead_of_raising(self):
        """This runs inside the watch loop; it must not take signals down."""
        result = sweep(FakeRouter(fail=True), self._config(), now=WEEKDAY)
        assert isinstance(result, Sweep)
        assert result.surveyed == 0
        assert result.alerts == []

    def test_crypto_mover_failure_does_not_lose_the_sweep(self):
        class PartlyBroken(FakeRouter):
            def scan_crypto_movers(self, limit=15):
                raise RuntimeError("binance unreachable")

        frames = {"BTCUSDT": synthetic_bars(drift=0.4, seed=1)}
        result = sweep(PartlyBroken(frames), self._config(crypto_movers=True),
                       now=WEEKDAY)
        assert result.alerts, "the hunt results must survive a mover failure"
        assert result.crypto_movers == []

    def test_reported_alerts_are_capped(self):
        frames = {s: synthetic_bars(drift=0.4, seed=i)
                  for i, s in enumerate(symbols_for(MarketWatchConfig(mode="markets",
                                                                     markets=["crypto"])))}
        result = sweep(FakeRouter(frames), self._config(max_reported=3), now=WEEKDAY)
        assert len(result.alerts) <= 3

    def test_disabled_sweep_does_no_work(self):
        router = FakeRouter({"BTCUSDT": synthetic_bars(seed=1)})
        result = sweep(router, MarketWatchConfig(mode="none"), now=WEEKDAY)
        assert result.surveyed == 0
        assert result.crypto_movers == []


class TestFormatting:
    def test_an_untrained_alert_says_it_is_not_a_trade(self):
        """The single most important sentence this module prints."""
        frames = {"BTCUSDT": synthetic_bars(drift=0.4, noise=0.4, seed=1)}
        config = MarketWatchConfig(mode="markets", markets=["crypto"],
                                   min_score=0.0, crypto_movers=False)
        result = sweep(FakeRouter(frames), config, now=WEEKDAY)
        text = format_sweep(result, watchlist=[])
        assert "never a trade to take" in text

    def test_watchlist_members_are_marked(self):
        frames = {"BTCUSDT": synthetic_bars(drift=0.4, noise=0.4, seed=1)}
        config = MarketWatchConfig(mode="markets", markets=["crypto"],
                                   min_score=0.0, crypto_movers=False)
        result = sweep(FakeRouter(frames), config, now=WEEKDAY)
        assert "* BTCUSDT" in format_sweep(result, watchlist=["BTCUSDT"])

    def test_empty_sweep_formats_without_error(self):
        empty = Sweep(WEEKDAY, "H1", 0, 0, [], [])
        assert "nothing to survey" in format_sweep(empty)

    def test_sweep_survives_a_json_round_trip(self):
        import json

        frames = {"BTCUSDT": synthetic_bars(drift=0.4, seed=1)}
        config = MarketWatchConfig(mode="markets", markets=["crypto"],
                                   min_score=0.0, crypto_movers=False)
        payload = json.dumps(sweep(FakeRouter(frames), config, now=WEEKDAY).to_dict())
        assert json.loads(payload)["timeframe"] == "H1"
