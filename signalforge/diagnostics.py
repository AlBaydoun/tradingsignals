"""Checks a person needs before trusting a signal, in one place.

Two questions decide whether the engine is usable at all, and neither is about
models: **is this symbol called what I think it is called**, and **can my
account actually take this trade**. Both are asked by `doctor` and by the
dashboard, so both live here rather than being written twice and drifting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from signalforge.config import Config
from signalforge.features import indicators as ta
from signalforge.risk import minimum_balance_for
from signalforge.universe import get_instrument, mt5_name


@dataclass
class SymbolMapping:
    """What one watchlist entry is called in MetaTrader 5."""

    symbol: str
    mt5_symbol: str
    market: str
    # "config" when the name was set explicitly, "default" when it is a guess.
    source: str
    spread_pips: float
    spread_price_units: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Affordability:
    """Whether one symbol/timeframe can be traded at the configured risk."""

    symbol: str
    timeframe: str
    stop_distance: float
    minimum_balance: float
    minimum_lot_risk: float
    affordable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def symbol_mappings(config: Config) -> list[SymbolMapping]:
    """The exact strings to search for in the MT5 Market Watch window."""
    out: list[SymbolMapping] = []
    for symbol in config.watchlist:
        try:
            instrument = get_instrument(symbol)
        except KeyError:
            continue
        out.append(
            SymbolMapping(
                symbol=symbol,
                mt5_symbol=mt5_name(symbol, config.mt5_symbol_suffix),
                market=instrument.market,
                source="config" if instrument.mt5_symbol_is_exact else "default",
                spread_pips=instrument.typical_spread_pips,
                spread_price_units=round(
                    instrument.typical_spread_pips * instrument.pip_size, 6
                ),
            )
        )
    return out


def affordability(
    config: Config,
    router,
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    bars: int = 200,
) -> list[Affordability]:
    """Which combinations the account cannot trade, and what they would need.

    Sizing rounds down to the broker's lot step so risk never exceeds the
    configured percentage. When one minimum lot risks more than the budget the
    result is not an error — it is silence, every signal skipped with no
    explanation. This turns that silence into a number.
    """
    risk_pct = config.risk.risk_percent_per_trade
    balance = config.risk.account_balance
    out: list[Affordability] = []

    for symbol in symbols or config.watchlist:
        try:
            instrument = get_instrument(symbol)
        except KeyError:
            continue
        for timeframe in timeframes or config.timeframes:
            try:
                df = router.get_bars(symbol, timeframe, bars)
            except Exception:
                continue
            if df.empty or len(df) < 20:
                continue
            atr = ta.atr(df["high"], df["low"], df["close"], 14).iloc[-1]
            if not atr or atr != atr:  # NaN
                continue

            stop_distance = config.risk.sl_atr_mult * float(atr)
            needed = minimum_balance_for(
                instrument,
                entry_price=float(df["close"].iloc[-1]),
                stop_distance=stop_distance,
                risk_percent=risk_pct,
                account_currency=config.risk.account_currency,
            )
            out.append(
                Affordability(
                    symbol=symbol,
                    timeframe=timeframe,
                    stop_distance=round(stop_distance, 6),
                    minimum_balance=round(needed, 2),
                    minimum_lot_risk=round(needed * risk_pct / 100.0, 2),
                    affordable=needed <= balance,
                )
            )
    return out
