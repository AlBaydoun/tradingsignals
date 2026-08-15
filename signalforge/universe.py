"""The instrument universe.

Every tradable symbol is described once, here: where its price data comes from,
what it is called inside MetaTrader 5, how big a pip is, what it typically costs
to trade, and when it is liquid enough to be worth trading at all.

Pip sizes and spreads are the two numbers that decide whether a strategy is
profitable or merely looks profitable, so they are explicit rather than guessed.
Spreads are conservative retail estimates — override them in config with your
own broker's real numbers before you trust any backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

# Trading session windows in UTC. Used both to gate signals and as features.
SESSIONS: dict[str, tuple[time, time]] = {
    "sydney": (time(21, 0), time(6, 0)),
    "tokyo": (time(0, 0), time(9, 0)),
    "london": (time(7, 0), time(16, 0)),
    "newyork": (time(12, 0), time(21, 0)),
}

# Overlaps are where most of the daily range is produced.
SESSION_OVERLAPS: dict[str, tuple[time, time]] = {
    "london_newyork": (time(12, 0), time(16, 0)),
    "tokyo_london": (time(7, 0), time(9, 0)),
}


@dataclass(frozen=True)
class Instrument:
    """A single tradable market."""

    symbol: str  # canonical name used throughout the engine
    market: str  # forex | crypto | metals | indices | energy
    provider: str  # binance | yahoo
    provider_symbol: str  # symbol as the data provider knows it
    mt5_symbol: str  # symbol as MetaTrader 5 knows it (before broker suffix)
    digits: int  # price decimal places
    pip_size: float  # price movement of one pip
    contract_size: float  # units per 1.00 lot
    typical_spread_pips: float  # round-trip cost estimate, in pips
    commission_per_lot: float = 0.0  # round-trip commission in account currency
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    # Currency the profit is denominated in (drives pip-value conversion).
    quote_currency: str = "USD"
    base_currency: str = ""
    # Hours (UTC) when this instrument is liquid. Empty means 24/7.
    active_sessions: tuple[str, ...] = field(default=())
    # 24/5 for forex, 24/7 for crypto, exchange hours for index CFDs.
    trades_weekends: bool = False
    notes: str = ""

    @property
    def point(self) -> float:
        """The smallest price increment."""
        return 10 ** (-self.digits)

    def pips_between(self, price_a: float, price_b: float) -> float:
        """Distance between two prices, expressed in pips."""
        return abs(price_a - price_b) / self.pip_size

    def round_price(self, price: float) -> float:
        return round(price, self.digits)


def _fx(
    symbol: str,
    spread: float,
    *,
    digits: int = 5,
    pip: float = 0.0001,
    yahoo: str | None = None,
) -> Instrument:
    """Helper for spot FX pairs, which all share the same shape."""
    base, quote = symbol[:3], symbol[3:]
    return Instrument(
        symbol=symbol,
        market="forex",
        provider="yahoo",
        provider_symbol=yahoo or f"{symbol}=X",
        mt5_symbol=symbol,
        digits=digits,
        pip_size=pip,
        contract_size=100_000.0,
        typical_spread_pips=spread,
        quote_currency=quote,
        base_currency=base,
        active_sessions=("london", "newyork", "tokyo", "sydney"),
        trades_weekends=False,
    )


def _crypto(symbol: str, spread_pips: float, digits: int, pip: float) -> Instrument:
    """Helper for Binance spot pairs. Crypto CFDs on MT5 are 1 unit per lot."""
    return Instrument(
        symbol=symbol,
        market="crypto",
        provider="binance",
        provider_symbol=symbol,
        mt5_symbol=symbol.replace("USDT", "USD"),
        digits=digits,
        pip_size=pip,
        contract_size=1.0,
        typical_spread_pips=spread_pips,
        quote_currency="USD",
        base_currency=symbol.replace("USDT", ""),
        active_sessions=(),
        trades_weekends=True,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=50.0,
    )


# JPY pairs quote to 3 decimals, so a pip is 0.01 rather than 0.0001.
INSTRUMENTS: dict[str, Instrument] = {
    # ---- Major FX -------------------------------------------------------
    "EURUSD": _fx("EURUSD", 0.8),
    "GBPUSD": _fx("GBPUSD", 1.2),
    "AUDUSD": _fx("AUDUSD", 1.0),
    "NZDUSD": _fx("NZDUSD", 1.6),
    "USDCAD": _fx("USDCAD", 1.4),
    "USDCHF": _fx("USDCHF", 1.3),
    "USDJPY": _fx("USDJPY", 0.9, digits=3, pip=0.01),
    # ---- FX crosses -----------------------------------------------------
    "EURGBP": _fx("EURGBP", 1.3),
    "EURJPY": _fx("EURJPY", 1.4, digits=3, pip=0.01),
    "GBPJPY": _fx("GBPJPY", 2.2, digits=3, pip=0.01),
    "AUDJPY": _fx("AUDJPY", 1.8, digits=3, pip=0.01),
    "EURAUD": _fx("EURAUD", 2.0),
    # ---- Metals ---------------------------------------------------------
    "XAUUSD": Instrument(
        symbol="XAUUSD",
        market="metals",
        provider="yahoo",
        provider_symbol="GC=F",
        mt5_symbol="XAUUSD",
        digits=2,
        pip_size=0.1,
        contract_size=100.0,
        typical_spread_pips=2.5,
        quote_currency="USD",
        base_currency="XAU",
        active_sessions=("london", "newyork"),
        notes="Gold futures proxy from Yahoo; spot gold differs slightly.",
    ),
    "XAGUSD": Instrument(
        symbol="XAGUSD",
        market="metals",
        provider="yahoo",
        provider_symbol="SI=F",
        mt5_symbol="XAGUSD",
        digits=3,
        pip_size=0.01,
        contract_size=5000.0,
        typical_spread_pips=3.0,
        quote_currency="USD",
        base_currency="XAG",
        active_sessions=("london", "newyork"),
    ),
    # ---- Energy ---------------------------------------------------------
    "USOIL": Instrument(
        symbol="USOIL",
        market="energy",
        provider="yahoo",
        provider_symbol="CL=F",
        mt5_symbol="USOIL",
        digits=2,
        pip_size=0.01,
        contract_size=1000.0,
        typical_spread_pips=3.0,
        quote_currency="USD",
        base_currency="WTI",
        active_sessions=("london", "newyork"),
    ),
    # ---- Index CFDs -----------------------------------------------------
    "US500": Instrument(
        symbol="US500",
        market="indices",
        provider="yahoo",
        provider_symbol="ES=F",
        mt5_symbol="US500",
        digits=2,
        pip_size=0.1,
        contract_size=50.0,
        typical_spread_pips=4.0,
        quote_currency="USD",
        base_currency="SPX",
        active_sessions=("newyork", "london"),
    ),
    "NAS100": Instrument(
        symbol="NAS100",
        market="indices",
        provider="yahoo",
        provider_symbol="NQ=F",
        mt5_symbol="NAS100",
        digits=2,
        pip_size=0.1,
        contract_size=20.0,
        typical_spread_pips=6.0,
        quote_currency="USD",
        base_currency="NDX",
        active_sessions=("newyork", "london"),
    ),
    "GER40": Instrument(
        symbol="GER40",
        market="indices",
        provider="yahoo",
        provider_symbol="^GDAXI",
        mt5_symbol="GER40",
        digits=2,
        pip_size=0.1,
        contract_size=25.0,
        typical_spread_pips=5.0,
        quote_currency="EUR",
        base_currency="DAX",
        active_sessions=("london",),
    ),
    "JP225": Instrument(
        symbol="JP225",
        market="indices",
        provider="yahoo",
        provider_symbol="^N225",
        mt5_symbol="JP225",
        digits=1,
        pip_size=1.0,
        contract_size=100.0,
        typical_spread_pips=8.0,
        quote_currency="JPY",
        base_currency="NKY",
        active_sessions=("tokyo",),
    ),
    # ---- Crypto ---------------------------------------------------------
    "BTCUSDT": _crypto("BTCUSDT", 30.0, 2, 1.0),
    "ETHUSDT": _crypto("ETHUSDT", 15.0, 2, 0.1),
    "SOLUSDT": _crypto("SOLUSDT", 8.0, 3, 0.01),
    "BNBUSDT": _crypto("BNBUSDT", 10.0, 2, 0.1),
    "XRPUSDT": _crypto("XRPUSDT", 6.0, 5, 0.0001),
    "ADAUSDT": _crypto("ADAUSDT", 5.0, 5, 0.0001),
    "DOGEUSDT": _crypto("DOGEUSDT", 6.0, 6, 0.00001),
    "AVAXUSDT": _crypto("AVAXUSDT", 8.0, 3, 0.01),
    "LINKUSDT": _crypto("LINKUSDT", 8.0, 3, 0.01),
    "MATICUSDT": _crypto("MATICUSDT", 7.0, 5, 0.0001),
}

# Which currencies each instrument is exposed to. Used to map economic-calendar
# events onto instruments: an ECB decision matters for EURUSD, not for BTCUSDT.
INSTRUMENT_CURRENCIES: dict[str, tuple[str, ...]] = {}
for _sym, _inst in INSTRUMENTS.items():
    if _inst.market == "forex":
        INSTRUMENT_CURRENCIES[_sym] = (_sym[:3], _sym[3:])
    elif _inst.market in {"metals", "energy"}:
        INSTRUMENT_CURRENCIES[_sym] = ("USD",)
    elif _inst.market == "indices":
        INSTRUMENT_CURRENCIES[_sym] = (_inst.quote_currency,)
    else:  # crypto reacts to USD macro, but weakly
        INSTRUMENT_CURRENCIES[_sym] = ("USD",)


def get_instrument(symbol: str) -> Instrument:
    """Look up an instrument, with a helpful error if it is unknown."""
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown instrument {symbol!r}. "
            f"Known symbols: {', '.join(sorted(INSTRUMENTS))}"
        ) from None


def mt5_name(symbol: str, suffix: str = "") -> str:
    """The exact string to type into the MT5 mobile search box."""
    return f"{get_instrument(symbol).mt5_symbol}{suffix}"


def instruments_by_market(market: str) -> list[Instrument]:
    return [i for i in INSTRUMENTS.values() if i.market == market]


def is_session_active(session: str, hour_utc: int, minute_utc: int = 0) -> bool:
    """Whether a named session is open at the given UTC time."""
    if session not in SESSIONS:
        return False
    start, end = SESSIONS[session]
    now = time(hour_utc, minute_utc)
    if start <= end:
        return start <= now < end
    # Session wraps past midnight (Sydney).
    return now >= start or now < end


def active_sessions(hour_utc: int, minute_utc: int = 0) -> list[str]:
    return [s for s in SESSIONS if is_session_active(s, hour_utc, minute_utc)]
