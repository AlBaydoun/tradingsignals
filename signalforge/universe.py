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

from dataclasses import dataclass, field, replace
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
    # True when mt5_symbol came from the user's config and is therefore already
    # the exact Market Watch string. The global broker suffix is not appended.
    mt5_symbol_is_exact: bool = False
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
    "BRENT": Instrument(
        symbol="BRENT",
        market="energy",
        provider="yahoo",
        provider_symbol="BZ=F",
        mt5_symbol="BRENT",
        digits=2,
        pip_size=0.01,
        contract_size=1000.0,
        typical_spread_pips=4.0,
        quote_currency="USD",
        base_currency="BRENT",
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
        # 6 pips at 0.1 = 0.6 index points.
        typical_spread_pips=6.0,
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
        # 18 pips at 0.1 = 1.8 index points, a realistic retail Nasdaq spread.
        typical_spread_pips=18.0,
        quote_currency="USD",
        base_currency="NDX",
        active_sessions=("newyork", "london"),
    ),
    "US30": Instrument(
        symbol="US30",
        market="indices",
        provider="yahoo",
        provider_symbol="YM=F",
        mt5_symbol="US30",
        digits=1,
        pip_size=1.0,
        contract_size=1.0,
        typical_spread_pips=3.0,
        quote_currency="USD",
        base_currency="DJI",
        active_sessions=("newyork", "london"),
        notes="Dow. Broker contract sizes vary widely — verify in MT5.",
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
        # 12 pips at 0.1 = 1.2 index points.
        typical_spread_pips=12.0,
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


def _rebuild_currency_map() -> None:
    """Recompute the currency exposure map from the current INSTRUMENTS."""
    INSTRUMENT_CURRENCIES.clear()
    for sym, inst in INSTRUMENTS.items():
        if inst.market == "forex":
            INSTRUMENT_CURRENCIES[sym] = (sym[:3], sym[3:])
        elif inst.market in {"metals", "energy"}:
            INSTRUMENT_CURRENCIES[sym] = ("USD",)
        elif inst.market == "indices":
            INSTRUMENT_CURRENCIES[sym] = (inst.quote_currency,)
        else:  # crypto reacts to USD macro, but weakly
            INSTRUMENT_CURRENCIES[sym] = ("USD",)


_rebuild_currency_map()


# ---------------------------------------------------------------------------
# Broker naming
# ---------------------------------------------------------------------------
# The same market has a different name at every broker. These are the names the
# engine will silently accept and translate to its own canonical symbol, so
# `train --symbols US100 WTI BTCUSD` works without anyone reading a table first.
ALIASES: dict[str, str] = {
    # Indices
    "US100": "NAS100",
    "USTEC": "NAS100",
    "NAS": "NAS100",
    "NDX100": "NAS100",
    "NASDAQ": "NAS100",
    "SPX500": "US500",
    "US500CASH": "US500",
    "SP500": "US500",
    "DJ30": "US30",
    "DOW": "US30",
    "WALLSTREET": "US30",
    "US30CASH": "US30",
    "DAX40": "GER40",
    "GER30": "GER40",
    "DE40": "GER40",
    "NIKKEI": "JP225",
    "JPN225": "JP225",
    # Energy
    "WTI": "USOIL",
    "CRUDE": "USOIL",
    "OIL": "USOIL",
    "USCRUDE": "USOIL",
    "XTIUSD": "USOIL",
    "UKOIL": "BRENT",
    "BRENTOIL": "BRENT",
    "XBRUSD": "BRENT",
    # Metals
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",
    "XAU": "XAUUSD",
    "XAG": "XAGUSD",
    # Crypto: MT5 brokers quote against USD, the data comes from a USDT pair.
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "BNBUSD": "BNBUSDT",
    "XRPUSD": "XRPUSDT",
    "ADAUSD": "ADAUSDT",
    "DOGEUSD": "DOGEUSDT",
    "AVAXUSD": "AVAXUSDT",
    "LINKUSD": "LINKUSDT",
    "MATICUSD": "MATICUSDT",
    "BITCOIN": "BTCUSDT",
    "ETHEREUM": "ETHUSDT",
}

# Fields a user may override per instrument in config.yaml. Anything else is
# rejected loudly rather than ignored quietly — a typo in a spread setting that
# silently does nothing is worse than an error.
OVERRIDABLE_FIELDS = frozenset(
    {
        "mt5_symbol",
        "typical_spread_pips",
        "commission_per_lot",
        "contract_size",
        "digits",
        "pip_size",
        "min_lot",
        "lot_step",
        "max_lot",
        "notes",
    }
)


def resolve_symbol(name: str) -> str:
    """Translate a broker or colloquial name into the engine's canonical one.

    `resolve_symbol("US100.std")` is `"NAS100"`, and so is `resolve_symbol("us100")`.
    Broker suffixes are stripped only when what remains is recognisable, so an
    unknown name still produces a useful error rather than a mangled one.
    """
    raw = name.strip().upper()
    if raw in INSTRUMENTS:
        return raw
    if raw in ALIASES:
        return ALIASES[raw]

    # Try again without a broker suffix: "US100.STD", "EURUSD_ECN", "BTCUSD#".
    stripped = raw
    for separator in (".", "_", "-", "#", "+"):
        stripped = stripped.split(separator, 1)[0]
    stripped = stripped.rstrip("#+")
    if stripped in INSTRUMENTS:
        return stripped
    if stripped in ALIASES:
        return ALIASES[stripped]

    # Also check any instrument whose configured MT5 name matches exactly. This
    # catches overrides the user set themselves, e.g. "WTI.M" -> USOIL.
    for sym, inst in INSTRUMENTS.items():
        if inst.mt5_symbol.upper() == raw:
            return sym
    return raw


def apply_overrides(overrides: dict[str, dict]) -> list[str]:
    """Apply per-instrument settings from config onto the universe.

    This exists because real brokers are inconsistent: the same account may
    carry `XAUUSD`, `US100.std` and `WTI.m` side by side, which no single global
    suffix can express. Setting `mt5_symbol` here marks it exact, so the global
    suffix is not appended on top of it.

    Returns the list of canonical symbols that were changed.
    """
    changed: list[str] = []
    for name, settings in (overrides or {}).items():
        if not settings:
            continue
        symbol = resolve_symbol(name)
        if symbol not in INSTRUMENTS:
            raise KeyError(
                f"Override for unknown instrument {name!r}. "
                f"Known symbols: {', '.join(sorted(INSTRUMENTS))}"
            )
        unknown = set(settings) - OVERRIDABLE_FIELDS
        if unknown:
            raise KeyError(
                f"Cannot override {sorted(unknown)} on {symbol}. "
                f"Overridable fields: {', '.join(sorted(OVERRIDABLE_FIELDS))}"
            )
        patch = dict(settings)
        if "mt5_symbol" in patch:
            patch["mt5_symbol_is_exact"] = True
        INSTRUMENTS[symbol] = replace(INSTRUMENTS[symbol], **patch)
        changed.append(symbol)

    if changed:
        _rebuild_currency_map()
    return changed


def get_instrument(symbol: str) -> Instrument:
    """Look up an instrument by canonical name, alias, or broker name."""
    resolved = resolve_symbol(symbol)
    try:
        return INSTRUMENTS[resolved]
    except KeyError:
        raise KeyError(
            f"Unknown instrument {symbol!r}. "
            f"Known symbols: {', '.join(sorted(INSTRUMENTS))}"
        ) from None


def mt5_name(symbol: str, suffix: str = "") -> str:
    """The exact string to type into the MT5 mobile search box.

    An instrument whose MT5 name was set explicitly in config is returned
    verbatim; the global suffix only applies to the ones that were not.
    """
    instrument = get_instrument(symbol)
    if instrument.mt5_symbol_is_exact:
        return instrument.mt5_symbol
    return f"{instrument.mt5_symbol}{suffix}"


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
