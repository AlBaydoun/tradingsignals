"""Position sizing for MetaTrader 5.

The lot size is the only number in a signal that directly controls how much
money you lose when you are wrong, so it gets computed properly rather than
approximated.

Pip value depends on which currency the pair is quoted in:

* **Quote currency is the account currency** (EURUSD on a USD account) — pip
  value per lot is constant: `pip_size x contract_size`.
* **Base currency is the account currency** (USDJPY on a USD account) — pip
  value depends on the current rate and must be divided by it.
* **Neither** (EURGBP on a USD account) — a conversion rate is required. The
  engine fetches it; if it cannot, it says so on the signal rather than
  silently sizing the trade wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from signalforge.universe import Instrument

log = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """A sizing decision, with everything needed to audit it."""

    lots: float
    risk_amount: float  # account currency at stake if the stop is hit
    pip_value_per_lot: float
    stop_distance_pips: float
    # True when a cross-rate conversion had to be approximated.
    conversion_approximated: bool = False
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    @property
    def is_tradable(self) -> bool:
        return self.lots > 0

    def to_dict(self) -> dict:
        return {
            "lots": self.lots,
            "risk_amount": round(self.risk_amount, 2),
            "pip_value_per_lot": round(self.pip_value_per_lot, 4),
            "stop_distance_pips": round(self.stop_distance_pips, 1),
            "conversion_approximated": self.conversion_approximated,
            "warnings": self.warnings,
        }


def pip_value_per_lot(
    instrument: Instrument,
    current_price: float,
    account_currency: str = "USD",
    conversion_rate: float | None = None,
) -> tuple[float, bool, list[str]]:
    """Value of one pip, for one standard lot, in the account currency.

    Returns (value, was_approximated, warnings).
    """
    warnings: list[str] = []
    base_value = instrument.pip_size * instrument.contract_size

    quote = instrument.quote_currency.upper()
    base = instrument.base_currency.upper()
    account = account_currency.upper()

    # Case 1: profit is already in the account currency.
    if quote == account:
        return base_value, False, warnings

    # Case 2: the account currency is the base — divide by the current rate.
    # A USDJPY lot earns JPY; converting back to USD needs the USDJPY rate.
    if base == account:
        if current_price <= 0:
            warnings.append("no price available for pip-value conversion")
            return base_value, True, warnings
        return base_value / current_price, False, warnings

    # Case 3: a genuine cross. A conversion rate is required.
    if conversion_rate and conversion_rate > 0:
        return base_value * conversion_rate, False, warnings

    warnings.append(
        f"Cannot convert {quote} profit into {account} without a "
        f"{quote}{account} rate — lot size is an estimate. Verify in MT5 "
        "before executing."
    )
    return base_value, True, warnings


def calculate_lots(
    instrument: Instrument,
    *,
    entry_price: float,
    stop_price: float,
    account_balance: float,
    risk_percent: float,
    account_currency: str = "USD",
    conversion_rate: float | None = None,
) -> PositionSize:
    """Lots such that being stopped out costs exactly `risk_percent` of equity."""
    warnings: list[str] = []

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return PositionSize(0.0, 0.0, 0.0, 0.0, warnings=["stop equals entry"])

    stop_pips = stop_distance / instrument.pip_size
    risk_amount = account_balance * (risk_percent / 100.0)

    value_per_pip, approximated, conv_warnings = pip_value_per_lot(
        instrument, entry_price, account_currency, conversion_rate
    )
    warnings.extend(conv_warnings)

    risk_per_lot = stop_pips * value_per_pip
    if risk_per_lot <= 0:
        return PositionSize(
            0.0, 0.0, value_per_pip, stop_pips, approximated,
            warnings + ["computed zero risk per lot"],
        )

    raw_lots = risk_amount / risk_per_lot

    # Round *down* to the broker's lot step so risk never exceeds the target.
    lots = np.floor(raw_lots / instrument.lot_step) * instrument.lot_step
    lots = float(round(lots, 2))

    if lots < instrument.min_lot:
        if raw_lots >= instrument.min_lot * 0.5:
            warnings.append(
                f"Calculated size {raw_lots:.3f} is below the {instrument.min_lot} "
                f"minimum lot. Trading the minimum would risk "
                f"{(instrument.min_lot * risk_per_lot / account_balance * 100):.2f}% "
                "instead — skip the trade or reduce the stop distance."
            )
        else:
            warnings.append(
                f"Account too small for this stop distance ({stop_pips:.0f} pips)."
            )
        return PositionSize(0.0, 0.0, value_per_pip, stop_pips, approximated, warnings)

    if lots > instrument.max_lot:
        lots = instrument.max_lot
        warnings.append(f"Capped at the {instrument.max_lot} lot maximum.")

    actual_risk = lots * risk_per_lot
    return PositionSize(
        lots=lots,
        risk_amount=actual_risk,
        pip_value_per_lot=value_per_pip,
        stop_distance_pips=stop_pips,
        conversion_approximated=approximated,
        warnings=warnings,
    )


def portfolio_heat(
    open_risks: list[float], account_balance: float
) -> tuple[float, bool]:
    """Total open risk as a percentage of equity.

    Sizing each trade at 1% is meaningless if eight are open at once — that is
    an 8% position dressed up as eight small ones. Correlated instruments make
    it worse, since they tend to lose together.
    """
    total = sum(open_risks)
    percent = 100.0 * total / account_balance if account_balance > 0 else 0.0
    return percent, percent > 6.0


def correlation_adjusted_risk(
    symbol: str, open_symbols: list[str], base_risk_percent: float
) -> tuple[float, str | None]:
    """Cut position size when a correlated position is already open.

    Long EURUSD and long GBPUSD is close to one large short-dollar bet. Sizing
    both at full risk doubles the real exposure.
    """
    if not open_symbols:
        return base_risk_percent, None

    groups = {
        "usd_short": {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
        "usd_long": {"USDCAD", "USDCHF", "USDJPY"},
        "metals": {"XAUUSD", "XAGUSD"},
        "crypto_major": {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"},
        "indices": {"US500", "NAS100", "GER40", "JP225"},
    }

    symbol = symbol.upper()
    for name, members in groups.items():
        if symbol not in members:
            continue
        overlapping = [s for s in open_symbols if s.upper() in members]
        if not overlapping:
            continue
        # Halve for the first correlated position, then keep shrinking.
        factor = 1.0 / (1.0 + len(overlapping))
        return (
            round(base_risk_percent * factor, 3),
            f"Risk reduced to {base_risk_percent * factor:.2f}% — already exposed to "
            f"{name.replace('_', ' ')} via {', '.join(overlapping)}",
        )

    return base_risk_percent, None
