"""The OHLCV data contract shared by every provider.

One rule holds the whole engine together: a bar timestamped `t` contains only
information that was available at the *close* of `t`. Every provider normalises
to that convention, and every feature is computed on top of it. Break this and
your backtest will look wonderful and lose money live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Timeframe:
    """A chart timeframe and its provider-specific spellings."""

    name: str
    minutes: int
    binance: str
    yahoo: str
    pandas_rule: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


TIMEFRAMES: dict[str, Timeframe] = {
    "M1": Timeframe("M1", 1, "1m", "1m", "1min"),
    "M5": Timeframe("M5", 5, "5m", "5m", "5min"),
    "M15": Timeframe("M15", 15, "15m", "15m", "15min"),
    "M30": Timeframe("M30", 30, "30m", "30m", "30min"),
    "H1": Timeframe("H1", 60, "1h", "60m", "1h"),
    "H4": Timeframe("H4", 240, "4h", "1h", "4h"),
    "D1": Timeframe("D1", 1440, "1d", "1d", "1D"),
}


def timeframe_minutes(timeframe: str) -> int:
    try:
        return TIMEFRAMES[timeframe.upper()].minutes
    except KeyError:
        raise KeyError(
            f"Unknown timeframe {timeframe!r}. Known: {', '.join(TIMEFRAMES)}"
        ) from None


def get_timeframe(timeframe: str) -> Timeframe:
    try:
        return TIMEFRAMES[timeframe.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown timeframe {timeframe!r}. Known: {', '.join(TIMEFRAMES)}"
        ) from None


class Provider(Protocol):
    """Anything that can hand us bars."""

    name: str

    def fetch(self, provider_symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Return a UTC-indexed OHLCV frame with at most `bars` rows."""
        ...


def validate_ohlcv(df: pd.DataFrame, *, symbol: str = "?") -> pd.DataFrame:
    """Normalise and sanity-check a raw provider frame.

    Removes the pathologies that silently poison a backtest: duplicate
    timestamps, unsorted indexes, zero-range bars, and bars whose high is below
    their low (which some providers emit during outages).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: provider frame missing columns {missing}")

    out = df[OHLCV_COLUMNS].copy()

    # Index must be a tz-aware UTC DatetimeIndex.
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: index must be a DatetimeIndex")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    # Pin the datetime resolution. Providers hand back milliseconds, the CSV
    # cache reads back microseconds, and pandas refuses to merge_asof across
    # mismatched units — so normalise once, here, at the boundary.
    out.index = out.index.as_unit("ns")

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.astype("float64")

    # Drop rows where price data is unusable.
    valid = (
        out[["open", "high", "low", "close"]].notna().all(axis=1)
        & (out[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (out["high"] >= out["low"])
    )
    dropped = int((~valid).sum())
    out = out[valid]
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

    # Some CFD feeds report high/low that do not bracket open/close. Repair
    # rather than drop, since the close is the number we actually trade on.
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)

    out.attrs["symbol"] = symbol
    out.attrs["rows_dropped"] = dropped
    return out


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate bars up to a higher timeframe.

    Used to build H4 from H1 (Yahoo has no native 4h) and to derive context
    timeframes without extra network calls.
    """
    tf = get_timeframe(timeframe)
    if df.empty:
        return df

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(tf.pandas_rule, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.attrs.update(df.attrs)
    return out


def align_to_signal_frame(
    context: pd.DataFrame, signal_index: pd.DatetimeIndex, prefix: str
) -> pd.DataFrame:
    """Merge higher-timeframe values onto a lower-timeframe index, safely.

    `merge_asof` with `direction="backward"` guarantees each signal bar only
    sees the most recent *completed* higher-timeframe bar. Using a plain
    reindex-and-forward-fill here is the single most common way to leak future
    information into a backtest.
    """
    if context.empty or len(signal_index) == 0:
        return pd.DataFrame(index=signal_index)

    ctx = context.copy()
    ctx.columns = [f"{prefix}_{c}" for c in ctx.columns]
    ctx = ctx.sort_index()
    ctx.index = pd.DatetimeIndex(ctx.index).as_unit("ns")
    signal_index = pd.DatetimeIndex(signal_index).as_unit("ns")

    left = pd.DataFrame(index=signal_index.sort_values()).reset_index()
    left.columns = ["_ts"]
    right = ctx.reset_index()
    right.columns = ["_ts"] + list(ctx.columns)

    merged = pd.merge_asof(left, right, on="_ts", direction="backward")
    merged = merged.set_index("_ts")
    merged.index.name = signal_index.name
    return merged


def bars_needed(timeframe: str, lookback_days: float) -> int:
    """How many bars of `timeframe` cover roughly `lookback_days` of calendar time."""
    minutes = timeframe_minutes(timeframe)
    return max(1, int((lookback_days * 24 * 60) / minutes))


def infer_gaps(df: pd.DataFrame, timeframe: str) -> pd.Series:
    """Flag bars that follow an unexpected gap in the series.

    Weekend gaps in FX are normal; a missing hour mid-London is a data problem
    the features should know about rather than silently smooth over.
    """
    if df.empty:
        return pd.Series(dtype="float64")
    expected = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    deltas = pd.Series(df.index, index=df.index).diff()
    return (deltas > expected * 1.5).astype("float64").fillna(0.0)


def latest_complete_bar(df: pd.DataFrame, timeframe: str) -> pd.Timestamp | None:
    """The timestamp of the last bar we are confident has closed.

    Providers return the in-progress bar, whose high/low/close will still move.
    Trading on it is a subtle form of lookahead, so the engine drops it.
    """
    if df.empty:
        return None
    now = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
    period = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    last = df.index[-1]
    if now - last < period:
        return df.index[-2] if len(df) > 1 else None
    return last


def drop_incomplete_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Remove the still-forming bar so features never see partial data."""
    cutoff = latest_complete_bar(df, timeframe)
    if cutoff is None:
        return df
    out = df.loc[:cutoff]
    out.attrs.update(df.attrs)
    return out


def summarize(df: pd.DataFrame) -> dict[str, object]:
    """A compact description of a price series, for logs and health checks."""
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "start": df.index[0].isoformat(),
        "end": df.index[-1].isoformat(),
        "last_close": float(df["close"].iloc[-1]),
        "median_volume": float(np.median(df["volume"])) if len(df) else 0.0,
    }
