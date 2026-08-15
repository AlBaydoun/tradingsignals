"""Candle anatomy and bar-level microstructure.

Retail data feeds give no order book, so the shape of each candle is the best
available proxy for what buyers and sellers actually did inside the bar. A long
upper wick on heavy volume is sellers defending a level; the same range with the
close on the high is buyers absorbing supply. These features encode that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Body, wick and range geometry, all normalised by the bar's own range."""
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    rng = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()

    out = pd.DataFrame(index=df.index)
    out["body_ratio"] = body / rng
    out["upper_wick_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / rng
    out["lower_wick_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / rng
    # Where the bar closed inside its range: 1.0 = on the high, 0.0 = on the low.
    out["close_position"] = (close - low) / rng
    out["is_bullish"] = (close > open_).astype("float64")
    out["range_pct"] = rng / close
    out["body_pct"] = body / close
    # Direction of the wick imbalance: positive means rejection from above.
    out["wick_skew"] = out["upper_wick_ratio"] - out["lower_wick_ratio"]
    return out


def gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Opening gaps relative to the previous close.

    Matters most for FX over the weekend and index CFDs at the cash open.
    """
    out = pd.DataFrame(index=df.index)
    prev_close = df["close"].shift(1)
    out["gap_pct"] = (df["open"] - prev_close) / prev_close
    out["gap_abs_pct"] = out["gap_pct"].abs()
    # Whether the bar traded back through its own opening gap.
    filled_up = (df["low"] <= prev_close) & (df["open"] > prev_close)
    filled_down = (df["high"] >= prev_close) & (df["open"] < prev_close)
    out["gap_filled"] = (filled_up | filled_down).astype("float64")
    return out


def sequences(df: pd.DataFrame, max_run: int = 10) -> pd.DataFrame:
    """Consecutive up/down bars and range expansion streaks."""
    out = pd.DataFrame(index=df.index)
    direction = np.sign(df["close"].diff()).fillna(0.0)

    # Length of the current unbroken run of same-direction bars.
    run = np.zeros(len(direction))
    values = direction.to_numpy()
    for i in range(1, len(values)):
        if values[i] != 0 and values[i] == values[i - 1]:
            run[i] = min(run[i - 1] + 1, max_run)
        elif values[i] != 0:
            run[i] = 1
    out["consecutive_run"] = run * values
    out["abs_consecutive_run"] = np.abs(run)

    rng = df["high"] - df["low"]
    out["range_expansion"] = rng / rng.rolling(20, min_periods=10).mean().replace(
        0.0, np.nan
    )
    out["is_inside_bar"] = (
        (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    ).astype("float64")
    out["is_outside_bar"] = (
        (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    ).astype("float64")
    return out


def engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """The two reversal patterns that actually survive statistical testing."""
    out = pd.DataFrame(index=df.index)
    open_, close = df["open"], df["close"]
    prev_open, prev_close = open_.shift(1), close.shift(1)

    bull = (
        (close > open_)
        & (prev_close < prev_open)
        & (close >= prev_open)
        & (open_ <= prev_close)
    )
    bear = (
        (close < open_)
        & (prev_close > prev_open)
        & (close <= prev_open)
        & (open_ >= prev_close)
    )
    out["bullish_engulfing"] = bull.astype("float64")
    out["bearish_engulfing"] = bear.astype("float64")
    return out


def pin_bars(df: pd.DataFrame, wick_ratio: float = 0.6) -> pd.DataFrame:
    """Rejection candles: a dominant wick with the body pushed to one end."""
    anatomy = candle_anatomy(df)
    out = pd.DataFrame(index=df.index)
    out["bullish_pin"] = (
        (anatomy["lower_wick_ratio"] > wick_ratio) & (anatomy["close_position"] > 0.6)
    ).astype("float64")
    out["bearish_pin"] = (
        (anatomy["upper_wick_ratio"] > wick_ratio) & (anatomy["close_position"] < 0.4)
    ).astype("float64")
    return out


def spread_proxy(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Corwin-Schultz style effective-spread estimate from high/low ranges.

    Retail feeds hide the real spread; this recovers a usable proxy from the
    bar geometry, which is enough to spot when liquidity has evaporated.
    """
    high, low = df["high"], df["low"]
    beta = (np.log(high / low) ** 2).rolling(2).sum()
    high_2 = high.rolling(2).max()
    low_2 = low.rolling(2).min()
    gamma = np.log(high_2 / low_2) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0.0)
    estimate = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return estimate.rolling(period, min_periods=period // 2).median()


def order_flow_proxy(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume split into buy/sell pressure by where each bar closed.

    Without tick data this is an approximation, but the imbalance it produces
    tracks real order-flow imbalance closely enough to be predictive.
    """
    out = pd.DataFrame(index=df.index)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    buy_fraction = ((df["close"] - df["low"]) / rng).clip(0.0, 1.0).fillna(0.5)

    buy_volume = df["volume"] * buy_fraction
    sell_volume = df["volume"] * (1.0 - buy_fraction)

    total = df["volume"].rolling(period, min_periods=period // 2).sum().replace(0.0, np.nan)
    out["flow_imbalance"] = (
        buy_volume.rolling(period, min_periods=period // 2).sum()
        - sell_volume.rolling(period, min_periods=period // 2).sum()
    ) / total
    out["buy_pressure"] = buy_fraction
    # Volume per unit of price movement: high means absorption, low means a
    # thin market where price slides on little trade.
    out["volume_per_range"] = df["volume"] / (rng / df["close"]).replace(0.0, np.nan)
    out["volume_per_range_z"] = (
        out["volume_per_range"]
        - out["volume_per_range"].rolling(100, min_periods=30).mean()
    ) / out["volume_per_range"].rolling(100, min_periods=30).std().replace(0.0, np.nan)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """All microstructure features for a price frame."""
    parts = [
        candle_anatomy(df),
        gaps(df),
        sequences(df),
        engulfing(df),
        pin_bars(df),
        order_flow_proxy(df),
    ]
    out = pd.concat(parts, axis=1)
    out["spread_proxy"] = spread_proxy(df)
    return out
