"""Technical indicators, implemented directly on numpy/pandas.

Written from the source definitions rather than pulled from a TA package, for
three reasons: no compiled dependency to break on install, every function is
strictly causal (no value at bar *t* uses data from *t+1*), and the smoothing
conventions are explicit — Wilder's smoothing and a simple EMA differ, and that
difference moves an RSI reading by enough to change a trade decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — the one RSI, ATR and ADX are actually defined with."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype="float64")
    return series.rolling(period, min_periods=period).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull moving average: fast, and much less laggy than an EMA."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    return wma(2 * wma(series, half) - wma(series, period), sqrt_p)


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_ema(true_range(high, low, close), period)


def realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualisation is deliberately omitted — we compare it to itself."""
    return np.log(close / close.shift(1)).rolling(period, min_periods=period).std()


def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Uses the high-low range, so it converges far faster than close-to-close."""
    factor = 1.0 / (4.0 * np.log(2.0))
    log_hl = np.log(high / low) ** 2
    return np.sqrt(factor * log_hl.rolling(period, min_periods=period).mean())


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Adds open/close information to Parkinson; the most efficient OHLC estimator."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    value = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(value.rolling(period, min_periods=period).mean().clip(lower=0))


def rogers_satchell_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Unlike Garman-Klass, stays unbiased when the market is trending."""
    term = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(
        low / open_
    )
    return np.sqrt(term.rolling(period, min_periods=period).mean().clip(lower=0))


def volatility_of_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """How unstable the volatility itself is — regime changes show up here first."""
    vol = realized_volatility(close, period)
    return vol.rolling(period, min_periods=period).std()


# --------------------------------------------------------------------------
# Momentum / oscillators
# --------------------------------------------------------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_ema(gain, period)
    avg_loss = wilder_ema(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # A run with no losses is RSI 100, not NaN.
    return out.where(avg_loss != 0.0, 100.0)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    percent_k = 100.0 * (close - lowest) / denom
    return percent_k, percent_k.rolling(d, min_periods=d).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period, min_periods=period).mean()
    # CCI uses mean absolute deviation, not standard deviation.
    mad = typical.rolling(period, min_periods=period).apply(
        lambda w: float(np.mean(np.abs(w - w.mean()))), raw=True
    )
    return (typical - mean) / (0.015 * mad.replace(0.0, np.nan))


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0.0, np.nan)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return 100.0 * (series / series.shift(period) - 1.0)


def trix(series: pd.Series, period: int = 15) -> pd.Series:
    """Triple-smoothed rate of change — filters out most intraday noise."""
    smoothed = ema(ema(ema(series, period), period), period)
    return 100.0 * smoothed.pct_change()


def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    median = (high + low) / 2.0
    return sma(median, 5) - sma(median, 34)


def ultimate_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    prev_close = close.shift(1)
    buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = true_range(high, low, close)
    out = pd.Series(0.0, index=close.index)
    for period, weight in ((7, 4.0), (14, 2.0), (28, 1.0)):
        bp_sum = buying_pressure.rolling(period, min_periods=period).sum()
        tr_sum = tr.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
        out = out + weight * (bp_sum / tr_sum)
    return 100.0 * out / 7.0


# --------------------------------------------------------------------------
# Trend strength and direction
# --------------------------------------------------------------------------


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index plus the two directional indicators.

    Returns (adx, plus_di, minus_di). ADX measures how *strongly* price trends,
    saying nothing about direction — that is what the DI pair is for.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr_ = wilder_ema(true_range(high, low, close), period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder_ema(plus_dm, period) / atr_
    minus_di = 100.0 * wilder_ema(minus_dm, period) / atr_

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder_ema(dx, period), plus_di, minus_di


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Supertrend line and its direction (+1 up, -1 down).

    The band-ratcheting logic is genuinely sequential, so this runs as an
    explicit loop rather than a vectorised expression.
    """
    atr_ = atr(high, low, close, period)
    median = (high + low) / 2.0
    upper = (median + multiplier * atr_).to_numpy()
    lower = (median - multiplier * atr_).to_numpy()
    close_arr = close.to_numpy()

    n = len(close_arr)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.full(n, 1.0)
    trend = np.full(n, np.nan)

    for i in range(n):
        if i == 0 or np.isnan(upper[i]) or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            continue
        # Bands only tighten in the direction of the trend; they never loosen.
        final_upper[i] = (
            min(upper[i], final_upper[i - 1])
            if close_arr[i - 1] <= final_upper[i - 1]
            else upper[i]
        )
        final_lower[i] = (
            max(lower[i], final_lower[i - 1])
            if close_arr[i - 1] >= final_lower[i - 1]
            else lower[i]
        )
        if close_arr[i] > final_upper[i - 1]:
            direction[i] = 1.0
        elif close_arr[i] < final_lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
        trend[i] = final_lower[i] if direction[i] > 0 else final_upper[i]

    return (
        pd.Series(trend, index=close.index),
        pd.Series(direction, index=close.index),
    )


def parabolic_sar(
    high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2
) -> pd.Series:
    """Wilder's stop-and-reverse. Inherently sequential."""
    high_arr, low_arr = high.to_numpy(), low.to_numpy()
    n = len(high_arr)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=high.index)

    bull = True
    af = step
    extreme = high_arr[0]
    sar[0] = low_arr[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (extreme - sar[i - 1])
        if bull:
            # The SAR may never move inside the previous two bars' range.
            sar[i] = min(sar[i], low_arr[i - 1], low_arr[max(0, i - 2)])
            if low_arr[i] < sar[i]:
                bull, sar[i], extreme, af = False, extreme, low_arr[i], step
            elif high_arr[i] > extreme:
                extreme, af = high_arr[i], min(af + step, max_step)
        else:
            sar[i] = max(sar[i], high_arr[i - 1], high_arr[max(0, i - 2)])
            if high_arr[i] > sar[i]:
                bull, sar[i], extreme, af = True, extreme, high_arr[i], step
            elif low_arr[i] < extreme:
                extreme, af = low_arr[i], min(af + step, max_step)

    return pd.Series(sar, index=high.index)


def vortex(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series]:
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr_sum = true_range(high, low, close).rolling(period, min_periods=period).sum()
    tr_sum = tr_sum.replace(0.0, np.nan)
    return (
        vm_plus.rolling(period, min_periods=period).sum() / tr_sum,
        vm_minus.rolling(period, min_periods=period).sum() / tr_sum,
    )


def efficiency_ratio(series: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman's ratio of net movement to total path length.

    Near 1.0 means a clean directional move; near 0 means the market covered a
    lot of distance and went nowhere. One of the best trend/chop separators.
    """
    direction = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period, min_periods=period).sum()
    return direction / volatility.replace(0.0, np.nan)


def choppiness_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Above ~61 the market is ranging; below ~38 it is trending."""
    tr_sum = true_range(high, low, close).rolling(period, min_periods=period).sum()
    rng = (
        high.rolling(period, min_periods=period).max()
        - low.rolling(period, min_periods=period).min()
    )
    ratio = tr_sum / rng.replace(0.0, np.nan)
    return 100.0 * np.log10(ratio.clip(lower=1e-10)) / np.log10(period)


def hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling rescaled-range Hurst estimate.

    H > 0.5 suggests persistence (trend-following works), H < 0.5 suggests
    mean reversion, and a pure random walk sits at 0.5. The estimate is noisy on
    short windows — treat it as a soft prior, not a decision rule.

    Computed on the log *price* path, not on returns: the 0.5 random-walk
    reference point only holds for the undifferenced series.
    """
    log_price = np.log(series)

    def _hurst(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]
        if len(values) < 20:
            return np.nan
        lags = range(2, min(20, len(values) // 2))
        tau = []
        for lag in lags:
            diff = values[lag:] - values[:-lag]
            # Root-mean-square, not standard deviation. `std` subtracts the
            # mean of the differences, which is exactly the drift term that
            # makes a trending series persistent — using it would report a
            # strong trend as *less* persistent than a random walk.
            rms = float(np.sqrt(np.mean(diff**2)))
            tau.append(rms if rms > 0 else 1e-10)
        try:
            slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return np.nan
        return float(slope)

    return log_price.rolling(window, min_periods=window // 2).apply(_hurst, raw=True)


def linreg_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Slope of a least-squares fit, normalised by price so it compares across symbols."""
    x = np.arange(period, dtype="float64")
    x_centred = x - x.mean()
    denom = float((x_centred**2).sum())

    def _slope(window: np.ndarray) -> float:
        return float(np.dot(x_centred, window - window.mean()) / denom)

    slope = series.rolling(period, min_periods=period).apply(_slope, raw=True)
    return slope / series.replace(0.0, np.nan)


def linreg_r2(series: pd.Series, period: int = 20) -> pd.Series:
    """How well a straight line explains recent price — trend 'cleanliness'."""
    x = np.arange(period, dtype="float64")

    def _r2(window: np.ndarray) -> float:
        if np.std(window) == 0:
            return 0.0
        corr = np.corrcoef(x, window)[0, 1]
        return float(corr**2) if np.isfinite(corr) else 0.0

    return series.rolling(period, min_periods=period).apply(_r2, raw=True)


# --------------------------------------------------------------------------
# Channels and bands
# --------------------------------------------------------------------------


def bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    return mid, mid + num_std * std, mid - num_std * std


def bollinger_width(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    mid, upper, lower = bollinger(series, period, num_std)
    return (upper - lower) / mid.replace(0.0, np.nan)


def bollinger_percent_b(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    """Where price sits inside the bands: 0 = lower band, 1 = upper band."""
    _, upper, lower = bollinger(series, period, num_std)
    return (series - lower) / (upper - lower).replace(0.0, np.nan)


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_mult: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(close, period)
    band = atr_mult * atr(high, low, close, period)
    return mid, mid + band, mid - band


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Note the shift: today's channel must be built from bars before today."""
    upper = high.rolling(period, min_periods=period).max().shift(1)
    lower = low.rolling(period, min_periods=period).min().shift(1)
    return (upper + lower) / 2.0, upper, lower


def squeeze_on(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Bollinger Bands inside Keltner Channels — the classic coiled spring.

    A squeeze does not say which way price will break, only that the break is
    likely to be violent when it comes.
    """
    _, bb_up, bb_low = bollinger(close, period, 2.0)
    _, kc_up, kc_low = keltner(high, low, close, period, 1.5)
    return ((bb_up < kc_up) & (bb_low > kc_low)).astype("float64")


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def money_flow_index(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """RSI weighted by volume."""
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    positive = raw_flow.where(typical > typical.shift(1), 0.0)
    negative = raw_flow.where(typical < typical.shift(1), 0.0)
    pos_sum = positive.rolling(period, min_periods=period).sum()
    neg_sum = negative.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def chaikin_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Where in each bar's range the close sits, weighted by volume."""
    span = (high - low).replace(0.0, np.nan)
    multiplier = ((close - low) - (high - close)) / span
    flow = multiplier * volume
    vol_sum = volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return flow.rolling(period, min_periods=period).sum() / vol_sum


def rolling_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(period, min_periods=period).sum()
    vol = volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return pv / vol


def volume_zscore(volume: pd.Series, period: int = 50) -> pd.Series:
    """How unusual current volume is. The primary ingredient of breakout detection."""
    mean = volume.rolling(period, min_periods=period).mean()
    std = volume.rolling(period, min_periods=period).std().replace(0.0, np.nan)
    return (volume - mean) / std


def relative_volume(volume: pd.Series, period: int = 50) -> pd.Series:
    mean = volume.rolling(period, min_periods=period).mean().replace(0.0, np.nan)
    return volume / mean


# --------------------------------------------------------------------------
# Ichimoku
# --------------------------------------------------------------------------


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> dict[str, pd.Series]:
    """Ichimoku components, kept strictly causal.

    The cloud is conventionally *plotted* 26 bars into the future. Here the
    spans are shifted forward, which means the value at bar t is derived from
    bar t-26 — past data, as required. The chikou span, which looks backwards
    from the future, is deliberately omitted: it cannot be used as a feature.
    """

    def midpoint(period: int) -> pd.Series:
        return (
            high.rolling(period, min_periods=period).max()
            + low.rolling(period, min_periods=period).min()
        ) / 2.0

    tenkan = midpoint(tenkan_period)
    kijun = midpoint(kijun_period)
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": ((tenkan + kijun) / 2.0).shift(kijun_period),
        "senkou_b": midpoint(senkou_b_period).shift(kijun_period),
    }


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def swing_highs(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Confirmed swing highs, marked at the bar where confirmation arrives.

    A swing high needs `right` bars after it to be confirmed, so the flag is
    shifted forward by `right`. Marking it on the pivot bar itself would be a
    lookahead bug — and a popular one.
    """
    rolling_max = high.rolling(left + right + 1, center=True).max()
    is_pivot = (high == rolling_max) & high.notna()
    return is_pivot.shift(right).fillna(False).astype("float64")


def swing_lows(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    rolling_min = low.rolling(left + right + 1, center=True).min()
    is_pivot = (low == rolling_min) & low.notna()
    return is_pivot.shift(right).fillna(False).astype("float64")


def last_swing_level(
    price: pd.Series, pivots: pd.Series, lookback: int = 50
) -> pd.Series:
    """Most recent confirmed pivot price, for structure-based stop placement."""
    values = price.where(pivots > 0)
    return values.ffill(limit=lookback)


def pivot_points(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, pd.Series]:
    """Classic floor-trader pivots from the previous bar."""
    prev_h, prev_l, prev_c = high.shift(1), low.shift(1), close.shift(1)
    pivot = (prev_h + prev_l + prev_c) / 3.0
    return {
        "pivot": pivot,
        "r1": 2 * pivot - prev_l,
        "s1": 2 * pivot - prev_h,
        "r2": pivot + (prev_h - prev_l),
        "s2": pivot - (prev_h - prev_l),
    }


def zscore(series: pd.Series, period: int = 100) -> pd.Series:
    mean = series.rolling(period, min_periods=period // 2).mean()
    std = series.rolling(period, min_periods=period // 2).std().replace(0.0, np.nan)
    return (series - mean) / std


def percentile_rank(series: pd.Series, period: int = 252) -> pd.Series:
    """Where the current value sits in its own recent history, from 0 to 1.

    More robust than a z-score for skewed quantities like ATR or volume.
    """
    return series.rolling(period, min_periods=period // 4).rank(pct=True)
