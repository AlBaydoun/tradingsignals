"""Assembles the model's feature matrix.

Three principles govern everything here:

1. **Causality.** No feature at bar `t` may use information from `t+1`. Every
   rolling window looks backwards; higher-timeframe context is joined with
   `merge_asof`, never a reindex-and-fill.
2. **Scale invariance.** Raw prices are useless to a model that must generalise
   across EURUSD at 1.08 and BTCUSDT at 63,000. Everything is expressed as a
   ratio, a z-score, a percentile, or a multiple of ATR.
3. **Stationarity.** A feature whose distribution drifts over the years teaches
   the model yesterday's market. Bounded oscillators and normalised distances
   are preferred over levels.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from signalforge.config import FeatureConfig
from signalforge.data.base import align_to_signal_frame, infer_gaps, timeframe_minutes
from signalforge.features import indicators as ta
from signalforge.features import microstructure as micro

log = logging.getLogger(__name__)


def _price_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Returns, distances from moving averages, and moving-average geometry."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # Log returns over several horizons: the model's raw view of momentum.
    log_close = np.log(close)
    for lag in cfg.return_lags:
        out[f"ret_{lag}"] = log_close.diff(lag)
        # Scaling by realised volatility makes a 20-pip move mean the same
        # thing in a quiet session and a violent one.
        out[f"ret_{lag}_vol_adj"] = out[f"ret_{lag}"] / ta.realized_volatility(
            close, cfg.volatility_lookback
        ).replace(0.0, np.nan)

    atr_ = ta.atr(df["high"], df["low"], close, cfg.atr_period)
    atr_safe = atr_.replace(0.0, np.nan)

    for period in cfg.ema_periods:
        ema_ = ta.ema(close, period)
        # Distance in ATR units, not percent: directly comparable across symbols.
        out[f"dist_ema_{period}_atr"] = (close - ema_) / atr_safe
        out[f"ema_{period}_slope"] = ema_.pct_change(5)

    # Moving-average stack: are the EMAs fanned out in order, or tangled?
    fast, slow = cfg.ema_periods[0], cfg.ema_periods[-1]
    out["ema_spread_atr"] = (ta.ema(close, fast) - ta.ema(close, slow)) / atr_safe
    ema_values = pd.concat([ta.ema(close, p) for p in cfg.ema_periods], axis=1)
    ordered_up = ema_values.apply(lambda r: r.is_monotonic_decreasing, axis=1)
    ordered_down = ema_values.apply(lambda r: r.is_monotonic_increasing, axis=1)
    out["ema_stack_bull"] = ordered_up.astype("float64")
    out["ema_stack_bear"] = ordered_down.astype("float64")

    out["hma_20_slope"] = ta.hma(close, 20).pct_change(3)
    out["linreg_slope_20"] = ta.linreg_slope(close, 20)
    out["linreg_slope_50"] = ta.linreg_slope(close, 50)
    out["linreg_r2_20"] = ta.linreg_r2(close, 20)
    return out


def _momentum_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    high, low, close = df["high"], df["low"], df["close"]

    for period in cfg.rsi_periods:
        rsi_ = ta.rsi(close, period)
        out[f"rsi_{period}"] = rsi_
        # Centred and scaled to roughly [-1, 1] for gradient stability.
        out[f"rsi_{period}_centered"] = (rsi_ - 50.0) / 50.0

    # RSI divergence: price makes a new extreme, momentum does not.
    rsi14 = ta.rsi(close, 14)
    price_high = close.rolling(14).max()
    rsi_high = rsi14.rolling(14).max()
    out["bear_divergence"] = (
        (close >= price_high * 0.999) & (rsi14 < rsi_high * 0.97)
    ).astype("float64")
    price_low = close.rolling(14).min()
    rsi_low = rsi14.rolling(14).min()
    out["bull_divergence"] = (
        (close <= price_low * 1.001) & (rsi14 > rsi_low * 1.03)
    ).astype("float64")

    macd_line, macd_signal, macd_hist = ta.macd(close)
    atr_safe = ta.atr(high, low, close, cfg.atr_period).replace(0.0, np.nan)
    out["macd_hist_atr"] = macd_hist / atr_safe
    out["macd_line_atr"] = macd_line / atr_safe
    out["macd_cross"] = np.sign(macd_line - macd_signal)
    out["macd_hist_slope"] = macd_hist.diff(3) / atr_safe

    k, d = ta.stochastic(high, low, close)
    out["stoch_k"] = k / 100.0
    out["stoch_d"] = d / 100.0
    out["stoch_cross"] = np.sign(k - d)

    out["cci_20"] = ta.cci(high, low, close, 20) / 100.0
    out["williams_r"] = ta.williams_r(high, low, close, 14) / 100.0
    out["roc_10"] = ta.roc(close, 10)
    out["roc_20"] = ta.roc(close, 20)
    out["trix_15"] = ta.trix(close, 15)
    out["awesome_osc"] = ta.awesome_oscillator(high, low) / atr_safe
    out["ultimate_osc"] = ta.ultimate_oscillator(high, low, close) / 100.0
    return out


def _trend_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    high, low, close = df["high"], df["low"], df["close"]

    adx_, plus_di, minus_di = ta.adx(high, low, close, cfg.adx_period)
    out["adx"] = adx_ / 100.0
    out["di_spread"] = (plus_di - minus_di) / 100.0
    out["adx_rising"] = (adx_.diff(3) > 0).astype("float64")

    st_line, st_dir = ta.supertrend(high, low, close)
    atr_safe = ta.atr(high, low, close, cfg.atr_period).replace(0.0, np.nan)
    out["supertrend_dir"] = st_dir
    out["supertrend_dist_atr"] = (close - st_line) / atr_safe

    sar = ta.parabolic_sar(high, low)
    out["sar_dist_atr"] = (close - sar) / atr_safe
    out["sar_bull"] = (close > sar).astype("float64")

    vi_plus, vi_minus = ta.vortex(high, low, close, 14)
    out["vortex_spread"] = vi_plus - vi_minus

    out["efficiency_ratio"] = ta.efficiency_ratio(close, 20)
    out["efficiency_ratio_50"] = ta.efficiency_ratio(close, 50)
    out["choppiness"] = ta.choppiness_index(high, low, close, 14) / 100.0
    out["hurst"] = ta.hurst_exponent(close, cfg.hurst_window)

    ich = ta.ichimoku(high, low)
    out["tenkan_kijun_atr"] = (ich["tenkan"] - ich["kijun"]) / atr_safe
    out["price_vs_cloud"] = np.sign(
        close - pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).max(axis=1)
    ) + np.sign(close - pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).min(axis=1))
    out["cloud_thickness_atr"] = (ich["senkou_a"] - ich["senkou_b"]).abs() / atr_safe
    return out


def _volatility_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    atr_ = ta.atr(high, low, close, cfg.atr_period)
    out["atr_pct"] = atr_ / close
    # Percentile rank is the key one: "is volatility high *for this symbol*".
    out["atr_percentile"] = ta.percentile_rank(atr_, 252)
    out["atr_zscore"] = ta.zscore(atr_, cfg.volatility_lookback)
    out["atr_ratio_fast_slow"] = ta.atr(high, low, close, 5) / atr_.replace(0.0, np.nan)

    out["realized_vol"] = ta.realized_volatility(close, 20)
    out["realized_vol_percentile"] = ta.percentile_rank(out["realized_vol"], 252)
    out["parkinson_vol"] = ta.parkinson_volatility(high, low, 20)
    out["garman_klass_vol"] = ta.garman_klass_volatility(open_, high, low, close, 20)
    out["rogers_satchell_vol"] = ta.rogers_satchell_volatility(
        open_, high, low, close, 20
    )
    out["vol_of_vol"] = ta.volatility_of_volatility(close, 20)
    # Rising short-term vol against stable long-term vol precedes expansion.
    out["vol_regime_shift"] = ta.realized_volatility(
        close, 10
    ) / ta.realized_volatility(close, 50).replace(0.0, np.nan)

    out["bb_width"] = ta.bollinger_width(close, cfg.bb_period, cfg.bb_std)
    out["bb_width_percentile"] = ta.percentile_rank(out["bb_width"], 252)
    out["bb_percent_b"] = ta.bollinger_percent_b(close, cfg.bb_period, cfg.bb_std)
    out["squeeze_on"] = ta.squeeze_on(high, low, close, cfg.bb_period)
    # How long the squeeze has been building — longer coils break harder.
    out["squeeze_duration"] = (
        out["squeeze_on"].groupby((out["squeeze_on"] == 0).cumsum()).cumsum()
    )

    kc_mid, kc_up, kc_low = ta.keltner(
        high, low, close, cfg.keltner_period, cfg.keltner_atr_mult
    )
    out["keltner_position"] = (close - kc_low) / (kc_up - kc_low).replace(0.0, np.nan)

    don_mid, don_up, don_low = ta.donchian(high, low, 20)
    out["donchian_position"] = (close - don_low) / (don_up - don_low).replace(0.0, np.nan)
    out["donchian_breakout_up"] = (close > don_up).astype("float64")
    out["donchian_breakout_down"] = (close < don_low).astype("float64")
    return out


def _volume_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    # Several instruments (spot FX via Yahoo) report no volume at all. Emit the
    # columns anyway so the feature matrix keeps a stable shape, and let the
    # model learn to ignore a constant.
    if volume.sum() <= 0:
        for name in (
            "volume_zscore",
            "relative_volume",
            "obv_slope",
            "mfi",
            "cmf",
            "vwap_dist_atr",
            "volume_trend_agreement",
        ):
            out[name] = 0.0
        out["has_volume"] = 0.0
        return out

    out["has_volume"] = 1.0
    out["volume_zscore"] = ta.volume_zscore(volume, cfg.volume_lookback)
    out["relative_volume"] = ta.relative_volume(volume, cfg.volume_lookback)
    out["obv_slope"] = ta.obv(close, volume).diff(10) / volume.rolling(
        cfg.volume_lookback, min_periods=10
    ).mean().replace(0.0, np.nan)
    out["mfi"] = ta.money_flow_index(high, low, close, volume, 14) / 100.0
    out["cmf"] = ta.chaikin_money_flow(high, low, close, volume, 20)

    vwap = ta.rolling_vwap(high, low, close, volume, 20)
    atr_safe = ta.atr(high, low, close, cfg.atr_period).replace(0.0, np.nan)
    out["vwap_dist_atr"] = (close - vwap) / atr_safe

    # Does volume confirm the price move, or is the move running on fumes?
    price_dir = np.sign(close.diff())
    vol_change = volume.pct_change()
    out["volume_trend_agreement"] = price_dir * np.sign(vol_change)
    return out


def _time_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Session and clock features.

    Hour-of-day is encoded as sine and cosine so that 23:00 and 00:00 are
    adjacent to the model rather than maximally distant.
    """
    out = pd.DataFrame(index=df.index)
    idx = df.index

    hour = idx.hour + idx.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

    out["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype("float64")
    out["session_london"] = ((hour >= 7) & (hour < 16)).astype("float64")
    out["session_newyork"] = ((hour >= 12) & (hour < 21)).astype("float64")
    # The overlap is where most of the daily range gets made.
    out["session_overlap"] = ((hour >= 12) & (hour < 16)).astype("float64")
    out["is_monday"] = (idx.dayofweek == 0).astype("float64")
    out["is_friday"] = (idx.dayofweek == 4).astype("float64")
    out["is_weekend"] = (idx.dayofweek >= 5).astype("float64")

    out["bar_gap"] = infer_gaps(df, timeframe)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    timeframe: str,
    cfg: FeatureConfig | None = None,
    context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix for one symbol on one timeframe.

    `context` maps a higher timeframe name to its price frame; a compact summary
    of each is joined onto the signal timeframe with `merge_asof`.
    """
    cfg = cfg or FeatureConfig()
    if df.empty or len(df) < 60:
        return pd.DataFrame(index=df.index)

    parts = [
        _price_features(df, cfg),
        _momentum_features(df, cfg),
        _trend_features(df, cfg),
        _volatility_features(df, cfg),
        _volume_features(df, cfg),
        micro.build(df),
        _time_features(df, timeframe),
    ]
    features = pd.concat(parts, axis=1)

    if context:
        for ctx_tf, ctx_df in context.items():
            if ctx_df is None or ctx_df.empty or len(ctx_df) < 60:
                continue
            summary = higher_timeframe_summary(ctx_df, cfg)
            aligned = align_to_signal_frame(summary, features.index, prefix=ctx_tf.lower())
            features = pd.concat([features, aligned], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)
    features.attrs["symbol"] = df.attrs.get("symbol", "?")
    features.attrs["timeframe"] = timeframe
    return features


def higher_timeframe_summary(
    df: pd.DataFrame, cfg: FeatureConfig | None = None
) -> pd.DataFrame:
    """A compact description of the higher-timeframe state.

    Deliberately small — joining 150 columns per context timeframe would
    triple the feature count for very little added signal, and every extra
    column costs sample efficiency.
    """
    cfg = cfg or FeatureConfig()
    high, low, close = df["high"], df["low"], df["close"]
    atr_ = ta.atr(high, low, close, cfg.atr_period)
    atr_safe = atr_.replace(0.0, np.nan)

    adx_, plus_di, minus_di = ta.adx(high, low, close, cfg.adx_period)
    _, st_dir = ta.supertrend(high, low, close)

    out = pd.DataFrame(index=df.index)
    out["trend_dir"] = st_dir
    out["adx"] = adx_ / 100.0
    out["di_spread"] = (plus_di - minus_di) / 100.0
    out["rsi"] = ta.rsi(close, 14) / 100.0
    out["dist_ema50_atr"] = (close - ta.ema(close, 50)) / atr_safe
    out["efficiency_ratio"] = ta.efficiency_ratio(close, 20)
    out["atr_percentile"] = ta.percentile_rank(atr_, 252)
    out["bb_percent_b"] = ta.bollinger_percent_b(close, cfg.bb_period, cfg.bb_std)
    out["ret_5"] = np.log(close).diff(5)
    out["donchian_position"] = ta.percentile_rank(close, 50)
    return out


def feature_names(matrix: pd.DataFrame) -> list[str]:
    """Model input columns, excluding anything the trainer adds later."""
    reserved = {"label", "label_return", "sample_weight", "barrier_hit", "event_end"}
    return [c for c in matrix.columns if c not in reserved]


def clean_for_model(
    matrix: pd.DataFrame, max_nan_fraction: float = 0.35
) -> tuple[pd.DataFrame, list[str]]:
    """Drop unusable columns and rows so the model sees a dense matrix.

    Columns that are constant carry no information; columns that are mostly NaN
    are usually an indicator whose warm-up exceeds the available history.
    """
    if matrix.empty:
        return matrix, []

    nan_fraction = matrix.isna().mean()
    keep = nan_fraction[nan_fraction <= max_nan_fraction].index.tolist()

    variance = matrix[keep].std(numeric_only=True)
    keep = [c for c in keep if c in variance.index and variance[c] > 1e-12]

    cleaned = matrix[keep].copy()
    # Forward-fill briefly for indicators that legitimately gap (weekend bars),
    # but never invent long stretches of data.
    cleaned = cleaned.ffill(limit=3)
    dropped = [c for c in matrix.columns if c not in keep]
    return cleaned, dropped
