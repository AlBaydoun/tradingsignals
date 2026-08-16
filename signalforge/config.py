"""Configuration objects and YAML loading.

Every tunable in the engine lives here so a user can change behaviour without
touching code. `config/config.yaml` overrides these defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"


@dataclass
class DataConfig:
    """Where price data comes from and how much of it we keep."""

    cache_dir: str = str(DEFAULT_DATA_DIR / "cache")
    # How many bars to pull per timeframe when training. More bars means a
    # slower but far more trustworthy walk-forward estimate.
    history_bars: dict[str, int] = field(
        default_factory=lambda: {
            "M1": 20000,
            "M5": 20000,
            "M15": 20000,
            "M30": 15000,
            "H1": 15000,
            # H4 is derived from hourly data, which Yahoo caps at ~730 days.
            "H4": 3000,
            "D1": 3000,
        }
    )
    # Bars fetched for live signal generation (must exceed the longest lookback).
    live_bars: int = 1500
    cache_ttl_seconds: int = 300
    request_timeout: int = 30
    max_retries: int = 3


@dataclass
class FeatureConfig:
    """Feature engineering knobs."""

    # Higher timeframes blended into each signal timeframe for context.
    context_timeframes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "M1": ["M15", "H1"],
            "M5": ["M30", "H4"],
            "M15": ["H1", "H4"],
            "M30": ["H4", "D1"],
            "H1": ["H4", "D1"],
            "H4": ["D1"],
            "D1": [],
        }
    )
    rsi_periods: list[int] = field(default_factory=lambda: [7, 14, 28])
    ema_periods: list[int] = field(default_factory=lambda: [8, 21, 55, 200])
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    keltner_period: int = 20
    keltner_atr_mult: float = 1.5
    volume_lookback: int = 50
    return_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 13, 21])
    volatility_lookback: int = 100
    hurst_window: int = 100


@dataclass
class LabelConfig:
    """Triple-barrier labelling parameters.

    The barriers are expressed in ATR multiples so a label means the same thing
    on a quiet EURUSD morning and a violent BTC candle.
    """

    upper_atr_mult: float = 1.5
    lower_atr_mult: float = 1.5
    max_horizon_bars: int = 24
    # A bar is only labelled if the move clears costs by this factor.
    min_cost_multiple: float = 1.5


@dataclass
class ModelConfig:
    """Model and validation settings."""

    model_dir: str = str(DEFAULT_DATA_DIR / "models")
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_child_samples: int = 60
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 1.0
    random_state: int = 7
    # Walk-forward validation
    n_splits: int = 5
    embargo_bars: int = 24
    min_train_bars: int = 2000
    min_test_bars: int = 300
    # Calibration
    calibration_bins: int = 10
    calibration_method: str = "isotonic"  # isotonic | sigmoid


@dataclass
class RiskConfig:
    """Stop placement and position sizing."""

    account_balance: float = 10000.0
    account_currency: str = "USD"
    risk_percent_per_trade: float = 0.5
    max_concurrent_risk_percent: float = 2.0
    sl_atr_mult: float = 1.5
    tp_atr_mults: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    # Prefer a swing high/low within this lookback if it sits beyond the ATR stop.
    structure_lookback: int = 20
    structure_buffer_atr: float = 0.25
    min_reward_risk: float = 1.2
    # Signals are discarded if the expected move does not clear this many
    # multiples of the round-trip trading cost.
    min_edge_cost_multiple: float = 2.0


@dataclass
class SignalConfig:
    """Thresholds that decide whether a signal is emitted at all."""

    min_confidence: float = 0.58
    min_directional_edge: float = 0.12
    max_signals_per_run: int = 8
    # Reject signals in the worst historical confidence bucket even if the raw
    # probability looks good.
    min_bucket_hit_rate: float = 0.50
    signal_validity_bars: int = 3
    # Block new entries this many minutes around a high-impact economic event.
    news_blackout_minutes: int = 30
    allow_medium_impact_trading: bool = True
    # Refuse signals in market conditions where this model has historically
    # lost money. Measured per regime and per session during training.
    enforce_conditional_edge: bool = True
    # A condition needs at least this many past trades before it is allowed to
    # veto anything — otherwise a thin slice of a backtest starts making rules.
    min_regime_trades: int = 25
    min_regime_profit_factor: float = 1.0


@dataclass
class ReasoningConfig:
    """The optional Claude reasoning layer."""

    enabled: bool = True
    model: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 8000
    # If the LLM disagrees with the quantitative model by more than this, the
    # signal is dropped rather than reconciled.
    max_confidence_override: float = 0.15
    timeout_seconds: int = 120


@dataclass
class LearningConfig:
    """Continuous learning loop."""

    journal_path: str = str(DEFAULT_DATA_DIR / "journal" / "trades.jsonl")
    state_path: str = str(DEFAULT_DATA_DIR / "state" / "engine_state.json")
    retrain_every_hours: int = 24
    # Population Stability Index above this means the live feature distribution
    # has drifted away from the training distribution.
    psi_alert_threshold: float = 0.25
    # Rolling window of closed trades used to detect live performance decay.
    performance_window: int = 50
    min_live_hit_rate: float = 0.40
    disable_after_consecutive_losses: int = 12


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)

    # Which instruments and timeframes the engine is allowed to trade.
    watchlist: list[str] = field(
        default_factory=lambda: [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "XAUUSD",
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "US500",
            "NAS100",
        ]
    )
    timeframes: list[str] = field(
        default_factory=lambda: ["M5", "M15", "M30", "H1", "H4"]
    )
    # MT5 brokers append suffixes like ".m" or "_ecn" to symbol names.
    mt5_symbol_suffix: str = ""
    timezone: str = "UTC"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: Any, override: Any) -> Any:
    """Recursively overlay `override` onto `base` dicts."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for key, value in override.items():
            out[key] = _merge(base.get(key), value) if key in base else value
        return out
    return override


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration, layering a YAML file over the dataclass defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    defaults = Config().to_dict()

    if cfg_path.exists():
        with open(cfg_path) as fh:
            user = yaml.safe_load(fh) or {}
        merged = _merge(defaults, user)
    else:
        merged = defaults

    # Environment overrides for the few settings people change per-machine.
    if os.getenv("SIGNALFORGE_ACCOUNT_BALANCE"):
        merged["risk"]["account_balance"] = float(
            os.environ["SIGNALFORGE_ACCOUNT_BALANCE"]
        )
    if os.getenv("SIGNALFORGE_RISK_PERCENT"):
        merged["risk"]["risk_percent_per_trade"] = float(
            os.environ["SIGNALFORGE_RISK_PERCENT"]
        )
    if os.getenv("SIGNALFORGE_MT5_SUFFIX"):
        merged["mt5_symbol_suffix"] = os.environ["SIGNALFORGE_MT5_SUFFIX"]

    return Config(
        data=DataConfig(**merged["data"]),
        features=FeatureConfig(**merged["features"]),
        labels=LabelConfig(**merged["labels"]),
        model=ModelConfig(**merged["model"]),
        risk=RiskConfig(**merged["risk"]),
        signals=SignalConfig(**merged["signals"]),
        reasoning=ReasoningConfig(**merged["reasoning"]),
        learning=LearningConfig(**merged["learning"]),
        watchlist=merged["watchlist"],
        timeframes=merged["timeframes"],
        mt5_symbol_suffix=merged["mt5_symbol_suffix"],
        timezone=merged["timezone"],
    )
