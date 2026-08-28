"""Event-driven backtester.

Written to be pessimistic wherever the data is ambiguous, because the purpose
of a backtest is not to produce an encouraging number — it is to find out
whether an idea survives contact with costs.

The specific pessimism:

* **Entry at the next bar's open, not this bar's close.** You cannot trade a
  close you have only just observed.
* **Spread and slippage on both sides**, taken from the instrument definition.
* **Stop before target when a single bar spans both.** OHLC data cannot say
  which came first; assuming the favourable one is how backtests learn to lie.
* **The stop is assumed to slip.** Gaps go through stops, especially over
  weekends and around news.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signalforge.backtest import metrics
from signalforge.universe import Instrument

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    starting_balance: float = 10_000.0
    risk_percent: float = 0.5
    slippage_pips: float = 0.5
    # Extra slippage applied when a stop is hit, reflecting gap-through risk.
    stop_slippage_multiplier: float = 1.5
    max_concurrent_positions: int = 1
    # Trades are closed automatically after this many bars.
    max_holding_bars: int = 200
    allow_shorts: bool = True
    commission_per_lot: float = 0.0


@dataclass
class Position:
    direction: int
    entry_time: pd.Timestamp
    entry_bar: int
    entry_price: float
    stop_loss: float
    take_profit: float
    size_lots: float
    risk_amount: float
    confidence: float = 0.0
    regime: str = ""
    # How far the entry bar's open sat from the signal bar's close, in ATR.
    # Large values are gaps, which used to quietly rewrite the trade's risk.
    entry_gap_atr: float = 0.0
    meta: dict = field(default_factory=dict)


class Backtester:
    """Simulates a signal series bar by bar."""

    def __init__(self, instrument: Instrument, config: BacktestConfig | None = None):
        self.instrument = instrument
        self.config = config or BacktestConfig()

    def _cost_per_lot(self) -> float:
        """Round-trip cost in price units per lot."""
        inst = self.instrument
        spread = inst.typical_spread_pips * inst.pip_size
        slip = 2.0 * self.config.slippage_pips * inst.pip_size
        return spread + slip

    def _pnl(self, position: Position, exit_price: float) -> tuple[float, float]:
        """Profit and cost for a closed position, in account currency."""
        inst = self.instrument
        gross = (
            (exit_price - position.entry_price)
            * position.direction
            * position.size_lots
            * inst.contract_size
        )
        cost = (
            self._cost_per_lot() * position.size_lots * inst.contract_size
            + self.config.commission_per_lot * position.size_lots
        )
        return gross - cost, cost

    def _size(self, entry: float, stop: float, balance: float) -> float:
        """Lots such that hitting the stop costs exactly `risk_percent`."""
        inst = self.instrument
        distance = abs(entry - stop)
        if distance <= 0:
            return 0.0

        risk_amount = balance * (self.config.risk_percent / 100.0)
        value_per_lot = distance * inst.contract_size
        if value_per_lot <= 0:
            return 0.0

        lots = risk_amount / value_per_lot
        lots = np.floor(lots / inst.lot_step) * inst.lot_step
        return float(np.clip(lots, 0.0, inst.max_lot))

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        *,
        regimes: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, metrics.PerformanceReport]:
        """Simulate trading `signals` over `df`.

        `signals` must carry `direction` (+1/-1/0), `stop_loss`, `take_profit`
        and optionally `confidence`, indexed like `df`.
        """
        cfg = self.config
        inst = self.instrument

        balance = cfg.starting_balance
        equity_curve: list[float] = []
        equity_index: list[pd.Timestamp] = []
        trades: list[dict] = []
        open_positions: list[Position] = []

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()
        index = df.index

        signal_dir = signals.reindex(df.index)["direction"].fillna(0).to_numpy()
        signal_sl = signals.reindex(df.index).get(
            "stop_loss", pd.Series(np.nan, index=df.index)
        ).to_numpy()
        signal_tp = signals.reindex(df.index).get(
            "take_profit", pd.Series(np.nan, index=df.index)
        ).to_numpy()
        signal_conf = signals.reindex(df.index).get(
            "confidence", pd.Series(0.0, index=df.index)
        ).fillna(0.0).to_numpy()
        # Optional: risk expressed as a distance rather than an absolute level.
        # Present on anything built by `signals_from_model`; absent on hand-made
        # signal frames, which fall back to the levels as given.
        aligned_signals = signals.reindex(df.index)
        signal_sl_dist = aligned_signals.get(
            "stop_distance", pd.Series(np.nan, index=df.index)
        ).to_numpy()
        signal_tp_dist = aligned_signals.get(
            "target_distance", pd.Series(np.nan, index=df.index)
        ).to_numpy()
        signal_atr = aligned_signals.get(
            "atr", pd.Series(np.nan, index=df.index)
        ).to_numpy()

        stop_slip = (
            cfg.slippage_pips * cfg.stop_slippage_multiplier * inst.pip_size
        )

        for i in range(len(df) - 1):
            # ---- manage open positions on this bar -----------------------
            still_open: list[Position] = []
            for position in open_positions:
                exit_price: float | None = None
                reason = ""

                hit_stop = (
                    lows[i] <= position.stop_loss
                    if position.direction > 0
                    else highs[i] >= position.stop_loss
                )
                hit_target = (
                    highs[i] >= position.take_profit
                    if position.direction > 0
                    else lows[i] <= position.take_profit
                )

                if hit_stop and hit_target:
                    # Ambiguous bar. Assume the stop — the pessimistic reading.
                    exit_price = position.stop_loss - position.direction * stop_slip
                    reason = "stop_ambiguous"
                elif hit_stop:
                    exit_price = position.stop_loss - position.direction * stop_slip
                    reason = "stop"
                elif hit_target:
                    exit_price = position.take_profit
                    reason = "target"
                elif i - position.entry_bar >= cfg.max_holding_bars:
                    exit_price = closes[i]
                    reason = "timeout"

                if exit_price is None:
                    still_open.append(position)
                    continue

                pnl, cost = self._pnl(position, exit_price)
                balance += pnl
                risk = max(position.risk_amount, 1e-9)
                trades.append(
                    {
                        "entry_time": position.entry_time,
                        "exit_time": index[i],
                        "direction": position.direction,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "entry_gap_atr": position.entry_gap_atr,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "size_lots": position.size_lots,
                        "pnl": pnl,
                        "cost": cost,
                        "r_multiple": pnl / risk,
                        "holding_bars": i - position.entry_bar,
                        "exit_reason": reason,
                        "confidence": position.confidence,
                        "regime": position.regime,
                        "balance": balance,
                    }
                )

            open_positions = still_open

            # ---- consider a new entry ------------------------------------
            direction = int(signal_dir[i])
            can_open = len(open_positions) < cfg.max_concurrent_positions
            if direction != 0 and can_open:
                if direction < 0 and not cfg.allow_shorts:
                    direction = 0

            if direction != 0 and can_open:
                stop = signal_sl[i]
                target = signal_tp[i]
                if np.isfinite(stop) and np.isfinite(target):
                    # Enter at the next bar's open, paying the spread.
                    raw_entry = opens[i + 1]
                    entry = raw_entry + direction * (
                        inst.typical_spread_pips * inst.pip_size / 2.0
                        + cfg.slippage_pips * inst.pip_size
                    )

                    # Re-anchor the risk to where we actually got in.
                    #
                    # Without this, a gap between the signal bar's close and the
                    # entry bar's open silently rewrites the trade. A real case
                    # from XAUUSD H4: gold gapped 66 points into a short signal,
                    # leaving the close-anchored stop 3.4 points from the entry
                    # instead of the intended 69. Risk-based sizing then bought
                    # 14x the normal lot against a 1:33 reward-to-risk; that one
                    # trade returned 31R and was 48% of the whole backtest's
                    # profit — a profit factor of 3.24 that fell to 1.68 without
                    # it. The signal's intent is "risk 1.5 ATR to make 1 ATR",
                    # and that intent is a distance, not a price.
                    gap_atr = 0.0
                    if np.isfinite(signal_sl_dist[i]) and signal_sl_dist[i] > 0:
                        stop = entry - direction * signal_sl_dist[i]
                        target = entry + direction * signal_tp_dist[i]
                        if np.isfinite(signal_atr[i]) and signal_atr[i] > 0:
                            gap_atr = float((raw_entry - closes[i]) / signal_atr[i])

                    # A stop on the wrong side of the entry is a bad signal.
                    valid = (
                        (stop < entry < target)
                        if direction > 0
                        else (target < entry < stop)
                    )
                    if valid:
                        lots = self._size(entry, stop, balance)
                        if lots > 0:
                            open_positions.append(
                                Position(
                                    direction=direction,
                                    entry_time=index[i + 1],
                                    entry_bar=i + 1,
                                    entry_price=entry,
                                    stop_loss=float(stop),
                                    take_profit=float(target),
                                    size_lots=lots,
                                    risk_amount=balance
                                    * (cfg.risk_percent / 100.0),
                                    confidence=float(signal_conf[i]),
                                    entry_gap_atr=gap_atr,
                                    regime=str(regimes.iloc[i])
                                    if regimes is not None and i < len(regimes)
                                    else "",
                                )
                            )

            # ---- mark to market -----------------------------------------
            unrealised = 0.0
            for position in open_positions:
                unrealised += (
                    (closes[i] - position.entry_price)
                    * position.direction
                    * position.size_lots
                    * inst.contract_size
                )
            equity_curve.append(balance + unrealised)
            equity_index.append(index[i])

        trade_frame = pd.DataFrame(trades)
        equity = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_index))

        bars_per_year = self._bars_per_year(df)
        report = metrics.compute(
            trade_frame,
            equity,
            starting_balance=cfg.starting_balance,
            periods_per_year=bars_per_year,
            total_bars=len(df),
        )
        return trade_frame, equity, report

    @staticmethod
    def _bars_per_year(df: pd.DataFrame) -> float:
        """Infer the annualisation factor from the actual bar spacing."""
        if len(df) < 3:
            return 252.0
        median_delta = pd.Series(df.index).diff().median()
        if pd.isna(median_delta) or median_delta.total_seconds() <= 0:
            return 252.0
        bars_per_day = 86400.0 / median_delta.total_seconds()
        return float(bars_per_day * 252.0)


def walk_forward_backtest(
    model,
    df: pd.DataFrame,
    instrument: Instrument,
    *,
    config: BacktestConfig | None = None,
    regimes: pd.Series | None = None,
    min_confidence: float = 0.55,
    min_edge: float = 0.10,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series, metrics.PerformanceReport]:
    """Backtest a trained model on its **walk-forward** predictions.

    This is the entry point to use. It draws on `model.oos_signals()`, where
    every prediction came from a model that had never seen the bar it is
    predicting. Feeding `model.predict_signal(X)` into `Backtester.run`
    instead measures how well the model memorised its own training set — which
    on a boosted-tree model looks spectacular and means nothing.
    """
    predictions = model.oos_signals()
    if predictions.empty:
        raise ValueError(
            "No walk-forward predictions available. The model must be trained "
            "in this process (they are not persisted with the model file)."
        )

    signals = signals_from_model(
        predictions,
        df,
        instrument,
        min_confidence=min_confidence,
        min_edge=min_edge,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
    )
    backtester = Backtester(instrument, config)
    return backtester.run(df, signals, regimes=regimes)


def signals_from_model(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    instrument: Instrument,
    *,
    min_confidence: float = 0.55,
    min_edge: float = 0.10,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Turn model output into concrete entries, stops and targets.

    Kept separate from the model so the same predictions can be backtested
    under different risk rules without retraining anything.
    """
    from signalforge.features import indicators as ta

    atr_ = ta.atr(df["high"], df["low"], df["close"], atr_period)
    close = df["close"]

    aligned = predictions.reindex(df.index)
    direction = aligned["direction"].fillna(0).astype("int64")
    confidence = aligned["confidence"].fillna(0.0)
    edge = aligned["edge"].fillna(0.0)

    take = (confidence >= min_confidence) & (edge.abs() >= min_edge)
    direction = direction.where(take, 0)

    out = pd.DataFrame(index=df.index)
    out["direction"] = direction
    out["confidence"] = confidence
    out["stop_loss"] = np.where(
        direction > 0,
        close - sl_atr_mult * atr_,
        np.where(direction < 0, close + sl_atr_mult * atr_, np.nan),
    )
    out["take_profit"] = np.where(
        direction > 0,
        close + tp_atr_mult * atr_,
        np.where(direction < 0, close - tp_atr_mult * atr_, np.nan),
    )
    # The absolute levels above are anchored to *this* bar's close, but entry
    # happens at the next bar's open. When those differ — a gap — the levels no
    # longer express the risk the signal intended. The distances do, so they are
    # carried separately and re-anchored to the real entry by the backtester.
    out["stop_distance"] = np.where(direction != 0, sl_atr_mult * atr_, np.nan)
    out["target_distance"] = np.where(direction != 0, tp_atr_mult * atr_, np.nan)
    out["atr"] = atr_
    return out
