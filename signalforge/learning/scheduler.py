"""The continuous learning loop.

Three jobs, on different clocks:

* **Resolve** (every run): check open signals against price and close the ones
  that hit a stop, a target, or their expiry. This is what feeds the journal.
* **Retrain** (daily, or on drift): refit models on fresh data and re-measure
  them out-of-sample.
* **Police** (every run): compare live results against what each model
  promised, and disable the ones that have stopped working.

The policing step is the one that matters. A system that only ever retrains
gets better at fitting whatever just happened; a system that also retires its
own failing models is the one that survives a regime change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from signalforge.config import Config
from signalforge.learning.journal import JournalEntry, TradeJournal
from signalforge.models import ModelRegistry, assess
from signalforge.universe import get_instrument

log = logging.getLogger(__name__)


@dataclass
class LearningReport:
    """What one pass of the learning loop did."""

    resolved: int = 0
    wins: int = 0
    losses: int = 0
    retrained: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    reinstated: list[str] = field(default_factory=list)
    drift_alerts: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        parts = []
        if self.resolved:
            parts.append(
                f"resolved {self.resolved} signals ({self.wins}W/{self.losses}L)"
            )
        if self.retrained:
            parts.append(f"retrained {len(self.retrained)}")
        if self.disabled:
            parts.append(f"disabled {', '.join(self.disabled)}")
        if self.reinstated:
            parts.append(f"reinstated {', '.join(self.reinstated)}")
        return "; ".join(parts) if parts else "nothing to do"


class EngineState:
    """Small persistent key-value store for loop bookkeeping."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("Engine state unreadable, resetting: %s", exc)
            return {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self._data, fh, indent=2, default=str)
        tmp.replace(self.path)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    def all(self) -> dict:
        return dict(self._data)


class LearningLoop:
    """Ties the journal, the registry and the drift detector together."""

    def __init__(self, config: Config, router=None):
        self.config = config
        self.journal = TradeJournal(config.learning.journal_path)
        self.registry = ModelRegistry(config.model.model_dir)
        self.state = EngineState(config.learning.state_path)
        self._router = router

    @property
    def router(self):
        if self._router is None:
            from signalforge.data import DataRouter

            self._router = DataRouter(self.config.data)
        return self._router

    # -- resolving open signals -----------------------------------------

    def resolve_open_signals(self) -> tuple[int, int, int]:
        """Check every open signal against subsequent price action.

        Uses the same pessimistic rule as the backtester: when a single bar
        spans both the stop and the target, the stop is assumed to have been
        hit first.
        """
        open_entries = self.journal.open_signals()
        if not open_entries:
            return 0, 0, 0

        resolved = wins = losses = 0
        by_symbol: dict[tuple[str, str], list[JournalEntry]] = {}
        for entry in open_entries:
            by_symbol.setdefault((entry.symbol, entry.timeframe), []).append(entry)

        for (symbol, timeframe), entries in by_symbol.items():
            try:
                df = self.router.get_bars(symbol, timeframe, 500)
            except Exception as exc:
                log.warning("Could not resolve %s %s: %s", symbol, timeframe, exc)
                continue
            if df.empty:
                continue

            instrument = get_instrument(symbol)
            for entry in entries:
                issued = pd.Timestamp(entry.issued_at)
                if issued.tzinfo is None:
                    issued = issued.tz_localize("UTC")

                future = df[df.index > issued]
                if future.empty:
                    continue

                direction = 1 if entry.direction == "BUY" else -1
                target = entry.take_profits[0] if entry.take_profits else None
                if target is None:
                    continue

                outcome = self._scan_outcome(future, direction, entry.stop_loss, target)
                if outcome is None:
                    # Still open, but expire it if it has run far past its window.
                    max_bars = self.config.signals.signal_validity_bars * 20
                    if len(future) > max_bars:
                        exit_price = float(future["close"].iloc[-1])
                        pnl = (
                            (exit_price - entry.entry)
                            * direction
                            * entry.lots
                            * instrument.contract_size
                        )
                        risk = max(abs(entry.entry - entry.stop_loss), 1e-9)
                        self.journal.update_outcome(
                            entry.signal_id,
                            status="expired",
                            exit_price=exit_price,
                            pnl=pnl,
                            r_multiple=(exit_price - entry.entry) * direction / risk,
                            exit_reason="expired_unresolved",
                        )
                        resolved += 1
                    continue

                exit_price, reason, mfe, mae = outcome
                pnl = (
                    (exit_price - entry.entry)
                    * direction
                    * entry.lots
                    * instrument.contract_size
                )
                risk_distance = max(abs(entry.entry - entry.stop_loss), 1e-9)
                r_multiple = (exit_price - entry.entry) * direction / risk_distance
                status = "won" if pnl > 0 else "lost"

                self.journal.update_outcome(
                    entry.signal_id,
                    status=status,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_multiple,
                    exit_reason=reason,
                    mfe=mfe,
                    mae=mae,
                )
                resolved += 1
                wins += status == "won"
                losses += status == "lost"

        return resolved, wins, losses

    @staticmethod
    def _scan_outcome(
        future: pd.DataFrame, direction: int, stop: float, target: float
    ) -> tuple[float, str, float, float] | None:
        """Walk forward until a barrier is touched."""
        mfe = mae = 0.0
        for _, bar in future.iterrows():
            high, low = float(bar["high"]), float(bar["low"])
            if direction > 0:
                mfe = max(mfe, high)
                mae = min(mae, low) if mae else low
                hit_stop, hit_target = low <= stop, high >= target
            else:
                mfe = min(mfe, low) if mfe else low
                mae = max(mae, high)
                hit_stop, hit_target = high >= stop, low <= target

            if hit_stop and hit_target:
                return stop, "stop_ambiguous", mfe, mae
            if hit_stop:
                return stop, "stop", mfe, mae
            if hit_target:
                return target, "target", mfe, mae
        return None

    # -- policing --------------------------------------------------------

    def police_models(self) -> tuple[list[str], list[str], list[dict]]:
        """Disable models whose live results contradict their promises."""
        disabled: list[str] = []
        reinstated: list[str] = []
        alerts: list[dict] = []
        cfg = self.config.learning

        for entry in self.registry.list_models():
            outcomes = self.journal.outcomes(entry.symbol, entry.timeframe)
            if len(outcomes) < 20:
                continue  # not enough evidence to judge either way

            model = self.registry.load(entry.symbol, entry.timeframe)
            expected = None
            if model and model.report and model.report.reliability:
                weighted = [
                    b.realised * b.count
                    for b in model.report.reliability
                    if b.count >= 20
                ]
                counts = [b.count for b in model.report.reliability if b.count >= 20]
                expected = sum(weighted) / sum(counts) if counts else None

            report = assess(
                None,
                None,
                outcomes,
                expected,
                psi_threshold=cfg.psi_alert_threshold,
                window=cfg.performance_window,
                min_hit_rate=cfg.min_live_hit_rate,
                max_consecutive_losses=cfg.disable_after_consecutive_losses,
            )

            key = f"{entry.symbol}/{entry.timeframe}"
            if report.action == "disable" and entry.enabled:
                self.registry.set_enabled(entry.symbol, entry.timeframe, False)
                disabled.append(key)
                alerts.append({"model": key, **report.to_dict()})
                log.warning("Disabled %s: %s", key, "; ".join(report.reasons))
            elif report.action == "none" and not entry.enabled:
                # It has recovered — a disabled model is not disabled forever.
                recent = outcomes[-cfg.performance_window :]
                if len(recent) >= 20 and sum(recent) / len(recent) > cfg.min_live_hit_rate + 0.05:
                    self.registry.set_enabled(entry.symbol, entry.timeframe, True)
                    reinstated.append(key)
            elif report.action in ("warn", "retrain"):
                alerts.append({"model": key, **report.to_dict()})

        return disabled, reinstated, alerts

    # -- retraining ------------------------------------------------------

    def models_due_for_retrain(self) -> list[tuple[str, str]]:
        due: list[tuple[str, str]] = []
        for symbol in self.config.watchlist:
            for timeframe in self.config.timeframes:
                if self.registry.needs_retrain(
                    symbol, timeframe, self.config.learning.retrain_every_hours
                ):
                    due.append((symbol, timeframe))
        return due

    def run_once(self, *, retrain: bool = False, trainer=None) -> LearningReport:
        """One full pass of the loop."""
        report = LearningReport()

        try:
            resolved, wins, losses = self.resolve_open_signals()
            report.resolved, report.wins, report.losses = resolved, wins, losses
        except Exception as exc:
            report.errors.append(f"resolve failed: {exc}")
            log.exception("Signal resolution failed")

        try:
            disabled, reinstated, alerts = self.police_models()
            report.disabled = disabled
            report.reinstated = reinstated
            report.drift_alerts = alerts
        except Exception as exc:
            report.errors.append(f"policing failed: {exc}")
            log.exception("Model policing failed")

        if retrain and trainer is not None:
            for symbol, timeframe in self.models_due_for_retrain():
                try:
                    trainer(symbol, timeframe)
                    report.retrained.append(f"{symbol}/{timeframe}")
                except Exception as exc:
                    report.errors.append(f"retrain {symbol}/{timeframe}: {exc}")

        self.state.set("last_run", datetime.now(timezone.utc).isoformat())
        self.state.set("last_report", report.to_dict())
        return report
