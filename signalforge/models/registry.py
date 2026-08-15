"""Model persistence: one trained model per symbol and timeframe."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from signalforge.models.ensemble import ModelReport, SignalModel

log = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    symbol: str
    timeframe: str
    path: Path
    trained_at: str
    directional_accuracy: float
    n_samples: int
    enabled: bool = True

    @property
    def key(self) -> str:
        return f"{self.symbol}_{self.timeframe}"

    def age_hours(self) -> float:
        try:
            trained = datetime.fromisoformat(self.trained_at)
        except (ValueError, TypeError):
            return float("inf")
        if trained.tzinfo is None:
            trained = trained.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - trained).total_seconds() / 3600.0


class ModelRegistry:
    """Stores trained models on disk with a searchable index.

    The index carries each model's out-of-sample accuracy and an `enabled`
    flag, so the learning loop can retire a model that has stopped working
    without deleting it.
    """

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        try:
            with open(self.index_path) as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("Model index unreadable, starting fresh: %s", exc)
            return {}

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self._index, fh, indent=2)
        tmp.replace(self.index_path)

    @staticmethod
    def _key(symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}_{timeframe.upper()}"

    def save(self, model: SignalModel, symbol: str, timeframe: str) -> Path:
        key = self._key(symbol, timeframe)
        path = self.dir / f"{key}.joblib"
        joblib.dump(
            {
                "model": model.model,
                "calibrators": model.calibrators,
                "feature_columns": model.feature_columns,
                "report": model.report.to_dict() if model.report else None,
                "config": model.config,
            },
            path,
            compress=3,
        )

        report = model.report
        self._index[key] = {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "path": str(path),
            "trained_at": report.trained_at if report else "",
            "directional_accuracy": report.directional_accuracy if report else 0.0,
            "coverage": report.coverage if report else 0.0,
            "n_samples": report.n_samples if report else 0,
            "brier_score": report.brier_score if report else 1.0,
            "warnings": report.warnings if report else [],
            "enabled": self._index.get(key, {}).get("enabled", True),
        }
        self._save_index()
        return path

    def load(self, symbol: str, timeframe: str) -> SignalModel | None:
        key = self._key(symbol, timeframe)
        entry = self._index.get(key)
        if not entry:
            return None

        path = Path(entry["path"])
        if not path.exists():
            log.warning("Model file missing for %s, dropping from index", key)
            self._index.pop(key, None)
            self._save_index()
            return None

        try:
            payload = joblib.load(path)
        except Exception as exc:
            log.warning("Could not load model %s: %s", key, exc)
            return None

        model = SignalModel(payload.get("config"))
        model.model = payload["model"]
        model.calibrators = payload.get("calibrators", {})
        model.feature_columns = payload.get("feature_columns", [])

        report_dict = payload.get("report")
        if report_dict:
            from signalforge.models.ensemble import ReliabilityBucket

            buckets = [
                ReliabilityBucket(**b) for b in report_dict.pop("reliability", [])
            ]
            model.report = ModelReport(**report_dict, reliability=buckets)
        return model

    def list_models(self) -> list[ModelEntry]:
        return [
            ModelEntry(
                symbol=e["symbol"],
                timeframe=e["timeframe"],
                path=Path(e["path"]),
                trained_at=e.get("trained_at", ""),
                directional_accuracy=e.get("directional_accuracy", 0.0),
                n_samples=e.get("n_samples", 0),
                enabled=e.get("enabled", True),
            )
            for e in self._index.values()
        ]

    def is_enabled(self, symbol: str, timeframe: str) -> bool:
        return self._index.get(self._key(symbol, timeframe), {}).get("enabled", False)

    def set_enabled(self, symbol: str, timeframe: str, enabled: bool) -> None:
        """Retire or reinstate a model without discarding it."""
        key = self._key(symbol, timeframe)
        if key in self._index:
            self._index[key]["enabled"] = enabled
            self._save_index()

    def needs_retrain(self, symbol: str, timeframe: str, max_age_hours: float) -> bool:
        entry = self._index.get(self._key(symbol, timeframe))
        if not entry:
            return True
        model_entry = ModelEntry(
            symbol=entry["symbol"],
            timeframe=entry["timeframe"],
            path=Path(entry["path"]),
            trained_at=entry.get("trained_at", ""),
            directional_accuracy=entry.get("directional_accuracy", 0.0),
            n_samples=entry.get("n_samples", 0),
        )
        return model_entry.age_hours() > max_age_hours

    def summary(self) -> dict[str, object]:
        models = self.list_models()
        accuracies = [m.directional_accuracy for m in models if m.enabled]
        return {
            "total": len(models),
            "enabled": sum(1 for m in models if m.enabled),
            "mean_accuracy": round(sum(accuracies) / len(accuracies), 4)
            if accuracies
            else 0.0,
            "dir": str(self.dir),
        }
