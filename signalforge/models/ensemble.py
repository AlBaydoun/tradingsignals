"""The prediction model and its calibration.

A gradient-boosted classifier predicts which triple barrier a trade from this
bar would hit first: up, down, or neither. Its raw output is *not* trusted as a
probability — boosted trees are systematically overconfident — so it is
calibrated against genuinely out-of-sample walk-forward predictions.

The number the engine reports to the user as "accuracy" is not the model's
self-reported confidence. It is the **realised out-of-sample hit rate of past
signals that carried a similar confidence**, read from a reliability table. If
the table says signals in the 0.65-0.70 bucket historically won 58% of the time,
that is what gets displayed, no matter what the model claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from signalforge.config import ModelConfig
from signalforge.models.validation import PurgedWalkForward, walk_forward_predict

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - fallback path
    HAS_LIGHTGBM = False
    log.warning("LightGBM unavailable; falling back to logistic regression")

# Model classes: short, neutral, long.
CLASSES = [-1, 0, 1]


@dataclass
class ReliabilityBucket:
    """One row of the calibration table."""

    lower: float
    upper: float
    count: int
    predicted: float  # mean confidence the model claimed
    realised: float  # fraction that actually won
    edge: float  # realised minus a coin flip

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelReport:
    """Everything measured about a trained model, all of it out-of-sample."""

    symbol: str
    timeframe: str
    n_samples: int
    n_features: int
    n_folds: int
    # Out-of-sample directional accuracy over bars where a signal fired.
    directional_accuracy: float
    coverage: float  # fraction of bars that produced a signal at all
    log_loss: float
    brier_score: float
    # Triple-barrier labels overlap, so `n_samples` badly overstates how much
    # independent evidence there is. These three fields say how much the
    # accuracy figure can actually be trusted.
    effective_sample_size: int = 0
    accuracy_ci_low: float = 0.0
    accuracy_ci_high: float = 1.0
    reliability: list[ReliabilityBucket] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    class_balance: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""
    warnings: list[str] = field(default_factory=list)
    # Where this model's edge actually lives, measured on the walk-forward
    # backtest. Used at signal time to veto trades in conditions the model has
    # historically lost money in. See models/conditional.py.
    conditional_edge: dict = field(default_factory=dict)
    # Backtest performance after costs, for the ranker and the CLI.
    backtest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["reliability"] = [b.to_dict() for b in self.reliability]
        return out

    def hit_rate_for_confidence(self, confidence: float) -> float | None:
        """The realised win rate for signals of this confidence.

        This is the honest answer to "how accurate is this signal?" — a
        measured frequency, not the model's opinion of itself.
        """
        for bucket in self.reliability:
            if bucket.lower <= confidence < bucket.upper:
                # A bucket with too few observations cannot support a claim.
                return bucket.realised if bucket.count >= 20 else None
        return None

    def top_features(self, n: int = 15) -> list[tuple[str, float]]:
        return sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:n]

    @property
    def edge_is_significant(self) -> bool:
        """Whether the measured accuracy is distinguishable from a coin flip.

        If the confidence interval straddles 0.5, the model has not
        demonstrated an edge — however good the point estimate looks.
        """
        return self.accuracy_ci_low > 0.5


def wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly at the small sample sizes this engine often faces.
    """
    if trials <= 0:
        return 0.0, 1.0
    phat = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = phat + z**2 / (2 * trials)
    spread = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return (
        float(max(0.0, (centre - spread) / denominator)),
        float(min(1.0, (centre + spread) / denominator)),
    )


class SignalModel:
    """Gradient-boosted 3-class classifier with out-of-sample calibration."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.calibrators: dict[int, object] = {}
        self.feature_columns: list[str] = []
        self.report: ModelReport | None = None
        self.classes_: list[int] = CLASSES
        # Walk-forward predictions retained from training. These are the only
        # predictions it is legitimate to backtest on — see `oos_signals`.
        self.oos_proba: pd.DataFrame | None = None

    # -- estimator ------------------------------------------------------

    def _make_estimator(self):
        cfg = self.config
        if HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                max_depth=cfg.max_depth,
                min_child_samples=cfg.min_child_samples,
                subsample=cfg.subsample,
                subsample_freq=1,
                colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda,
                random_state=cfg.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        return LogisticRegression(
            max_iter=1000, multi_class="multinomial", random_state=cfg.random_state
        )

    def _fit_predict(self, X_train, y_train, w_train, X_test):
        """One fold: train from scratch, predict the held-out block."""
        if len(np.unique(y_train)) < 2:
            return None
        model = self._make_estimator()
        try:
            if HAS_LIGHTGBM:
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train, sample_weight=w_train)
        except Exception as exc:
            log.warning("Fold training failed: %s", exc)
            return None

        proba = model.predict_proba(X_test)
        # Re-order columns into our canonical [-1, 0, 1] ordering, filling any
        # class the fold never saw with zeros.
        aligned = np.zeros((len(X_test), len(CLASSES)))
        for col, cls in enumerate(model.classes_):
            if cls in CLASSES:
                aligned[:, CLASSES.index(cls)] = proba[:, col]
        return aligned

    # -- training -------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
        event_end_time: pd.Series | None = None,
        symbol: str = "?",
        timeframe: str = "?",
    ) -> ModelReport:
        """Train the model and measure it honestly.

        Two passes: a walk-forward pass that produces out-of-sample predictions
        for calibration and scoring, then a final fit on all the data for live
        use. The metrics all come from the first pass.
        """
        cfg = self.config
        self.feature_columns = list(X.columns)
        warnings_: list[str] = []

        mask = y.notna()
        if sample_weight is not None:
            mask &= sample_weight > 0
        X, y = X[mask], y[mask].astype("int64")
        weights = sample_weight[mask] if sample_weight is not None else None

        if len(X) < cfg.min_train_bars + cfg.min_test_bars:
            raise ValueError(
                f"{symbol} {timeframe}: only {len(X)} usable rows, need at least "
                f"{cfg.min_train_bars + cfg.min_test_bars}"
            )

        # Convert each label's resolution *time* into a row position within the
        # filtered matrix. Doing this by timestamp rather than by original
        # position is essential: masking removes rows, so a position computed
        # against the unfiltered frame points somewhere else entirely.
        ends = None
        if event_end_time is not None:
            end_times = pd.to_datetime(event_end_time[mask], utc=True)
            # Compare as int64 nanoseconds. `.to_numpy()` on a tz-aware
            # DatetimeIndex yields an *object* array of Timestamps, and
            # np.searchsorted over object dtype returns silently wrong
            # positions — which would leave purging half-disabled and inflate
            # every accuracy figure the engine reports.
            index_ns = np.asarray(X.index.asi8, dtype="int64")
            missing = end_times.isna().to_numpy()
            end_ns = np.where(
                missing, np.iinfo("int64").max, end_times.astype("int64").to_numpy()
            )
            positions = np.searchsorted(index_ns, end_ns, side="left")
            positions = np.where(missing, -1, positions)
            # A label resolving past the end of the matrix still blocks every
            # remaining row, so clamp rather than discard.
            positions = np.where(positions >= len(X), len(X) - 1, positions)
            ends = pd.Series(positions.astype("int64"), index=X.index)

        splitter = PurgedWalkForward(
            n_splits=cfg.n_splits,
            embargo_bars=cfg.embargo_bars,
            min_train_bars=cfg.min_train_bars,
            min_test_bars=cfg.min_test_bars,
        )
        oos, folds = walk_forward_predict(
            self._fit_predict,
            X,
            y,
            sample_weight=weights,
            event_end=ends,
            splitter=splitter,
        )

        if not folds:
            raise ValueError(
                f"{symbol} {timeframe}: not enough history for walk-forward validation"
            )

        proba_cols = [c for c in oos.columns if c != "fold"]
        evaluated = oos[oos["fold"].notna()]
        y_eval = y.loc[evaluated.index]

        self._fit_calibrators(evaluated[proba_cols].to_numpy(), y_eval.to_numpy())
        self.oos_proba = evaluated[proba_cols].astype("float64")

        # How much a single label overlaps its neighbours. Labels resolving over
        # 8 bars means ~8 consecutive rows describe the same market move, so the
        # independent evidence is roughly one eighth of the row count.
        overlap = 1.0
        if ends is not None:
            positions = np.arange(len(X))
            spans = ends.to_numpy() - positions
            valid_spans = spans[spans > 0]
            if len(valid_spans):
                overlap = max(1.0, float(np.mean(valid_spans)))

        report = self._build_report(
            evaluated[proba_cols].to_numpy(),
            y_eval.to_numpy(),
            symbol=symbol,
            timeframe=timeframe,
            n_features=X.shape[1],
            n_folds=len(folds),
            warnings_=warnings_,
            label_overlap=overlap,
        )

        # Final production model, trained on everything.
        self.model = self._make_estimator()
        self.model.fit(
            X, y, sample_weight=weights.to_numpy() if weights is not None else None
        )

        if HAS_LIGHTGBM and hasattr(self.model, "feature_importances_"):
            total = float(np.sum(self.model.feature_importances_)) or 1.0
            report.feature_importance = {
                col: float(imp) / total
                for col, imp in zip(X.columns, self.model.feature_importances_)
            }

        self.report = report
        return report

    def _fit_calibrators(self, proba: np.ndarray, y_true: np.ndarray) -> None:
        """Map raw scores onto frequencies that hold up out of sample.

        Fitted per class, one-vs-rest, on walk-forward predictions only.
        """
        self.calibrators = {}
        for i, cls in enumerate(CLASSES):
            target = (y_true == cls).astype("float64")
            scores = proba[:, i]
            valid = np.isfinite(scores) & np.isfinite(target)
            if valid.sum() < 100 or len(np.unique(target[valid])) < 2:
                continue
            try:
                if self.config.calibration_method == "isotonic":
                    calibrator = IsotonicRegression(
                        y_min=0.0, y_max=1.0, out_of_bounds="clip"
                    )
                    calibrator.fit(scores[valid], target[valid])
                else:
                    calibrator = LogisticRegression()
                    calibrator.fit(scores[valid].reshape(-1, 1), target[valid])
                self.calibrators[cls] = calibrator
            except Exception as exc:
                log.warning("Calibration failed for class %s: %s", cls, exc)

    def _apply_calibration(self, proba: np.ndarray) -> np.ndarray:
        """Calibrate then renormalise so the row still sums to one."""
        if not self.calibrators:
            return proba

        out = proba.copy()
        for i, cls in enumerate(CLASSES):
            calibrator = self.calibrators.get(cls)
            if calibrator is None:
                continue
            scores = proba[:, i]
            if isinstance(calibrator, IsotonicRegression):
                out[:, i] = calibrator.predict(scores)
            else:
                out[:, i] = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

        totals = out.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return out / totals

    def _build_report(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        *,
        symbol: str,
        timeframe: str,
        n_features: int,
        n_folds: int,
        warnings_: list[str],
        label_overlap: float = 1.0,
    ) -> ModelReport:
        calibrated = self._apply_calibration(proba)

        # Directional edge = P(up) - P(down); its sign is the trade direction.
        edge = calibrated[:, CLASSES.index(1)] - calibrated[:, CLASSES.index(-1)]
        predicted_direction = np.sign(edge)
        confidence = np.max(calibrated, axis=1)

        # Score only where the model actually committed to a direction, and the
        # market actually moved — a timeout is not a directional error.
        fired = (predicted_direction != 0) & (y_true != 0)
        if fired.sum() > 0:
            accuracy = float((predicted_direction[fired] == y_true[fired]).mean())
        else:
            accuracy = 0.0
            warnings_.append("Model never produced a directional prediction")

        eps = 1e-12
        true_index = np.array([CLASSES.index(int(v)) for v in y_true])
        true_proba = calibrated[np.arange(len(y_true)), true_index]
        logloss = float(-np.mean(np.log(np.clip(true_proba, eps, 1.0))))

        onehot = np.zeros_like(calibrated)
        onehot[np.arange(len(y_true)), true_index] = 1.0
        brier = float(np.mean(np.sum((calibrated - onehot) ** 2, axis=1)))

        reliability = self._reliability_table(
            confidence, predicted_direction, y_true, self.config.calibration_bins
        )

        counts = pd.Series(y_true).value_counts(normalize=True)
        balance = {str(k): round(float(v), 4) for k, v in counts.items()}

        # Discount the sample for label overlap, then put error bars on the
        # accuracy. A 63% hit rate over 600 heavily-overlapping labels can
        # easily be a 50% hit rate wearing a disguise.
        n_fired = int(fired.sum())
        effective_n = max(1, int(n_fired / max(label_overlap, 1.0)))
        successes = int(round(accuracy * effective_n))
        ci_low, ci_high = wilson_interval(successes, effective_n)

        if accuracy < 0.5 and n_fired > 50:
            warnings_.append(
                f"Out-of-sample directional accuracy is {accuracy:.1%} — "
                "at or below a coin flip. Do not trade this model."
            )
        elif ci_low <= 0.5:
            warnings_.append(
                f"Accuracy {accuracy:.1%} has a 95% confidence interval of "
                f"{ci_low:.1%}-{ci_high:.1%} on ~{effective_n} independent "
                "observations. That interval includes a coin flip, so this "
                "model has not demonstrated a real edge."
            )
        if effective_n < 100:
            warnings_.append(
                f"Only ~{effective_n} independent observations after adjusting "
                f"for label overlap ({label_overlap:.1f} bars per label). Treat "
                "every number from this model as provisional."
            )

        return ModelReport(
            symbol=symbol,
            timeframe=timeframe,
            n_samples=len(y_true),
            n_features=n_features,
            n_folds=n_folds,
            directional_accuracy=round(accuracy, 4),
            coverage=round(float(fired.mean()), 4),
            log_loss=round(logloss, 4),
            brier_score=round(brier, 4),
            effective_sample_size=effective_n,
            accuracy_ci_low=round(ci_low, 4),
            accuracy_ci_high=round(ci_high, 4),
            reliability=reliability,
            class_balance=balance,
            trained_at=pd.Timestamp.utcnow().isoformat(),
            warnings=warnings_,
        )

    @staticmethod
    def _reliability_table(
        confidence: np.ndarray,
        predicted: np.ndarray,
        y_true: np.ndarray,
        n_bins: int,
    ) -> list[ReliabilityBucket]:
        """Confidence bucket versus what actually happened.

        This table is the engine's conscience: if the model says 70% and this
        table says 52%, the user is told 52%.
        """
        buckets: list[ReliabilityBucket] = []
        # Only directional calls where a directional outcome existed.
        mask = (predicted != 0) & (y_true != 0)
        if mask.sum() == 0:
            return buckets

        conf, pred, truth = confidence[mask], predicted[mask], y_true[mask]
        edges = np.linspace(0.3, 1.0, n_bins + 1)

        for lower, upper in zip(edges[:-1], edges[1:]):
            in_bin = (conf >= lower) & (conf < upper)
            count = int(in_bin.sum())
            if count == 0:
                continue
            realised = float((pred[in_bin] == truth[in_bin]).mean())
            buckets.append(
                ReliabilityBucket(
                    lower=round(float(lower), 3),
                    upper=round(float(upper), 3),
                    count=count,
                    predicted=round(float(conf[in_bin].mean()), 4),
                    realised=round(realised, 4),
                    edge=round(realised - 0.5, 4),
                )
            )
        return buckets

    # -- inference ------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated class probabilities in [-1, 0, +1] order."""
        if self.model is None:
            raise RuntimeError("Model is not trained")

        aligned = X.reindex(columns=self.feature_columns)
        raw = self.model.predict_proba(aligned)

        out = np.zeros((len(aligned), len(CLASSES)))
        for col, cls in enumerate(self.model.classes_):
            if cls in CLASSES:
                out[:, CLASSES.index(cls)] = raw[:, col]
        return self._apply_calibration(out)

    def oos_signals(self) -> pd.DataFrame:
        """Signals built from walk-forward predictions only.

        **This is the frame to backtest.** `predict_signal` runs the final
        model, which was fitted on every row including the ones you would be
        evaluating — backtesting that measures memorisation, not skill, and it
        will happily report a 90% win rate on a model with no edge whatsoever.

        Returns an empty frame if the model has been loaded from disk rather
        than freshly trained, since walk-forward predictions are not persisted.
        """
        if self.oos_proba is None or self.oos_proba.empty:
            return pd.DataFrame(
                columns=["p_down", "p_flat", "p_up", "edge", "direction", "confidence"]
            )

        calibrated = self._apply_calibration(self.oos_proba.to_numpy())
        return self._signal_frame(calibrated, self.oos_proba.index)

    def _signal_frame(self, proba: np.ndarray, index: pd.Index) -> pd.DataFrame:
        p_down, p_flat, p_up = proba[:, 0], proba[:, 1], proba[:, 2]
        edge = p_up - p_down

        out = pd.DataFrame(index=index)
        out["p_down"] = p_down
        out["p_flat"] = p_flat
        out["p_up"] = p_up
        out["edge"] = edge
        out["direction"] = np.sign(edge).astype("int64")
        out["confidence"] = np.maximum(p_up, p_down)

        if self.report is not None:
            out["measured_accuracy"] = [
                self.report.hit_rate_for_confidence(c) for c in out["confidence"]
            ]
        else:
            out["measured_accuracy"] = np.nan
        return out

    def predict_signal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Direction, confidence, edge and measured accuracy per row.

        For **live inference only**. Backtesting this output is a leakage bug —
        use `oos_signals()` instead.
        """
        proba = self.predict_proba(X)
        return self._signal_frame(proba, X.index)
