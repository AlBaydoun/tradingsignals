"""Purged walk-forward validation.

Standard k-fold cross-validation is invalid on financial data and will hand you
a beautiful, entirely fictional accuracy score. Two reasons:

1. **Shuffling breaks time.** Training on Thursday to predict Wednesday is not a
   thing you can do with money.
2. **Label overlap leaks.** A triple-barrier label starting at bar 1000 may not
   resolve until bar 1024. If bar 1000 is in the training set and bar 1010 is in
   the test set, the training label already contains the test period's outcome.

This module fixes both. Folds move strictly forward in time; training samples
whose label windows overlap the test set are **purged**; and a further
**embargo** of bars after the test set is dropped, because serial correlation
means the bars immediately following a test period still carry its information.

Reference: López de Prado, *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One walk-forward split."""

    index: int
    train: np.ndarray
    test: np.ndarray
    purged: int
    embargoed: int

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"Fold({self.index}: train={len(self.train)} test={len(self.test)} "
            f"purged={self.purged} embargoed={self.embargoed})"
        )


class PurgedWalkForward:
    """Expanding-window walk-forward splitter with purging and embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 24,
        min_train_bars: int = 2000,
        min_test_bars: int = 300,
        expanding: bool = True,
    ):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars
        self.min_test_bars = min_test_bars
        self.expanding = expanding

    def split(
        self, n_samples: int, event_end: np.ndarray | None = None
    ) -> Iterator[Fold]:
        """Yield folds over `n_samples` observations.

        `event_end[i]` is the sample index at which observation i's label
        resolved. Supplying it enables purging; without it, only the embargo
        protects the test set.
        """
        usable = n_samples - self.min_train_bars
        if usable < self.min_test_bars:
            return

        test_size = max(self.min_test_bars, usable // self.n_splits)
        positions = np.arange(n_samples)

        fold_index = 0
        test_start = self.min_train_bars
        while test_start + test_size <= n_samples and fold_index < self.n_splits:
            test_end = min(test_start + test_size, n_samples)
            test = positions[test_start:test_end]

            train_end = test_start
            train_start = 0 if self.expanding else max(0, train_end - self.min_train_bars)
            train = positions[train_start:train_end]

            purged_count = 0
            if event_end is not None and len(train):
                # Drop any training sample whose label resolves at or after the
                # first test bar — its outcome overlaps the test period.
                ends = event_end[train]
                keep = (ends >= 0) & (ends < test_start)
                purged_count = int((~keep).sum())
                train = train[keep]

            # Embargo the bars right after the test window so the *next* fold's
            # training set does not start inside the tail of this test period.
            embargo_end = min(test_end + self.embargo_bars, n_samples)
            embargoed = embargo_end - test_end

            if len(train) >= self.min_train_bars // 2 and len(test) > 0:
                yield Fold(fold_index, train, test, purged_count, embargoed)
                fold_index += 1

            test_start = embargo_end

    def n_folds(self, n_samples: int, event_end: np.ndarray | None = None) -> int:
        return sum(1 for _ in self.split(n_samples, event_end))


def walk_forward_predict(
    fit_predict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: pd.Series | None = None,
    event_end: pd.Series | None = None,
    splitter: PurgedWalkForward | None = None,
) -> tuple[pd.DataFrame, list[Fold]]:
    """Produce genuinely out-of-sample predictions across the whole series.

    `fit_predict(X_train, y_train, w_train, X_test) -> ndarray[n_test, n_classes]`

    Every returned probability was produced by a model that had never seen the
    bar it is predicting, nor any bar whose label overlapped it. This is the
    only prediction set the engine will quote accuracy numbers from.
    """
    splitter = splitter or PurgedWalkForward()
    ends = event_end.to_numpy() if event_end is not None else None
    folds = list(splitter.split(len(X), ends))

    if not folds:
        return pd.DataFrame(index=X.index), []

    columns = sorted(pd.unique(y.dropna()))
    out = pd.DataFrame(index=X.index, columns=columns, dtype="float64")
    out["fold"] = np.nan

    for fold in folds:
        X_train, y_train = X.iloc[fold.train], y.iloc[fold.train]
        w_train = (
            sample_weight.iloc[fold.train].to_numpy()
            if sample_weight is not None
            else None
        )
        X_test = X.iloc[fold.test]

        proba = fit_predict(X_train, y_train, w_train, X_test)
        if proba is None:
            continue
        out.iloc[fold.test, : len(columns)] = proba
        out.iloc[fold.test, out.columns.get_loc("fold")] = fold.index

    return out, folds


def check_no_leakage(
    train_idx: np.ndarray, test_idx: np.ndarray, event_end: np.ndarray
) -> list[str]:
    """Assert a fold is clean. Used by the test suite and the `doctor` command."""
    problems: list[str] = []
    if len(train_idx) == 0 or len(test_idx) == 0:
        return problems

    if train_idx.max() >= test_idx.min():
        problems.append(
            f"train index {train_idx.max()} is at or after test start {test_idx.min()}"
        )

    overlapping = [i for i in train_idx if 0 <= event_end[i] >= test_idx.min()]
    if overlapping:
        problems.append(
            f"{len(overlapping)} training labels resolve inside the test window"
        )
    return problems
