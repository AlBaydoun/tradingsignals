"""The engine: turns market data into ranked, risk-sized signals.

The pipeline for each (symbol, timeframe):

    data -> features -> regime -> model -> ranking -> levels -> sizing
         -> event/news filter -> grading -> reasoning -> journal

Every stage can veto. A signal only survives if the model has an edge, the edge
survives costs, the market is liquid, no major release is imminent, the risk is
sizeable, and the confidence band has a real track record. Most of the time
nothing survives, and that is the intended behaviour.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from signalforge.anomaly import detect as detect_anomaly
from signalforge.config import Config
from signalforge.data import DataRouter, timeframe_minutes
from signalforge.features import build_feature_matrix, clean_for_model
from signalforge.labeling import apply_triple_barrier, cost_in_price_units
from signalforge.learning import JournalEntry, TradeJournal
from signalforge.models import ModelRegistry, SignalModel
from signalforge.news import EconomicCalendar, NewsAggregator
from signalforge.regime import RegimeDetector
from signalforge.reasoning import (
    ReasoningEngine,
    build_evidence_packet,
    build_warnings,
    describe_evidence,
    market_summary,
    multi_timeframe_agreement,
)
from signalforge.risk import calculate_lots, compute_levels, correlation_adjusted_risk
from signalforge.selection import Ranking, rank_candidate, summarise, top_opportunities
from signalforge.signals.schema import (
    Direction,
    Evidence,
    Signal,
    SignalBundle,
    SignalQuality,
    WatchItem,
    grade,
)
from signalforge.universe import get_instrument, mt5_name

log = logging.getLogger(__name__)


class SignalEngine:
    """Top-level orchestrator."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.router = DataRouter(self.config.data)
        self.registry = ModelRegistry(self.config.model.model_dir)
        self.calendar = EconomicCalendar()
        self.news = NewsAggregator()
        self.reasoner = ReasoningEngine(self.config.reasoning)
        self.journal = TradeJournal(self.config.learning.journal_path)
        self._regime_detectors: dict[str, RegimeDetector] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, symbol: str, timeframe: str) -> dict:
        """Fit and persist a model for one symbol/timeframe."""
        cfg = self.config
        instrument = get_instrument(symbol)

        bars = cfg.data.history_bars.get(timeframe.upper(), 15000)
        df = self.router.get_bars(symbol, timeframe, bars)
        if df.empty or len(df) < cfg.model.min_train_bars + cfg.model.min_test_bars:
            raise ValueError(
                f"{symbol} {timeframe}: only {len(df)} bars available, need "
                f"{cfg.model.min_train_bars + cfg.model.min_test_bars}"
            )

        context = self._context_frames(symbol, timeframe)
        features = build_feature_matrix(df, timeframe, cfg.features, context)
        X, dropped = clean_for_model(features)

        cost = cost_in_price_units(
            self.router.effective_spread_pips(symbol), instrument.pip_size
        )
        labels = apply_triple_barrier(
            df,
            upper_atr_mult=cfg.labels.upper_atr_mult,
            lower_atr_mult=cfg.labels.lower_atr_mult,
            horizon=cfg.labels.max_horizon_bars,
            atr_period=cfg.features.atr_period,
            cost_price_units=cost,
            min_cost_multiple=cfg.labels.min_cost_multiple,
        )
        label_frame = labels.to_frame()

        usable = (
            label_frame["tradable"]
            & ~label_frame["ambiguous"]
            & X.notna().all(axis=1)
        )
        if usable.sum() < cfg.model.min_train_bars:
            raise ValueError(
                f"{symbol} {timeframe}: only {int(usable.sum())} usable rows after "
                "cost filtering — the spread is likely too wide for this timeframe"
            )

        model = SignalModel(cfg.model)
        report = model.fit(
            X[usable],
            label_frame["label"][usable],
            sample_weight=label_frame["sample_weight"][usable],
            event_end_time=label_frame["event_end_time"][usable],
            symbol=symbol,
            timeframe=timeframe,
        )

        # Measure where this model's edge actually lives. A blended accuracy
        # figure averages a real edge in one regime with a bleed in another;
        # the map is what lets signal generation refuse the losing conditions.
        self._measure_conditional_edge(model, report, df, instrument)

        self.registry.save(model, symbol, timeframe)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "conditional_edge": report.conditional_edge,
            "backtest": report.backtest,
            "accuracy": report.directional_accuracy,
            "ci_low": report.accuracy_ci_low,
            "ci_high": report.accuracy_ci_high,
            "significant": report.edge_is_significant,
            "coverage": report.coverage,
            "brier": report.brier_score,
            "samples": report.n_samples,
            "effective_samples": report.effective_sample_size,
            "folds": report.n_folds,
            "features": X.shape[1],
            "dropped_features": len(dropped),
            "label_summary": labels.summary(),
            "warnings": report.warnings,
        }

    def _measure_conditional_edge(self, model, report, df, instrument) -> None:
        """Backtest the freshly trained model and record where it makes money.

        Runs on walk-forward predictions only. Failure here degrades to an
        empty map (which gates nothing) rather than blocking training.
        """
        cfg = self.config
        try:
            from signalforge.backtest import BacktestConfig, walk_forward_backtest
            from signalforge.models import build_conditional_edge

            detector = RegimeDetector().fit(df)
            regimes = detector.classify_series(df)["trend_regime"]

            trades, _, performance = walk_forward_backtest(
                model,
                df,
                instrument,
                config=BacktestConfig(
                    starting_balance=cfg.risk.account_balance,
                    risk_percent=cfg.risk.risk_percent_per_trade,
                ),
                regimes=regimes,
                min_confidence=cfg.signals.min_confidence - 0.06,
                min_edge=cfg.signals.min_directional_edge - 0.04,
                sl_atr_mult=cfg.risk.sl_atr_mult,
                tp_atr_mult=cfg.risk.tp_atr_mults[0],
            )

            edge = build_conditional_edge(
                trades,
                min_trades=cfg.signals.min_regime_trades,
                min_profit_factor=cfg.signals.min_regime_profit_factor,
            )
            report.conditional_edge = edge.to_dict()
            report.backtest = {
                "trades": performance.trades,
                "win_rate": performance.win_rate,
                "profit_factor": performance.profit_factor,
                "expectancy_r": performance.expectancy_r,
                "max_drawdown_pct": performance.max_drawdown_pct,
                "verdict": performance.verdict(),
            }

            blocked = [s.condition for s in edge.worst_conditions() if s.is_losing]
            if blocked:
                report.warnings.append(
                    "Loses money in "
                    + ", ".join(c.replace("_", " ") for c in blocked)
                    + " — signals in those conditions will be blocked."
                )
        except Exception as exc:
            log.warning("Could not measure conditional edge: %s", exc)
            report.conditional_edge = {}
            report.backtest = {}

    def _context_frames(self, symbol: str, timeframe: str) -> dict[str, pd.DataFrame]:
        """Higher-timeframe frames used for multi-timeframe context."""
        context: dict[str, pd.DataFrame] = {}
        for ctx_tf in self.config.features.context_timeframes.get(timeframe.upper(), []):
            try:
                bars = max(500, self.config.data.history_bars.get(ctx_tf, 3000) // 3)
                frame = self.router.get_bars(symbol, ctx_tf, bars)
                if not frame.empty:
                    context[ctx_tf] = frame
            except Exception as exc:
                log.warning("Context %s for %s unavailable: %s", ctx_tf, symbol, exc)
        return context

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        *,
        use_reasoning: bool = True,
        refresh_news: bool = True,
    ) -> SignalBundle:
        """Run the full pipeline and return everything it produced."""
        cfg = self.config
        symbols = symbols or cfg.watchlist
        timeframes = timeframes or cfg.timeframes
        now = datetime.now(timezone.utc)

        bundle = SignalBundle(generated_at=now)
        diagnostics: dict[str, object] = {
            "symbols_requested": len(symbols),
            "timeframes": timeframes,
            "reasoning_available": self.reasoner.available,
            "reasoning_unavailable_reason": self.reasoner.unavailable_reason,
        }

        if refresh_news:
            try:
                self.calendar.refresh()
                self.news.fetch()
            except Exception as exc:
                log.warning("News refresh failed: %s", exc)
                diagnostics["news_error"] = str(exc)[:200]

        rankings: list[Ranking] = []
        candidates: list[dict] = []
        regimes: dict[str, object] = {}

        for symbol in symbols:
            try:
                symbol_result = self._evaluate_symbol(symbol, timeframes, now)
            except Exception as exc:
                log.warning("Evaluation failed for %s: %s", symbol, exc)
                diagnostics.setdefault("errors", []).append(f"{symbol}: {exc}")
                continue

            rankings.extend(symbol_result["rankings"])
            candidates.extend(symbol_result["candidates"])
            if symbol_result.get("regime"):
                regimes[symbol] = symbol_result["regime"]
            bundle.watchlist.extend(symbol_result["watch"])
            bundle.blocked.extend(symbol_result["blocked"])

        # Rank, then build full signals only for the best few. Building a signal
        # is expensive (levels, sizing, reasoning) so it is not done for every
        # candidate.
        best = top_opportunities(rankings, cfg.signals.max_signals_per_run)
        best_keys = {(r.symbol, r.timeframe) for r in best}

        open_symbols: list[str] = []
        for candidate in candidates:
            key = (candidate["symbol"], candidate["timeframe"])
            if key not in best_keys:
                continue
            try:
                signal = self._build_signal(
                    candidate,
                    ranking=next(
                        r for r in best if (r.symbol, r.timeframe) == key
                    ),
                    open_symbols=open_symbols,
                    use_reasoning=use_reasoning,
                    now=now,
                )
            except Exception as exc:
                log.warning("Signal construction failed for %s: %s", key, exc)
                continue

            if signal is None:
                continue
            bundle.signals.append(signal)
            if signal.is_actionable:
                open_symbols.append(signal.symbol)

        bundle.rankings = [r.to_dict() for r in sorted(
            rankings, key=lambda r: r.score, reverse=True
        )]
        bundle.market_summary = (
            market_summary(regimes, len([r for r in rankings if r.tradable]), len(rankings))
            + " "
            + summarise(rankings)
        )
        diagnostics["candidates_evaluated"] = len(candidates)
        diagnostics["rankings"] = len(rankings)
        bundle.diagnostics = diagnostics

        for signal in bundle.actionable:
            self._journal_signal(signal)

        return bundle

    def _evaluate_symbol(
        self, symbol: str, timeframes: list[str], now: datetime
    ) -> dict:
        """Score every timeframe for one symbol."""
        cfg = self.config
        instrument = get_instrument(symbol)
        result: dict = {
            "rankings": [],
            "candidates": [],
            "watch": [],
            "blocked": [],
            "regime": None,
        }

        # --- event risk gate (applies to every timeframe) -----------------
        event_risk = self.calendar.assess(
            symbol,
            blackout_minutes=cfg.signals.news_blackout_minutes,
            allow_medium_impact=cfg.signals.allow_medium_impact_trading,
        )
        if event_risk.blocked:
            result["blocked"].append({"symbol": symbol, "reason": event_risk.reason})
            return result

        is_weekend = now.weekday() >= 5
        if is_weekend and not instrument.trades_weekends:
            result["blocked"].append(
                {"symbol": symbol, "reason": "market closed for the weekend"}
            )
            return result

        news_sentiment = self.news.sentiment_for(symbol)
        headlines = self.news.headlines_for_briefing(symbol, 5)

        for timeframe in timeframes:
            model = self.registry.load(symbol, timeframe)
            if model is None or not self.registry.is_enabled(symbol, timeframe):
                continue

            df = self.router.get_bars(symbol, timeframe, cfg.data.live_bars)
            if df.empty or len(df) < 200:
                continue

            # Regime, computed once per symbol on the primary timeframe.
            if result["regime"] is None:
                detector = self._regime_detectors.get(symbol)
                if detector is None:
                    detector = RegimeDetector().fit(df)
                    self._regime_detectors[symbol] = detector
                result["regime"] = detector.current(df)

            regime_state = result["regime"]

            anomaly = detect_anomaly(df, symbol, timeframe)
            if anomaly.is_igniting or anomaly.is_coiling:
                result["watch"].append(
                    WatchItem(
                        symbol=symbol,
                        mt5_symbol=mt5_name(symbol, cfg.mt5_symbol_suffix),
                        timeframe=timeframe,
                        reason=anomaly.describe(),
                        ignition_score=anomaly.ignition_score,
                        coiling_score=anomaly.coiling_score,
                        direction_hint=anomaly.direction,
                        price=float(df["close"].iloc[-1]),
                        price_change_pct=anomaly.price_change_pct,
                    )
                )

            context = self._context_frames(symbol, timeframe)
            features = build_feature_matrix(df, timeframe, cfg.features, context)
            X, _ = clean_for_model(features)
            if X.empty:
                continue

            latest = X.iloc[[-1]].reindex(columns=model.feature_columns)
            if latest.isna().all(axis=1).iloc[0]:
                continue
            latest = latest.fillna(0.0)

            prediction = model.predict_signal(latest).iloc[0]
            direction = int(prediction["direction"])
            if direction == 0:
                continue

            measured = prediction.get("measured_accuracy")
            measured = float(measured) if pd.notna(measured) else None

            from signalforge.features import indicators as ta

            atr_value = float(
                ta.atr(df["high"], df["low"], df["close"], cfg.features.atr_period).iloc[-1]
            )
            spread_pips = self.router.effective_spread_pips(symbol)
            reward_risk = cfg.risk.tp_atr_mults[0] / cfg.risk.sl_atr_mult

            regime_fit = max(
                regime_state.trend_following_score,
                regime_state.mean_reversion_score,
                regime_state.breakout_score,
            )

            # Use the overlap-adjusted sample, not the raw row count, so a model
            # trained on 600 heavily-overlapping labels is ranked as the thin
            # evidence it actually is.
            sample_size = (
                model.report.effective_sample_size if model.report else 0
            )
            ranking = rank_candidate(
                symbol=symbol,
                timeframe=timeframe,
                model_accuracy=model.report.directional_accuracy if model.report else 0.5,
                measured_accuracy=measured,
                sample_size=sample_size,
                reward_risk=reward_risk,
                atr=atr_value,
                spread_pips=spread_pips,
                pip_size=instrument.pip_size,
                market=instrument.market,
                hour_utc=now.hour,
                trades_weekends=instrument.trades_weekends,
                is_weekend=is_weekend,
                regime_fit=regime_fit,
                data_rows=len(df),
                expected_rows=cfg.data.live_bars,
            )
            result["rankings"].append(ranking)

            if not ranking.tradable:
                continue

            confidence = float(prediction["confidence"])
            edge = float(prediction["edge"])
            if (
                confidence < cfg.signals.min_confidence
                or abs(edge) < cfg.signals.min_directional_edge
            ):
                continue

            # Conditional-edge gate. The model may be confident, but if it has
            # historically lost money in the regime or session holding right
            # now, that confidence is not worth acting on.
            conditional_note = ""
            if cfg.signals.enforce_conditional_edge and model.report:
                from signalforge.models import ConditionalEdge

                conditional = ConditionalEdge.from_dict(
                    model.report.conditional_edge or {}
                )
                allowed, note = conditional.regime_verdict(
                    self._current_trend_regime(symbol, df)
                )
                if not allowed:
                    result["blocked"].append(
                        {"symbol": symbol, "reason": f"{timeframe}: {note}"}
                    )
                    continue
                conditional_note = note

                session_ok, session_note = conditional.session_verdict(now.hour)
                if not session_ok:
                    result["blocked"].append(
                        {"symbol": symbol, "reason": f"{timeframe}: {session_note}"}
                    )
                    continue

            result["candidates"].append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "df": df,
                    "features": X,
                    "model": model,
                    "prediction": prediction,
                    "direction": direction,
                    "confidence": confidence,
                    "edge": edge,
                    "measured_accuracy": measured,
                    "regime": regime_state,
                    "anomaly": anomaly,
                    "event_risk": event_risk,
                    "news_sentiment": news_sentiment,
                    "headlines": headlines,
                    "atr": atr_value,
                    "spread_pips": spread_pips,
                    "instrument": instrument,
                    "conditional_note": conditional_note,
                }
            )

        return result

    def _current_trend_regime(self, symbol: str, df: pd.DataFrame) -> str:
        """Trend regime for the latest bar, reusing the cached detector."""
        detector = self._regime_detectors.get(symbol)
        if detector is None:
            detector = RegimeDetector().fit(df)
            self._regime_detectors[symbol] = detector
        series = detector.classify_series(df)["trend_regime"]
        return str(series.iloc[-1]) if len(series) else "range"

    def _build_signal(
        self,
        candidate: dict,
        *,
        ranking: Ranking,
        open_symbols: list[str],
        use_reasoning: bool,
        now: datetime,
    ) -> Signal | None:
        """Assemble one complete, risk-sized signal."""
        cfg = self.config
        instrument = candidate["instrument"]
        symbol = candidate["symbol"]
        timeframe = candidate["timeframe"]
        direction = candidate["direction"]
        df = candidate["df"]

        levels = compute_levels(
            df,
            instrument,
            direction,
            sl_atr_mult=cfg.risk.sl_atr_mult,
            tp_atr_mults=cfg.risk.tp_atr_mults,
            atr_period=cfg.features.atr_period,
            structure_lookback=cfg.risk.structure_lookback,
            structure_buffer_atr=cfg.risk.structure_buffer_atr,
            min_reward_risk=cfg.risk.min_reward_risk,
        )

        # Scale risk down when a correlated position is already open.
        risk_percent, correlation_note = correlation_adjusted_risk(
            symbol, open_symbols, cfg.risk.risk_percent_per_trade
        )

        conversion_rate = self._conversion_rate(instrument)
        size = calculate_lots(
            instrument,
            entry_price=levels.entry,
            stop_price=levels.stop_loss,
            account_balance=cfg.risk.account_balance,
            risk_percent=risk_percent,
            account_currency=cfg.risk.account_currency,
            conversion_rate=conversion_rate,
        )

        measured = candidate["measured_accuracy"]
        confidence = candidate["confidence"]
        quality = grade(
            confidence,
            measured,
            levels.reward_risk,
            candidate["edge"],
            min_confidence=cfg.signals.min_confidence,
        )

        feature_row = candidate["features"].iloc[-1].to_dict()
        agreement, _ = multi_timeframe_agreement(feature_row, direction)
        regime_state = candidate["regime"]

        anomaly = candidate["anomaly"]
        anomaly_note = anomaly.describe() if (anomaly.is_igniting or anomaly.is_coiling) else ""

        event_risk = candidate["event_risk"]
        event_note = ""
        if event_risk.next_event and event_risk.minutes_to_next:
            event_note = (
                f"Next scheduled event: {event_risk.next_event.title} in "
                f"{event_risk.minutes_to_next:.0f} minutes."
            )

        reasoning_text = describe_evidence(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            regime=regime_state,
            features=feature_row,
            measured_accuracy=measured,
            model_confidence=confidence,
            reward_risk=levels.reward_risk,
            news_sentiment=candidate["news_sentiment"].score,
            anomaly_note=anomaly_note,
            event_note=event_note,
        )

        warnings = build_warnings(
            regime=regime_state,
            reward_risk=levels.reward_risk,
            measured_accuracy=measured,
            cost_ratio=ranking.cost_ratio,
            agreement=agreement,
            direction=direction,
            event_risk_score=event_risk.risk_score,
            conversion_approximated=size.conversion_approximated,
        )
        warnings.extend(levels.warnings)
        warnings.extend(size.warnings)
        if correlation_note:
            warnings.append(correlation_note)

        # The regime gate passed; say why, since "this model makes money here"
        # is stronger evidence than any single-bar indicator reading.
        conditional_note = candidate.get("conditional_note")
        if conditional_note:
            reasoning_text += f" {conditional_note[0].upper()}{conditional_note[1:]}."

        model = candidate["model"]
        top_features = model.report.top_features(10) if model.report else []

        # --- optional Claude review ---------------------------------------
        if use_reasoning and self.reasoner.available:
            packet = build_evidence_packet(
                symbol=symbol,
                timeframe=timeframe,
                direction=Direction.from_int(direction).value,
                entry=levels.entry,
                stop_loss=levels.stop_loss,
                take_profits=levels.take_profits,
                reward_risk=levels.reward_risk,
                model_confidence=confidence,
                measured_accuracy=measured,
                regime=regime_state.to_dict(),
                top_features=top_features,
                news_sentiment=candidate["news_sentiment"].score,
                headlines=candidate["headlines"],
                event_risk=event_risk.to_dict(),
                anomaly=anomaly.to_dict(),
                backtest={
                    "out_of_sample_accuracy": model.report.directional_accuracy
                    if model.report
                    else None,
                    "sample_size": model.report.n_samples if model.report else 0,
                    "expected_r": ranking.expected_r,
                    "cost_ratio": ranking.cost_ratio,
                },
            )
            assessment = self.reasoner.review(packet, fallback_reasoning=reasoning_text)
            if assessment.rejected:
                log.info("Reasoning layer rejected %s %s", symbol, timeframe)
                return None
            reasoning_text = assessment.reasoning or reasoning_text
            warnings.extend(assessment.key_risks)
            if assessment.invalidation:
                warnings.append(f"Invalidated if: {assessment.invalidation}")
            confidence = float(
                np.clip(confidence + assessment.confidence_adjustment, 0.0, 1.0)
            )

        validity_minutes = (
            timeframe_minutes(timeframe) * cfg.signals.signal_validity_bars
        )

        return Signal(
            symbol=symbol,
            mt5_symbol=mt5_name(symbol, cfg.mt5_symbol_suffix),
            market=instrument.market,
            timeframe=timeframe,
            direction=Direction.from_int(direction),
            generated_at=now,
            valid_until=now + timedelta(minutes=validity_minutes),
            entry=levels.entry,
            stop_loss=levels.stop_loss,
            take_profits=levels.take_profits,
            lots=size.lots,
            risk_amount=size.risk_amount,
            risk_percent=risk_percent,
            stop_distance_pips=levels.stop_distance_pips,
            reward_risk=levels.reward_risk,
            model_confidence=confidence,
            measured_accuracy=measured,
            directional_edge=candidate["edge"],
            quality=quality,
            evidence=Evidence(
                regime=regime_state.describe(),
                regime_scores={
                    "trend_following": regime_state.trend_following_score,
                    "mean_reversion": regime_state.mean_reversion_score,
                    "breakout": regime_state.breakout_score,
                },
                top_features=top_features,
                multi_timeframe_agreement=agreement,
                anomaly=anomaly.to_dict(),
                news_sentiment=candidate["news_sentiment"].score,
                news_headlines=candidate["headlines"],
                event_risk=event_risk.to_dict(),
                backtest={
                    "out_of_sample_accuracy": model.report.directional_accuracy
                    if model.report
                    else None,
                    "expected_r": ranking.expected_r,
                    "cost_ratio": ranking.cost_ratio,
                    "rank_score": ranking.score,
                },
            ),
            reasoning=reasoning_text,
            warnings=warnings,
            stop_basis=levels.stop_basis,
            atr=levels.atr,
            spread_pips=candidate["spread_pips"],
        )

    def _conversion_rate(self, instrument) -> float | None:
        """Fetch a quote-currency to account-currency rate for cross pairs."""
        account = self.config.risk.account_currency.upper()
        quote = instrument.quote_currency.upper()
        if quote == account or instrument.base_currency.upper() == account:
            return None

        for pair in (f"{quote}{account}", f"{account}{quote}"):
            try:
                get_instrument(pair)
            except KeyError:
                continue
            try:
                quote_data = self.router.live_quote(pair)
                price = quote_data.get("price") or quote_data.get("bid")
                if price and price > 0:
                    return float(price) if pair.startswith(quote) else 1.0 / float(price)
            except Exception:
                continue
        return None

    def _journal_signal(self, signal: Signal) -> None:
        """Record the signal so its outcome can be measured later."""
        try:
            self.journal.record(
                JournalEntry(
                    signal_id=str(uuid.uuid4()),
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    direction=signal.direction.value,
                    issued_at=signal.generated_at.isoformat(),
                    entry=signal.entry,
                    stop_loss=signal.stop_loss,
                    take_profits=signal.take_profits,
                    lots=signal.lots,
                    risk_amount=signal.risk_amount,
                    model_confidence=signal.model_confidence,
                    measured_accuracy=signal.measured_accuracy,
                    reward_risk=signal.reward_risk,
                    regime=signal.evidence.regime,
                    quality=signal.quality.value,
                )
            )
        except Exception as exc:
            log.warning("Could not journal signal for %s: %s", signal.symbol, exc)
