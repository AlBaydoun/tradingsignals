"""Command-line interface.

    signalforge train      fit models for the watchlist
    signalforge signals    generate signals now
    signalforge backtest   measure a model out-of-sample
    signalforge scan       find markets moving abnormally
    signalforge learn      resolve open signals and police models
    signalforge journal    live performance versus what was promised
    signalforge doctor     check data, models and configuration
    signalforge watch      run continuously
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

from signalforge import __version__
from signalforge.config import Config, load_config

log = logging.getLogger("signalforge")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty and rarely useful at INFO.
    for noisy in ("urllib3", "requests", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_train(args, config: Config) -> int:
    from signalforge.signals import SignalEngine

    engine = SignalEngine(config)
    symbols = args.symbols or config.watchlist
    timeframes = args.timeframes or config.timeframes

    print(f"Training {len(symbols)} symbols x {len(timeframes)} timeframes")
    print("This fits a fresh model per combination and validates it walk-forward.")
    print()

    results, failures = [], []
    total = len(symbols) * len(timeframes)
    done = 0

    for symbol in symbols:
        for timeframe in timeframes:
            done += 1
            prefix = f"[{done}/{total}] {symbol} {timeframe}"
            try:
                result = engine.train(symbol, timeframe)
                results.append(result)
                significant = result["ci_low"] > 0.5
                flag = "" if significant else "  <- edge not significant"
                print(
                    f"{prefix}: accuracy {result['accuracy']:.4f} "
                    f"[95% CI {result['ci_low']:.3f}-{result['ci_high']:.3f}] "
                    f"eff.n {result['effective_samples']}{flag}"
                )
                for warning in result["warnings"]:
                    print(f"           ! {warning}")
            except Exception as exc:
                failures.append((symbol, timeframe, str(exc)))
                print(f"{prefix}: skipped — {exc}")

    print()
    print("=" * 60)
    if results:
        from signalforge.models import assess_batch, describe_batch

        accuracies = [r["accuracy"] for r in results]
        print(f"Trained {len(results)} models, {len(failures)} skipped.")
        print(f"Mean out-of-sample accuracy: {sum(accuracies) / len(accuracies):.4f}")

        # Correct for the size of the search. Testing 40 models produces about
        # two false positives at the 5% level even when every model is useless.
        for result in results:
            result["key"] = f"{result['symbol']}/{result['timeframe']}"
        assessed = assess_batch(results, alpha=0.05)
        by_key = {a.key: a for a in assessed}

        print()
        print(describe_batch(assessed))

        survivors = [a for a in assessed if a.survives_correction]
        if survivors:
            print()
            print("Models that survive multiple-comparison correction:")
            header = (
                f"  {'MODEL':18s} {'ACC':>6s} {'95% CI':>15s} "
                f"{'eff.n':>6s} {'q':>7s}  WHERE IT WORKS"
            )
            print(header)
            for entry in sorted(survivors, key=lambda a: a.q_value):
                result = next(r for r in results if r["key"] == entry.key)
                edge = result.get("conditional_edge") or {}
                regimes = edge.get("by_regime", {})
                good = [
                    name.replace("_", " ")
                    for name, slice_ in regimes.items()
                    if slice_.get("trades", 0) >= 25
                    and slice_.get("profit_factor", 0) >= 1.0
                ]
                where = ", ".join(good[:2]) if good else "not yet mapped"
                print(
                    f"  {entry.key:18s} {entry.accuracy:6.3f} "
                    f"[{result['ci_low']:.3f}-{result['ci_high']:.3f}] "
                    f"{entry.effective_n:6d} {entry.q_value:7.4f}  {where}"
                )
        else:
            print()
            print("Nothing survived. That is a real result, not a bug — most")
            print("markets on most timeframes are efficient enough that costs")
            print("eat any small predictive signal. Trading them anyway is how")
            print("accounts die.")

        demoted = [a for a in assessed if a.demoted]
        if demoted:
            print()
            print("Demoted by the correction (looked good alone, not in a batch):")
            for entry in demoted:
                print(
                    f"  {entry.key:18s} {entry.accuracy:6.3f} "
                    f"p={entry.p_value:.4f} -> q={entry.q_value:.4f}"
                )

        # Where the backtest found each model actually loses money.
        blocked_any = [
            r for r in results if (r.get("conditional_edge") or {}).get("by_regime")
        ]
        if blocked_any:
            print()
            print("Conditional edge maps (regimes that will be blocked live):")
            for result in blocked_any:
                regimes = result["conditional_edge"]["by_regime"]
                losing = [
                    f"{name.replace('_', ' ')} (PF {s['profit_factor']:.2f}, "
                    f"{s['trades']} trades)"
                    for name, s in regimes.items()
                    if s.get("trades", 0) >= 25 and s.get("profit_factor", 99) < 1.0
                ]
                if losing:
                    print(f"  {result['key']:18s} {'; '.join(losing)}")
    else:
        print("No models trained successfully.")
    return 0 if results else 1


def cmd_signals(args, config: Config) -> int:
    from signalforge.signals import SignalEngine, format_bundle, format_compact

    engine = SignalEngine(config)
    bundle = engine.generate(
        symbols=args.symbols,
        timeframes=args.timeframes,
        use_reasoning=not args.no_reasoning,
    )

    if args.json:
        print(json.dumps(bundle.to_dict(), indent=2, default=str))
    elif args.compact:
        print(format_compact(bundle))
    else:
        print(format_bundle(bundle, verbose=not args.brief))

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(bundle.to_dict(), fh, indent=2, default=str)
        print(f"\nWritten to {args.output}")

    return 0


def cmd_backtest(args, config: Config) -> int:
    from signalforge.backtest import BacktestConfig, by_hour, by_regime, walk_forward_backtest
    from signalforge.features import build_feature_matrix, clean_for_model
    from signalforge.labeling import apply_triple_barrier, cost_in_price_units
    from signalforge.models import SignalModel
    from signalforge.regime import RegimeDetector
    from signalforge.signals import SignalEngine
    from signalforge.universe import get_instrument

    engine = SignalEngine(config)
    symbol, timeframe = args.symbol.upper(), args.timeframe.upper()
    instrument = get_instrument(symbol)

    print(f"Backtesting {symbol} {timeframe}")
    print("Training a model and evaluating it on walk-forward predictions only.")
    print()

    bars = config.data.history_bars.get(timeframe, 15000)
    df = engine.router.get_bars(symbol, timeframe, bars)
    if df.empty:
        print(f"No data for {symbol} {timeframe}")
        return 1
    print(f"Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")

    context = engine._context_frames(symbol, timeframe)
    features = build_feature_matrix(df, timeframe, config.features, context)
    X, _ = clean_for_model(features)

    cost = cost_in_price_units(
        engine.router.effective_spread_pips(symbol), instrument.pip_size
    )
    labels = apply_triple_barrier(
        df,
        upper_atr_mult=config.labels.upper_atr_mult,
        lower_atr_mult=config.labels.lower_atr_mult,
        horizon=config.labels.max_horizon_bars,
        cost_price_units=cost,
        min_cost_multiple=config.labels.min_cost_multiple,
    )
    label_frame = labels.to_frame()
    usable = (
        label_frame["tradable"] & ~label_frame["ambiguous"] & X.notna().all(axis=1)
    )
    print(f"Usable labelled rows: {int(usable.sum())}")

    model = SignalModel(config.model)
    report = model.fit(
        X[usable],
        label_frame["label"][usable],
        sample_weight=label_frame["sample_weight"][usable],
        event_end_time=label_frame["event_end_time"][usable],
        symbol=symbol,
        timeframe=timeframe,
    )
    print(
        f"Out-of-sample accuracy: {report.directional_accuracy:.4f} "
        f"over {report.n_folds} folds"
    )
    print()

    detector = RegimeDetector().fit(df)
    regimes = detector.classify_series(df)["trend_regime"]

    trades, equity, performance = walk_forward_backtest(
        model,
        df,
        instrument,
        config=BacktestConfig(
            starting_balance=config.risk.account_balance,
            risk_percent=config.risk.risk_percent_per_trade,
        ),
        regimes=regimes,
        min_confidence=args.min_confidence,
        min_edge=args.min_edge,
        sl_atr_mult=config.risk.sl_atr_mult,
        tp_atr_mult=config.risk.tp_atr_mults[0],
    )

    print("=" * 60)
    print("WALK-FORWARD BACKTEST (after spread, slippage and commission)")
    print("=" * 60)
    for key, value in performance.to_dict().items():
        print(f"  {key:30s} {value}")
    print()
    print("VERDICT:", performance.verdict())

    if not trades.empty and args.detail:
        print()
        print("By regime:")
        print(by_regime(trades).to_string(index=False))
        print()
        print("By hour (UTC):")
        print(by_hour(trades).to_string(index=False))

    return 0


def cmd_scan(args, config: Config) -> int:
    from signalforge.anomaly import scan
    from signalforge.data import DataRouter

    router = DataRouter(config.data)
    symbols = args.symbols or config.watchlist
    timeframe = args.timeframe

    print(f"Scanning {len(symbols)} instruments on {timeframe} for unusual activity")
    print()

    frames = router.get_many(symbols, timeframe, 500)
    reports = scan(frames, timeframe, min_score=args.min_score)

    if reports:
        print(f"{len(reports)} instruments showing abnormal behaviour:")
        print()
        for report in reports:
            print(f"  {report.describe()}")
            print(
                f"     ignition {report.ignition_score:.0f} | "
                f"coiling {report.coiling_score:.0f} | "
                f"20-bar change {report.price_change_pct:+.2f}%"
            )
            if report.triggers:
                print(f"     triggers: {', '.join(report.triggers)}")
            print()
    else:
        print("Nothing unusual. All instruments within normal ranges.")

    if args.crypto_wide:
        print()
        print("Market-wide crypto movers (24h, beyond the watchlist):")
        for mover in router.scan_crypto_movers(12):
            print(
                f"  {mover['symbol']:12s} {mover['price_change_pct']:+7.2f}%  "
                f"vol ${mover['quote_volume'] / 1e6:.0f}M"
            )

    return 0


def cmd_learn(args, config: Config) -> int:
    from signalforge.learning import LearningLoop
    from signalforge.signals import SignalEngine

    engine = SignalEngine(config)
    loop = LearningLoop(config, router=engine.router)

    print("Running the learning loop")
    print()
    report = loop.run_once(
        retrain=args.retrain,
        trainer=engine.train if args.retrain else None,
    )

    print(f"Result: {report.describe()}")
    if report.drift_alerts:
        print()
        print("Drift alerts:")
        for alert in report.drift_alerts:
            print(f"  {alert['model']}: {alert['action']}")
            for reason in alert.get("reasons", []):
                print(f"     - {reason}")
    if report.errors:
        print()
        print("Errors:")
        for error in report.errors:
            print(f"  {error}")
    return 0


def cmd_journal(args, config: Config) -> int:
    from signalforge.learning import TradeJournal

    journal = TradeJournal(config.learning.journal_path)
    summary = journal.summary()

    print("=" * 60)
    print("TRADE JOURNAL")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:28s} {value}")

    buckets = journal.by_confidence_bucket()
    if buckets:
        print()
        print("Live calibration — did the stated confidence hold up?")
        print(f"  {'range':12s} {'trades':>7s} {'win rate':>9s} {'claimed':>9s}")
        for bucket in buckets:
            print(
                f"  {bucket['range']:12s} {bucket['trades']:7d} "
                f"{bucket['win_rate']:9.3f} {bucket['mean_confidence']:9.3f}"
            )

    delivered = summary.get("delivered_minus_promised")
    if delivered is not None:
        print()
        if delivered < -0.10:
            print(
                f"  Live results are {abs(delivered):.1%} WORSE than the models "
                "promised. Reduce size and investigate."
            )
        elif delivered > 0.05:
            print(f"  Live results are {delivered:.1%} better than promised.")
        else:
            print("  Live results are tracking the models' predictions.")

    return 0


def cmd_doctor(args, config: Config) -> int:
    from signalforge.data import DataRouter
    from signalforge.models import ModelRegistry
    from signalforge.news import EconomicCalendar, NewsAggregator
    from signalforge.reasoning import ReasoningEngine

    print("=" * 60)
    print(f"SIGNALFORGE {__version__} — SYSTEM CHECK")
    print("=" * 60)

    problems: list[str] = []

    print("\nData providers:")
    router = DataRouter(config.data)
    health = router.health(config.watchlist[:6], "H1")
    for symbol, status in health.items():
        mark = "ok " if status.get("ok") else "FAIL"
        rows = status.get("rows", 0)
        print(f"  [{mark}] {symbol:10s} {status.get('provider', '?'):8s} {rows} bars")
        if not status.get("ok"):
            problems.append(f"no data for {symbol}")

    print("\nModels:")
    registry = ModelRegistry(config.model.model_dir)
    summary = registry.summary()
    print(f"  {summary['total']} trained, {summary['enabled']} enabled")
    print(f"  mean out-of-sample accuracy: {summary['mean_accuracy']}")
    if summary["total"] == 0:
        problems.append("no models trained yet — run `signalforge train`")
    else:
        weak = [m for m in registry.list_models() if m.directional_accuracy < 0.5]
        if weak:
            print(f"  {len(weak)} model(s) below 50% accuracy:")
            for entry in weak[:5]:
                print(
                    f"     {entry.symbol}/{entry.timeframe}: "
                    f"{entry.directional_accuracy:.3f}"
                )

    print("\nEconomic calendar:")
    calendar = EconomicCalendar()
    events = calendar.refresh()
    print(f"  {len(events)} events this week")
    if not events:
        problems.append("calendar unavailable — event filtering is disabled")

    print("\nNews feeds:")
    news = NewsAggregator()
    items = news.fetch()
    print(f"  {len(items)} headlines from {len({i.source for i in items})} sources")
    if len(items) < 10:
        problems.append("few headlines retrieved — sentiment will be weak")

    print("\nReasoning layer:")
    reasoner = ReasoningEngine(config.reasoning)
    if reasoner.available:
        print(f"  enabled, model {config.reasoning.model}")
    else:
        print(f"  disabled ({reasoner.unavailable_reason})")
        print("  The engine works without it, using deterministic explanations.")

    print("\nRisk configuration:")
    print(f"  balance {config.risk.account_balance} {config.risk.account_currency}")
    print(f"  risk per trade {config.risk.risk_percent_per_trade}%")
    print(f"  MT5 symbol suffix {config.mt5_symbol_suffix!r}")
    if config.risk.risk_percent_per_trade > 2.0:
        problems.append(
            f"risk per trade is {config.risk.risk_percent_per_trade}% — "
            "above 2% is aggressive enough to matter"
        )

    print()
    print("=" * 60)
    if problems:
        print(f"{len(problems)} issue(s) found:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("All checks passed.")
    return 1 if problems else 0


def cmd_watch(args, config: Config) -> int:
    from signalforge.learning import LearningLoop
    from signalforge.signals import SignalEngine, format_bundle

    engine = SignalEngine(config)
    loop = LearningLoop(config, router=engine.router)
    interval = args.interval

    print(f"Watching. Regenerating signals every {interval}s. Ctrl-C to stop.")
    print()

    try:
        while True:
            started = time.time()
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"\n[{stamp}] running...")

            try:
                learning = loop.run_once(retrain=False)
                if learning.resolved:
                    print(f"  learning: {learning.describe()}")
            except Exception as exc:
                print(f"  learning loop failed: {exc}")

            try:
                bundle = engine.generate(use_reasoning=not args.no_reasoning)
                actionable = bundle.actionable
                if actionable:
                    print(format_bundle(bundle, verbose=False))
                else:
                    print(f"  no signals ({len(bundle.watchlist)} on watch)")
            except Exception as exc:
                print(f"  generation failed: {exc}")

            elapsed = time.time() - started
            time.sleep(max(5.0, interval - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signalforge",
        description="Adaptive market signal engine for MetaTrader 5.",
    )
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"signalforge {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit models for the watchlist")
    train.add_argument("--symbols", nargs="+")
    train.add_argument("--timeframes", nargs="+")
    train.set_defaults(func=cmd_train)

    signals = sub.add_parser("signals", help="generate signals now")
    signals.add_argument("--symbols", nargs="+")
    signals.add_argument("--timeframes", nargs="+")
    signals.add_argument("--json", action="store_true")
    signals.add_argument("--compact", action="store_true", help="one line per signal")
    signals.add_argument("--brief", action="store_true", help="omit the reasoning")
    signals.add_argument("--no-reasoning", action="store_true", help="skip the LLM review")
    signals.add_argument("--output", help="also write JSON to this path")
    signals.set_defaults(func=cmd_signals)

    backtest = sub.add_parser("backtest", help="measure a model out-of-sample")
    backtest.add_argument("symbol")
    backtest.add_argument("timeframe")
    backtest.add_argument("--min-confidence", type=float, default=0.55)
    backtest.add_argument("--min-edge", type=float, default=0.08)
    backtest.add_argument("--detail", action="store_true", help="break down by regime and hour")
    backtest.set_defaults(func=cmd_backtest)

    scan = sub.add_parser("scan", help="find markets moving abnormally")
    scan.add_argument("--symbols", nargs="+")
    scan.add_argument("--timeframe", default="H1")
    scan.add_argument("--min-score", type=float, default=60.0)
    scan.add_argument("--crypto-wide", action="store_true", help="scan all Binance pairs")
    scan.set_defaults(func=cmd_scan)

    learn = sub.add_parser("learn", help="resolve signals and police models")
    learn.add_argument("--retrain", action="store_true")
    learn.set_defaults(func=cmd_learn)

    journal = sub.add_parser("journal", help="live performance versus promises")
    journal.set_defaults(func=cmd_journal)

    doctor = sub.add_parser("doctor", help="check data, models and configuration")
    doctor.set_defaults(func=cmd_doctor)

    watch = sub.add_parser("watch", help="run continuously")
    watch.add_argument("--interval", type=int, default=300)
    watch.add_argument("--no-reasoning", action="store_true")
    watch.set_defaults(func=cmd_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Could not load configuration: {exc}", file=sys.stderr)
        return 1

    try:
        return args.func(args, config)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.exception("Command failed")
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
