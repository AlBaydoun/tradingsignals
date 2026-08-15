"""Rendering signals for humans — phone-first.

The primary consumer is someone holding a phone with the MT5 app open. That
constrains the format: short lines, the numbers in the order the MT5 order
ticket asks for them, and no decoration that survives copy-paste badly.

The accuracy line always states the *measured* out-of-sample hit rate and the
sample it came from, or says plainly that there isn't one.
"""

from __future__ import annotations

from signalforge.signals.schema import Signal, SignalBundle, SignalQuality, WatchItem

QUALITY_MARK = {
    SignalQuality.STRONG: "***",
    SignalQuality.MODERATE: "**",
    SignalQuality.WEAK: "*",
    SignalQuality.WATCH_ONLY: "-",
}


def format_signal(signal: Signal, *, verbose: bool = True) -> str:
    """A single signal, formatted for a phone screen."""
    lines: list[str] = []
    mark = QUALITY_MARK[signal.quality]

    lines.append(f"{mark} {signal.direction.value} {signal.mt5_symbol} [{signal.timeframe}]")
    lines.append("")
    lines.append(f"Entry      {signal.entry}")
    lines.append(f"Stop loss  {signal.stop_loss}   ({signal.stop_distance_pips:.0f} pips)")

    for i, target in enumerate(signal.take_profits, start=1):
        lines.append(f"Take profit {i}  {target}")

    lines.append(f"Lot size   {signal.lots}")
    lines.append(
        f"Risk       {signal.risk_amount:.2f} ({signal.risk_percent:.2f}% of account)"
    )
    lines.append(f"R:R        1:{signal.reward_risk:.1f}")
    lines.append("")

    # The honest accuracy statement.
    if signal.measured_accuracy is not None:
        lines.append(
            f"Historical accuracy at this confidence: "
            f"{signal.measured_accuracy:.0%} (measured out-of-sample)"
        )
        lines.append(f"Expectancy: {signal.expectancy_r:+.2f}R per trade")
    else:
        lines.append(
            "Historical accuracy: not enough past signals at this confidence "
            "to quote a number. Treat as unproven."
        )
    lines.append(f"Model confidence: {signal.model_confidence:.0%}")
    lines.append("")
    lines.append(f"Valid for {signal.minutes_remaining():.0f} more minutes")

    if verbose:
        if signal.reasoning:
            lines.append("")
            lines.append("Why:")
            for chunk in _wrap(signal.reasoning, 62):
                lines.append(f"  {chunk}")

        if signal.evidence.regime:
            lines.append("")
            lines.append(f"Market state: {signal.evidence.regime}")

        if signal.stop_basis:
            lines.append(f"Stop placed: {signal.stop_basis}")

        if signal.evidence.news_headlines:
            lines.append("")
            lines.append("Recent news:")
            for item in signal.evidence.news_headlines[:3]:
                lines.append(f"  [{item['sentiment']}] {item['title'][:58]}")

    if signal.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in signal.warnings:
            for chunk in _wrap(warning, 60):
                lines.append(f"  ! {chunk}")

    return "\n".join(lines)


def format_mt5_ticket(signal: Signal) -> str:
    """The bare numbers, in MT5 order-ticket order, for fast manual entry."""
    targets = " / ".join(str(t) for t in signal.take_profits)
    return (
        f"{signal.mt5_symbol} | {signal.direction.value} | "
        f"Vol {signal.lots} | SL {signal.stop_loss} | TP {targets}"
    )


def format_watch_item(item: WatchItem) -> str:
    arrow = "up" if item.direction_hint > 0 else "down" if item.direction_hint < 0 else "?"
    return (
        f"  {item.mt5_symbol} [{item.timeframe}] {item.price} "
        f"({item.price_change_pct:+.1f}%) — {item.reason} "
        f"[direction: {arrow}]"
    )


def format_bundle(bundle: SignalBundle, *, verbose: bool = True) -> str:
    """The full run output."""
    lines: list[str] = []
    timestamp = bundle.generated_at.strftime("%Y-%m-%d %H:%M UTC")

    lines.append("=" * 64)
    lines.append(f"  SIGNALFORGE — {timestamp}")
    lines.append("=" * 64)

    if bundle.market_summary:
        lines.append("")
        for chunk in _wrap(bundle.market_summary, 62):
            lines.append(chunk)

    actionable = bundle.actionable
    lines.append("")
    lines.append(f"TRADE SIGNALS ({len(actionable)})")
    lines.append("-" * 64)

    if not actionable:
        lines.append("")
        lines.append("  No signals clear the quality bar right now.")
        lines.append("  Not trading is a position. This is the engine working,")
        lines.append("  not the engine failing.")
    else:
        for signal in actionable:
            lines.append("")
            lines.append(format_signal(signal, verbose=verbose))
            lines.append("")
            lines.append("-" * 64)

    if bundle.watchlist:
        lines.append("")
        lines.append(f"WATCHLIST — unusual activity ({len(bundle.watchlist)})")
        lines.append("-" * 64)
        for item in bundle.watchlist:
            lines.append(format_watch_item(item))

    if bundle.rankings:
        lines.append("")
        lines.append("WHERE THE EDGE IS (top ranked)")
        lines.append("-" * 64)
        header = f"  {'SYMBOL':10s} {'TF':5s} {'SCORE':>6s} {'exp.R':>7s} {'cost×':>6s} {'acc':>6s}"
        lines.append(header)
        for row in bundle.rankings[:10]:
            accuracy = (
                f"{row['measured_accuracy']:.0%}"
                if row.get("measured_accuracy") is not None
                else "  n/a"
            )
            lines.append(
                f"  {row['symbol']:10s} {row['timeframe']:5s} "
                f"{row['score']:6.1f} {row['expected_r']:+7.3f} "
                f"{row['cost_ratio']:6.1f} {accuracy:>6s}"
            )

    if bundle.blocked:
        lines.append("")
        lines.append("BLOCKED")
        lines.append("-" * 64)
        for entry in bundle.blocked:
            lines.append(f"  {entry.get('symbol', '?')}: {entry.get('reason', '')}")

    lines.append("")
    lines.append("=" * 64)
    lines.append("Signals are probabilistic, not predictions. Accuracy figures are")
    lines.append("measured on past out-of-sample data and do not carry a guarantee.")
    lines.append("Risk only what you can afford to lose entirely.")
    lines.append("=" * 64)

    return "\n".join(lines)


def format_telegram(bundle: SignalBundle) -> list[str]:
    """One message per signal, sized for a Telegram push."""
    messages: list[str] = []
    for signal in bundle.actionable:
        targets = "\n".join(
            f"TP{i}: `{t}`" for i, t in enumerate(signal.take_profits, 1)
        )
        accuracy = (
            f"{signal.measured_accuracy:.0%} measured"
            if signal.measured_accuracy is not None
            else "unproven"
        )
        messages.append(
            f"*{signal.direction.value} {signal.mt5_symbol}* `{signal.timeframe}`\n"
            f"Entry: `{signal.entry}`\n"
            f"SL: `{signal.stop_loss}` ({signal.stop_distance_pips:.0f} pips)\n"
            f"{targets}\n"
            f"Lots: `{signal.lots}` | Risk: {signal.risk_percent:.2f}%\n"
            f"R:R 1:{signal.reward_risk:.1f} | Accuracy: {accuracy}\n\n"
            f"_{signal.reasoning[:280]}_"
        )
    return messages


def format_compact(bundle: SignalBundle) -> str:
    """One line per signal, for a status bar or a quick glance."""
    if not bundle.actionable:
        return "No signals."
    return "\n".join(format_mt5_ticket(s) for s in bundle.actionable)


def _wrap(text: str, width: int) -> list[str]:
    """Simple greedy word wrap — avoids a textwrap import for one call site."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + len(current) > width and current:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]
