"""Performance metrics.

Deliberately includes the unflattering ones. Any backtest can be made to look
good by quoting total return and hiding the drawdown, the trade count and the
cost assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    trades: int
    wins: int
    losses: int
    win_rate: float
    # Gross profit divided by gross loss. Below 1.0 means the strategy loses money.
    profit_factor: float
    # Average profit per trade in account currency, after costs.
    expectancy: float
    # The same in R multiples (risk units) — comparable across instruments.
    expectancy_r: float
    total_return_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    sharpe: float
    sortino: float
    calmar: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_bars: float
    total_costs: float
    # Costs as a share of gross profit — the number that kills scalping systems.
    cost_drag_pct: float
    consecutive_losses: int
    exposure_pct: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_viable(self) -> bool:
        """A deliberately strict bar for calling a strategy tradable."""
        return (
            self.trades >= 30
            and self.profit_factor > 1.15
            and self.expectancy_r > 0.05
            and self.max_drawdown_pct < 35.0
        )

    def verdict(self) -> str:
        if self.trades < 30:
            return f"Not enough trades ({self.trades}) to judge. Treat as unproven."
        if self.profit_factor <= 1.0:
            return (
                f"Loses money after costs (profit factor {self.profit_factor:.2f}). "
                "Do not trade."
            )
        if self.profit_factor < 1.15:
            return (
                f"Marginal (profit factor {self.profit_factor:.2f}). The edge is "
                "inside the error bars — a slightly worse spread erases it."
            )
        if self.max_drawdown_pct > 35.0:
            return (
                f"Profitable but the {self.max_drawdown_pct:.1f}% drawdown is more "
                "than most people can sit through."
            )
        return (
            f"Viable on this sample: profit factor {self.profit_factor:.2f}, "
            f"{self.expectancy_r:.2f}R per trade over {self.trades} trades."
        )


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Deepest peak-to-trough decline, and how long it lasted."""
    if equity.empty:
        return 0.0, 0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0.0, np.nan)
    max_dd = float(abs(drawdown.min()) * 100.0) if len(drawdown) else 0.0

    # Longest stretch spent below a previous peak.
    underwater = equity < running_max
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return max_dd, longest


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Return per unit of total volatility."""
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Like Sharpe, but only penalises downside volatility.

    Usually the fairer measure for a strategy that takes small losses and
    occasional large wins.
    """
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def compute(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    starting_balance: float,
    periods_per_year: float = 252.0,
    total_bars: int = 0,
) -> PerformanceReport:
    """Build the full report from a trade log and an equity curve."""
    if trades.empty:
        return PerformanceReport(
            0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0,
        )

    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_dd, dd_duration = max_drawdown(equity)
    returns = equity.pct_change().dropna()

    total_return = (
        float((equity.iloc[-1] / starting_balance - 1.0) * 100.0)
        if len(equity)
        else 0.0
    )
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    # Longest losing streak.
    streak = longest_streak = 0
    for value in pnl:
        streak = streak + 1 if value <= 0 else 0
        longest_streak = max(longest_streak, streak)

    total_costs = float(trades.get("cost", pd.Series(dtype="float64")).sum())
    cost_drag = (
        100.0 * total_costs / (gross_profit + total_costs)
        if (gross_profit + total_costs) > 0
        else 0.0
    )

    bars_in_market = float(trades.get("holding_bars", pd.Series([0])).sum())
    exposure = 100.0 * bars_in_market / total_bars if total_bars else 0.0

    r_multiples = trades.get("r_multiple", pd.Series(dtype="float64"))

    return PerformanceReport(
        trades=int(len(trades)),
        wins=int(len(wins)),
        losses=int(len(losses)),
        win_rate=round(float(len(wins) / len(trades)), 4),
        profit_factor=round(profit_factor, 3),
        expectancy=round(float(pnl.mean()), 2),
        expectancy_r=round(float(r_multiples.mean()), 3) if len(r_multiples) else 0.0,
        total_return_pct=round(total_return, 2),
        max_drawdown_pct=round(max_dd, 2),
        max_drawdown_duration_bars=int(dd_duration),
        sharpe=round(sharpe_ratio(returns, periods_per_year), 3),
        sortino=round(sortino_ratio(returns, periods_per_year), 3),
        calmar=round(calmar, 3),
        avg_win=round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 2) if len(losses) else 0.0,
        largest_win=round(float(pnl.max()), 2),
        largest_loss=round(float(pnl.min()), 2),
        avg_holding_bars=round(float(trades.get("holding_bars", pd.Series([0])).mean()), 1),
        total_costs=round(total_costs, 2),
        cost_drag_pct=round(cost_drag, 2),
        consecutive_losses=int(longest_streak),
        exposure_pct=round(exposure, 2),
    )


def by_regime(trades: pd.DataFrame, regime_column: str = "regime") -> pd.DataFrame:
    """Break performance down by market regime.

    Frequently the whole edge lives in one regime and the others bleed. Knowing
    which is the difference between a filter and a rewrite.
    """
    if trades.empty or regime_column not in trades.columns:
        return pd.DataFrame()

    rows = []
    for regime, group in trades.groupby(regime_column):
        pnl = group["pnl"]
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(abs(pnl[pnl <= 0].sum()))
        rows.append(
            {
                "regime": regime,
                "trades": len(group),
                "win_rate": round(float((pnl > 0).mean()), 3),
                "profit_factor": round(
                    gross_profit / gross_loss if gross_loss else float("inf"), 3
                ),
                "total_pnl": round(float(pnl.sum()), 2),
                "expectancy": round(float(pnl.mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    """Performance by hour of day — reveals which sessions actually pay."""
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame()

    frame = trades.copy()
    frame["hour"] = pd.DatetimeIndex(frame["entry_time"]).hour
    rows = []
    for hour, group in frame.groupby("hour"):
        pnl = group["pnl"]
        rows.append(
            {
                "hour_utc": int(hour),
                "trades": len(group),
                "win_rate": round(float((pnl > 0).mean()), 3),
                "total_pnl": round(float(pnl.sum()), 2),
            }
        )
    return pd.DataFrame(rows)
