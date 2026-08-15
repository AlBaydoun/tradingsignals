"""HTTP API.

Exists so the engine can feed a phone, a Telegram bot or a dashboard without
anyone shelling into the machine. Signal generation is slow (data fetch,
features, inference across the watchlist), so results are cached briefly and
served from the cache rather than recomputed per request.

    uvicorn signalforge.api.server:app --host 0.0.0.0 --port 8000

There is no authentication. Do not expose this to the internet as-is — put it
behind a reverse proxy, or bind it to localhost and reach it over a VPN.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from signalforge import __version__
from signalforge.config import load_config
from signalforge.signals import SignalEngine, format_bundle, format_compact

log = logging.getLogger(__name__)

app = FastAPI(
    title="SignalForge",
    version=__version__,
    description="Adaptive market signal engine for MetaTrader 5.",
)

_config = load_config()
_engine: SignalEngine | None = None
_engine_lock = Lock()

# Signal generation is expensive; serve a recent bundle rather than recompute.
_cache: dict[str, object] = {"bundle": None, "at": None}
_CACHE_SECONDS = 120


def get_engine() -> SignalEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SignalEngine(_config)
        return _engine


def _cached_bundle(force: bool = False):
    now = datetime.now(timezone.utc)
    cached_at = _cache.get("at")
    if (
        not force
        and _cache.get("bundle") is not None
        and cached_at is not None
        and (now - cached_at).total_seconds() < _CACHE_SECONDS
    ):
        return _cache["bundle"]

    bundle = get_engine().generate()
    _cache["bundle"] = bundle
    _cache["at"] = now
    return bundle


@app.get("/")
def root() -> dict:
    return {
        "service": "signalforge",
        "version": __version__,
        "endpoints": [
            "/health",
            "/signals",
            "/signals/text",
            "/watchlist",
            "/rankings",
            "/scan",
            "/journal",
            "/models",
            "/calendar",
            "/dashboard",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "time": datetime.now(timezone.utc).isoformat(),
        "cached_signals_age_seconds": (
            (datetime.now(timezone.utc) - _cache["at"]).total_seconds()
            if _cache.get("at")
            else None
        ),
    }


@app.get("/signals")
def signals(force: bool = Query(False, description="bypass the cache")) -> dict:
    try:
        return _cached_bundle(force).to_dict()
    except Exception as exc:
        log.exception("Signal generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/signals/text", response_class=PlainTextResponse)
def signals_text(
    compact: bool = Query(False), force: bool = Query(False)
) -> str:
    bundle = _cached_bundle(force)
    return format_compact(bundle) if compact else format_bundle(bundle)


@app.get("/watchlist")
def watchlist() -> dict:
    bundle = _cached_bundle()
    return {
        "generated_at": bundle.generated_at.isoformat(),
        "items": [w.to_dict() for w in bundle.watchlist],
    }


@app.get("/rankings")
def rankings(limit: int = Query(20, ge=1, le=100)) -> dict:
    bundle = _cached_bundle()
    return {
        "generated_at": bundle.generated_at.isoformat(),
        "summary": bundle.market_summary,
        "rankings": bundle.rankings[:limit],
    }


@app.get("/scan")
def scan(
    timeframe: str = Query("H1"),
    min_score: float = Query(60.0, ge=0, le=100),
) -> dict:
    from signalforge.anomaly import scan as run_scan

    engine = get_engine()
    frames = engine.router.get_many(_config.watchlist, timeframe, 500)
    reports = run_scan(frames, timeframe, min_score=min_score)
    return {
        "timeframe": timeframe,
        "count": len(reports),
        "reports": [r.to_dict() for r in reports],
    }


@app.get("/journal")
def journal() -> dict:
    from signalforge.learning import TradeJournal

    trade_journal = TradeJournal(_config.learning.journal_path)
    return {
        "summary": trade_journal.summary(),
        "calibration": trade_journal.by_confidence_bucket(),
    }


@app.get("/models")
def models() -> dict:
    from signalforge.models import ModelRegistry

    registry = ModelRegistry(_config.model.model_dir)
    return {
        "summary": registry.summary(),
        "models": [
            {
                "symbol": m.symbol,
                "timeframe": m.timeframe,
                "accuracy": m.directional_accuracy,
                "samples": m.n_samples,
                "trained_at": m.trained_at,
                "age_hours": round(m.age_hours(), 1),
                "enabled": m.enabled,
            }
            for m in registry.list_models()
        ],
    }


@app.get("/calendar")
def calendar(hours: float = Query(48.0, ge=1, le=168)) -> dict:
    from signalforge.news import EconomicCalendar

    economic_calendar = EconomicCalendar()
    economic_calendar.refresh()
    return {
        "events": economic_calendar.week_ahead(_config.watchlist),
        "horizon_hours": hours,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """A minimal self-contained dashboard, readable on a phone."""
    bundle = _cached_bundle()
    rows = []

    for signal in bundle.actionable:
        accuracy = (
            f"{signal.measured_accuracy:.0%}"
            if signal.measured_accuracy is not None
            else "unproven"
        )
        colour = "#2e7d32" if signal.direction.value == "BUY" else "#c62828"
        targets = " / ".join(str(t) for t in signal.take_profits)
        rows.append(
            f"""
            <div class="card">
              <div class="head" style="color:{colour}">
                {signal.direction.value} {signal.mt5_symbol}
                <span class="tf">{signal.timeframe}</span>
              </div>
              <table>
                <tr><td>Entry</td><td>{signal.entry}</td></tr>
                <tr><td>Stop loss</td><td>{signal.stop_loss}
                    ({signal.stop_distance_pips:.0f} pips)</td></tr>
                <tr><td>Targets</td><td>{targets}</td></tr>
                <tr><td>Lots</td><td>{signal.lots}</td></tr>
                <tr><td>Risk</td><td>{signal.risk_percent:.2f}%</td></tr>
                <tr><td>R:R</td><td>1:{signal.reward_risk:.1f}</td></tr>
                <tr><td>Measured accuracy</td><td>{accuracy}</td></tr>
              </table>
              <p class="why">{signal.reasoning}</p>
            </div>"""
        )

    body = "".join(rows) or (
        '<p class="empty">No signals clear the quality bar right now. '
        "Not trading is a position.</p>"
    )

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignalForge</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #101418; color: #e6e9ec; padding: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 16px; }}
  .card {{ background: #1a1f26; border-radius: 10px; padding: 14px;
           margin-bottom: 12px; }}
  .head {{ font-weight: 600; font-size: 17px; margin-bottom: 8px; }}
  .tf {{ background: #2a313a; color: #8b949e; font-size: 12px;
         padding: 2px 6px; border-radius: 4px; margin-left: 6px; }}
  table {{ width: 100%; font-size: 14px; border-collapse: collapse; }}
  td {{ padding: 3px 0; }}
  td:first-child {{ color: #8b949e; }}
  td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .why {{ font-size: 13px; color: #b6bec7; margin: 10px 0 0;
          line-height: 1.45; }}
  .empty {{ color: #8b949e; }}
  .foot {{ color: #6e7681; font-size: 12px; margin-top: 20px;
           line-height: 1.5; }}
</style></head>
<body>
  <h1>SignalForge</h1>
  <div class="meta">{bundle.generated_at.strftime('%Y-%m-%d %H:%M UTC')} —
    {len(bundle.actionable)} signals, {len(bundle.watchlist)} on watch</div>
  {body}
  <div class="foot">Accuracy figures are measured on past out-of-sample data
  and carry no guarantee. Risk only what you can afford to lose.</div>
</body></html>"""
