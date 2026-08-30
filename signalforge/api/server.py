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
from pathlib import Path
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
                # The count and interval that account for label overlap. A
                # client drawing error bars must use these, never n_samples.
                "effective_samples": m.effective_samples,
                "accuracy_ci_low": m.accuracy_ci_low,
                "accuracy_ci_high": m.accuracy_ci_high,
                "significant": m.edge_is_significant,
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


@app.get("/api/setup")
def setup() -> dict:
    """Symbol names and affordability — the two things to check before trading.

    Slow (it prices every watchlist combination), so it is not on the signal
    path. The dashboard loads it on demand.
    """
    from signalforge.diagnostics import affordability, symbol_mappings

    engine = get_engine()
    checks = affordability(_config, engine.router)
    return {
        "account_balance": _config.risk.account_balance,
        "account_currency": _config.risk.account_currency,
        "risk_percent": _config.risk.risk_percent_per_trade,
        "risk_budget": round(
            _config.risk.account_balance
            * _config.risk.risk_percent_per_trade
            / 100.0,
            2,
        ),
        "mt5_symbol_suffix": _config.mt5_symbol_suffix,
        "timeframes": _config.timeframes,
        "mappings": [m.to_dict() for m in symbol_mappings(_config)],
        "affordability": [a.to_dict() for a in checks],
        "unaffordable": [a.to_dict() for a in checks if not a.affordable],
    }


@app.get("/api/hunt")
def api_hunt(
    timeframe: str = Query("H1"),
    limit: int = Query(25, ge=1, le=60),
    markets: str | None = Query(None, description="comma-separated filter"),
) -> dict:
    """Rank the whole universe by movement per unit of trading cost."""
    from signalforge.models import ModelRegistry
    from signalforge.selection import default_universe, describe, hunt

    engine = get_engine()
    wanted = [m.strip() for m in markets.split(",")] if markets else None
    symbols = sorted(set(_config.watchlist) | set(default_universe(wanted)))

    now = datetime.now(timezone.utc)
    frames = engine.router.get_many(symbols, timeframe, 500)
    trained = {e.symbol for e in ModelRegistry(_config.model.model_dir).list_models()}

    results = hunt(
        frames,
        timeframe,
        hour_utc=now.hour,
        is_weekend=now.weekday() >= 5,
        models=trained,
        limit=limit,
    )
    # Binance lists far more than the engine's universe. These are reported as
    # market context only — there is no model, no cost model and no CFD for
    # most of them, so they are never candidates for a trade.
    movers: list[dict] = []
    if _config.market_watch.crypto_movers:
        try:
            movers = engine.router.scan_crypto_movers(8)
        except Exception as exc:
            log.debug("Crypto mover scan failed: %s", exc)

    return {
        "timeframe": timeframe,
        "generated_at": now.isoformat(),
        "summary": describe(results),
        "results": [r.to_dict() for r in results],
        "crypto_movers": movers,
    }


@app.get("/api/sweep")
def api_sweep() -> dict:
    """One pass over the wider market: what is moving, nothing about direction."""
    from signalforge.marketwatch import sweep
    from signalforge.models import ModelRegistry

    engine = get_engine()
    trained = {e.symbol for e in ModelRegistry(_config.model.model_dir).list_models()}
    result = sweep(engine.router, _config.market_watch, trained=trained)
    return {
        **result.to_dict(),
        "summary": result.describe(),
        "watchlist": _config.watchlist,
    }


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """The dashboard: one self-contained page, no build step, no CDN.

    Served as a static file rather than assembled here, so the markup can be
    edited without restarting a mental model of Python string escaping.
    """
    page = Path(__file__).parent / "static" / "index.html"
    try:
        return HTMLResponse(page.read_text(encoding="utf-8"))
    except OSError:
        raise HTTPException(500, f"Dashboard file missing at {page}") from None
