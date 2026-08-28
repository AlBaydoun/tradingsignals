# SignalForge

An adaptive, self-learning market signal engine that produces risk-sized trade
signals you can execute by hand on the MetaTrader 5 mobile app.

It reads price history across multiple timeframes, classifies the market
regime, tracks the economic calendar and news flow, detects markets that are
exploding or coiling, and emits ranked signals with an entry, a stop, a target
ladder and a lot size. It learns continuously from its own results and retires
its own models when they stop working.

**It also tells you, frequently and bluntly, when there is no trade worth
taking.** That is the feature that makes the rest of it worth having.

---

## The one thing to understand first

Most published trading systems are wrong in the same way: they measure
themselves on data they were trained on, report a spectacular win rate, and
lose money in production. This engine is built specifically to avoid that, and
much of its complexity exists for that reason alone.

Here is the same model measured both ways — BTCUSDT H4, identical code, run
during development of this repository:

| Measured on | Win rate | Profit factor | Return |
|---|---|---|---|
| Its own training data | 83.3% | 5.63 | +11,151% |
| Genuinely out-of-sample | 43.9% | 0.93 | −13.4% |

Same model. Same period. The first number is what you get by backtesting a
model's predictions on the data it was fitted to; the second is the truth. The
engine only ever reports the second, and `walk_forward_backtest()` is the only
backtest entry point that is easy to call.

---

## What it actually produces

```
*** BUY BTCUSD [H4]

Entry      63248.10
Stop loss  62105.40   (1143 pips)
Take profit 1  64390.80
Take profit 2  65533.50
Take profit 3  66676.20
Lot size   0.04
Risk       50.00 (0.50% of account)
R:R        1:1.0

Historical accuracy at this confidence: 56% (measured out-of-sample)
Expectancy: +0.12R per trade
Model confidence: 61%

Valid for 180 more minutes

Why: Buy setup on BTCUSDT H4 in a market that is weak uptrend, normal
  volatility. Drivers: directional pressure is upward; price is 1.2 ATR
  above its 21-period mean. Higher timeframes support the direction.
  Signals at this confidence have historically resolved correctly 56% of
  the time out-of-sample, against a 50% break-even requirement at 1:1.0.

Market state: weak uptrend, normal volatility
Stop placed: below recent swing low

Warnings:
  ! Expected move is only 2.8x the round-trip cost. A wider-than-usual
    spread erases this trade.
```

And when nothing qualifies — which is most of the time:

```
TRADE SIGNALS (0)
----------------------------------------------------------------

  No signals clear the quality bar right now.
  Not trading is a position. This is the engine working,
  not the engine failing.
```

---

## Quick start

```bash
git clone https://github.com/AlBaydoun/tradingsignals.git
cd tradingsignals

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Check data providers, calendar and news are reachable
python -m signalforge.cli doctor

# 2. Train models (start small — each one takes 1-3 minutes)
python -m signalforge.cli train --symbols BTCUSDT USDJPY --timeframes H1 H4

# 3. Generate signals
python -m signalforge.cli signals
```

No API keys are required. Price data comes from Binance (crypto) and Yahoo
Finance (forex, metals, indices); the economic calendar and news feeds are
public. The optional Claude reasoning layer needs `ANTHROPIC_API_KEY`, and the
engine runs identically without it.

**New here?** [`docs/QUICKSTART.md`](docs/QUICKSTART.md) is the seven-step
version, including where to run the thing and how to read the output on a
phone.

### Before risking money, change two settings

In `config/config.yaml`:

```yaml
risk:
  account_balance: 10000.0     # your real balance

# Your broker's exact Market Watch names. Real accounts mix suffixes freely,
# so each symbol is named individually rather than sharing one global suffix.
instruments:
  XAUUSD:  {mt5_symbol: "XAUUSD"}
  NAS100:  {mt5_symbol: "US100.std"}
  USOIL:   {mt5_symbol: "WTI.m"}
  BTCUSDT: {mt5_symbol: "BTCUSD"}
```

`doctor` prints the resulting mapping in full so you can check every line
against Market Watch. Broker names also work as input everywhere — `backtest
US100.std H1` and `backtest NAS100 H1` are the same command.

The other number worth your attention is `typical_spread_pips`. The defaults
are conservative retail guesses; every backtest result depends on them, and
replacing them with your broker's measured spreads is the single highest-value
change you can make.

---

## Commands

| Command | What it does |
|---|---|
| `doctor` | Data, models, calendar, news, symbol mapping and affordability |
| `hunt` | Rank the whole universe by movement per unit of cost, before training |
| `train` | Fit and walk-forward validate models |
| `signals` | Generate signals now (`--json`, `--compact`, `--brief`) |
| `backtest SYMBOL TF` | Honest out-of-sample backtest (`--detail` for regime/hour breakdown) |
| `scan` | Find markets moving abnormally (`--crypto-wide` scans all Binance pairs) |
| `learn` | Resolve open signals, police failing models (`--retrain`) |
| `journal` | Live results versus what the models promised |
| `watch` | Run continuously (`--interval 300`) |

### Finding what is worth trading

`hunt` answers *which market* before any model exists. It surveys every
instrument the engine can price and ranks on movement relative to each
instrument's own history, **divided by what that instrument costs to trade**:

```
   SYMBOL     MT5           SCORE    ATR%   COST  VOL%ILE  EXPAND   EFF   20-BAR
--------------------------------------------------------------------------------
   XAGUSD     XAGUSD         71.3   0.79%    9.3      82%    1.14  0.35   +3.58%
     active and affordable — worth training
   SOLUSDT    SOLUSD         58.1   1.35%    9.0      71%    1.14  0.12   +1.54%
     - movement is choppy (efficiency 0.12) — range, not trend
```

Cost is a multiplier on the score rather than one term among several, because
no amount of volatility rescues an instrument whose spread is the size of its
range. Anything below 1.5 round trips per ATR is dropped outright.

Volatility is always measured against the instrument's *own* distribution.
Gold moving 1% and EURUSD moving 1% are not the same event, so nothing is
compared cross-sectionally.

**`hunt` says nothing about direction** — a high score means the market is
worth studying, not that it is predictable. `train` and `backtest` answer that,
and usually the answer is no.

An HTTP API is included for feeding phones and bots:

```bash
uvicorn signalforge.api.server:app --host 127.0.0.1 --port 8000
# /signals  /signals/text  /rankings  /scan  /journal  /models  /dashboard
```

`/dashboard` is a self-contained mobile-readable page. There is no
authentication — bind it to localhost or put it behind a proxy.

---

## How it works

```
  Binance / Yahoo ─┐
  ForexFactory    ─┼─→ data ─→ features ─→ regime ─→ model ─→ ranking
  RSS news        ─┘            (151)      (GMM)    (LightGBM)  (cost-aware)
                                                                    │
                          journal ←─ signal ←─ sizing ←─ levels ←────┘
                             │                              (ATR + structure)
                             └─→ drift detection ─→ retrain / disable
```

**Data.** Binance klines for crypto (keyless), Yahoo chart API for FX, metals,
energy and index futures. Everything is normalised so a bar timestamped `t`
contains only information known at the close of `t`, cached on disk, and
degraded to stale data rather than failing when a provider throttles.

**Features.** 151 per bar: trend (ADX, Supertrend, Ichimoku, Parabolic SAR,
efficiency ratio, Hurst), momentum (RSI, MACD, Stochastic, CCI, TRIX,
Ultimate Oscillator), volatility (ATR percentile, Parkinson, Garman-Klass,
Rogers-Satchell, Bollinger width, squeeze duration), volume (OBV, MFI, CMF,
VWAP distance, volume z-score), microstructure (candle anatomy, wick skew,
order-flow proxy, Corwin-Schultz spread estimate) and session/clock features.
Higher timeframes are joined with `merge_asof`, never a reindex-and-fill.

**Regime.** A rule-based classifier (volatility state × trend state) plus an
unsupervised Gaussian mixture. Both feed the model and gate which strategy
family is allowed to fire.

**Labels.** Triple-barrier: for each bar, does price hit the profit barrier,
the stop barrier or the time barrier first? Barriers are ATR multiples, so a
label means the same thing on a quiet EURUSD morning and a violent BTC candle.
Bars whose barrier is narrower than the spread are excluded outright.

**Model.** LightGBM 3-class classifier, isotonic-calibrated on walk-forward
predictions, with sample weights that down-weight overlapping labels.

**Validation.** Purged walk-forward with embargo. Folds move strictly forward;
training samples whose label windows overlap the test set are removed; a
further embargo drops the bars immediately after each test window.

**Risk.** Stops are the wider of an ATR stop and a structure stop beyond the
last confirmed swing, never inside three spreads of entry. Lots are computed
from real pip-value arithmetic — including the cross-currency case, which is
flagged when it has to be approximated rather than silently sized wrong.

**Conditional edge.** After training, each model is backtested and its
performance measured *per regime and per session*. The map is stored with the
model and consulted at signal time: if this model has historically lost money
in the conditions holding right now, the signal is vetoed however confident the
model is. A condition needs 25+ past trades before it may veto anything.

**Learning.** Every signal is journalled. The loop resolves open signals
against subsequent price, compares live results against the model's own
promises, and disables models whose live hit rate falls below the floor.
Feature drift (PSI) triggers retraining; live losses trigger disabling.

---

## What it reports honestly, by design

**Accuracy is a measured frequency, not a model opinion.** The number on a
signal is the realised out-of-sample hit rate of past signals in that
confidence band, read from a reliability table. If the model says 70% and the
table says 52%, you are shown 52%. If a confidence band has fewer than 20
observations, you are told there is no track record rather than given a
number.

**Accuracy comes with error bars.** Triple-barrier labels overlap, so 600 rows
is nowhere near 600 independent observations. The engine divides by the mean
label span and reports a Wilson confidence interval on the effective sample:

```
BTCUSDT H4: accuracy 0.5638 [95% CI 0.515-0.612] eff.n 399
XAUUSD  H4: accuracy 0.6000 [95% CI 0.496-0.692] eff.n 92   <- edge not significant
```

The 60% model is *worse evidence* than the 56% one. A model whose interval
includes 0.5 has not demonstrated an edge, whatever its point estimate says.

**A backtest that looks too good is treated as a bug report.** During
development XAUUSD H4 reported a 72.3% win rate and a profit factor of 3.24 on
a correctly purged walk-forward run. The model and the validation were both
fine; the backtester was anchoring stops to the signal bar's close while
entering at the next bar's open. A 66-point gap in gold left a 3.4-point stop
where 69 was intended, sizing bought 14x the normal lot to keep the dollar risk
constant, and the resulting 31R trade was 48% of all profit in the run.

Re-anchored to the actual fill, the same model returns **profit factor 1.06
over 13 trades** — graded unproven. Every trade now carries `entry_gap_atr`,
and [`docs/HONEST_LIMITATIONS.md`](docs/HONEST_LIMITATIONS.md) §10 walks
through the whole thing, because the interesting part is that no number looked
wrong at any point.

**Costs are charged everywhere.** Spread, two-sided slippage and commission on
every simulated trade. Extra slippage on stops, because gaps go through them.
When one bar spans both the stop and the target, the stop is assumed to have
been hit first — OHLC data cannot say which came first, and guessing favourably
is how backtests learn to lie.

**Nothing is graded STRONG without a track record.** A signal in an unproven
confidence band is capped at WEAK no matter how certain the model is.

**Training a batch of models is priced in.** Fitting 12 models means roughly
one will look significant by chance alone. Every training run applies a
Benjamini-Hochberg correction across the whole batch and reports which models
were *demoted* — significant on their own, not significant as one of twelve
attempts:

```
Tested 12 models. 2 clear the naive 5% bar; 0 survive Benjamini-Hochberg.

Demoted by the correction (looked good alone, not in a batch):
  XAUUSD/H4   0.620  p=0.0110 -> q=0.0661
  USDJPY/H4   0.638  p=0.0053 -> q=0.0633
```

A 63.8% model is exactly what twelve coin-flip attempts produce. Without this
correction it would have been reported as an edge.

---

## Realistic expectations

From development runs in this repository, on real market data:

- Out-of-sample directional accuracy lands between **51% and 64%**.
- Across a 12-model run, **none survived multiple-comparison correction**. Two
  looked significant individually and were demoted once the size of the search
  was priced in.
- A backtest of the best model returned **profit factor 1.07** after costs —
  the engine's own verdict was *"Marginal. The edge is inside the error bars."*
- **Every instrument lost money in range-bound markets** (profit factors 0.59
  to 0.89), and the edge concentrated in trends. That pattern held across
  crypto, forex and index futures.

That last finding is the sort of thing this engine exists to produce, and it is
now acted on automatically rather than merely reported. On EURUSD H1 the
blended backtest says *"profit factor 0.83 — do not trade"*, but strong
downtrends alone run at 1.12. Gating on regime turns a losing model into a
narrow usable one; a single blended number would have thrown it away.

**Expect the honest answer to usually be "no trade."** A run that produces
nothing is the common case, not a malfunction.

**M1 and M5 almost never survive the cost filter.** At a 0.8-pip EURUSD spread
against a 1.5-pip M1 ATR, the round trip consumes most of the expected move.
The engine will tell you this rather than emit signals anyway.

---

## Limitations you should read before trusting it

See [`docs/HONEST_LIMITATIONS.md`](docs/HONEST_LIMITATIONS.md) for the full
list. The short version:

- Data is free retail data. Yahoo FX prices are indicative, not your broker's.
- Backtests assume your configured spread. Real spreads widen exactly when you
  most want to trade.
- Two years of history is a handful of regimes, not a representative sample.
- The engine cannot execute trades. It is deliberately advisory.
- Markets are close to efficient. A 55% edge is a *good* result, not a
  disappointing one, and it is fragile.

---

## Testing

```bash
pytest tests/ -q     # 131 tests
```

The suite is weighted toward the properties that matter: every indicator is
checked for causality by recomputing on truncated data, purged walk-forward is
checked for leakage, the model is checked for near-50% accuracy on pure noise,
and position sizing is checked to never round *up* into more risk than
configured.

---

## Licence

MIT. Provided as-is, with no warranty of any kind.

Trading leveraged instruments carries substantial risk of loss. Nothing this
software produces is financial advice. Test on a demo account for a meaningful
period before considering real money, and never risk more than you can afford
to lose entirely.
