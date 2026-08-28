# Honest limitations

Read this before you risk money. It is not boilerplate; every item below is a
concrete way this system can be wrong, and several are things it cannot fix.

---

## 1. The edge is small, and it may not exist

Out-of-sample directional accuracy in development ran between 51% and 64%. In a
12-model run, two cleared a naive significance test and **neither survived
correction for the number of models tested**. The best backtest returned a
profit factor of roughly 1.07 after costs.

A profit factor of 1.07 means that for every $100 lost, $107 is made. That is a
real edge if it holds, but it is thin enough that a broker with slightly worse
fills than you assumed erases it entirely.

Be clear-eyed about what "no model survived correction" means: it does not
prove there is no edge, but it does mean **this system, on this data, has not
demonstrated one**. Anything you trade on that basis is a bet on a hypothesis,
not on a measured result.

**If a version of this system ever shows you a 70% win rate and a profit factor
of 3, something is broken.** Check whether you are backtesting
`model.predict_signal()` output instead of `model.oos_signals()`.

---

## 2. Free data is not your broker's data

- **Yahoo FX prices are indicative.** They are not the bid/ask your broker
  fills you at. Small differences move a stop that sits 15 pips away.
- **Yahoo index data is futures** (`ES=F`, `NQ=F`), not your broker's CFD.
  They track closely, but not exactly, and they gap at different times.
- **Binance spot is not a crypto CFD.** Binance spreads are far tighter than
  any CFD broker will give you. The engine floors the measured spread at your
  configured estimate for exactly this reason, but the price series still
  differs.
- **Yahoo throttles aggressively.** The engine retries and degrades to cached
  data, but during a throttling episode you may be looking at bars that are
  minutes old.

**Consequence:** your live fills will differ from the backtest. Configure
`typical_spread_pips` in `signalforge/universe.py` with your broker's real
average spread, measured during the hours you actually trade, before believing
any backtest number.

---

## 3. Backtests are optimistic even when done correctly

This engine does the standard things right — purged walk-forward validation,
costs on every trade, stop-before-target on ambiguous bars, entry at the next
bar's open. It still cannot model:

- **Spread widening.** Spreads triple around news and at the daily rollover.
  The backtest uses one fixed number.
- **Slippage on gaps.** A weekend gap through your stop fills wherever the
  market opens. The engine applies a fixed 1.5x stop-slippage multiplier, which
  is a guess.
- **Requotes and rejections.** Not modelled at all.
- **Your own behaviour.** The backtest takes every signal without hesitation,
  never moves a stop, and never doubles down after a loss. You are unlikely to
  match it.
- **Swap and financing.** Overnight positions on leveraged instruments accrue
  swap. Not modelled.

---

## 4. The sample is smaller than it looks

Two problems compound here:

**Label overlap.** Triple-barrier labels resolve over several bars, so
consecutive rows describe overlapping slices of the same market move. 600 rows
with a 6-bar mean span is closer to 100 independent observations. The engine
corrects for this — it reports `eff.n` and a Wilson confidence interval — but
the correction is an approximation, not a proof.

**Limited history.** Free intraday data reaches back around two years for FX
and indices. That covers a handful of regimes, not a representative sample of
market conditions. A model trained on it has never seen a 2008, a 2020 March,
or a sustained high-rate environment.

**Consequence:** a model that looks good may simply have been lucky in the
particular two years it saw.

---

## 5. Multiple-comparison bias

Training 10 symbols across 4 timeframes fits 40 models. Some will look good by
chance alone — with 40 tries at the 95% level, you expect around two false
positives even if every model is worthless.

The engine now corrects for this. Every training run applies a
Benjamini-Hochberg FDR correction across the whole batch and reports which
models were demoted. From a real 12-model run:

```
Tested 12 models. 2 clear the naive 5% bar; 0 survive Benjamini-Hochberg.

Demoted:  XAUUSD/H4  0.620  p=0.0110 -> q=0.0661
          USDJPY/H4  0.638  p=0.0053 -> q=0.0633
```

Both would have been reported as edges without it.

**What this still does not cover:** the correction only sees the models in *one
run*. If you train a batch, tweak the features, and train again, you are
searching across runs and nothing is counting those attempts. Every time you
adjust a parameter and retrain, you are drawing another sample. The honest
defence is to decide which instruments and settings you are testing before you
look at results, and to treat a model discovered on the fifth retune with far
more suspicion than one found on the first.

---

## 6. The conditional edge map can itself be overfitted

Training measures each model's performance per regime and per session, then
blocks the losing conditions live. This is a genuine improvement — but it is
another way of slicing the same backtest, and slicing finely enough will always
produce a flattering subset.

The guards are a 25-trade minimum per condition before it may veto anything,
and only six session buckets rather than 24 hourly ones. Neither makes the map
*proof*; a regime showing profit factor 1.12 over 40 trades is weak evidence.

Treat the map as a filter that removes obviously bad conditions, not as a
discovery that a narrow slice is reliably profitable. Set
`enforce_conditional_edge: false` in the config to trade the blend instead and
compare — if the gated version does not beat the blend in your live journal,
the map was fitting noise.

---

## 7. Regime change breaks everything

The model learns relationships that held in its training window. When the
market's character changes — a new rate regime, a structural shift in crypto
correlation, a volatility regime break — those relationships can stop working
overnight, and nothing in the model will complain.

The learning loop is the defence: it tracks live results per model and disables
one whose hit rate falls below the floor or that strings together too many
losses. But it is inherently *reactive*. It finds out after the losses, not
before.

---

## 8. News and sentiment are weak signals

The sentiment model is a hand-built lexicon, not a language model. It:

- Misreads context. "Dollar strengthens" is bullish generically and bearish for
  gold; the lexicon does not know which instrument it is scoring.
- Cannot read the article, only the headline.
- Sees news *after* the market has. By the time a story reaches a public RSS
  feed, the move has usually happened.

Sentiment is used here as a tiebreaker and a risk flag. It never triggers a
trade, and it should not.

The economic calendar is more useful — knowing that CPI lands in ten minutes is
genuinely actionable — but it only covers *scheduled* events. Unscheduled
shocks arrive without warning.

---

## 9. What the anomaly detector can and cannot tell you

**Ignition** (a market moving abnormally right now) is detectable from volume,
velocity and range expansion. That part works.

**Coiling** (compression that precedes a violent move) tells you a break is
coming. It does **not** tell you which way. Any system claiming to predict
breakout direction from a squeeze is fitting noise. The engine reports coiling
as a reason to watch and to size down, never as a direction.

---

## 10. What `hunt` can and cannot tell you

`hunt` ranks markets by how much they are moving relative to their own history,
divided by what they cost to trade. Both halves are measurements, and both are
useful. Neither is a prediction.

**Volatility is a necessary condition for profit, never a sufficient one.** A
market at the 95th percentile of its own ATR range is a market where a correct
call pays and an incorrect one costs, in equal measure. The score says the
first is possible; it says nothing about which you will get.

The specific trap: an instrument scoring 70 on `hunt` and failing every
significance test in `train` is a completely coherent result, and a common one.
Movement you cannot predict is not opportunity, it is variance.

The cost half is more solid, because a spread is a fact rather than an
estimate — but the spread the engine uses is the one in your config, and the
defaults are guesses. A `COST` column computed from a wrong spread is
confidently wrong.

---

## 11. Broker symbols are yours to verify

The engine cannot see your Market Watch. It prints the symbol names from your
config and assumes they are right. If your broker's Nasdaq is `US100.cash` and
you configured `US100.std`, nothing in the engine will notice — you will simply
get signals for a symbol you cannot find.

Worse, and quieter: `contract_size` varies between brokers on exactly the
instruments where it matters most. Index CFDs in particular range from 1 to 50
units per lot depending on the broker. A wrong contract size produces a lot
size that risks a multiple of what you configured, and nothing will flag it.
Check one trade's margin requirement against the engine's stated risk before
trusting the sizing.

---

## 12. The engine cannot trade for you

There is deliberately no broker integration and no auto-execution. It produces
signals; you decide and execute. This is a design choice: an unattended system
with a subtle bug can lose money faster than you can notice.

That also means every signal depends on you reading it correctly, typing the
numbers correctly, and being awake.

---

## 13. Things that will silently mislead you

| Trap | Symptom | Fix |
|---|---|---|
| Backtesting `predict_signal()` | Win rate above 75% | Use `walk_forward_backtest()` |
| Wrong `mt5_symbol` override | Signals reference symbols you cannot find | Check `doctor`'s mapping table |
| Wrong `contract_size` | Lot sizes risk a multiple of what you set | Compare margin against stated risk |
| Trading a high `hunt` score | Volatility is not predictability | Only trade what `train` validates |
| Stale `account_balance` | Lot sizes risk the wrong amount | Update `config.yaml` |
| Trading a `WATCH_ONLY` signal | It was graded unprofitable at that R:R | Only trade actionable grades |
| Ignoring `eff.n` | Trusting a 60% model built on 90 observations | Prefer tight intervals over high point estimates |
| Reading accuracy, ignoring `q` | A demoted model looks like a winner | Trade only what survives the correction |
| Retuning until something passes | Searching across runs, uncounted | Pre-commit to settings before looking |
| `enforce_conditional_edge: false` | Trading regimes the model loses in | Leave the gate on unless comparing |

---

## 14. What would make this genuinely better

If you want to take this further, in rough order of expected value:

1. **Your broker's own tick data.** Replaces every assumption in the cost model
   with a measurement.
2. **Longer history.** Paid intraday data going back 10+ years would let the
   model see more than a couple of regimes.
3. **Forward testing on a demo account** for at least 100 trades before risking
   real money. The journal exists for this.
4. **Order book / depth data**, which would replace the microstructure proxies
   with the real thing.
5. **Correcting for multiple comparisons** when selecting which models to trade.

---

## The short version

This is a carefully built research tool that is honest about small edges. It is
not a money printer, and the parts of it that look most like a money printer
are the parts most likely to be a bug.

Markets are close to efficient. A genuine 55% edge, held with discipline over
hundreds of trades and modest position sizes, is a good outcome. Anything that
looks dramatically better than that deserves suspicion before celebration.
