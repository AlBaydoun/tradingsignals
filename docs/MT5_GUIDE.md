# Using SignalForge with MetaTrader 5 on your phone

The engine produces signals; you place the orders. This is how to do that
without fat-fingering the numbers.

---

## One-time setup

### 1. Find your broker's exact symbol names

Open MT5 → **Quotes** → tap **+** (top right) → browse or search.

Brokers rename things. The instrument this engine calls `BTCUSDT` might be
`BTCUSD`, `BTCUSD.m`, `Bitcoin`, or `BTCUSD_ecn`. Note the suffix your broker
uses and put it in `config/config.yaml`:

```yaml
mt5_symbol_suffix: ".m"     # or "_ecn", ".pro", "#", or "" for none
```

The `mt5_symbol` field printed on every signal is then exactly what you type
into the search box.

### 2. Set your real account balance

```yaml
risk:
  account_balance: 10000.0
  account_currency: USD
  risk_percent_per_trade: 0.5
```

Lot sizes are computed from these. A stale balance means every position is
mis-sized. Update it as your balance changes, or set it from the environment:

```bash
export SIGNALFORGE_ACCOUNT_BALANCE=8500
```

### 3. Verify one lot size by hand before trusting the rest

Take the first signal and check it:

```
Entry 1.08500, Stop 1.08000 → 50 pips
0.5% of $10,000 = $50 risk
EURUSD pip value = $10 per lot
$50 / (50 pips × $10) = 0.10 lots
```

If the signal says 0.10 lots, the arithmetic chain is working. If it does not,
stop and work out why before placing anything.

---

## Placing a signal

Given:

```
*** BUY EURUSD [H1]

Entry      1.08500
Stop loss  1.08000   (50 pips)
Take profit 1  1.09000
Take profit 2  1.09500
Take profit 3  1.10000
Lot size   0.10
```

In MT5 mobile:

1. **Quotes** → tap the symbol → **New Order**
2. Set **Volume** to `0.10`
3. Set **Stop Loss** to `1.08000`
4. Set **Take Profit** to `1.09000` (the first target)
5. Tap **BUY BY MARKET**

**Set the stop loss in the same action as the entry.** Not afterwards. The
gap between "I'll add the stop in a second" and an adverse spike is where
accounts die.

### Working the target ladder

The three targets exist so you can scale out. The simplest version that works:

- Enter full size with TP at target 1.
- When target 1 fills, you are done. Flat, profitable, no decisions.

If you want to run winners, place **three separate orders** at one third size
each, with TP1, TP2 and TP3 respectively, and move the stops on the remaining
two to breakeven once TP1 fills. MT5 mobile has no partial-close automation, so
three orders is the practical way to do it.

### Validity windows

Every signal shows `Valid for N more minutes`. After that the conditions that
produced it have likely changed — the entry price has moved, or the bar that
triggered it has closed. **Do not place an expired signal at a worse price.**
Wait for the next run.

---

## Reading the signal correctly

### Signal grades

| Grade | Meaning |
|---|---|
| `***` STRONG | Measured accuracy comfortably clears break-even at this R:R |
| `**` MODERATE | Clears break-even with a smaller margin |
| `*` WEAK | Marginal, or the confidence band has no measured track record |
| `-` WATCH_ONLY | Not tradable as it stands — shown for context only |

Never trade a `WATCH_ONLY`. It is displayed because the market is interesting,
not because the trade is good.

### The accuracy line is the important one

```
Historical accuracy at this confidence: 56% (measured out-of-sample)
Expectancy: +0.12R per trade
Model confidence: 61%
```

The **56%** is what past signals in this confidence band actually achieved on
data the model had never seen. The **61%** is the model's opinion of itself.
When they differ, believe the 56%.

If you see this instead:

```
Historical accuracy: not enough past signals at this confidence
to quote a number. Treat as unproven.
```

then the model is extrapolating. Size down or skip.

### Warnings are not decoration

```
Warnings:
  ! Expected move is only 2.8x the round-trip cost.
  ! This trades against the higher-timeframe trend.
```

Each of these measurably lowers the probability of the trade working. Two or
more, and the sensible response is usually to pass.

---

## Running it

### On demand

```bash
python -m signalforge.cli signals --compact
```

`--compact` gives one line per signal, sized for a phone screen:

```
EURUSD | BUY | Vol 0.10 | SL 1.08000 | TP 1.09000 / 1.09500 / 1.10000
```

### Continuously

```bash
python -m signalforge.cli watch --interval 300
```

Regenerates every five minutes and resolves open signals as they close.

### From your phone, over the network

Run the API on a machine that stays on:

```bash
uvicorn signalforge.api.server:app --host 127.0.0.1 --port 8000
```

Then reach `http://<host>:8000/dashboard` from your phone browser. There is no
authentication — use a VPN or an SSH tunnel, do not expose it to the internet.

---

## A sane operating routine

**Daily**

```bash
python -m signalforge.cli learn      # resolve yesterday's signals
python -m signalforge.cli signals    # today's candidates
```

**Weekly**

```bash
python -m signalforge.cli learn --retrain   # refit on fresh data
python -m signalforge.cli journal           # live vs promised
```

The `journal` output is the one that matters over time:

```
delivered_minus_promised   -0.03
```

Within a few points of zero means the models are calibrated. Persistently
negative by more than 10 points means reduce size and investigate — the models
are promising more than they deliver.

---

## Before real money

1. **Demo account, 100+ trades minimum.** Fewer than that tells you nothing.
2. **Compare journal results against the backtest.** A large gap means the cost
   assumptions are wrong for your broker.
3. **Start at 0.25% risk**, not 0.5%, and raise only after a positive live
   record.
4. **Re-measure your spread.** Sit with MT5 open during the hours you actually
   trade, note the real spread, and put that number in
   `signalforge/universe.py`. Every backtest depends on it.

---

## When the engine says nothing

```
No signals clear the quality bar right now.
Not trading is a position.
```

This is the normal state. On a system with a genuine 55% edge and strict cost
filtering, most hours produce nothing worth trading. The instinct to lower
thresholds until signals appear is the single most reliable way to lose money
with this tool.

If you want more signals, the honest levers are: add instruments to the
watchlist, or accept a longer timeframe where costs matter less. Not lowering
`min_confidence`.
