# Quick start — where to run it, and how

Seven steps to your first signal. Nothing here needs an API key, a paid data
feed, or a credit card.

Your instruments are already configured:

| Engine name | Your MT5 symbol | Market |
|---|---|---|
| `XAUUSD` | `XAUUSD` | Gold |
| `XAGUSD` | `XAGUSD` | Silver |
| `BTCUSDT` | `BTCUSD` | Bitcoin |
| `NAS100` | `US100.std` | Nasdaq 100 |
| `US30` | `US30.std` | Dow Jones |
| `USOIL` | `WTI.m` | WTI crude |
| `BRENT` | `BRENT.m` | Brent crude |

You can type either name anywhere the engine asks for a symbol. `hunt --symbols
US100.std` and `hunt --symbols NAS100` do the same thing.

---

## Step 0 — Decide where it runs

The engine is a Python program. It needs to run somewhere that stays on, and
you read the results on your phone. **It does not run on the phone itself, and
it does not connect to your broker.** It tells you what to type; you type it.

Pick one:

| Where | Cost | Good for | The catch |
|---|---|---|---|
| **Your laptop** | Free | Getting started today | Signals stop when the lid closes |
| **A cheap VPS** | ~$5/month | Running 24/7 | 20 minutes of setup |
| **A spare PC / Raspberry Pi at home** | Free | Running 24/7 for nothing | Dies with your power or internet |

**Start on your laptop.** Move to a VPS once you have decided the thing is
worth running continuously — Step 6 covers that, and nothing you set up now is
wasted.

Requirements: Python 3.11 or newer, about 2 GB of disk, and an internet
connection. Windows, macOS and Linux all work.

---

## Step 1 — Install it

```bash
git clone https://github.com/AlBaydoun/tradingsignals.git
cd tradingsignals

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That is the whole installation. Every command below assumes you are in the
`tradingsignals` folder with the virtual environment active — if you open a new
terminal later, run `cd tradingsignals && source .venv/bin/activate` again.

---

## Step 2 — Check it can see the markets

```bash
python -m signalforge.cli doctor
```

You should get bar counts for all seven instruments, a calendar event count,
a headline count, and — importantly — this table:

```
MT5 symbol mapping — check every one of these against Market Watch:
  XAUUSD     -> XAUUSD         (config, spread 2.5 pips = 0.25 price units)
  NAS100     -> US100.std      (config, spread 18 pips = 1.8 price units)
  USOIL      -> WTI.m          (config, spread 3 pips = 0.03 price units)
  ...
```

**Open MT5 on your phone, go to Quotes, tap the `+`, and search for each name
on the right.** If your broker spells one differently — `US100` instead of
`US100.std`, say — fix it in `config/config.yaml`:

```yaml
instruments:
  NAS100:
    mt5_symbol: "US100"     # whatever Market Watch actually says
```

This is the one step people skip and then wonder why a signal names a symbol
they cannot find.

`doctor` also tells you which instruments your account cannot afford:

```
Affordability at 10,000 USD and 0.5% risk (50.00 per trade):
  XAUUSD     H4   needs ~10,655 (smallest lot risks 53.27)
  XAGUSD     H4   needs ~16,382 (smallest lot risks 81.91)
```

Sizing rounds *down* so risk never exceeds what you set, which means an
instrument whose smallest lot risks more than your budget silently never fires.
Gold on H4 at a $10,000 balance is one of those. Now you know why, rather than
concluding the model is broken.

---

## Step 3 — Set your real numbers

Open `config/config.yaml` and change two things:

```yaml
risk:
  account_balance: 10000.0        # your actual balance
  risk_percent_per_trade: 0.5     # leave this alone until you have a track record
```

0.5% means twenty losses in a row costs about 10% of the account. That is the
point. Raising it is the fastest way to turn a small edge into a blown account.

While you are in there, the `typical_spread_pips` values under `instruments:`
are conservative guesses. Once you have watched your own spreads for a day,
put your real ones in. **Every backtest number the engine produces depends on
this**, and it is the difference between an honest result and a flattering one.

---

## Step 4 — Find out what is worth trading right now

```bash
python -m signalforge.cli hunt --timeframe H1
```

This is the "hunt the volatile trades" command. It surveys all 31 instruments
the engine knows — not just your seven — from price data alone, and ranks them
by **how much they are moving relative to their own normal, divided by what
they cost to trade**.

```
   SYMBOL     MT5           SCORE    ATR%   COST  VOL%ILE  EXPAND   EFF   20-BAR
--------------------------------------------------------------------------------
   XAGUSD     XAGUSD         71.3   0.79%    9.3      82%    1.14  0.35   +3.58%
     active and affordable — worth training
   SOLUSDT    SOLUSD         58.1   1.35%    9.0      71%    1.14  0.12   +1.54%
     - movement is choppy (efficiency 0.12) — range, not trend
```

Read it like this:

- **COST** — how many round-trip spreads one ATR pays for. Below 1.5 the
  instrument is dropped entirely: it cannot be traded profitably on that
  timeframe no matter what it does.
- **VOL%ILE / EXPAND** — how volatile it is *for itself*. `EXPAND 1.4` means
  its range is 40% above its own median. Gold moving 1% and EURUSD moving 1%
  are not the same event, so nothing here is compared across instruments.
- **EFF** — 0 to 1. High means the movement is going somewhere; low means it is
  expensive chop.

Useful variations:

```bash
python -m signalforge.cli hunt --timeframe H4          # slower, cheaper timeframe
python -m signalforge.cli hunt --markets metals energy # one sector
python -m signalforge.cli hunt --live-spreads          # measure spreads now
```

**What `hunt` does not tell you is direction.** A high score means the market
is worth studying, not that it is predictable. That is Step 5's job.

---

## Step 5 — Train, then generate signals

```bash
# Takes 1-3 minutes per model. Seven symbols x two timeframes ~ 30 minutes.
python -m signalforge.cli train --timeframes H1 H4

# Then, any time:
python -m signalforge.cli signals
```

Training is where the honesty lives. At the end it prints something like:

```
Tested 14 models. 1 clears the naive 5% bar; 0 survive Benjamini-Hochberg.
```

That means: of fourteen attempts, one looked good, and once you account for
having *made* fourteen attempts, none did. Train fourteen coin flips and one
will look brilliant. This is the engine refusing to sell you that one.

**Expect "no signals" most of the time.** That is the system working:

```
TRADE SIGNALS (0)

  No signals clear the quality bar right now.
  Not trading is a position.
```

To trade one that does appear, on your phone: open MT5 → Quotes → find the
symbol → New Order → set the volume to the lot size shown → set Stop Loss and
Take Profit to the prices shown → place it. Nothing is automated; you are the
execution layer, deliberately.

Retrain about once a week: `python -m signalforge.cli learn --retrain`.

---

## Step 6 — Keep it running (optional, do this later)

Once you want signals while you are asleep, run it continuously:

```bash
python -m signalforge.cli watch --interval 300
```

That checks every five minutes and prints new signals. On a laptop, leave the
terminal open. On a VPS, run it under `screen` or `tmux` so it survives you
disconnecting:

```bash
screen -S forge
python -m signalforge.cli watch --interval 300
# Ctrl-A then D to detach. `screen -r forge` to come back.
```

**A $5 VPS** (Hetzner, DigitalOcean, Contabo — any of them) is enough. Pick
Ubuntu 22.04 or newer, then:

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/AlBaydoun/tradingsignals.git
cd tradingsignals && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

...and carry on from Step 2.

---

## Step 7 — Read it on your phone

The engine ships a small web dashboard:

```bash
uvicorn signalforge.api.server:app --host 0.0.0.0 --port 8000
```

Then open `http://<the machine's IP>:8000/dashboard` on your phone. On a home
network that is your laptop's local IP; on a VPS it is the server's address.

**There is no password on it.** On a VPS, either bind it to `127.0.0.1` and
reach it through an SSH tunnel, or put it behind a proxy with authentication.
Do not leave it open to the internet.

Simplest alternative if you would rather not run a server: just run
`python -m signalforge.cli signals --compact` and read the terminal.

---

## What to do in your first month

1. **Paper trade.** Write down every signal and what it would have done. Do not
   put money on it yet.
2. **Check `journal` weekly** — `python -m signalforge.cli journal` compares
   what the models promised against what actually happened. If live results are
   materially worse than the backtest, believe the live results.
3. **Fix your spreads.** After a week of watching, replace the estimates in
   `config.yaml` with what you actually see, and retrain. Numbers will get
   worse. That is the point.
4. **Then demo trade for 100 trades** before real money.

The honest expectation, spelled out in
[`HONEST_LIMITATIONS.md`](HONEST_LIMITATIONS.md): out-of-sample accuracy in
development ran 51–64%, and in a 12-model batch **none survived correction for
the number of models tested**. A genuine 55% edge held with discipline is a
good outcome. Anything that looks dramatically better than that is a bug, not a
discovery.

---

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown instrument 'XYZ'` | Not in the universe | `hunt --all` lists everything it knows |
| Signal names a symbol you can't find in MT5 | Broker spells it differently | Step 2 — fix `instruments:` in config |
| `doctor` shows 0 bars for a symbol | Provider throttling | Wait a few minutes and rerun |
| Every `hunt` row shows COST below 1.5 | Timeframe too fast for your spreads | Try `--timeframe H4` |
| `signals` returns nothing, repeatedly | Normal | It is meant to. See Step 5 |
| Lot sizes look wrong | Stale balance, or broker contract size differs | Update `account_balance`; check `contract_size` |
