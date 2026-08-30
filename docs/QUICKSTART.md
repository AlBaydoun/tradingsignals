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

### Windows

Open **Command Prompt** (press Start, type `cmd`, hit Enter) and run:

```bat
cd %USERPROFILE%\Documents
git clone https://github.com/AlBaydoun/tradingsignals.git
cd tradingsignals
```

Then **double-click `setup.bat`** in that folder. It creates the environment,
installs everything, and runs the health check. Nothing else to type.

After that you have five clickable files:

| Double-click | What it does |
|---|---|
| `setup.bat` | One-time install (run once) |
| `check.bat` | Verify data, symbol names and affordability |
| `hunt.bat` | Rank every market by movement per unit of cost |
| `train.bat` | Fit the models (~30 min, weekly) |
| `signals.bat` | Get signals right now |
| `watch.bat` | Run continuously until you close the window |
| `dashboard.bat` | Open the visual dashboard, and serve it to your phone |

If you prefer to type it yourself, the Windows commands are:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m signalforge.cli doctor
```

Note `.venv\Scripts\activate` — **not** `source .venv/bin/activate`. `source`
is a Mac/Linux command and Windows will tell you it is "not recognized".

### Mac / Linux

```bash
git clone https://github.com/AlBaydoun/tradingsignals.git
cd tradingsignals

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Do I need Visual Studio or VS Code?

**No.** Command Prompt is enough, and the `.bat` files mean you barely need
that. VS Code is only worth installing if you want a nicer editor for
`config/config.yaml` — Notepad opens it perfectly well.

### Python version

Anything from **3.10 upward**, including 3.14. Every dependency has a Windows
wheel. If `python` is not recognised at all, install it from
[python.org/downloads](https://www.python.org/downloads/) and **tick "Add
python.exe to PATH"** on the first screen of the installer — that checkbox is
the single most common cause of trouble.

Every command below assumes you are in the `tradingsignals` folder with the
environment active. Opening a new terminal later? Run `cd tradingsignals` then
`.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac).

---

## Step 2 — Check it can see the markets

```bat
REM Windows: double-click check.bat, or type
.venv\Scripts\python -m signalforge.cli doctor
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

```bat
REM Windows: double-click hunt.bat, or type
.venv\Scripts\python -m signalforge.cli hunt --timeframe H1
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

```bat
REM Windows: double-click train.bat, then signals.bat. Or type:
.venv\Scripts\python -m signalforge.cli train --timeframes H1 H4
.venv\Scripts\python -m signalforge.cli signals
```

```bash
# Mac / Linux
python -m signalforge.cli train --timeframes H1 H4
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

## Step 5b — Watching everything, not just your seven

`watch.bat` does two jobs at once.

Every cycle it generates **signals** for your seven trained instruments. Every
fourth cycle it **sweeps all 31 instruments** the engine can price, plus the
market-wide Binance movers, and reports anything unusual:

```
  Swept 31 instruments (10 open) on H1. 3 worth a look: XRPUSDT, SOLUSDT, BTCUSDT
    XRPUSDT     35.2   +0.61%  but the movement is chop, not travel
               no model — this is a market to look at, never a trade to take

  Market-wide crypto movers (24h, beyond the universe):
    ZKCUSDT       +51.15%  vol $7M
    PROMUSDT      +43.45%  vol $28M
```

**The sweep never produces a trade.** It cannot: those instruments have no
trained model, so the engine has no measured edge on them and says so on every
line. It tells you where to look. Whether to add one to your watchlist and
train it is your call — and the more you train, the harsher the statistical
correction becomes for all of them.

Turn it off with `--no-sweep`, or edit `market_watch:` in `config/config.yaml`
to narrow it to particular markets.

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

**A $5 VPS** (Hostinger, Hetzner, Contabo, DigitalOcean — any of them) is
enough. Pick Ubuntu 22.04 or newer, SSH in, and run one command:

```bash
curl -fsSL https://raw.githubusercontent.com/AlBaydoun/tradingsignals/main/deploy/vps-setup.sh | bash
```

That installs everything and registers systemd services so it starts at boot
and restarts itself if it crashes. Full detail, including how to read the
dashboard safely over an SSH tunnel, is in [`HOSTING.md`](HOSTING.md).

Note: this needs Hostinger's **VPS** product, not their shared web hosting.
Shared hosting cannot run a long-lived process or install LightGBM.

---

## Step 7 — The dashboard

**Windows: double-click `dashboard.bat`.** Your browser opens at the dashboard,
and the terminal prints the addresses to type into your phone.

```bash
# Mac / Linux, or if you prefer typing
python -m signalforge.cli dashboard --host 0.0.0.0
```

Five tabs:

| Tab | What it is for |
|---|---|
| **Signals** | The trades. Every value is tap-to-copy for MetaTrader |
| **Hunt** | Which markets are moving enough to be worth their cost |
| **Models** | Accuracy with error bars — read the bar, not the number |
| **Journal** | What the models promised against what actually happened |
| **Setup** | Your MT5 symbol names and what your account can afford |

The Signals tab is built for the job you actually do: read a number, type it
into MT5, don't mistype it. **Tap any row and it copies.** Mistyping a stop is
how a 0.5% risk becomes 5%.

### Reading it on your phone

Both machines on the same Wi-Fi. The terminal prints something like:

```
On your phone, on the same Wi-Fi, open:
  http://192.168.1.42:8000/dashboard
```

Type that into your phone's browser. Add it to your home screen and it behaves
like an app.

**There is no password on this page.** `--host 0.0.0.0` exposes it to everyone
on your network, which is fine at home and not fine on public Wi-Fi. On a VPS,
bind to `127.0.0.1` and reach it through an SSH tunnel, or put it behind a
proxy that asks for a login. Never expose it to the open internet.

Prefer not to run a server at all? `signals.bat` prints the same trades as
text.

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
| `No module named signalforge` | Clone is empty, or you are in the wrong folder | `dir` should show `requirements.txt`; if not, re-clone |
| `No such file: requirements.txt` | Same cause as above | Same fix |
| `'source' is not recognized` | Mac command on Windows | Use `.venv\Scripts\activate` |
| `'python' is not recognized` | Python not on PATH | Reinstall, tick "Add python.exe to PATH" |
| `pip` says "Defaulting to user installation" | You are outside the venv | Activate it, or use the `.bat` files |
| `Unknown instrument 'XYZ'` | Not in the universe | `hunt --all` lists everything it knows |
| Signal names a symbol you can't find in MT5 | Broker spells it differently | Step 2 — fix `instruments:` in config |
| `doctor` shows 0 bars for a symbol | Provider throttling | Wait a few minutes and rerun |
| Every `hunt` row shows COST below 1.5 | Timeframe too fast for your spreads | Try `--timeframe H4` |
| `signals` returns nothing, repeatedly | Normal | It is meant to. See Step 5 |
| Lot sizes look wrong | Stale balance, or broker contract size differs | Update `account_balance`; check `contract_size` |
