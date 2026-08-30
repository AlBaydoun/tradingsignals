# Hostinger VPS, step by step

Everything from buying the server to reading the dashboard on your phone.
Written for someone who has never used a Linux server. Roughly 45 minutes,
most of it waiting.

**Before you start, be sure you want to.** A VPS is only worth paying for once
you have a track record you believe. Read *"Should you do this yet?"* at the
bottom first — it is a genuine question, not a formality.

---

## Part 1 — Buy the server

### 1.1 Pick the right product

Go to **hostinger.com → Hosting → VPS Hosting**.

> ⚠️ **It must be VPS.** Hostinger's cheaper "Web Hosting" / "Cloud Hosting"
> plans cannot run this. They give you a website, not a computer — no
> long-running programs, no installing LightGBM. If the page talks about
> WordPress and websites, you are on the wrong product.

**KVM 1** is enough: 1 vCPU, 4 GB RAM, 50 GB disk. The engine needs about 1 GB
of RAM and 3 GB of disk.

Longer terms are cheaper per month, but a 12-month commitment on something you
have not tested yet is the wrong trade. Take the shortest term available.

### 1.2 Set it up

Hostinger asks a few questions after purchase:

| Question | Answer |
|---|---|
| Location | Whichever is closest to you — it only affects your SSH speed |
| Operating system | **Ubuntu 24.04** (plain, from "OS only" — *not* one with a control panel) |
| Root password | Generate a strong one and **save it in your password manager now** |
| SSH key | Skip it. Password is fine to start |

> If it offers "Ubuntu with CyberPanel/cPanel/Plesk", say no. Those install a
> web control panel you do not need and that will fight you for port 8000.

### 1.3 Write down your IP

When setup finishes, hPanel shows an **IP address** like `31.220.90.14`. You
need it for every step below. Write it down.

---

## Part 2 — Connect to it

You have two ways in. Try the browser one first — it always works.

### Option A — Hostinger's browser terminal (easiest)

In hPanel, open your VPS and click **Browser terminal**. A black window opens,
already logged in as root. Done — skip to Part 3.

### Option B — SSH from your own PC (better)

You need this eventually anyway, for the dashboard tunnel. Windows 10 and 11
have SSH built in.

Open **Command Prompt** and type, with your own IP:

```
ssh root@31.220.90.14
```

First time only, it asks:

```
The authenticity of host ... can't be established.
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Enter. Then paste your root password.

> **The password will not appear as you type.** No dots, no stars, nothing.
> That is normal — Linux hides it completely. Paste and press Enter.

When you see `root@srv123:~#`, you are in.

---

## Part 3 — Install everything (one command)

Paste this and press Enter:

```bash
curl -fsSL https://raw.githubusercontent.com/AlBaydoun/tradingsignals/main/deploy/vps-setup.sh | bash
```

It takes 3–6 minutes and tells you what it is doing:

```
==> Installing system packages (python3, git, curl)
==> Creating the unprivileged service account 'signalforge'
==> Downloading SignalForge into /opt/signalforge
==> Building the Python environment (2-5 minutes)
==> Checking data providers and symbol configuration
```

Then a health check listing your instruments, and a summary of what to do next.

**What it just did:** installed Python, created a locked-down user called
`signalforge` that cannot log in, put the code in `/opt/signalforge`, installed
the dependencies, and registered two background services that start
automatically when the server reboots.

**What it deliberately did not do:** start the trading watcher. There are no
trained models yet. An engine with nothing to say only teaches you to ignore
its output.

---

## Part 4 — Your settings

### 4.1 Open the config file

```bash
cd /opt/signalforge
nano config/config.yaml
```

`nano` is a text editor inside the terminal. Arrow keys to move. No mouse.

### 4.2 Change your balance

Scroll down to:

```yaml
risk:
  account_balance: 10000.0
```

Change `10000.0` to your real balance.

### 4.3 Check your symbol names

Further down:

```yaml
instruments:
  XAUUSD:
    mt5_symbol: "XAUUSD"
  NAS100:
    mt5_symbol: "US100.std"
```

Open MT5 on your phone → **Quotes** → **+** → search each name on the right. If
your broker spells one differently, correct it here.

### 4.4 Save and exit

`Ctrl+O` then `Enter` to save. `Ctrl+X` to exit.

### 4.5 Verify

```bash
sudo -u signalforge ./.venv/bin/python -m signalforge.cli doctor
```

It prints your symbol mapping and tells you which instruments your balance
cannot afford. Fix anything it flags before moving on.

---

## Part 5 — Train

```bash
sudo -u signalforge ./.venv/bin/python -m signalforge.cli train --timeframes H1 H4
```

**About 30 minutes.** It prints each model as it finishes:

```
[1/14] XAUUSD H1: accuracy 0.5147 [95% CI 0.484-0.545] eff.n 1038  <- edge not significant
```

At the end it tells you how many survived the multiple-comparison correction.
Usually very few. That is the honest answer, not a failure.

> **Do not close the window during training.** If you are on SSH and your
> connection drops, training dies with it. To be safe, run it inside `screen`:
>
> ```bash
> screen -S train
> sudo -u signalforge ./.venv/bin/python -m signalforge.cli train --timeframes H1 H4
> ```
>
> Press `Ctrl+A` then `D` to detach — training keeps going. `screen -r train`
> to come back. Hostinger's browser terminal disconnects if you close the tab,
> so this matters there.

---

## Part 6 — Start it

```bash
systemctl start signalforge-watch
systemctl status signalforge-watch
```

You want to see `active (running)` in green.

Watch it work:

```bash
tail -f /opt/signalforge/data/watch.log
```

```
[08:47:56] cycle 1
  no signals (0 on watch)
  Swept 31 instruments (10 open) on H1. Nothing moving unusually.
```

`Ctrl+C` stops watching the log. **It does not stop the engine** — that keeps
running in the background, and restarts itself if it crashes or the server
reboots.

---

## Part 7 — Read the dashboard

The dashboard has **no password**, and your VPS has a public IP that anyone can
reach. So the installer bound it to localhost. You reach it through an
encrypted tunnel.

On **your own Windows PC**, open a *new* Command Prompt:

```
ssh -L 8000:127.0.0.1:8000 root@31.220.90.14
```

Enter your password. **Leave this window open** — closing it closes the tunnel.

Now open your browser to:

```
http://localhost:8000/dashboard
```

That is your VPS's dashboard, arriving through the tunnel.

> **Do not** change the service to `--host 0.0.0.0` to skip the tunnel. That
> puts an unauthenticated page showing your positions, balance and risk on the
> open internet.

### On your phone

Install **Termius** (free, iOS and Android). Add a host with your IP and root
password, then add a port forward: local `8000` → `127.0.0.1:8000`. Connect,
then open `http://localhost:8000/dashboard` in your phone browser.

---

## Part 8 — Keep it fed

Retrain weekly. Set it once:

```bash
crontab -e
```

Choose `1` for nano if asked. Add this at the bottom:

```
0 2 * * 0 cd /opt/signalforge && sudo -u signalforge ./.venv/bin/python -m signalforge.cli learn --retrain >> data/retrain.log 2>&1
```

`Ctrl+O`, `Enter`, `Ctrl+X`. That retrains every Sunday at 02:00 UTC.

---

## Commands you will actually use

```bash
systemctl status signalforge-watch      # is it alive?
systemctl restart signalforge-watch     # after changing config
systemctl stop signalforge-watch        # pause it
tail -f /opt/signalforge/data/watch.log # what is it doing?

cd /opt/signalforge && git pull         # get updates
systemctl restart signalforge-watch     #   then restart
```

---

## When something breaks

| What you see | What it means | Fix |
|---|---|---|
| `Permission denied (publickey)` | Wrong password, or key-only login | Reset the root password in hPanel |
| `Connection refused` on SSH | Server still booting | Wait 2 minutes |
| `active (running)` but log is empty | No models trained yet | Do Part 5 |
| `failed` on status | Read the error | `journalctl -u signalforge-watch -n 50` |
| Dashboard won't load | Tunnel closed | Reopen the `ssh -L` window |
| `command not found: curl` | Very minimal image | `apt update && apt install -y curl` |
| Training killed partway | Ran out of RAM | Train fewer symbols: `--symbols XAUUSD US30` |

---

## Should you do this yet?

Honestly: **probably not, and here is the test.**

A VPS makes the engine *awake* 24/7. It does nothing whatsoever for whether the
models have an edge. On the most recent run, one model of fourteen survived the
multiple-comparison correction, and its own backtest then produced 13 trades —
not enough to judge. A coin flip running on a paid server is still a coin flip.

Pay for a VPS when **both** of these are true:

1. Your journal shows results you believe, over 100+ signals, with **your
   broker's real spreads** in the config rather than my estimates.
2. You are demonstrably missing signals because your PC was asleep.

Until then, `watch.bat` on your own machine costs nothing and teaches you the
same things. See [`HOSTING.md`](HOSTING.md) for the comparison, and
[`HONEST_LIMITATIONS.md`](HONEST_LIMITATIONS.md) for what the engine cannot do
regardless of where it runs.
