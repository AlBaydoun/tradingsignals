# Where to run it

The engine is a Python program that fetches prices, thinks, and prints trades.
It has to run *somewhere that stays on*. That is the whole question.

**GitHub is not that place, and there is nothing wrong with GitHub.** It stores
the code. It does not run it. No hosting problem is being worked around by
moving away from it — you keep using GitHub for the code wherever you decide to
run things.

---

## The three real options

| | Cost | Runs while you sleep | Setup |
|---|---|---|---|
| **Your Windows PC** | free | only if you leave it on | 5 minutes |
| **A VPS** (Hostinger, Hetzner, Contabo…) | ~$5–8/mo | yes | 20 minutes |
| **A spare PC / Raspberry Pi at home** | free | yes, until the power blinks | 30 minutes |

### The honest recommendation

**Start on your Windows PC today.** Not as a stepping stone — because until you
have watched a hundred signals and replaced the spread estimates with your
broker's real numbers, there is nothing worth running 24/7. A model you have
not verified running unattended overnight is not an advantage.

**Move to a VPS when two things are true:** you have a track record in the
journal that you believe, and you are missing signals because your PC is
asleep. Not before.

---

## Windows PC

Covered in [`QUICKSTART.md`](QUICKSTART.md). Double-click `setup.bat`, then
`train.bat`, then `watch.bat`. Leave the window open.

The catch: Windows sleeps. `Settings → System → Power → Screen and sleep →
When plugged in, put my device to sleep after → Never` stops it. The signal
engine cannot fetch prices while the machine is suspended.

---

## VPS — one command

Any Ubuntu 22.04+ box. Hostinger's cheapest VPS is plenty; so is Hetzner's
CX22 or a Contabo VPS S. You need 1 GB of RAM and about 3 GB of disk.

SSH in, then:

```bash
curl -fsSL https://raw.githubusercontent.com/AlBaydoun/tradingsignals/main/deploy/vps-setup.sh | bash
```

That installs everything and registers two systemd services, so the engine
starts at boot and restarts itself if it crashes. It deliberately does **not**
start the watcher yet — there are no trained models on a fresh box.

Then:

```bash
cd ~/tradingsignals
nano config/config.yaml                                    # your balance, your symbols
./.venv/bin/python -m signalforge.cli doctor               # check symbol names
./.venv/bin/python -m signalforge.cli train --timeframes H1 H4   # ~30 min
sudo systemctl start signalforge-watch
tail -f data/watch.log
```

### Reading the dashboard on a VPS

The dashboard has **no password**, and a VPS has a public IP. The installer
binds it to `127.0.0.1` for that reason. Reach it through an SSH tunnel from
your own machine:

```bash
ssh -L 8000:127.0.0.1:8000 youruser@your-server-ip
```

Leave that running, then open `http://localhost:8000/dashboard` in your
browser. It works on a phone too, using any SSH client with port forwarding.

**Do not** change the service to `--host 0.0.0.0` unless you have put a
password-protected reverse proxy in front of it. An open dashboard tells
anyone who finds it what you trade and how much you risk.

### Keeping it fed

The engine retrains on demand, not automatically. Add a weekly cron:

```bash
crontab -e
# Sundays at 02:00 — retrain and re-police the models
0 2 * * 0 cd ~/tradingsignals && ./.venv/bin/python -m signalforge.cli learn --retrain >> data/retrain.log 2>&1
```

---

## What about Hostinger's shared hosting?

Shared web hosting — the cPanel kind, meant for WordPress — will not work.
There is no long-running process, no ability to install compiled Python
packages like LightGBM, and no background scheduler you control. You need their
**VPS** product, not their web hosting.

---

## What none of these change

Hosting decides whether the engine is *awake*. It has no effect on whether the
models have an edge. A coin flip running on a $200/month server is still a coin
flip — see [`HONEST_LIMITATIONS.md`](HONEST_LIMITATIONS.md).

Spend the money on a VPS when you have something worth running continuously,
and not one day earlier.
