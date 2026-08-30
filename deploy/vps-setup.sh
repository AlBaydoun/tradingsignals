#!/usr/bin/env bash
# =============================================================================
#  SignalForge — one-command setup on a fresh Ubuntu/Debian VPS
#  (Hostinger, Hetzner, Contabo, DigitalOcean — any of them.)
#
#    curl -fsSL https://raw.githubusercontent.com/AlBaydoun/tradingsignals/main/deploy/vps-setup.sh | bash
#
#  Or, having cloned already:   bash deploy/vps-setup.sh
#
#  Installs the engine, trains nothing (that is a deliberate second step), and
#  registers two systemd services so it survives reboots and disconnections.
# =============================================================================
set -euo pipefail

REPO="${SIGNALFORGE_REPO:-https://github.com/AlBaydoun/tradingsignals.git}"
DIR="${SIGNALFORGE_DIR:-$HOME/tradingsignals}"
PORT="${SIGNALFORGE_PORT:-8000}"

say()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m!!\033[0m  %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ] && [ -z "${SIGNALFORGE_ALLOW_ROOT:-}" ]; then
  warn "Running as root. That works, but a normal user with sudo is safer."
  warn "Set SIGNALFORGE_ALLOW_ROOT=1 to proceed anyway."
  exit 1
fi

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# --- 1. System packages ------------------------------------------------------
say "Installing system packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3 python3-venv python3-pip git curl

# --- 2. Code -----------------------------------------------------------------
if [ -d "$DIR/.git" ]; then
  say "Updating existing checkout at $DIR"
  git -C "$DIR" pull --ff-only
else
  say "Cloning into $DIR"
  git clone --depth 1 "$REPO" "$DIR"
fi
cd "$DIR"

# --- 3. Python environment ---------------------------------------------------
say "Creating the Python environment"
python3 -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r requirements.txt

# --- 4. Health check ---------------------------------------------------------
say "Checking data providers and symbol configuration"
./.venv/bin/python -m signalforge.cli doctor || \
  warn "doctor reported issues — read them above before trading."

# --- 5. systemd services -----------------------------------------------------
# Two units. The watcher is the engine; the dashboard is how you read it.
# Both restart on failure and start at boot.
say "Registering systemd services"
USER_NAME="$(id -un)"

$SUDO tee /etc/systemd/system/signalforge-watch.service >/dev/null <<UNIT
[Unit]
Description=SignalForge market watcher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${DIR}
ExecStart=${DIR}/.venv/bin/python -m signalforge.cli watch --interval 300
Restart=always
RestartSec=30
StandardOutput=append:${DIR}/data/watch.log
StandardError=append:${DIR}/data/watch.log

[Install]
WantedBy=multi-user.target
UNIT

# Bound to localhost on purpose: the dashboard has no authentication, and a
# VPS has a public IP. Reach it over an SSH tunnel (see the printout below)
# or put a password-protected reverse proxy in front of it.
$SUDO tee /etc/systemd/system/signalforge-dashboard.service >/dev/null <<UNIT
[Unit]
Description=SignalForge dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${DIR}
ExecStart=${DIR}/.venv/bin/python -m signalforge.cli dashboard --host 127.0.0.1 --port ${PORT}
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT

$SUDO systemctl daemon-reload
$SUDO systemctl enable --now signalforge-dashboard.service
say "Dashboard service started. The watcher stays stopped until models exist."

# --- 6. What to do next ------------------------------------------------------
cat <<NEXT

============================================================================
 Installed at $DIR

 STEP 1 — set your balance and check your broker's symbol names
     nano config/config.yaml
     ./.venv/bin/python -m signalforge.cli doctor

 STEP 2 — train (about 30 minutes; safe to disconnect)
     ./.venv/bin/python -m signalforge.cli train --timeframes H1 H4

 STEP 3 — start the watcher
     sudo systemctl start signalforge-watch
     tail -f data/watch.log

 READING THE DASHBOARD
     It is bound to localhost because it has no password and this machine
     has a public IP. From your own computer, open an SSH tunnel:

         ssh -L ${PORT}:127.0.0.1:${PORT} ${USER_NAME}@<this-server-ip>

     then browse to  http://localhost:${PORT}/dashboard

 USEFUL
     sudo systemctl status  signalforge-watch
     sudo systemctl restart signalforge-watch
     sudo systemctl stop    signalforge-watch
     tail -f $DIR/data/watch.log
============================================================================

NEXT
