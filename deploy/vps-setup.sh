#!/usr/bin/env bash
# =============================================================================
#  SignalForge — one-command install on a fresh Ubuntu/Debian VPS
#  (Hostinger, Hetzner, Contabo, DigitalOcean — any of them.)
#
#      curl -fsSL https://raw.githubusercontent.com/AlBaydoun/tradingsignals/main/deploy/vps-setup.sh | sudo bash
#
#  Hostinger hands you a root login, so that is the case this is written for.
#  Running as root, it creates a dedicated unprivileged `signalforge` user and
#  installs under /opt/signalforge — the engine never needs root, and a service
#  that runs continuously should not have it.
#
#  It installs and configures. It deliberately does NOT start the watcher: a
#  fresh box has no trained models, and starting an engine with nothing to say
#  only teaches you to ignore its output.
#
#  Environment overrides:
#    SIGNALFORGE_DIR   install location   (default /opt/signalforge, or ~/tradingsignals)
#    SIGNALFORGE_PORT  dashboard port     (default 8000)
#    SIGNALFORGE_REPO  git URL
#    SKIP_SYSTEMD=1    install only, register no services
# =============================================================================
set -euo pipefail

REPO="${SIGNALFORGE_REPO:-https://github.com/AlBaydoun/tradingsignals.git}"
PORT="${SIGNALFORGE_PORT:-8000}"
SERVICE_USER="signalforge"

say()  { printf '\n\033[1;36m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
ok()   { printf '    \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m !\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[1;31m ✗\033[0m  %s\n\n' "$*"; exit 1; }

# --- Who are we, and where does this go? -------------------------------------
if [ "$(id -u)" -eq 0 ]; then
  AS_ROOT=1
  SUDO=""
  DIR="${SIGNALFORGE_DIR:-/opt/signalforge}"
else
  AS_ROOT=0
  command -v sudo >/dev/null || die "Not root and sudo is missing. Log in as root."
  SUDO="sudo"
  SERVICE_USER="$(id -un)"
  DIR="${SIGNALFORGE_DIR:-$HOME/tradingsignals}"
fi

command -v apt-get >/dev/null || die \
  "This installer expects Ubuntu or Debian. On another distro, follow docs/QUICKSTART.md by hand."

printf '\n\033[1m  SignalForge — VPS install\033[0m\n'
printf '  target: %s\n' "$DIR"
printf '  user:   %s\n' "$SERVICE_USER"

# --- 1. System packages ------------------------------------------------------
say "Installing system packages (python3, git, curl)"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3 python3-venv python3-pip git curl ca-certificates
ok "$(python3 --version)"

# --- 2. Service account ------------------------------------------------------
if [ "$AS_ROOT" -eq 1 ]; then
  if id "$SERVICE_USER" >/dev/null 2>&1; then
    ok "user $SERVICE_USER already exists"
  else
    say "Creating the unprivileged service account '$SERVICE_USER'"
    useradd --system --create-home --home-dir "/home/$SERVICE_USER" \
            --shell /bin/bash "$SERVICE_USER"
    ok "created (no password, no login — services only)"
  fi
fi

# --- 3. Code -----------------------------------------------------------------
if [ -d "$DIR/.git" ]; then
  say "Updating the existing checkout"
  $SUDO git -C "$DIR" pull --ff-only
else
  say "Downloading SignalForge into $DIR"
  $SUDO mkdir -p "$(dirname "$DIR")"
  $SUDO git clone --depth 1 "$REPO" "$DIR"
fi
[ "$AS_ROOT" -eq 1 ] && chown -R "$SERVICE_USER:$SERVICE_USER" "$DIR"
ok "code in place"

# --- 4. Python environment ---------------------------------------------------
say "Building the Python environment (2-5 minutes)"
run_as() {
  if [ "$AS_ROOT" -eq 1 ]; then
    su - "$SERVICE_USER" -c "cd '$DIR' && $*"
  else
    ( cd "$DIR" && eval "$*" )
  fi
}
run_as "python3 -m venv .venv"
run_as "./.venv/bin/python -m pip install --quiet --upgrade pip"
run_as "./.venv/bin/python -m pip install --quiet -r requirements.txt"
ok "dependencies installed"

# --- 5. Health check ---------------------------------------------------------
say "Checking data providers and symbol configuration"
if ! run_as "./.venv/bin/python -m signalforge.cli doctor"; then
  warn "doctor reported issues. Read them above — most are symbol names you"
  warn "need to confirm against your broker's Market Watch."
fi

# --- 6. systemd services -----------------------------------------------------
if [ "${SKIP_SYSTEMD:-0}" = "1" ]; then
  warn "SKIP_SYSTEMD=1 — no services registered."
else
  say "Registering services so it survives reboots"

  $SUDO tee /etc/systemd/system/signalforge-watch.service >/dev/null <<UNIT
[Unit]
Description=SignalForge market watcher
Documentation=https://github.com/AlBaydoun/tradingsignals
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${DIR}
ExecStart=${DIR}/.venv/bin/python -m signalforge.cli watch --interval 300
Restart=always
RestartSec=30
StandardOutput=append:${DIR}/data/watch.log
StandardError=append:${DIR}/data/watch.log
# Hardening that is safe to apply blind. ProtectSystem=full covers /usr,
# /boot and /etc, which is where damage would matter. `strict` would also make
# the install directory read-only, and a service that dies with "Read-only
# file system" on someone's first VPS is worse than the marginal gain.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectKernelTunables=true
ProtectControlGroups=true

[Install]
WantedBy=multi-user.target
UNIT

  # Bound to localhost deliberately: the dashboard has no authentication and a
  # VPS has a public IP. It is reached over an SSH tunnel — see the printout.
  $SUDO tee /etc/systemd/system/signalforge-dashboard.service >/dev/null <<UNIT
[Unit]
Description=SignalForge dashboard
Documentation=https://github.com/AlBaydoun/tradingsignals
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${DIR}
ExecStart=${DIR}/.venv/bin/python -m signalforge.cli dashboard --host 127.0.0.1 --port ${PORT}
Restart=always
RestartSec=15
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectKernelTunables=true
ProtectControlGroups=true

[Install]
WantedBy=multi-user.target
UNIT

  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now signalforge-dashboard.service
  ok "dashboard service running on 127.0.0.1:${PORT}"
  ok "watcher service registered but NOT started — it has no models yet"
fi

# --- 7. What to do next ------------------------------------------------------
SUDO_HINT=""
[ "$AS_ROOT" -eq 0 ] && SUDO_HINT="sudo "
AS="$([ "$AS_ROOT" -eq 1 ] && echo "sudo -u $SERVICE_USER " || echo "")"

cat <<NEXT

$(printf '\033[1;32m═══════════════════════════════════════════════════════════════\033[0m')
$(printf '\033[1m  Installed at %s\033[0m' "$DIR")

  Three things left, in order.

$(printf '\033[1m  1. Your balance and your broker'"'"'s symbol names\033[0m')

     cd $DIR
     ${SUDO_HINT}nano config/config.yaml
     ${AS}./.venv/bin/python -m signalforge.cli doctor

     Change account_balance. Check every symbol under instruments:
     against the Market Watch window in MetaTrader 5.

$(printf '\033[1m  2. Train (about 30 minutes — you can disconnect)\033[0m')

     ${AS}./.venv/bin/python -m signalforge.cli train --timeframes H1 H4

$(printf '\033[1m  3. Start the watcher\033[0m')

     ${SUDO_HINT}systemctl start signalforge-watch
     tail -f $DIR/data/watch.log

$(printf '\033[1m  Reading the dashboard\033[0m')

     It has no password, so it is bound to localhost. From YOUR OWN
     computer — not this server — run:

         ssh -L ${PORT}:127.0.0.1:${PORT} $([ "$AS_ROOT" -eq 1 ] && echo root || echo "$SERVICE_USER")@$(hostname -I 2>/dev/null | awk '{print $1}')

     Leave that window open, then browse to:

         http://localhost:${PORT}/dashboard

$(printf '\033[1m  Day to day\033[0m')

     ${SUDO_HINT}systemctl status signalforge-watch
     ${SUDO_HINT}systemctl restart signalforge-watch
     tail -f $DIR/data/watch.log
$(printf '\033[1;32m═══════════════════════════════════════════════════════════════\033[0m')

NEXT
