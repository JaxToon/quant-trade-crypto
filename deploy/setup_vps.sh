#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 24.04 server, run as root:
#   bash setup_vps.sh https://github.com/<you>/<repo>.git
# Then copy .env and data/ from the laptop (see deploy/README.md) and start the services.
set -euo pipefail
REPO="${1:?usage: setup_vps.sh <git clone URL>}"
timedatectl set-timezone UTC
apt-get update -y && apt-get install -y python3-venv python3-pip git ufw
ufw allow OpenSSH && ufw --force enable          # nothing else is exposed; dashboard stays on 127.0.0.1
id trader >/dev/null 2>&1 || adduser --disabled-password --gecos "" trader
sudo -u trader git clone "$REPO" /home/trader/TraderStuff
cd /home/trader/TraderStuff
sudo -u trader python3 -m venv .venv
sudo -u trader .venv/bin/pip install -q -r requirements-fund.txt
sudo -u trader mkdir -p data
cp deploy/fund-*.service /etc/systemd/system/ && systemctl daemon-reload
sudo -u trader .venv/bin/python deploy/check_access.py
echo "Next: copy .env and data/fund_ledger_paper.json here, then: systemctl enable --now fund-daemon fund-bot fund-dash"
