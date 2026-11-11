# Moving the fund to a VPS

Recommended: DigitalOcean, Droplet "Basic / Regular", 1 GB RAM, $6/month,
region **Singapore**, image **Ubuntu 24.04**. Any 1 GB Ubuntu server outside
Indonesia works the same.

1. On the server (as root): `bash setup_vps.sh <git clone URL>`
   (download the script first: `curl -O https://raw.githubusercontent.com/<you>/<repo>/master/deploy/setup_vps.sh`;
   a private repo needs a GitHub token in the URL or a deploy key).
   It ends with `check_access.py`: all lines must say OK.
2. On the laptop, stop the local fund first (close the Claude session running it,
   or `fund\setup_task.ps1 -Remove`), so two bots never trade one ledger or poll one Telegram bot.
3. Copy the secrets and the book (from the repo folder on the laptop):
   `scp .env data/fund_ledger_paper.json root@<server-ip>:/home/trader/TraderStuff/`
   then on the server: `mv /home/trader/TraderStuff/fund_ledger_paper.json /home/trader/TraderStuff/data/ && chown -R trader:trader /home/trader/TraderStuff`
4. Start: `systemctl enable --now fund-daemon fund-bot fund-dash`
5. Check: `journalctl -u fund-daemon -n 20`, `cat /home/trader/TraderStuff/data/fund_heartbeat.json`, `/status` on Telegram.
6. Dashboard from the laptop: `ssh -L 8090:127.0.0.1:8090 root@<server-ip>` then open http://127.0.0.1:8090.

Before real money: whitelist the server's IP (printed by `check_access.py`) on the
Tokocrypto API key, and never enable withdrawals on it.
