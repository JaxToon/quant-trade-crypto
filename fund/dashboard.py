"""Fund dashboard: status, performance, positions, kill switch.

    python -m fund.dashboard            # http://127.0.0.1:8090

Bound to 127.0.0.1 only. The kill/resume buttons need FUND_DASHBOARD_TOKEN
(set it in .env); without it the page is read-only.
"""
from __future__ import annotations

import hmac
import os

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from fund import config, ledger
from fund.market import today_utc

app = FastAPI(title="TraderStuff Fund")
TOKEN = os.getenv("FUND_DASHBOARD_TOKEN", "")


def _status(mode: str) -> dict:
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
    if not os.path.exists(path):
        return {"mode": mode, "exists": False}
    b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    last = b.runs[-1] if b.runs else {}
    prices = last.get("prices", {})
    positions = [{"symbol": k, "qty": q, "value": q * prices.get(k, 0.0)}
                 for k, q in b.qty.items() if q > 0]
    equity = b.cash + sum(p["value"] for p in positions)
    curve = [[r["date"], r["equity"]] for r in b.runs]
    fees = sum(t.get("fee", t.get("commission", 0.0)) or 0.0 for t in b.trades)
    days_since = (today_utc() - pd.Timestamp(last["date"], tz="UTC")).days if last else None
    light = "red" if b.halted else ("yellow" if days_since is None or days_since > 1 else "green")
    return {"mode": mode, "exists": True, "light": light, "halted": b.halted,
            "halt_reason": b.halt_reason, "kill_file": os.path.exists(config.KILL_FILE),
            "start_capital": b.start_capital, "equity": equity, "cash": b.cash,
            "return": equity / b.start_capital - 1 if b.start_capital else 0.0,
            "drawdown": 1 - equity / b.peak_equity if b.peak_equity else 0.0,
            "peak": b.peak_equity, "positions": positions, "curve": curve,
            "last_run": last.get("date"), "days_since_run": days_since,
            "n_runs": len(b.runs), "n_trades": len(b.trades), "fees": fees,
            "recent": b.runs[-10:][::-1]}


@app.get("/api/status")
def status(mode: str = "paper"):
    return _status(mode)


def _auth(token: str | None):
    if not TOKEN:
        raise HTTPException(403, "FUND_DASHBOARD_TOKEN is not set: controls are disabled")
    if not token or not hmac.compare_digest(token, TOKEN):
        raise HTTPException(401, "bad token")


@app.post("/api/kill")
def kill(x_token: str | None = Header(default=None)):
    _auth(x_token)
    with open(config.KILL_FILE, "w") as f:
        f.write("stopped from dashboard\n")
    return {"ok": True, "note": "no new buys from the next run on; exits still happen"}


@app.post("/api/resume")
def resume(x_token: str | None = Header(default=None), mode: str = "paper"):
    _auth(x_token)
    if os.path.exists(config.KILL_FILE):
        os.remove(config.KILL_FILE)
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
    if os.path.exists(path):
        b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
        b.halted, b.halt_reason = False, ""
        ledger.save(b, path)
    return {"ok": True}


PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fund Monitor</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ec;--mut:#8a91a0;--g:#2fbf71;--y:#e0b03a;--r:#e05252;--line:#2a2f3a}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:24px;font-weight:600}
.light{display:flex;align-items:center;gap:12px;font-size:18px}.dot{width:22px;height:22px;border-radius:50%}
table{width:100%;border-collapse:collapse}td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500;font-size:12px}.pos{color:var(--g)}.neg{color:var(--r)}
button{background:#252a35;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer}
button.stop{border-color:var(--r);color:var(--r)}svg{width:100%;height:180px}select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px}
</style></head><body><main>
<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
<h2 style="margin:0">Fund Monitor <small style="color:var(--mut);font-weight:400">C4 DONCH-VT</small></h2>
<select id="mode"><option value="paper">Paper</option><option value="live">Live</option></select></div>
<div class="card light" style="margin-top:12px"><div class="dot" id="dot"></div><div id="lt"></div></div>
<div class="row" id="kpi"></div>
<div class="card"><div class="k">Kurva ekuitas</div><svg id="curve" viewBox="0 0 600 180" preserveAspectRatio="none"></svg></div>
<div class="row"><div class="card"><div class="k">Posisi terbuka</div><table id="pos"></table></div>
<div class="card"><div class="k">10 hari terakhir</div><table id="runs"></table></div></div>
<div class="card"><div class="k">Kontrol</div><p style="color:var(--mut);margin:6px 0">Stop: tidak ada pembelian baru mulai run berikutnya; posisi tetap keluar sesuai sinyal.</p>
<button class="stop" onclick="ctl('kill')">Stop pembelian</button> <button onclick="ctl('resume')">Lanjutkan</button></div>
</main><script>
const $=id=>document.getElementById(id),usd=x=>'$'+x.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}),pct=x=>(x*100).toFixed(2)+'%';
async function load(){const m=$('mode').value,s=await (await fetch('/api/status?mode='+m)).json();
if(!s.exists){$('lt').textContent='Belum ada ledger untuk mode '+m;$('dot').style.background='var(--mut)';['kpi','pos','runs'].forEach(i=>$(i).innerHTML='');$('curve').innerHTML='';return}
const col={green:'var(--g)',yellow:'var(--y)',red:'var(--r)'}[s.light];$('dot').style.background=col;
$('lt').textContent=s.halted?'BERHENTI: '+s.halt_reason:(s.light==='green'?'Berjalan normal, run terakhir '+s.last_run:'Peringatan: run terakhir '+s.last_run+' ('+s.days_since_run+' hari lalu)');
$('kpi').innerHTML=[['Ekuitas',usd(s.equity)],['Return',`<span class="${s.return>=0?'pos':'neg'}">${pct(s.return)}</span>`],['Drawdown',pct(s.drawdown)],['Posisi',s.positions.length],['Total fee',usd(s.fees)],['Hari berjalan',s.n_runs]].map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
const c=s.curve.map(p=>p[1]);if(c.length){const lo=Math.min(...c),hi=Math.max(...c),r=hi-lo||1,n=Math.max(c.length-1,1);
$('curve').innerHTML=`<polyline fill="none" stroke="var(--g)" stroke-width="2" points="${c.map((v,i)=>(i/n*600).toFixed(1)+','+(170-(v-lo)/r*160).toFixed(1)).join(' ')}"/>`}
$('pos').innerHTML='<tr><th>Koin</th><th>Qty</th><th>Nilai</th></tr>'+(s.positions.map(p=>`<tr><td>${p.symbol}</td><td>${p.qty.toPrecision(4)}</td><td>${usd(p.value)}</td></tr>`).join('')||'<tr><td colspan=3>Kas penuh</td></tr>');
$('runs').innerHTML='<tr><th>Tanggal</th><th>Ekuitas</th><th>Trade</th></tr>'+s.recent.map(r=>`<tr><td>${r.date}</td><td>${usd(r.equity)}</td><td>${r.orders.length}</td></tr>`).join('')}
async function ctl(a){const t=prompt('Token dashboard:');if(!t)return;const r=await fetch('/api/'+a+'?mode='+$('mode').value,{method:'POST',headers:{'X-Token':t}});alert(r.ok?'OK':'Gagal: '+(await r.json()).detail);load()}
$('mode').onchange=load;load();setInterval(load,60000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("FUND_DASHBOARD_PORT", "8090")), log_level="warning")
