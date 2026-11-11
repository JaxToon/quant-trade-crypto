"""Fund dashboard: status, performance vs buy-and-hold, prices, signals, trades, logs.

    python -m fund.dashboard            # http://127.0.0.1:8090

Bound to 127.0.0.1 only. The kill/resume buttons need FUND_DASHBOARD_TOKEN
(set it in .env); without it the page is read-only.

Panels carried over from the old SMC dashboard (dashboard/index.html): equity
curve, price movement, trade history, balance/return, signal cards, recent
logs. Regime, bear veto, GNN and knowledge-graph panels are not carried over:
the fund's strategy does not use them.
"""
from __future__ import annotations

import hmac
import os
from collections import deque
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from fund import config, core, ledger, market
from fund.market import today_utc
from research import strategies

app = FastAPI(title="TraderStuff Fund")
TOKEN = os.getenv("FUND_DASHBOARD_TOKEN", "")


def _ledger_path(mode: str) -> str:
    return config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER


_live_cache: dict = {"t": 0.0, "prices": {}}


def _live_prices() -> dict[str, float]:
    """Current prices for display only (the fund decides once a day on the daily open).
    One request for all symbols, cached 30 s; empty on any network error."""
    import time
    if time.time() - _live_cache["t"] < 30:
        return _live_cache["prices"]
    prices = market.live_prices()
    _live_cache.update(t=time.time(), prices=prices)
    return prices


def _closes() -> pd.DataFrame:
    """Complete daily closes from the fund's own kline cache (no network)."""
    frames = {}
    for s in config.UNIVERSE:
        p = os.path.join(config.KLINE_DIR, f"{s}_1d.parquet")
        if os.path.exists(p):
            frames[s] = pd.read_parquet(p)["close"]
    return pd.DataFrame(frames).sort_index()


@lru_cache(maxsize=4)
def _signals_for(last_bar: str) -> dict:
    close = _closes()
    sig, tgt = core.targets(close)
    vol = strategies.sigma90(close)
    d = close.index[-1]
    out = {}
    for s in close.columns:
        c = close[s]
        out[s] = {"active": int(round(sig.at[d, s] * len(strategies.LOOKBACKS))),
                  "of": len(strategies.LOOKBACKS),
                  "target": float(tgt.at[d, s]),
                  "vol90": float(vol.at[d, s]),
                  "close": float(c.iloc[-1]),
                  "hi20": float(c.iloc[-21:-1].max()),   # shortest-but-one breakout level
                  "chg30": float(c.iloc[-1] / c.iloc[-31] - 1)}
    return {"as_of": str(d.date()), "coins": out}


def _status(mode: str) -> dict:
    path = _ledger_path(mode)
    if not os.path.exists(path):
        return {"mode": mode, "exists": False}
    b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    last = b.runs[-1] if b.runs else {}
    prices = last.get("prices", {})
    positions = [{"symbol": k, "qty": q, "price": prices.get(k, 0.0), "value": q * prices.get(k, 0.0)}
                 for k, q in b.qty.items() if q > 0]
    equity = b.cash + sum(p["value"] for p in positions)
    for p in positions:
        p["weight"] = p["value"] / equity if equity else 0.0
    live = _live_prices()
    for p in positions:
        lp = live.get(p["symbol"])
        p["live_price"], p["live_value"] = lp, (p["qty"] * lp if lp else None)
    live_equity = (b.cash + sum(p["live_value"] for p in positions)
                   if positions and all(p["live_value"] is not None for p in positions) else (b.cash if not positions else None))

    # Buy-and-hold equal weight across the universe, from the first run's prices.
    curve, bh = [], []
    p0 = b.runs[0]["prices"] if b.runs else {}
    for r in b.runs:
        curve.append([r["date"], r["equity"]])
        rel = [r["prices"][k] / p0[k] for k in config.UNIVERSE if k in r["prices"] and p0.get(k)]
        bh.append([r["date"], b.start_capital * float(np.mean(rel))] if rel else [r["date"], None])

    # Backtest of the same period with the fund's own code (forward-test criterion 1).
    bt_curve, gap = [], None
    try:
        from fund import report
        c = report.compare(mode)
        if c:
            bt_curve = [[d, c["backtest"].get(d)] for d, _ in curve]
            gap = c["gap"]
    except Exception:
        pass

    fees = sum(t.get("fee", t.get("commission", 0.0)) or 0.0 for t in b.trades)
    days_since = (today_utc() - pd.Timestamp(last["date"], tz="UTC")).days if last else None
    light = "red" if b.halted else ("yellow" if days_since is None or days_since > 1 else "green")
    return {"mode": mode, "exists": True, "light": light, "halted": b.halted,
            "halt_reason": b.halt_reason, "kill_file": os.path.exists(config.KILL_FILE),
            "start_capital": b.start_capital, "equity": equity, "cash": b.cash,
            "exposure": 1 - b.cash / equity if equity else 0.0,
            "return": equity / b.start_capital - 1 if b.start_capital else 0.0,
            "bh_return": (bh[-1][1] / b.start_capital - 1) if bh and bh[-1][1] else None,
            "drawdown": 1 - equity / b.peak_equity if b.peak_equity else 0.0,
            "peak": b.peak_equity, "positions": positions, "curve": curve, "bh_curve": bh,
            "bt_curve": bt_curve, "bt_gap": gap,
            "live_equity": live_equity,
            "live_return": (live_equity / b.start_capital - 1) if live_equity and b.start_capital else None,
            "live_since_run": (live_equity / equity - 1) if live_equity and equity else None,
            "last_run": last.get("date"), "days_since_run": days_since,
            "started": b.runs[0]["date"] if b.runs else None,
            "n_runs": len(b.runs), "n_trades": len(b.trades), "fees": fees,
            "trades": b.trades[-30:][::-1], "recent": b.runs[-10:][::-1]}


@app.get("/api/status")
def status(mode: str = "paper"):
    return _status(mode)


_preview_cache: dict = {"t": 0.0, "data": None}


@app.get("/api/preview")
def preview(mode: str = "paper"):
    """What the next run would do if today closed at the current live price.
    Display only: the bot still decides once a day on the real daily close."""
    import time
    if time.time() - _preview_cache["t"] < 60 and _preview_cache["data"] and _preview_cache["data"].get("mode") == mode:
        return _preview_cache["data"]
    path = _ledger_path(mode)
    live = _live_prices()
    close = _closes()
    if not os.path.exists(path) or close.empty or not live:
        return {"mode": mode, "ok": False}
    syms = [s for s in config.UNIVERSE if s in close.columns and s in live]
    today = today_utc()
    close = close[syms].copy()
    close.loc[today] = pd.Series(live)[syms]
    sig, tgt = core.targets(close)
    b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    t = tgt.loc[today].to_numpy(dtype=float)
    s = sig.loc[today].to_numpy(dtype=float)
    q = np.array([b.qty.get(k, 0.0) for k in syms])
    px = np.array([live[k] for k in syms])
    equity = b.cash + float((q * px).sum())
    if b.halted or os.path.exists(config.KILL_FILE):
        t = np.minimum(t, q * px / equity if equity else 0.0)
    last = np.array([b.last_sig.get(k, 0.0) for k in syms])
    orders, *_ = core.plan_orders(syms, dict(zip(syms, q)), b.cash, live, t, s, last,
                                  today.dayofweek == 0, min_order=config.MIN_ORDER_USDT)
    coins = []
    for k in syms:
        lv = core.submodel_levels(close[k].dropna())
        on = sum(x["on"] for x in lv)
        ups = [x["breakout"] for x in lv if not x["on"] and x["breakout"]]
        downs = [x["stop"] for x in lv if x["on"] and x["stop"]]
        coins.append({"symbol": k, "price": live[k], "active": on,
                      "buy_above": min(ups) if ups else None, "sell_below": max(downs) if downs else None,
                      "target": float(t[syms.index(k)])})
    data = {"mode": mode, "ok": True, "as_of": time.strftime("%H:%M:%S"), "equity": equity,
            "orders": [{"symbol": o.symbol, "side": o.side, "notional": o.notional} for o in orders],
            "coins": coins}
    _preview_cache.update(t=time.time(), data=data)
    return data


@app.get("/api/market")
def market_moves(days: int = 30):
    """% change of each coin over the last `days` complete daily bars."""
    close = _closes()
    if close.empty:
        return {"dates": [], "series": {}}
    w = close.iloc[-(days + 1):]
    rel = (w / w.iloc[0] - 1) * 100
    return {"dates": [str(d.date()) for d in rel.index],
            "series": {s: [None if pd.isna(v) else round(float(v), 2) for v in rel[s]] for s in rel.columns}}


@app.get("/api/signals")
def signals():
    close = _closes()
    if close.empty:
        return {"as_of": None, "coins": {}}
    return _signals_for(str(close.index[-1].date()))


@app.get("/api/logs")
def logs(n: int = 40):
    path = os.path.join(config.DATA_DIR, "fund.log")
    if not os.path.exists(path):
        return {"lines": []}
    with open(path, encoding="utf-8", errors="replace") as f:
        return {"lines": list(deque(f, maxlen=n))[::-1]}


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
    path = _ledger_path(mode)
    if os.path.exists(path):
        b = ledger.load(path, mode, config.PAPER_START_CAPITAL)
        b.halted, b.halt_reason = False, ""
        ledger.save(b, path)
    return {"ok": True}


PAGE = r"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fund Monitor</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ec;--mut:#8a91a0;--g:#2fbf71;--y:#e0b03a;--r:#e05252;--b:#5b8def;--line:#2a2f3a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h2{margin:0;font-size:20px}.sub{color:var(--mut);font-weight:400;font-size:13px}
.grid{display:grid;gap:12px;margin:12px 0}.g6{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}.g4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}.v{font-size:22px;font-weight:600}
.light{display:flex;align-items:center;gap:12px;font-size:16px}.dot{width:20px;height:20px;border-radius:50%;flex:none}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:500;font-size:11px}.pos{color:var(--g)}.neg{color:var(--r)}.mut{color:var(--mut)}
.scroll{overflow-x:auto}button{background:#252a35;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer}
button.stop{border-color:var(--r);color:var(--r)}svg{width:100%;display:block}
select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--mut);margin-top:6px}.legend i{display:inline-block;width:10px;height:3px;margin-right:4px;vertical-align:middle}
.bar{height:6px;background:#252a35;border-radius:3px;overflow:hidden;margin:6px 0}.bar>div{height:100%;background:var(--g)}
.coin{display:flex;justify-content:space-between;align-items:baseline}.coin b{font-size:15px}
.logs{font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--mut);max-height:220px;overflow:auto;white-space:pre-wrap;word-break:break-all}
.note{color:var(--mut);font-size:12px;margin:6px 0 0}
</style></head><body><main>
<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
<h2>Fund Monitor <span class="sub">C4 DONCH-VT ×2 · trend following harian · 8 koin + emas (PAXG) · berhenti beli di drawdown 30%</span></h2>
<select id="mode"><option value="paper">Paper</option><option value="live">Live</option></select></div>

<div class="card light" style="margin-top:12px"><div class="dot" id="dot"></div><div id="lt"></div></div>
<div class="grid g6" id="kpi"></div>

<div class="grid g2">
 <div class="card"><div class="k">Kurva ekuitas vs buy &amp; hold</div><svg id="curve" viewBox="0 0 600 200" preserveAspectRatio="none" style="height:200px"></svg>
  <div class="legend"><span><i style="background:var(--g)"></i>Fund (paper)</span><span><i style="background:var(--b)"></i>Backtest periode sama</span><span><i style="background:var(--mut)"></i>Buy &amp; hold semua aset</span></div>
  <p class="note" id="curvenote"></p></div>
 <div class="card"><div class="k">Pergerakan harga (% , 30 hari)</div><svg id="mkt" viewBox="0 0 600 200" preserveAspectRatio="none" style="height:200px"></svg>
  <div class="legend" id="mktleg"></div></div>
</div>

<div class="card"><div class="k">Pratinjau live <span id="pvasof" class="mut"></span></div>
<p class="note" style="margin:0 0 8px">Kalau hari ini ditutup di harga sekarang, inilah yang akan dilakukan bot besok pagi (07:10 WIB). Bot hanya memutuskan sekali sehari dari harga penutupan, jadi daftar ini bisa berubah sampai 07:00 WIB.</p>
<div id="pvorders" style="margin-bottom:10px"></div><div class="scroll"><table id="pvcoins"></table></div></div>
<div class="card"><div class="k">Sinyal per koin <span id="sigasof" class="mut"></span></div>
<p class="note" style="margin:0 0 8px">Strategi punya 9 sub-model (lookback 5 s/d 360 hari). Makin banyak yang aktif, makin besar porsi koin itu. Bobot dikecilkan untuk koin yang lebih liar (target volatilitas 25%).</p>
<div class="grid g4" id="sig" style="margin:0"></div></div>

<div class="grid g2">
 <div class="card"><div class="k">Posisi terbuka</div><div class="scroll"><table id="pos"></table></div></div>
 <div class="card"><div class="k">Riwayat trade</div><div class="scroll" style="max-height:260px"><table id="trades"></table></div></div>
</div>

<div class="grid g2">
 <div class="card"><div class="k">Log terbaru</div><div class="logs" id="logs"></div></div>
 <div class="card"><div class="k">Kontrol</div><p class="note">Stop: tidak ada pembelian baru mulai run berikutnya; posisi tetap keluar sesuai sinyal. Butuh token (FUND_DASHBOARD_TOKEN di .env).</p>
 <button class="stop" onclick="ctl('kill')">Stop pembelian</button> <button onclick="ctl('resume')">Lanjutkan</button></div>
</div>
</main><script>
const $=id=>document.getElementById(id);
const usd=x=>'$'+Number(x).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const pct=(x,d=2)=>(x*100).toFixed(d)+'%',sgn=x=>x>=0?'pos':'neg';
const COLORS=['#f7931a','#8c8cff','#f0b90b','#23a7e0','#3cc8c8','#9945ff','#e84142','#e6007a','#c9a227'];
function line(pts,lo,hi,color,w){const r=hi-lo||1,n=Math.max(pts.length-1,1);
 return `<polyline fill="none" stroke="${color}" stroke-width="${w}" vector-effect="non-scaling-stroke" points="${pts.map((v,i)=>v==null?'':(i/n*600).toFixed(1)+','+(190-(v-lo)/r*180).toFixed(1)).filter(Boolean).join(' ')}"/>`}
async function j(u){return (await fetch(u)).json()}
async function load(){
 const m=$('mode').value,s=await j('/api/status?mode='+m);
 if(!s.exists){$('lt').textContent='Belum ada ledger untuk mode '+m;$('dot').style.background='var(--mut)';
  ['kpi','pos','trades'].forEach(i=>$(i).innerHTML='');$('curve').innerHTML='';$('curvenote').textContent='';}
 else{
 $('dot').style.background={green:'var(--g)',yellow:'var(--y)',red:'var(--r)'}[s.light];
 $('lt').textContent=s.halted?'BERHENTI: '+s.halt_reason:(s.light==='green'?`Berjalan normal · run terakhir ${s.last_run} · mulai ${s.started}`:`Peringatan: run terakhir ${s.last_run} (${s.days_since_run} hari lalu)`);
 const bh=s.bh_return==null?'—':`<span class="${sgn(s.bh_return)}">${pct(s.bh_return)}</span>`;
 const liveTxt=s.live_equity==null?'—':`${usd(s.live_equity)}<div class="mut" style="font-size:12px;font-weight:400">${s.live_since_run>=0?'+':''}${pct(s.live_since_run)} sejak run · resmi ${usd(s.equity)}</div>`;
 $('kpi').innerHTML=[['Ekuitas live',liveTxt],['Return fund (resmi)',`<span class="${sgn(s.return)}">${pct(s.return)}</span>`],['Buy &amp; hold',bh],
  ['Drawdown',pct(s.drawdown)],['Di pasar',pct(s.exposure,0)],['Total fee',usd(s.fees)],
  ['Selisih vs backtest',s.bt_gap==null?'—':`<span class="${Math.abs(s.bt_gap)<=0.01?'pos':'neg'}">${pct(s.bt_gap)}</span>`]]
  .map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
 const a=s.curve.map(p=>p[1]),b=s.bh_curve.map(p=>p[1]),c=(s.bt_curve||[]).map(p=>p[1]),all=a.concat(b.filter(x=>x!=null),c.filter(x=>x!=null));
 if(a.length>=2){const lo=Math.min(...all),hi=Math.max(...all);$('curve').innerHTML=line(b,lo,hi,'var(--mut)',1.5)+line(c,lo,hi,'var(--b)',1.5)+line(a,lo,hi,'var(--g)',2.5)}
 else $('curve').innerHTML='<text x="300" y="100" text-anchor="middle" fill="#8a91a0" font-size="14">Kurva muncul setelah 2 hari run</text>';
 $('curvenote').textContent=s.n_runs<30?`Baru ${s.n_runs} hari data: terlalu sedikit untuk menilai strategi.`:'';
 $('pos').innerHTML='<tr><th>Koin</th><th>Qty</th><th>Harga beli/run</th><th>Harga live</th><th>Nilai live</th><th>+/-</th></tr>'+(s.positions.map(p=>{const ch=p.live_price?p.live_price/p.price-1:null;return `<tr><td>${p.symbol.replace('USDT','')}</td><td>${p.qty.toPrecision(4)}</td><td>${usd(p.price)}</td><td>${p.live_price?usd(p.live_price):'—'}</td><td>${p.live_value!=null?usd(p.live_value):usd(p.value)}</td><td class="${ch==null?'mut':sgn(ch)}">${ch==null?'—':(ch>=0?'+':'')+pct(ch)}</td></tr>`}).join('')||'<tr><td colspan=6 class="mut">Kas penuh</td></tr>');
 $('trades').innerHTML='<tr><th>Tanggal</th><th>Koin</th><th>Arah</th><th>Nilai</th><th>Fee</th></tr>'+(s.trades.map(t=>
  `<tr><td>${t.date}</td><td>${t.symbol.replace('USDT','')}</td><td class="${t.side==='BUY'?'pos':'neg'}">${t.side==='BUY'?'BELI':'JUAL'}</td><td>${usd(t.notional??(t.qty*(t.price||0)))}</td><td>${usd(t.fee??t.commission??0)}</td></tr>`).join('')||'<tr><td colspan=5 class="mut">Belum ada trade</td></tr>');
 }
 const mk=await j('/api/market?days=30'),names=Object.keys(mk.series);
 if(names.length){const vals=names.flatMap(n=>mk.series[n].filter(v=>v!=null)),lo=Math.min(0,...vals),hi=Math.max(0,...vals);
  $('mkt').innerHTML=line(mk.dates.map(()=>0),lo,hi,'var(--line)',1)+names.map((n,i)=>line(mk.series[n],lo,hi,COLORS[i%9],1.6)).join('');
  $('mktleg').innerHTML=names.map((n,i)=>{const v=mk.series[n].at(-1);return `<span><i style="background:${COLORS[i%9]}"></i>${n.replace('USDT','')} <b class="${sgn(v)}">${v>0?'+':''}${v}%</b></span>`}).join('')}
 const pv=await j('/api/preview?mode='+m);
 if(pv.ok){$('pvasof').textContent='· dihitung '+pv.as_of;
  $('pvorders').innerHTML=pv.orders.length?pv.orders.map(o=>`<span class="${o.side==='BUY'?'pos':'neg'}" style="margin-right:14px"><b>${o.side==='BUY'?'BELI':'JUAL'}</b> ${o.symbol.replace('USDT','')} ${usd(o.notional)}</span>`).join(''):'<span class="mut">Tidak ada beli/jual kalau harga bertahan seperti sekarang.</span>';
  $('pvcoins').innerHTML='<tr><th>Aset</th><th>Harga live</th><th>Sub-model</th><th>Tambah beli kalau tutup di atas</th><th>Jual sebagian kalau tutup di bawah</th></tr>'+pv.coins.map(c=>{
   const up=c.buy_above?`${usd(c.buy_above)} <span class="mut">(+${((c.buy_above/c.price-1)*100).toFixed(1)}%)</span>`:'<span class="mut">semua aktif</span>';
   const dn=c.sell_below?`${usd(c.sell_below)} <span class="mut">(${((c.sell_below/c.price-1)*100).toFixed(1)}%)</span>`:'<span class="mut">tidak ada posisi</span>';
   return `<tr><td>${c.symbol.replace('USDT','')}</td><td>${usd(c.price)}</td><td>${c.active}/9</td><td>${up}</td><td>${dn}</td></tr>`}).join('')}
 else{$('pvorders').innerHTML='<span class="mut">Harga live belum tersedia.</span>';$('pvcoins').innerHTML='';}
 const sg=await j('/api/signals');$('sigasof').textContent=sg.as_of?'· data '+sg.as_of:'';
 $('sig').innerHTML=Object.entries(sg.coins).map(([k,c])=>{const st=c.active===0?'Tidak ada tren naik':(c.active>=6?'Tren naik kuat':'Tren naik sebagian');
  const gap=c.close<c.hi20?`perlu +${((c.hi20/c.close-1)*100).toFixed(1)}% untuk tembus high 20 hari`:'di atas high 20 hari';
  return `<div class="card"><div class="coin"><b>${k.replace('USDT','')}</b><span class="${sgn(c.chg30)}">${c.chg30>0?'+':''}${(c.chg30*100).toFixed(1)}% 30h</span></div>
  <div class="bar"><div style="width:${c.active/c.of*100}%"></div></div><div>${c.active}/${c.of} sub-model aktif · ${st}</div>
  <div class="mut">Target porsi ${pct(c.target,1)} · volatilitas ${pct(c.vol90,0)}</div><div class="mut">${gap}</div></div>`}).join('')||'<div class="mut">Belum ada data</div>';
 const lg=await j('/api/logs?n=40');$('logs').textContent=lg.lines.join('')||'Belum ada log';
}
async function ctl(a){const t=prompt('Token dashboard:');if(!t)return;const r=await fetch('/api/'+a+'?mode='+$('mode').value,{method:'POST',headers:{'X-Token':t}});alert(r.ok?'OK':'Gagal: '+(await r.json()).detail);load()}
$('mode').onchange=load;load();setInterval(load,30000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("FUND_DASHBOARD_PORT", "8090")), log_level="warning")
