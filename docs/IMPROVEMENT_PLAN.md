# Rencana Improvement v2 (dievaluasi ulang 11 Oktober 2026)

Rencana ini menggantikan v1 (masih bisa dilihat di git history, commit `7f9035e`).
Dasarnya:

- [`EVALUASI_2026-10-11.md`](EVALUASI_2026-10-11.md): kenapa sistem lama tidak punya edge.
- [`RISET_STRATEGI_2026-10.md`](RISET_STRATEGI_2026-10.md): riset luar + uji 10 strategi dengan aturan yang dikunci.

Label: **[TERUKUR]**, **[DUGAAN]**, **[SUMBER LUAR]**.

---

## Ringkasan dalam 5 baris

1. **SMC dipensiunkan** sebagai strategi trading. Ia yang terburuk di setiap tes, dan dua versinya bangkrut. [TERUKUR]
2. **Trend following** (Donchian ensemble, C4) adalah kandidat terbaik. [TERUKUR]
   Tapi C4 rugi −3.9% di data ujian 2025–2026, jadi **belum terbukti untung**.
3. Data historis sudah "terpakai". **Bukti baru hanya bisa datang dari waktu ke depan**:
   jalankan C4 sebagai paper trading harian (forward test).
4. Sistem baru dibuat **ramping**: cukup jalan sekali sehari, tanpa Neo4j, GNN, LLM, atau
   WebSocket. Lebih sedikit yang bisa rusak.
5. Uang asli hanya kalau forward test lolos syarat yang ditulis **sebelum** dimulai.

---

## Apa yang berubah dari v1 (evaluasi ulang rekomendasi saya)

| v1 bilang | Setelah riset | Status |
|---|---|---|
| Fase 0: perbaiki cara mengukur dulu | Sudah dikerjakan untuk riset (`research/`): eksekusi di open hari berikutnya, biaya asli, data Binance, holdout, uji acak, 8 test | **Selesai untuk riset** |
| Fase 1: cari edge, maksimal 5 kandidat | Diuji 10 (termasuk SMC). Trend following terbaik di DEV, tapi tidak ada yang lolos ujian holdout | **Selesai: tidak ada yang lolos** |
| "Time-series momentum harian, kandidat dengan bukti akademik" | Lolos uji acak di DEV (Sharpe 1.54) tapi drawdown 52.6%, dan −14.8% di holdout | Benar soal buktinya, tapi bukan pemenang |
| Fase 2: turunkan fee dulu | Untuk strategi harian yang jarang trading, biayanya kecil (C4 ±1.9%/tahun). Fee bukan masalah utama lagi | **Prioritas turun** |
| Fase 3: testnet lalu uang kecil | Diganti dengan **forward test** paper C4 yang syaratnya dikunci dulu | **Diubah** |
| Fase 4: dashboard 5 tab | Dibandingkan dengan FreqUI, Hummingbot, QuantConnect, FIA. Kurang: login, kill switch, notifikasi, panel risiko, grafik trade | **Diperbaiki jadi v2 (6 tab)** |
| Fase 5: sederhanakan arsitektur (opsional) | Strategi harian **tidak butuh** Neo4j, GNN, LLM, atau stream WebSocket | **Jadi inti rencana** |
| "Target minimal 3x biaya bolak-balik" untuk SMC | SMC tetap rugi di semua timeframe (1h, 4h, 1d), bahkan di harian | SMC dihentikan |

---

## Langkah A: Keputusan kamu, pensiunkan SMC sebagai strategi trading

Bukti [TERUKUR], semuanya dengan harga masuk jujur dan fee Tokocrypto:

| SMC | DEV 2018–2024 | Holdout 2025–2026 |
|---|---|---|
| 1h | bangkrut 25 Feb 2021 | −46.7% |
| 4h | bangkrut 14 Mei 2022 | −74.6% |
| 1d | −67% | −28.0% |

Kodenya tidak dihapus. Tetap di git, dan bisa dipakai untuk riset.

## Langkah B: Bot trend harian yang ramping (mode paper dulu)

| Bagian | Desain |
|---|---|
| Jadwal | Sekali sehari, 5 menit setelah candle harian UTC tutup (07:05 WIB) |
| Sinyal | **Kode yang sama persis** dengan riset (`research/strategies.py`), jadi tidak ada beda antara backtest dan live |
| Eksekusi paper | Dicatat di harga **open** hari itu (jujur), ke ledger **baru**. Ledger lama tidak disentuh |
| Eksekusi nyata (nanti) | Order Tokocrypto; order < $5 dilewati atau ditumpuk; rekonsiliasi saldo exchange vs ledger tiap hari |
| Pengaman (FIA 2024) | Kill switch, batas rugi harian, batas masuk-ulang otomatis, blokir order kalau data basi, batas ukuran order |
| Notifikasi | Telegram: ringkasan harian, setiap trade, peringatan |
| Yang **tidak** dipakai | Neo4j, GNN, LLM, WebSocket, monitor SL/TP per detik (stop dicek di harga close harian) |

## Langkah C: Forward test (syarat dikunci sebelum mulai)

- **Durasi:** minimal 6 bulan sejak bot paper berjalan.
- **Utama:** C4. **Ikut dicatat:** C5, karena sinyalnya sama dan eksposurnya lebih besar. Paper tidak ada biayanya.
- **Lulus kalau:**
  - return bersih > 0;
  - drawdown terburuk < 20%;
  - hasil paper tidak menyimpang lebih dari 1% per bulan dari backtest periode yang sama
    (cek bahwa bot menjalankan strategi yang benar).
- **Juga dilaporkan:** vs buy-and-hold 8 koin, vs BTC.
- **Kalau gagal:** tidak pakai uang asli, evaluasi ulang. Jangan ubah parameter lalu langsung pakai uang.

## Langkah D: Riset putaran 2 (paralel, dikunci dulu sebelum jalan)

| # | Ide | Kenapa | Catatan |
|---|---|---|---|
| D1 | **Trend long/short di futures** | Paper Zarattini menyebut versi long-short lebih kuat; 2025–2026 adalah pasar turun yang dilewatkan strategi long-only | Butuh fee Bittime resmi (belum terverifikasi); data funding Binance sudah tersedia [TERUKUR] |
| D2 | **Universe lebih luas**: 20 koin paling likuid, termasuk koin yang sudah mati | Mengurangi bias survivorship dan menambah diversifikasi; hasil utama paper memakai 20 koin | Data koin yang sudah dihapus perlu dicek ketersediaannya |
| D3 | **Imbal hasil untuk kas yang menganggur** | C4 rata-rata 89% di kas | Cek apakah Tokocrypto punya produk USDT yang aman; [DUGAAN] |

Penting: data 2018–2026 sudah terlihat. Pemenang putaran 2 tetap **wajib** forward test.

## Langkah E: Dashboard v2 (dievaluasi ulang dengan referensi luar)

### Pembanding yang dipelajari [SUMBER LUAR]

| Referensi | Yang mereka punya |
|---|---|
| **FreqUI** (dashboard bot open-source Freqtrade) | Grafik candle dengan tanda entry/exit, tombol start/stop bot, ringkasan performa semua bot, grafik saldo termasuk PnL belum terealisasi; via Telegram: `/status`, `/profit`, `/daily`, `/performance` per koin, `/forceexit`, `/stopentry` |
| **Hummingbot Dashboard** | Kartu ringkasan (PnL bersih, jumlah trade, akurasi, profit factor); menurut DeepWiki (belum saya cek di kode) juga PnL terealisasi vs belum dan statistik alasan keluar (kena TP / kena SL) |
| **QuantConnect** (laporan statistik) | Total order, rata-rata untung/rugi, CAGR, drawdown, Sharpe, Sortino, win rate, **total fee**, turnover, **lama pemulihan drawdown** |
| **FIA, *Best Practices for Automated Trading Risk Controls* (Juli 2024)** | Batas ukuran order, batas posisi (dengan peringatan saat mendekati batas), toleransi harga, cancel-on-disconnect, **kill switch**, cek kewajaran data pasar (data basi = blokir order), **batas masuk-ulang otomatis**, rekonsiliasi eksekusi exchange vs catatan internal |

### Evaluasi rencana lama (5 tab)

| Aspek | Rencana lama | Penilaian |
|---|---|---|
| Metrik jujur (ekspektasi, PF, R, WR + CI, peringatan n kecil) | Ada | **Tepat**, sesuai QuantConnect/Hummingbot |
| Biaya per trade | Ada | Tepat; tambah **total fee** dan **lama pemulihan drawdown** |
| Status order di exchange + rekonsiliasi | Ada | Tepat, sesuai FIA 4.1 |
| **Tombol kontrol** (stop entry, tutup semua) | **Tidak ada** | Kurang. FreqUI dan FIA 1.5 menjadikannya standar |
| **Login / keamanan** | **Tidak ada** | **Celah nyata [TERUKUR]:** `main.py:649` menjalankan dashboard di `0.0.0.0` tanpa login, jadi saldo, posisi, dan potongan log bisa dibuka siapa pun di jaringan yang sama |
| **Notifikasi HP** | Hanya disebut di Fase 3 | Kurang. Telegram adalah standar FreqUI dan Hummingbot |
| **Panel batas risiko** | Tidak ada | Kurang. FIA 1.2: posisi vs batas, dengan peringatan sebelum tembus |
| **Batas masuk-ulang** | Tidak ada | Kurang. FIA 3.2, dan sudah terbukti perlu: 3 long TSLA masuk ulang dan masing-masing rugi −$526.08 dalam satu jam |
| **Grafik trade** (candle + tanda entry/exit) | Tidak ada | Kurang. Paling mudah dipahami sekilas |
| **Live vs backtest** | Hanya selisih harga masuk | Kurang. Perlu satu panel "hasil nyata vs hasil simulasi periode yang sama" |
| Kalender return harian/bulanan | Tidak ada | Tambahan yang berguna (`/daily` FreqUI) |
| GNN dan Knowledge Graph di baris teratas | Dipindah ke "Lab" | Tepat |

### Desain v2: 6 tab, layar pertama dibuat sesederhana mungkin

| Tab | Isi |
|---|---|
| **1. Status** (layar pertama) | Satu lampu besar: hijau = jalan normal, kuning = ada peringatan, merah = berhenti. Di bawahnya hanya 4 angka: ekuitas, PnL hari ini, drawdown dari puncak, jumlah posisi terbuka. Tombol **Stop entry** dan **Tutup semua & stop** (dengan konfirmasi) |
| **2. Performa** | Kurva ekuitas + drawdown vs buy-and-hold; ekspektasi per trade, profit factor, win rate dengan CI 95%, Sharpe, total fee, lama pemulihan drawdown; peringatan "n < 30, belum bermakna"; kalender return; per koin dan per alasan keluar |
| **3. Posisi & order** | Posisi terbuka, jarak ke stop dalam %, status order di exchange (terpasang/terisi/hilang), hasil rekonsiliasi exchange vs ledger |
| **4. Risiko** | Eksposur vs batas (cap notional 1.0x), batas rugi harian (5%) dengan peringatan di 80%, penghitung masuk-ulang per setup, status data (umur candle terakhir per koin; basi = entry diblokir) |
| **5. Sinyal & grafik** | Grafik candle per koin dengan tanda entry/exit; untuk strategi trend: status sinyal tiap koin (misalnya 6/9 sub-model aktif) dan jarak ke trailing stop; log penolakan |
| **6. Riset** | Live vs backtest periode yang sama, buku catatan percobaan, Deflated Sharpe; GNN dan Knowledge Graph dipindah ke sini |

**Teknis:**

- Tetap FastAPI.
- **Wajib login** (token) atau hanya `127.0.0.1` sebelum tombol kontrol ada.
- Endpoint: `/api/status`, `/api/performance`, `/api/positions`, `/api/risk`, `/api/signals`, `/api/research`, plus `POST /api/control` (terlindungi).
- Grafik pakai TradingView Lightweight Charts (open source, ringan).
- Bot Telegram kecil untuk `/status`, `/daily`, `/stop`, plus notifikasi otomatis kalau proteksi gagal, data basi, atau batas rugi tersentuh.

**Syarat selesai:**

- Setiap angka bisa direproduksi dari ledger dengan satu script.
- Tombol kontrol tidak bisa dipakai tanpa login.
- Kill switch diuji sungguhan di testnet.

## Langkah F: Uang asli (paling akhir)

- **Syarat:** forward test lolos, kill switch diuji di testnet, dan login dashboard aktif.
- **Modal:** C4 ±$900–1.200 supaya setiap perubahan posisi ≥ $5 [DUGAAN, dari order minimum
  $5 yang TERUKUR]. C5 ±$360. Bisa lebih kecil, tapi hasil nyata akan lebih menyimpang dari simulasi.
- **Saklar:** 4 saklar sengaja, sama seperti sekarang. Ditambah batas rugi harian dan
  notifikasi setiap order.

---

## Harapan yang realistis [TERUKUR dari backtest, bukan janji]

| | Tahun bagus | Tahun buruk | Win rate | Drawdown terburuk |
|---|---|---|---|---|
| C4 | +15% s/d +44% | −2% s/d −7% | 10–15% | 13.7% |
| C5 | +42% s/d +392% | −7% s/d −23% | 10–14% | 38.8% |
| Buy & hold 8 koin | +39% s/d +919% | −17% s/d −77% | — | 82.7% |

- Sebagian besar trade **kecil dan rugi**. Untung datang dari sedikit tren besar.
  Bot harus dibiarkan jalan tanpa diganggu.
- Ada periode panjang **tanpa untung**, contohnya seluruh 2025–2026.
- Angka tahun bagus terlalu tinggi karena bias survivorship. Lihat bagian 8 di laporan riset.

---

## Yang perlu kamu putuskan

1. **Pensiunkan SMC sebagai strategi trading?** Saran: ya.
2. **Bangun bot trend harian (Langkah B) dan mulai forward test (Langkah C)?** Saran: ya. Ini jalan tercepat ke bukti yang jujur.
3. **Riset putaran 2 (Langkah D)?** Saran: ya, mulai dari D1, setelah fee Bittime resmi kamu cek.
4. **Dashboard v2 (Langkah E):** dikerjakan bersama bot? Saran: ya, versi sederhana dulu (tab Status + Performa + Posisi).

## Yang tidak akan dilakukan

- Uang asli sebelum forward test lolos.
- Menambah ML.
- Memilih strategi setelah melihat hasil ujian (misalnya RSI2 karena positif di holdout).
- Menyentuh ledger paper lama tanpa perintah langsung darimu.

---

## Kamus singkat

| Istilah | Arti |
|---|---|
| **Trend following** | Beli saat harga menembus ke atas, keluar saat tren patah. Tidak menebak puncak atau dasar |
| **Donchian** | Batas tertinggi dan terendah harga dalam N hari terakhir |
| **Ensemble** | Gabungan beberapa versi (9 lookback) supaya tidak bergantung pada satu angka |
| **Target volatilitas** | Koin yang lebih liar dibeli lebih sedikit |
| **Forward test** | Menjalankan strategi ke depan tanpa uang asli, untuk bukti yang belum pernah dilihat |
| **Holdout** | Data ujian yang disegel sampai pilihan dibuat |
| **Survivorship bias** | Hasil terlalu bagus karena hanya menguji koin yang masih hidup sekarang |
| **Kill switch** | Tombol darurat: hentikan semua order baru dan tutup posisi |
