# Rencana Improvement TraderStuff (dibuat 11 Oktober 2026)

Dari fondasi sampai hasil akhir: **profit yang terbukti**, **teknik trade yang lebih
baik**, dan **dashboard yang jujur dan enak dipakai**.

Dasar rencana ini adalah [`EVALUASI_2026-10-11.md`](EVALUASI_2026-10-11.md).
Label yang dipakai sama: **[TERUKUR]**, **[DUGAAN]**, **[SUMBER LUAR]**.

---

## Tujuan akhir (kapan proyek ini disebut "berhasil")

1. Ada strategi yang **untung setelah semua biaya, di harga pasar asli**, dan terbukti
   di data yang belum pernah dilihat (holdout).
2. Hasil trading nyata **cocok** dengan hasil simulasi.
3. Dashboard menampilkan angka yang **jujur**: setelah biaya, dengan pembanding, dengan
   peringatan kalau sampelnya masih kecil.
4. Uang asli masuk **bertahap**: kecil dulu, dengan pengaman otomatis.

---

## 3 aturan main

1. **Ekspektasi per trade setelah biaya** adalah ukuran utama, bukan win rate.
   Buktinya [TERUKUR]: market order dengan win rate 63.1% tetap rugi $26,573.
2. **Satu hipotesis, ditulis dulu, baru diuji.** Data holdout dibuka satu kali per
   kandidat final.
3. **Kalau tidak ada yang lolos, kita berhenti.** "Tidak ada edge" adalah hasil yang sah,
   dan hasil itu menyelamatkan uang.

---

## Peta fase

| Fase | Isi | Syarat lulus (gate) | Ukuran kerja |
|---|---|---|---|
| 0 | Fondasi: timbangan yang jujur | Mesin baru mereproduksi hasil "rugi di harga pasar" | Sedang |
| 1 | Teknik trade: cari edge | Untung di holdout, CI di atas nol, mengalahkan entry acak | Besar |
| 2 | Turunkan biaya: venue dan fee | Fee terverifikasi, ada stop di exchange | Kecil–sedang |
| 3 | Eksekusi nyata bertahap | ≥ 30 trade nyata cocok dengan simulasi | Sedang + waktu |
| 4 | Dashboard baru | Semua angka bisa direproduksi dari ledger | Sedang |
| 5 | Sederhanakan arsitektur | Keputusan kamu | Kecil |

Fase 4 dan 5 bisa dikerjakan **paralel** dengan Fase 1. Fase 3 hanya boleh dimulai
setelah Fase 1 lulus.

---

## Fase 0: Fondasi, "timbangan yang jujur"

Kenapa paling dulu: semua angka setelahnya bergantung pada ini. Selama ini timbangannya
salah ±0.5R per trade [TERUKUR].

| # | Pekerjaan | Di mana |
|---|---|---|
| 0.1 | **Model harga masuk yang benar.** Default: masuk di harga pasar saat sinyal (plus slippage). Opsi kedua: limit order yang hanya terisi kalau harga benar-benar menembusnya. Stop dan target dihitung dari harga masuk yang **asli**. | `backtrack/paper_trading.py` (`maybe_open`), `strategy/entry_insight.py` |
| 0.2 | **Biaya realistis jadi default:** fee venue (`VENUE_FEE_PCT` sudah ada), slippage stop 0.119% [TERUKUR], slippage entry diukur. | `config/settings.py` |
| 0.3 | **Data dari Binance sendiri, disimpan lokal.** Kline 5m/15m/1h/4h/1d, 8 koin, Januari 2022 sampai sekarang, dari `data-api.binance.vision` (sudah dicek bisa [TERUKUR]). Backtest pakai **tanggal tetap**, bukan "730 hari terakhir", supaya hasilnya bisa diulang. | script fetch baru, folder `data/klines/` |
| 0.4 | **Holdout dikunci:** 6 bulan terakhir (mis. 1 April 2026 sampai sekarang) tidak boleh disentuh selama pengembangan. Program menolak membacanya kecuali dengan flag `--final`. | konfigurasi backtest |
| 0.5 | **Buku catatan percobaan otomatis:** setiap backtest mencatat config dan hasilnya. Jumlah percobaan dipakai untuk menghitung Deflated Sharpe. | file `research/trials.jsonl` |
| 0.6 | **Pembanding wajib di setiap laporan:** (a) buy-and-hold 8 koin, (b) **entry acak** dengan stop, target, dan biaya yang sama, diulang ±1.000 kali. Strategi harus lebih baik dari entry acak, bukan cuma lebih baik dari nol. | modul benchmark baru |
| 0.7 | **Test regresi:** harga masuk tidak boleh lebih bagus dari harga pasar; biaya selalu terpotong. | `tests/` |

**Gate 0:** mesin baru mereproduksi hasil eksperimen fill realism (market order rugi),
semua test lulus, dan pembanding entry acak sudah jalan.

---

## Fase 1: Teknik trade, mencari strategi yang benar-benar untung

Semua kandidat ditulis dulu (pre-registered), diuji di data pengembangan, lalu yang
terbaik dibuka sekali di holdout. **Maksimal 5 kandidat** sebelum kita berhenti dan
evaluasi ulang.

| # | Kandidat | Kenapa | Label |
|---|---|---|---|
| 1.1 | **SMC/FVG versi jujur** di 1h, 4h, dan 1d: masuk di harga pasar, target minimal **3x biaya bolak-balik** | Timeframe besar bergerak lebih jauh dibanding biaya. Trade kita sekarang targetnya cuma ±0.63% padahal biaya 0.76% [TERUKUR] | DUGAAN |
| 1.2 | **Tren harian + entry SMC:** LONG hanya kalau tren harian naik (mis. harga di atas rata-rata 50 hari), lalu masuk di FVG | Tidak melawan arus besar | DUGAAN |
| 1.3 | **Time-series momentum harian** (tanpa SMC): beli kalau return 4–12 minggu positif, keluar kalau berbalik, ukuran posisi disesuaikan volatilitas | Satu-satunya kandidat dengan bukti akademik di crypto (Liu & Tsyvinski 2021) | SUMBER LUAR |
| 1.4 | **Uji ulang filter yang dulu "ditolak"** (bias timeframe besar, EMA50, regime) di atas kandidat 1.1 | Bias lama menghukum filter yang mengurangi jumlah trade | DUGAAN |
| 1.5 | **Exit dalam R dari harga masuk asli** (1.5R / 2R / 3R) plus batas waktu | Matriks target lama diukur di harga tengah gap | DUGAAN |

Aturan tambahan:

- **Sizing:** tetap risk 0.25% per trade + cap notional 1.0x (sudah ada). Volatility targeting diuji sebagai opsi. [DUGAAN]
- **Arah:** di spot hanya LONG. SHORT hanya kalau nanti pakai futures **dan** SHORT terbukti sendiri.
- **Timeframe live** hanya yang lulus uji. 5m/15m tidak boleh live tanpa uji multi-tahun.

**Gate 1** (semua harus terpenuhi):

1. Di holdout: ekspektasi per trade setelah biaya > 0, batas bawah CI 95% (block bootstrap) > 0.
2. Lebih baik dari 95% hasil entry acak.
3. Untung di ≥ 4 dari 6 periode data pengembangan.
4. Drawdown terburuk < 20%.
5. Deflated Sharpe dilaporkan dengan jumlah percobaan yang jujur.

Kalau tidak ada kandidat yang lolos: **berhenti, jangan pakai uang asli.**

---

## Fase 2: Turunkan biaya (tuas profit terbesar setelah ada edge)

| # | Pekerjaan | Catatan |
|---|---|---|
| 2.1 | Cek **jadwal fee resmi Bittime futures** | Satu sumber pihak ketiga menyebut taker ±0.054% / maker ±0.024%, tapi datanya tidak konsisten. **Belum terverifikasi.** Kalau benar, biaya ±7x lebih murah dari Tokocrypto spot. [SUMBER LUAR] |
| 2.2 | Jangan berharap fee murah saja menyelamatkan strategi lama | Di fee 0.1%, market order tetap rugi −$54,728, 0/6 periode [TERUKUR] |
| 2.3 | Kalau pakai futures: leverage 1–2x saja, hitung funding dan likuidasi di backtest dulu | DUGAAN |
| 2.4 | Lebih sedikit trade, gerakan lebih besar per trade | Biaya dibayar per trade |

**Gate 2:** venue punya stop di exchange, fee terverifikasi, API teruji.

---

## Fase 3: Eksekusi nyata, bertahap

| # | Langkah | Syarat |
|---|---|---|
| 3.1 | Jalankan di **testnet** dengan ledger yang sudah diperbaiki; bandingkan harga isi order vs ledger | Setelah Fase 0 |
| 3.2 | Lengkapi pengaman: batas rugi harian (sudah ada, 5%), tombol stop darurat, batas jumlah posisi, **notifikasi** (mis. Telegram) kalau proteksi gagal | — |
| 3.3 | **Uang asli kecil** (jumlah yang siap hilang), 4 saklar dinyalakan sengaja | Setelah Gate 1 dan Gate 2 |
| 3.4 | Jalankan ≥ 30 trade, bandingkan dengan backtest periode yang sama | — |
| 3.5 | Naikkan modal **bertahap** hanya kalau live cocok dengan backtest | — |

**Gate 3:** ≥ 30 trade nyata, hasil di dalam rentang backtest, tidak ada insiden posisi
tanpa proteksi.

---

## Fase 4: Dashboard baru

**Kondisi sekarang [TERUKUR, dari kode]:** satu file HTML 827 baris; dashboard crypto dan
saham terpisah di dua port (8080/8081); dua baris teratas diisi regime, sinyal GNN, dan
Knowledge Graph (Neo4j), padahal regime dan graph tidak dipakai untuk keputusan Path 1 dan
GNN hanya bisa memveto di ±1.1% panggilan (sisanya terblokir threshold); PnL yang
tampil adalah PnL "dicatat" (dengan diskon); kata fee, slippage, benchmark, confidence,
dan OCO tidak muncul sama sekali di halamannya.

**Desain baru, 5 tab:**

| Tab | Isi |
|---|---|
| **Ringkasan** | Kurva ekuitas dan drawdown; vs buy-and-hold periode yang sama; PnL hari ini, minggu ini, bulan ini; status sistem satu baris (hijau/kuning/merah) |
| **Performa jujur** | Ekspektasi per trade, profit factor, rata-rata R, win rate **dengan CI 95%**, jumlah trade plus peringatan kalau n < 30; dipecah per arah, timeframe, dan koin; biaya per trade (fee + slippage + selisih harga masuk) |
| **Posisi & order** | Posisi terbuka, jarak ke stop/target dalam R, status OCO di exchange (terpasang / terisi / hilang), hasil rekonsiliasi |
| **Sinyal & penolakan** | Sinyal terbaru dan alasan ditolak (datanya sudah ada di `rejection_history`) |
| **Kesehatan & riset** | Kesegaran data per koin, koneksi WebSocket, proses, watchdog; buku catatan percobaan dan Deflated Sharpe |

**Teknis [DUGAAN, bisa diganti]:**

- Tetap FastAPI. Endpoint baru: `/api/summary`, `/api/performance`, `/api/positions`, `/api/health`, `/api/research`.
- **Satu dashboard** untuk crypto dan saham.
- Grafik pakai library chart (mis. TradingView Lightweight Charts), mode gelap, bisa dibuka di HP.
- GNN dan Neo4j dipindah ke halaman "Lab".

**Syarat selesai:** setiap angka di dashboard bisa direproduksi dari ledger dengan satu
script, dan tidak ada angka "dicatat" yang tampil tanpa label.

---

## Fase 5: Sederhanakan arsitektur (keputusan kamu)

| Komponen | Fakta | Usul |
|---|---|---|
| Neo4j | Wajib hidup untuk start (langkah 1 di `main.py`), watchdog tidak bisa menyalakannya; ke Path 1 hanya lewat GNN [TERUKUR] | Jadikan **opsional**: sistem tetap jalan tanpa Neo4j |
| GNN | 98.9% terblokir; tanpa threshold pun tidak menambah edge [TERUKUR] | Keluarkan dari loop trading, kode disimpan |
| LLM Path 2 | 0 trade dari 3,050+ keputusan [TERUKUR] | Keluarkan dari loop trading, kode disimpan |
| Saham | Belum ada broker yang bisa dipakai | Bekukan sampai ada jalur eksekusi |

Hasil yang diharapkan [DUGAAN]: start lebih cepat, lebih sedikit titik gagal, biaya
OpenAI nol. Sesuai juga dengan keputusanmu 22 Agustus 2026: fokus ke sistem inti, tanpa ML.

---

## Yang perlu kamu putuskan

1. **Mulai Fase 0?** Rekomendasi: ya, ini prasyarat semuanya.
2. **Fase 5** (Neo4j opsional, GNN dan LLM keluar dari loop, kode disimpan)? Rekomendasi: ya.
3. **Paper trading tetap mati sampai Fase 0 selesai?** Rekomendasi: ya. Ledger lama masih
   mencatat dengan diskon, jadi data barunya tetap bias.
4. **Fee Bittime:** kalau kamu punya akun, cek halaman fee resminya. Itu informasi
   termurah yang bisa mengubah Fase 2.

## Yang TIDAK akan dilakukan

- Uang asli sebelum Gate 1 dan Gate 3 lulus.
- Menambah ML (sesuai keputusanmu).
- Mengejar win rate tinggi dengan mengorbankan ekspektasi.
- Mengubah ledger paper tanpa perintah langsung darimu.

---

## Kamus singkat

| Istilah | Arti |
|---|---|
| **R** | Jarak dari harga masuk ke stop. "+2R" artinya untung 2x dari risiko |
| **Ekspektasi** | Rata-rata hasil per trade setelah biaya. Harus positif |
| **Biaya bolak-balik** | Fee masuk + fee keluar. Tokocrypto: 0.76% |
| **Holdout** | Data yang disimpan dan baru dibuka di akhir, seperti soal ujian |
| **Overfitting** | Strategi terlihat bagus karena terlalu sering dicoba di data yang sama |
| **Entry acak** | Pembanding: masuk di waktu acak dengan aturan keluar yang sama. Kalau strategi tidak mengalahkannya, sinyalnya tidak berguna |
| **Drawdown** | Penurunan dari puncak modal. 100% berarti modal habis |

## Sumber luar

- [Liu & Tsyvinski (2021), Risks and Returns of Cryptocurrency, RFS 34(6)](https://ideas.repec.org/a/oup/rfinst/v34y2021i6p2689-2727..html): time-series momentum yang kuat di crypto
- [TradersUnion, fee Bittime (pihak ketiga, tidak konsisten, belum terverifikasi)](https://tradersunion.com/brokers/crypto/view/bittime/fees/)
