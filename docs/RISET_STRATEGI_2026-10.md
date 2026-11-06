# Riset Strategi Trading Otomatis (11 Oktober 2026)

> **Kesimpulan:**
>
> 1. **Strategi SMC yang dipakai sekarang adalah yang terburuk** dari 10 strategi, di setiap tes.
>    Dua versinya membuat akun bangkrut di data 2018–2024.
> 2. Keluarga **trend following** (terutama Donchian ensemble, kode C4) adalah yang terbaik
>    di 2018–2024 dan yang paling tahan saat pasar turun di 2025–2026.
> 3. Tapi C4 **rugi −3.9%** di data ujian 2025–2026 (pasar turun −45%). Menurut aturan yang
>    dikunci sejak awal, hasilnya **GAGAL: belum boleh pakai uang asli.**
> 4. Langkah jujur berikutnya: **uji maju (forward test)** C4 dengan paper trading, sambil riset
>    putaran 2. Rencananya ada di [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).

Label: **[TERUKUR]** = saya jalankan/hitung sendiri. **[DUGAAN]** = kesimpulan logis,
belum diuji. **[SUMBER LUAR]** = dari literatur/web.

---

## 1. Cara pengujiannya (supaya hasilnya bisa dipercaya)

- **Aturan dikunci dulu.** Daftar strategi, parameter, periode, biaya, dan syarat lulus
  ditulis di [`research/PREREGISTRATION.md`](../research/PREREGISTRATION.md) dan di-commit
  (`393f758`) **sebelum** satu strategi pun dijalankan.
- **Data:** kline harian Binance untuk 8 koin watchlist, 2017–2026.
- **Dua periode:**
  - **DEV** (pengembangan): 2018-08-17 s/d 2024-12-31. Semua pemilihan dilakukan di sini.
  - **HOLDOUT** (ujian): 2025-01-01 s/d 2026-10-09. Data ini disegel sampai pemenang dipilih
    dan di-commit (`6fcc198`), lalu dibuka **satu kali** (`05688a9`).
- **Eksekusi jujur:** keputusan diambil di harga close, order dieksekusi di harga open hari
  berikutnya. Biaya **0.4322% per sisi** (fee Tokocrypto 0.3822% + slippage 0.05%).
  Tanpa leverage, long saja (spot).
- **Cek kebocoran [TERUKUR]:** sinyal di periode DEV identik bit per bit, baik dihitung
  dengan maupun tanpa data 2025–2026. Tidak ada informasi masa depan yang bocor.
- **Mesin backtest diuji** dengan 8 test otomatis: tidak mengintip masa depan, biaya terpotong
  di dua sisi, tidak pernah ber-leverage.

---

## 2. Riset luar: strategi apa yang punya bukti? [SUMBER LUAR]

| Keluarga strategi | Bukti yang ditemukan | Dipakai sebagai kandidat? |
|---|---|---|
| **Trend following, ensemble Donchian** | Zarattini, Pagani & Barbon (2025), *Catching Crypto Trends*: Sharpe di atas 1.5 setelah fee, alpha ±10.8%/tahun vs BTC. Ringkasan dari kelompok penulisnya (Concretum): BTC sejak 2015 CAGR 30%, Sharpe 1.56, MaxDD 19%; portofolio 20 koin CAGR 18%, MaxDD 11%. Ringkasan pihak ketiga lain menyebut angka berbeda; replikasi independen yang bisa saya verifikasi tidak ditemukan | Ya: C3, C4, C5 |
| **Time-series momentum** | Liu & Tsyvinski (2021), *Review of Financial Studies*: efek time-series momentum yang kuat di crypto | Ya: C2 |
| **Cross-sectional momentum** | Liu, Tsyvinski & Wu (2022), *Journal of Finance*: momentum adalah salah satu dari 3 faktor crypto. Survei arXiv 2510.14435: momentum 2 minggu tetap signifikan setelah 2020 (spread mingguan 2.1%, belum dikurangi biaya, di ratusan koin) | Ya: C6 (hanya 8 koin, jadi lebih lemah) |
| **Filter rata-rata bergerak** | Le & Ruthbah (2024) menguji SMA 20/65/150/200 di BTC/ETH dengan biaya 0.1–0.5% (angka hasilnya tidak berhasil saya akses) | Ya: C1 |
| **Carry / funding rate** (long spot + short perpetual) | Schmeling, Schrimpf & Todorov (BIS WP 1087): carry kadang di atas 40%/tahun. Survei arXiv: Sharpe 6.45 (2020–2025), turun ke 4.06 sejak 2024, negatif di 2025 | Analisis terpisah (butuh venue futures) |
| **Mean reversion** | Tidak ada bukti yang tahan biaya di timeframe harian/mingguan. Di intraday ada efeknya, tapi kecil: ±1.3 bp per trade vs biaya 5 bp (Kitron & Wengrowicz, preprint 2026) | Ya, sebagai pembanding: C7 |
| **SMC / ICT** | Tidak ditemukan studi peer-review. Satu backtest vendor (FVG di Nasdaq) kalah jauh dari buy-and-hold | Ya: C8–C10 (strategi kita sekarang, dengan harga masuk jujur) |

**Pelajaran utama dari literatur:** strategi yang terbukti di crypto adalah strategi
**trading jarang** (harian/mingguan) yang menangkap tren besar. Strategi yang trading
sering habis dimakan biaya, apalagi di biaya kita yang 0.76% bolak-balik.

---

## 3. Hasil DEV, 2018–2024 [TERUKUR]

| Strategi | CAGR | Sharpe | CI 95% Sharpe | MaxDD | Calmar | Win rate | Trade/thn | Ekspektasi/trade | Eksposur rata-rata | Uji acak p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| BH-BTC | +52.3% | 0.97 | [0.16, 1.83] | 76.6% | 0.68 | — | — | — | 100% | n/a |
| BH-EW | +77.8% | 1.13 | [0.25, 2.04] | 82.7% | 0.94 | — | — | — | 100% | n/a |
| C1 SMA200 | +67.9% | 1.25 | [0.35, 2.16] | 55.3% | 1.23 | 21% | 32.1 | +3.87% | 51% | 1.24 |
| C2 TSMOM4W | +95.4% | 1.54 | [0.69, 2.42] | 52.6% | 1.81 | 41% | 35.1 | +3.07% | 52% | 1.20 |
| C3 DONCH-BTC | +20.5% | 1.27 | [0.39, 2.10] | 22.0% | 0.93 | 13% | 9.7 | +2.53% | 19% | 1.12 |
| C4 DONCH-VT | +18.5% | 1.64 | [0.73, 2.53] | 13.7% | 1.35 | 15% | 83.1 | +0.25% | 11% | 1.02 |
| C5 DONCH-EW | +65.7% | 1.56 | [0.65, 2.41] | 38.8% | 1.69 | 14% | 83.1 | +1.18% | 34% | 1.04 |
| C6 XSMOM3 | +81.0% | 1.22 | [0.24, 2.14] | 87.4% | 0.93 | 40% | 47.8 | +2.29% | 67% | 0.99 |
| C7 RSI2 | -0.4% | 0.09 | [-0.32, 0.59] | 28.7% | -0.01 | 60% | 49.7 | +0.01% | 8% | 0.56 |
| C8 SMC-1H | -100.0% | -1.28 | [-2.45, -1.02] | 100.0% | -1.00 | 63% | 316.6 | — | — | n/a |
| C9 SMC-4H | -100.0% | -1.33 | [-1.88, -0.95] | 100.0% | -1.00 | 62% | 265.6 | — | — | n/a |
| C10 SMC-1D | -16.1% | -1.61 | [-2.39, -0.79] | 69.6% | -0.23 | 56% | 93.3 | -0.18% | — | n/a |

Cara membaca:

- **CAGR**: rata-rata pertumbuhan per tahun.
- **Sharpe**: return dibagi risiko; lebih tinggi lebih baik.
- **MaxDD**: penurunan terdalam dari puncak.
- **Calmar**: CAGR dibagi MaxDD.
- **Uji acak p95**: Sharpe yang bisa dicapai dengan waktu masuk **acak** (1.000 kali coba).
  Strategi harus mengalahkannya.

**Pemilihan (aturan yang dikunci):**

| Strategi | CI > 0 | Kalahkan uji acak | MaxDD < 40% | Calmar > buy-and-hold | Lolos? |
|---|---|---|---|---|---|
| C1 SMA200 | ya | ya | **tidak** | ya | tidak |
| C2 TSMOM4W | ya | ya | **tidak** | ya | tidak |
| C3 DONCH-BTC | ya | ya | ya | **tidak** (0.93 vs 0.94) | tidak |
| **C4 DONCH-VT** | ya | ya | ya | ya | **ya, Sharpe tertinggi → TERPILIH** |
| C5 DONCH-EW | ya | ya | ya | ya | ya (Sharpe 1.56) |
| C6 XSMOM3 | ya | ya | tidak | tidak | tidak |
| C7 RSI2 | tidak | tidak | ya | tidak | tidak |
| C8–C10 SMC | tidak | — | tidak | tidak | tidak |

---

## 4. Hasil HOLDOUT, 2025 – Oktober 2026 (dibuka satu kali) [TERUKUR]

| Strategi | CAGR | Sharpe | CI 95% Sharpe | MaxDD | Calmar | Win rate | Trade/thn | Ekspektasi/trade | Eksposur rata-rata |
|---|---|---|---|---|---|---|---|---|---|
| BH-BTC | -6.8% | 0.05 | [-1.30, 1.50] | 53.0% | -0.13 | — | — | — | 100% |
| BH-EW | -28.9% | -0.21 | [-1.49, 1.14] | 70.3% | -0.41 | — | — | — | 100% |
| C1 SMA200 | -14.6% | -0.33 | [-1.68, 1.04] | 42.7% | -0.34 | 19% | 38.9 | -0.12% | 35% |
| C2 TSMOM4W | -8.6% | -0.09 | [-1.50, 1.14] | 43.4% | -0.20 | 38% | 48.0 | -0.10% | 45% |
| C3 DONCH-BTC | -5.5% | -0.47 | [-1.73, 0.81] | 17.9% | -0.31 | 5% | 11.3 | -0.23% | 22% |
| C4 DONCH-VT | -2.2% | -0.28 | [-1.81, 1.00] | 11.0% | -0.20 | 10% | 130.9 | -0.01% | 10% |
| C5 DONCH-EW | -12.1% | -0.63 | [-2.10, 0.67] | 32.0% | -0.38 | 10% | 130.9 | -0.07% | 24% |
| C6 XSMOM3 | -14.1% | -0.09 | [-1.53, 1.23] | 55.5% | -0.25 | 34% | 50.8 | -0.08% | 65% |
| C7 RSI2 | +2.2% | 0.25 | [-0.44, 0.95] | 7.0% | 0.32 | 60% | 43.4 | +0.06% | 5% |
| C8 SMC-1H | -29.8% | -3.40 | [-4.46, -2.27] | 48.3% | -0.62 | 62% | 160.5 | -0.22% | — |
| C9 SMC-4H | -53.8% | -3.76 | [-5.05, -2.49] | 75.4% | -0.71 | 56% | 226.4 | -0.33% | — |
| C10 SMC-1D | -16.9% | -2.22 | [-3.34, -1.08] | 29.0% | -0.58 | 55% | 107.6 | -0.17% | — |

Total selama holdout:

| Strategi | Total | Drawdown terdalam |
|---|---|---|
| Buy & hold 8 koin | **−45.3%** | 70.3% |
| Buy & hold BTC | −11.7% | 53.0% |
| **C4 DONCH-VT (terpilih)** | **−3.9%** | **11.0%** |
| C5 DONCH-EW | −20.4% | 32.0% |
| C2 TSMOM4W | −14.8% | 43.4% |
| C7 RSI2 | +4.0% | 7.0% |
| SMC 1h / 4h / 1d | −46.7% / −74.6% / −28.0% | 48% / 75% / 29% |

**Vonis C4: GAGAL.** Syaratnya untung > 0, dan hasilnya −3.9%.

Yang terjadi: pasar 8 koin turun 45%. C4 melindungi modal (rugi kecil, drawdown 11%)
tapi **tidak menghasilkan untung**. Untuk orang yang alternatifnya memegang USDT,
itu belum cukup.

RSI2 adalah satu-satunya yang positif di holdout (+4.0%), tapi gagal total di DEV
(Sharpe 0.09). Memilihnya sekarang berarti memilih setelah melihat jawaban ujian.
Itu tidak sah, dan kemungkinan besar hanya kebetulan.

---

## 5. Per tahun [TERUKUR]

| Tahun | BH-BTC | BH-EW | C2 TSMOM4W | C4 DONCH-VT | C5 DONCH-EW | C10 SMC-1D |
|---|---|---|---|---|---|---|
| 2018 | -42% | -47% | -17% | -3% | -8% | +1% |
| 2019 | +94% | +39% | +65% | +15% | +42% | -7% |
| 2020 | +302% | +310% | +240% | +35% | +105% | -7% |
| 2021 | +60% | +919% | +854% | +44% | +392% | -15% |
| 2022 | -64% | -77% | -43% | -7% | -23% | -28% |
| 2023 | +156% | +185% | +104% | +22% | +59% | -12% |
| 2024 | +119% | +92% | +40% | +20% | +53% | -30% |
| 2025 | -6% | -34% | +2% | -2% | -14% | -21% |
| 2026 | -6% | -17% | -17% | -2% | -7% | -9% |

2018 hanya dihitung mulai 17 Agustus, dan 2026 sampai 9 Oktober.

**Pola paling penting:**

- Di tahun pasar hancur, C4 hanya turun sedikit: 2018 −3% (pasar −47%), 2022 −7% (pasar −77%).
- Di tahun bagus, C4 naik jauh lebih kecil dari pasar, karena rata-rata hanya **11%** modal
  yang dipakai; sisanya tunai.
- C5 adalah sinyal yang sama dengan eksposur lebih besar: naik dan turunnya lebih besar.

---

## 6. Pelajaran terpenting: win rate tinggi ≠ untung [TERUKUR]

| Strategi | Win rate | Hasil |
|---|---|---|
| SMC 1h (sistem sekarang) | **63%** (DEV), **62%** (holdout) | Bangkrut di DEV, −46.7% di holdout |
| RSI2 (mean reversion) | **60%** | Sharpe 0.09 di DEV |
| C4 Donchian (terbaik) | **15%** | Sharpe 1.64 di DEV, drawdown paling kecil |

Strategi Donchian dan SMA200 hanya menang 13–21% dari trade-nya di DEV. Tapi rata-rata
untungnya **22–56x lebih besar** dari rata-rata ruginya. Contohnya C5: menang +9.77%, kalah
−0.21% per trade.

Mengejar win rate tinggi justru membawa ke strategi yang rugi.

---

## 7. Carry / funding rate (long spot + short futures) [TERUKUR]

| Tahun | BTC carry kotor | ETH carry kotor |
|---|---|---|
| 2021 | 30.6% | 37.5% |
| 2024 | 12.0% | 13.0% |
| 2025 | 5.1% | 4.9% |
| **12 bulan terakhir** | **+3.4%** | **+2.5%** |

Dulu sangat menguntungkan, sekarang tinggal ±3% per tahun **sebelum** biaya dan modal
margin. Tidak sebanding dengan repotnya dua venue.

---

## 8. Batasan dan kejujuran statistik

- **Deflated Sharpe** (koreksi karena mencoba banyak strategi) untuk C4:

  | Cara hitung | DSR | Status |
  |---|---|---|
  | Sesuai pre-registration (10 strategi, varians termasuk SMC) | 0.11 | gagal (< 0.95) |
  | Tanpa SMC (7 strategi) | 0.99 | lolos |
  | Varians dari noise estimasi | 0.995 | lolos |
  | Seluruh proyek (±160 percobaan) | 0.93 | tipis di bawah |

  Hanya baris pertama yang pre-registered. Sisanya sensitivitas. Kesimpulannya: bukti untuk
  C4 **kuat tapi tidak tanpa celah**.
- **Bias survivorship:** 8 koin ini dipilih tahun 2026 karena masih besar. Koin yang mati
  (LUNA, FTT, dan lain-lain) tidak ada. Hasil long-only di DEV terlalu bagus karena ini.
  Benchmark punya bias yang sama.
- **Holdout sudah terpakai.** Riset apa pun berikutnya di data 2018–2026 bukan lagi ujian
  bersih. Bukti baru yang jujur hanya bisa datang dari **waktu ke depan** (forward test).
- **Kontaminasi:** pengetahuan saya mencakup sebagian 2025–2026, dan satu ringkasan web
  menyebut tahun negatif untuk BTC Donchian di 2025. Parameter diambil dari literatur, tidak
  disetel.
- Universe kecil (8 koin), long saja, dan biaya slippage diasumsikan 0.05%.

---

## 9. Kebutuhan modal [DUGAAN, dari data TERUKUR]

Order minimum Tokocrypto **$5** untuk 8 koin ini [TERUKUR, dari API]. Posisi berubah per
1/9 sub-model, jadi supaya setiap perubahan ≥ $5:

- **C4:** modal ±**$900–1.200**
- **C5:** modal ±**$360**

Di bawah itu, sebagian perubahan kecil harus dilewati, dan hasil nyata akan menyimpang
dari simulasi.

---

## Cara cek ulang

```
python -m research.data --intervals 1d 4h 1h        # unduh data (tersimpan lokal)
python -m pytest tests/test_research_engine.py       # 8 test mesin
python -m research.run_comparison dev                # DEV (holdout tidak dimuat)
python -m research.smc_adapter --tf 1h --phase dev   # SMC (juga 4h, 1d, dan --phase holdout)
python -m research.run_comparison select
python -m research.run_comparison holdout
python -m research.carry_analysis
```

Hasil mentah ada di `research/results/`.

## Sumber luar

- [Zarattini, Pagani & Barbon (2025), Catching Crypto Trends](https://abarbon.com/papers/catching-crypto-trends) · [ringkasan Concretum](https://concretumgroup.substack.com/p/catching-crypto-trends)
- [Liu & Tsyvinski (2021), Risks and Returns of Cryptocurrency, RFS](https://ideas.repec.org/a/oup/rfinst/v34y2021i6p2689-2727..html)
- [Liu, Tsyvinski & Wu (2022), Common Risk Factors in Cryptocurrency](https://papers.ssrn.com/abstract=3379131)
- [Cryptocurrency as an Investable Asset Class (arXiv 2510.14435)](https://arxiv.org/abs/2510.14435)
- [Schmeling, Schrimpf & Todorov, Crypto Carry (BIS WP 1087)](https://www.bis.org/publ/work1087.htm)
- [Le & Ruthbah, simple trend rules for crypto (ringkasan CXO)](https://cxoadvisory.com/technical-trading/applying-simple-trend-following-rules-to-cryptocurrencies)
- [Kitron & Wengrowicz (2026), short-horizon mean reversion (arXiv 2608.21888)](https://arxiv.org/pdf/2608.21888)
- [Backtest FVG ICT di Nasdaq (vendor)](https://backtrex.com/en/backtests/ict-fair-value-gap-nasdaq)
- [Pajak kripto PMK 50/2025 (Bisnis.com)](https://ekonomi.bisnis.com/read/20250731/259/1898322/ppn-dihapus-dan-pph-naik-skema-baru-pajak-kripto-berlaku-mulai-1-agustus-2025)
