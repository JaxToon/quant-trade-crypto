# Riset intraday: metode regresi tren dari screenshot (6 November 2026)

Permintaan pemilik: coba taktik intra-jam seperti di screenshot TradingView (SOPHUSDT
15m, indikator sudut regresi + Pearson R dengan latar rezim merah/hijau, posisi short),
dengan target 40% per minggu.

Aturan dikunci dulu: [`research/PREREGISTRATION_INTRADAY.md`](../research/PREREGISTRATION_INTRADAY.md).
Kode: `research/intraday_data.py`, `research/intraday.py`. Hasil: `research/results/intraday.json`.

## Cara uji

- 8 koin likuid, harga **perpetual futures Binance asli** 2020–2026, timeframe 15m dan 1h.
- Masuk long kalau tren naik lurus (Pearson R > batas), short kalau tren turun lurus, keluar
  saat rezim berbalik (R melewati 0). Ini inti dari indikator di screenshot.
- Biaya futures 0,10% per sisi (dan 0,15% sebagai uji tekanan) + funding rate asli.
  Biaya ini **jauh lebih murah** dari Tokocrypto spot (0,4322%), jadi hasil ini adalah versi
  yang paling menguntungkan bagi strategi.
- 12 variasi (2 timeframe × 3 panjang jendela × 2 batas R), semuanya dilaporkan.

## Hasil [TERUKUR]

| Variasi | Sharpe 2020–24 | Return/thn 2020–24 | Hasil 2025–26 (biaya 0,15%) | Minggu terbaik | Minggu > +40% |
|---|---|---|---|---|---|
| 1h N96 R0,8 (terbaik) | 1,08 | +16,8% | −19,5% | +9,5% | 0 |
| 1h N96 R0,6 | 0,99 | +16,8% | −20,4% | +10,6% | 0 |
| 1h N48 R0,8 | 0,59 | +7,9% | −9,1% | +8,2% | 0 |
| 1h N24 R0,6 | −0,02 | −1,5% | −38,3% | +7,8% | 0 |
| 15m N96 R0,6 | 0,03 | −0,8% | −34,6% | +8,0% | 0 |
| 15m N24 R0,6 | −4,96 | −52,2% | −86,8% | +5,1% | 0 |

(12 variasi lengkap di `intraday.json`.)

**Keputusan sesuai aturan: DITOLAK.** Variasi terbaik gagal 3 dari 4 syarat: Deflated Sharpe
0,00 (bagusnya di 2020–24 bisa dijelaskan oleh kebetulan dari 12 percobaan), dan rugi −19,5%
dengan Sharpe −0,67 di 2025–26.

## Apa artinya

1. **Makin cepat timeframe-nya, makin parah.** Versi 15 menit dengan jendela pendek rugi
   sampai −52% per tahun: terlalu banyak trade (puluhan ribu), biaya dan sinyal palsu
   memakan semuanya. Ini pola yang sama dengan SMC intraday dulu.
2. **Tidak ada satu minggu pun di atas +40%** di 12 variasi selama hampir 7 tahun
   (sekitar 350 minggu masing-masing). Minggu terbaik yang pernah terjadi +10,6%.
3. Versi 1 jam yang lambat sempat bagus di 2020–24 (mirip trend following harian, tapi
   lebih mahal), lalu rugi di 2025–26. C4 harian tetap lebih baik dan lebih murah.
4. Satu trade SOPH di screenshot yang menang besar tidak mewakili ribuan trade yang
   dihasilkan aturan yang sama.

Sesuai aturan, parameter metode ini tidak dioprek lagi di data yang sama.
