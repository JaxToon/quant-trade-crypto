# Riset koin untuk C4 (6 November 2026)

Pertanyaan: apakah 8 koin sekarang (BTC, ETH, BNB, XRP, ADA, SOL, AVAX, DOT) yang terbaik
untuk strategi C4, atau ada koin lain yang lebih baik?

Aturan ujian dikunci sebelum ada hasil: [`research/PREREGISTRATION_UNIVERSE.md`](../research/PREREGISTRATION_UNIVERSE.md).
Kode: `research/universe_data.py` (data), `research/universe_study.py` (ujian).
Hasil mentah: `research/results/universe.json`.

## Cara mengujinya supaya jujur

- Aturan C4 tidak diubah sama sekali. Yang diganti hanya daftar koinnya.
- Setiap daftar dipilih **hanya dengan data yang tersedia saat itu**. Memilih koin karena
  backtest-nya paling bagus sama dengan curang: kita sudah tahu koin mana yang naik.
- Kumpulan koinnya adalah **semua 691 pair USDT yang pernah listing di Binance**, termasuk yang
  sudah mati atau delisting (LUNA lama, BCC, VEN, FTT, MATIC, EOS). Tanpa koin mati, hasil
  hanya melihat koin yang selamat.
- Biaya, eksekusi, dan periode sama dengan riset C4 sebelumnya.

## Hasil [TERUKUR]

| Universe | Cara memilih | Sharpe DEV | CAGR DEV | Drawdown DEV | Sharpe holdout | Return holdout |
|---|---|---|---|---|---|---|
| **U8 (sekarang)** | tetap | **1,64** | **+18,5%** | 13,7% | −0,28 | −3,9% |
| T10 | 10 koin paling likuid, dipilih ulang tiap bulan | 1,20 | +10,4% | 15,1% | −0,02 | −0,6% |
| T20 | 20 koin paling likuid, dipilih ulang tiap bulan | 1,00 | +7,2% | 13,8% | −0,23 | −2,5% |
| WF4 | 4 terbaik dari 8 koin, berdasarkan hasil C4 setahun terakhir | 1,48 | +16,5% | 13,9% | −0,10 | −2,2% |
| WF8 | 8 terbaik dari 20 terlikuid, berdasarkan hasil C4 setahun terakhir | 0,89 | +7,3% | 15,6% | −0,09 | −1,4% |

DEV = 2018-08-17 s/d 2024-12-31. Holdout = 2025-01-01 s/d 2026-10-09.

**Keputusan sesuai aturan: tetap pakai 8 koin sekarang.** Tidak ada kandidat yang punya Sharpe
lebih tinggi di kedua periode.

## Apa artinya

1. **Memilih koin "terbaik" dari hasil masa lalu tidak membantu.** WF4 dan WF8 memilih koin yang
   performa C4-nya paling bagus setahun terakhir, dan hasilnya lebih buruk dari daftar tetap di
   periode DEV. Koin yang tren-nya bagus tahun ini belum tentu bagus tahun depan.
2. **Menambah koin juga tidak membantu.** Dengan 10 atau 20 koin, modal per koin makin kecil,
   sementara koin tambahan (yang lebih kecil dan lebih liar) tidak menambah tren yang bisa ditangkap.
3. **Angka backtest U8 kemungkinan terlalu bagus [DUGAAN, alasannya terukur].** 8 koin itu
   dipilih dari koin besar yang *masih hidup sekarang*. T10/T20 dipilih tanpa tahu masa depan,
   dan hasil DEV-nya lebih rendah (CAGR +7% s/d +10%, Sharpe 1,0 s/d 1,2). Harapan yang lebih
   realistis untuk C4 ke depan ada di kisaran itu, bukan +18,5%.
4. Di holdout semua versi tetap negatif. Belum ada daftar koin yang membuat C4 terbukti untung;
   forward test paper tetap jadi penentu.

## Keterbatasan

- Koin yang delisting saat dipegang dihitung keluar di harga terakhirnya. Kenyataannya harga
  biasanya jatuh setelah pengumuman delisting, jadi hasil T10/T20/WF8 sedikit terlalu bagus.
- 2 meme coin bernama Mandarin tidak terunduh (baru listing, belum punya 365 hari data, jadi
  tidak mungkin terpilih).
- Holdout sudah pernah dilihat sekali (keputusan C4), jadi di sini hanya pemeriksaan kedua.
