# Forward test fund (dikunci 6 November 2026, sebelum ada hasil)

**Strategi:** C4 DONCH-VT dengan setelan risiko **×2** (pilihan pemilik, toleransi
penurunan 25–30%). Sinyal sama dengan riset; ukuran posisi dikali dua, maksimal 25% per
koin dan 100% total modal (tanpa pinjaman).

**Mulai:** 2026-11-06, paper, modal $1.000, ledger `data/fund_ledger_paper.json`.
(Hari pertama sempat dijalankan di ×1; ledger itu diarsipkan sebagai
`data/fund_ledger_paper_x1_archived_2026-11-06.json` dan tidak dihitung.)

## Patokan dari backtest ×2 [TERUKUR, bukan janji]

| | 2018–2024 | 2025–2026 |
|---|---|---|
| Return per tahun | +38,6% | −8,9% total |
| Penurunan terdalam | 26,0% | 21,1% |
| Tahun terbaik / terburuk | +101% / −13,3% | |
| Rata-rata modal di pasar | ~22% | |

Angka 2018–2024 kemungkinan terlalu bagus (8 koin dipilih dari yang masih hidup sekarang;
lihat `docs/RISET_KOIN_2026-11.md`).

## Syarat lulus (sebelum uang asli)

Dievaluasi paling cepat setelah **3 bulan** (2027-02-06), idealnya 6 bulan:

1. Bot menjalankan strategi dengan benar: hasil paper tidak menyimpang lebih dari 1% per
   bulan dari backtest periode yang sama (cek dengan `fund/core.py` di data yang sama).
2. Penurunan terdalam < 30% (batas pemilik; bot berhenti membeli otomatis di 30%).
3. Return bersih ≥ return buy-and-hold BTC di periode yang sama, **atau** penurunan terdalam
   kurang dari setengah penurunan buy-and-hold BTC.

Syarat 3 sengaja tidak menuntut "pasti untung": strategi trend following bisa rugi kecil
berbulan-bulan saat pasar turun atau mendatar. Yang diuji adalah apakah ia melindungi
modal lebih baik daripada hanya memegang BTC.

## Kalau pemilik memilih mulai uang asli lebih cepat

Itu hak pemilik. Syarat minimum: modal yang benar-benar siap direlakan, API key Tokocrypto
tanpa izin withdraw, dan uang asli diperlakukan sebagai forward test-nya sendiri dengan
syarat di atas.

## Tidak boleh

- Mengubah parameter strategi lalu langsung memakai uang asli.
- Menaikkan setelan risiko di atas ×2 tanpa keputusan baru dari pemilik.
