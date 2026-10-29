# Skrip analisis & verifikasi

Semua skrip di sini **mengukur**, tidak mengubah apa pun. Dipakai untuk membuktikan
klaim di README dan TODO dengan angka, bukan perkiraan.

Butuh: `numpy`, `scipy`, `soundfile`, `librosa`.
Jalankan dari folder `song/` (skrip memakai path relatif ke `stems/` dan `out3/`).

| Skrip | Menjawab pertanyaan |
|---|---|
| `verify_orchestra.py` | Apakah dinamika orkestra mengikuti peta bagian A–E? Apakah nadanya ikut chord? |
| `verify_stems.py` | Apakah 7 stem diskret kalau dijumlah = mix asli? Berapa RMS per bagian? |
| `tempo_check.py` | Apakah lagu benar-benar 80,000 BPM? Berapa drift per bar? |
| `grid_fit.py` | Di mana posisi beat presisi? Fit `t_n = t0 + n·P` dari onset drum+bass. |
| `sync_check.py` | Apakah grid lagu dan grid orkestra benar-benar sejajar? |
| `feasibility.py` | Mungkinkah peak < −1,0 dBFS **dan** korelasi > 0,98 tercapai bersamaan? |
| `peak_check.py` | Di mana sampel yang memuncak? Stem mana penyumbangnya? |
| `click_check.py` | Ada sampel loncat (klik) di mix? |
| `sum7_check.py` | Berapa peak dari penjumlahan 7 stem asli saja, tanpa orkestra? |
| `gain_optimize.py` | Struktur gain terbaik: crest orkestra ditekan, mix tetap linier. |
| `stem_diag.py` | Energi per bar per stem, onset pertama, uji chord dari Piano saja. |
| `analyze_alt.py` | Versi alternatif deteksi chord (chroma harmonik + constraint bass). |

## Angka acuan

Hasil pengukuran terakhir, untuk membandingkan kalau ada yang berubah:

| Metrik | Nilai | Skrip |
|---|---|---|
| Tempo | 80,000 BPM (drift 0,086 ms/beat) | `tempo_check.py` |
| Grid offset | 0,065 s | `grid_fit.py` |
| Peak 7 stem asli saja | **+0,34 dBFS** (59 sampel > 0 dB) | `sum7_check.py` |
| Peak mix_v3 | −1,50 dBFS | `peak_check.py` |
| Korelasi stem asli vs sumber | `maxdiff 0.000e+00` | `verify_stems.py` |
| Chord cocok | 45/49 bar, cosine median 0,73 | `analyze_alt.py` |

## Catatan

`peak +0,34 dBFS` dari tujuh stem asli itu temuan penting: artinya syarat
peak < −1,0 dBFS **secara matematis mewajibkan** atenuasi ≥1,34 dB. Jadi mix yang
4,4 dB lebih pelan dari referensi bukan bug — itu konsekuensi dari syarat tersebut.
Lihat `TODO.md` bagian 2.
