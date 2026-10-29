# Kulproy AI Music Studio

Toolkit untuk **menambah lapisan orkestra sinematik di atas lagu yang sudah jadi**, tanpa memotong, tanpa mengubah urutan, tanpa menyentuh satu sampel pun dari rekaman aslinya.

Proyek studi kasus: *"NANA HARAHAP — UNTUKMU"* (80 BPM, C major, 77 bar, 229,62 detik) diberi lapisan string ensemble, cello, dan timpani bergaya *The Sound of Silence* (versi Disturbed).

---

## Prinsip

1. **Lagu asli tidak diubah.** Tidak ada pemotongan, tidak ada penyambungan ulang, tidak ada normalisasi ulang stem asli. Tujuh stem asli disalin sampel-identik (`maxdiff = 0.000e+00`).
2. **Semua keputusan berbasis angka.** BPM diukur, chord per bar dideteksi, dinamika ditentukan dari RMS per bagian. Tidak ada tebakan.
3. **Verifikasi bisa diulang.** Setiap langkah punya skrip pemeriksa yang mencetak angka, bukan klaim.

---

## Struktur

```
orchestra.py        Sintesis 3 stem orkestra (Strings, Cello, Timpani)
                    -> numpy murni: osilator, ADSR, reverb, stereo spread
export_stems.py     Ekspor 10 stem sinkron + mix, gain staging per bagian
make_v3.py          Bangun .flp: 10 channel, 10 clip, track bernama
build_all.py        Utilitas .flp: patch pyflp, buang channel, tulis playlist
analyze.py          Deteksi chord per bar dari chroma CQT + chroma bass
song.py             Mesin komposisi awal (osilator + ADSR + ekspor MIDI)

chords.json         Chord per bar hasil deteksi librosa (77 bar)
orch_levels.json    Level RMS tiap stem per bagian (A-E)
orch_meta.json      Metadata sintesis: tuning, divisi, envelope
stems_manifest.json Manifest 10 stem: path, panjang, checksum

arrangement_clean.flp   Template: 7 track, clip di bar 1
arrangement_v3.flp      Hasil akhir: 10 track + orkestra
```

---

## Peta dinamika

Mengikuti pola *Sound of Silence* (Disturbed) — dari nyaris senyap ke klimaks lalu turun.

| Bagian | Bar | Waktu | Lagu | Strings | Cello | Timpani |
|---|---|---|---|---|---|---|
| A — intro/verse intim | 1–17 | 0:00–0:51 | −15,4 dB | −34,4 | senyap | senyap |
| B — naik | 18–33 | 0:51–1:39 | −13,5 | −25,5 | −27,5 | senyap |
| C — band masuk | 34–48 | 1:39–2:24 | −11,0 | −19,0 | −20,0 | −25,0 |
| **D — klimaks** | 49–74 | 2:24–3:42 | −10,4 | **−15,9** | **−16,9** | **−18,4** |
| E — outro | 75–77 | 3:42–3:49 | −19,2 | −28,2 | −29,2 | −31,2 |

String pad masuk sangat tipis di intro (drone satu oktaf rendah), lalu naik bertahap. Timpani baru muncul di bagian C dan memukul tiap dua bar di klimaks. Outro turun ke nada panjang dan decay mengikuti fade lagu.

---

## Cara pakai

Butuh Python 3.12+ dengan `numpy`, `scipy`, `soundfile`, `librosa`, dan `pyflp` (untuk .flp).

```bash
# 1. Siapkan 7 stem asli di song/stems/ (dari stem separation)
#    Format nama: "<Nama> - <Judul> - 80bpm - Cmaj.wav"

# 2. Deteksi chord per bar
python analyze.py            # -> chords.json

# 3. Sintesis orkestra
cd song && python orchestra.py    # -> out3/stems/08_Strings.wav, 09_Cello, 10_Timpani

# 4. Ekspor 10 stem + mix
python export_stems.py            # -> out3/stems/01..10_*.wav + mix_v3.wav

# 5. Bangun project FL Studio
python make_v3.py                 # -> song/arrangement_v3.flp
```

---

## Catatan teknis

**pyflp butuh dua patch** untuk Python 3.12+ — keduanya sudah ada di `build_all.py`:

1. `EventEnum` kosong di Python modern → disuntik anggota `_DUMMY = 255` supaya parsing tidak gagal.
2. `PlaylistEvent.STRUCT` didefinisikan 32 byte (bukan default) supaya `position`, `length`, `track_rvidx` terbaca benar.

**Panjang clip** = `round(durasi_detik × BPM/60 × PPQ)`. Untuk 229,62 s pada 80 BPM / PPQ 96 → **29.391 tick**. Salah satu iterasi sempat menulis 293.913 (10× lipat, jadi 38 menit) — sudah dikoreksi.

**`track_rvidx`** 499 = track 1, turun 1 per track. **`item_index`** = iid channel.

**Nama track** disisipkan sebagai `UnicodeEvent(TrackID.Name, utf-16-le)` tepat **setelah** event `TrackID.Data` milik track itu. Menyisipkan di indeks 0 membuat semua nama bergeser satu track.

---

## Verifikasi

```bash
python verify.py
```

Mencetak:

- 10 channel, 10 clip, semua `position = 0`, `length = 29391`
- nama track cocok dengan nama channel
- tempo 80, PPQ 96
- semua path sample benar-benar ada di disk
- korelasi `(mix − orkestra)` vs jumlah 7 stem asli
- peak mix, deteksi klik

Angka acuan hasil terakhir:

| Metrik | Nilai |
|---|---|
| Panjang | 229,62 s (identik semua stem) |
| Peak mix | −1,50 dBFS |
| Stem asli vs sumber | `maxdiff 0.000e+00` (identik) |
| Chord terdeteksi | 77 bar, cosine median 0,73 |

---

## Yang belum sempurna

- **Timbre orkestra masih sintetis.** Strings dibangun dari osilator sawtooth detuned + lowpass + vibrato — belum sampel nyata. Ini kelemahan terbesar aransemen sekarang.
- **Mix lebih pelan 4,4 dB** dari referensi. Penyebabnya matematis: tujuh stem asli kalau dijumlah saja sudah peak +0,34 dBFS, jadi untuk mencapai <−1 dB harus ada atenuasi ≥1,34 dB. Kalau mau lebih keras, perlu limiter — dan itu merusak linearitas mix.
- **Chord detection 45/49 bar cocok** dengan template progresi; sisanya ambigu dan diambil dari progresi tetangga.

Rincian lengkap setiap masalah — angka terukur, penyebab, dan langkah perbaikannya —
ada di **[TODO.md](TODO.md)**. Baca itu dulu sebelum melanjutkan.

---

## Lisensi & hak cipta

Kode di repo ini bebas dipakai. **Rekaman asli tidak disertakan** dan tidak boleh didistribusikan tanpa izin pemegang master.

Lagu *"UNTUKMU"* — pencipta: Nana Harahap (tercatat di Aquarius Musicindo); hak distribusi dipegang Aquarius Musicindo. Menambah aransemen di atas master adalah *derivative work* dan butuh izin pemegang master, terpisah dari hak cipta komposisi.
