# Sisa Pekerjaan

Catatan lanjutan untuk repo ini. Semua angka di sini **hasil pengukuran**, bukan perkiraan.
Terakhir diperbarui setelah commit `be4b1dc`.

Status sekarang: `arrangement_v3.flp` **LULUS** `verify.py` — 10 channel, 10 clip,
semua `position=0`, semua `length=29391` tick (229,617 s), nama track cocok,
tempo 80, PPQ 96.

---

## 1. Timbre orkestra masih sintetis — prioritas tertinggi

**Masalah.** Strings, Cello, dan Timpani dibangun dari osilator numpy
(`orchestra.py`): sawtooth detuned + lowpass + vibrato + reverb. Secara harmoni dan
dinamika sudah benar, tapi secara timbre masih terdengar seperti MIDI, bukan ensembel
gesek nyata.

**Kenapa ini yang paling penting.** Struktur lagu sudah dinilai user "sudah mendekati
referensi". Yang tersisa adalah kualitas bunyi. Ini satu-satunya kelemahan yang
langsung terdengar oleh telinga, bukan oleh angka.

**Pilihan perbaikan, dari termurah ke terbaik:**

| Cara | Usaha | Hasil |
|---|---|---|
| Tambah lapisan detune + variasi bow noise di `orchestra.py` | rendah | sedikit lebih hidup |
| Ganti osilator dengan sampel sustain dari SoundFont gratis (mis. VSCO 2 Community Edition, CC0) | sedang | jauh lebih nyata |
| Generate materi orkestra via ElevenLabs Music, lalu stem-separate | sedang | kualitas studio, tapi berbayar |
| Rekam pemain sungguhan | tinggi | terbaik |

**Kalau pakai SoundFont:** yang dibutuhkan cuma satu sampel sustain per nada
(loop), lalu pitch-shift. VSCO 2 CC0 aman untuk komersial. Cello butuh register
C2–C4, strings butuh C3–C6.

**Titik edit:** fungsi sintesis di `orchestra.py`, bagian `strings_note()`,
`cello_note()`, `timpani_hit()`.

---

## 2. Mix 4,4 dB lebih pelan dari referensi

**Angka.** `mix_v3.wav` RMS ≈ −15,6 dB, sedangkan `mix.wav` referensi −11,23 dB.

**Penyebab — ini matematis, bukan bug.** Tujuh stem asli kalau dijumlah apa adanya
sudah memuncak **+0,34 dBFS** (59 sampel melewati 0 dBFS di sekitar bar 72,5).
Untuk memenuhi syarat peak < −1,0 dBFS, atenuasi minimal **1,34 dB** wajib.
Total gain master yang dipakai: −1,50 dBFS peak.

**Pilihan:**

- **Biarkan** — paling aman, mix linear, tidak ada distorsi. Cocok kalau nanti
  di-master ulang di DAW.
- **Limiter** di master — bisa naik ~4 dB, tapi merusak linearitas mix dan
  menambah distorsi intermodulasi. Tidak disarankan untuk arsip.
- **Mastering terpisah** — render stem, master di DAW dengan limiter + loudness
  target (mis. −14 LUFS untuk streaming). Ini jalur yang benar kalau tujuannya rilis.

**Catatan:** peak 0,877 ada di stem Drums, 0,988 di Vocals-lead. Dua ini penyumbang
peak terbesar. Kalau mau mix lebih keras tanpa limiter, turunkan Drums ~1,5 dB dan
Vocals-lead ~0,8 dB lebih dulu, lalu naikkan master.

---

## 3. Chord detection: 45/49 bar cocok

**Angka.** Dari `chords.json`, 77 bar terdeteksi. Evaluasi terhadap template
progresi: **45 dari 49 bar cocok**, cosine median 0,73. Sisanya ambigu dan diambil
dari progresi tetangga.

**Kenapa belum 100%.** Lagu ini banyak padat harmoni (Cmaj7, Csus2, Cadd9 sering
saling tumpang tindih — lihat bar 1–3). Chroma CQT sulit membedakan maj7 vs add9
vs sus2 kalau ada vokal dan gitar fingerstyle bersamaan.

**Kalau mau diperbaiki:** analisis per stem (bukan campuran), khususnya Piano dan
Guitar saja. Piano biasanya paling jelas menyuarakan voicing. `analyze.py` sudah
memisahkan chroma harmonik dari Instrumental — tinggal ganti sumbernya ke stem Piano.

**Dampak ke orkestra:** rendah. Nada orkestra diambil dari root + fifth, jadi
kesalahan maj7/add9 tidak menyebabkan fals — hanya kurang kaya warna.

---

## 4. Word timestamps belum diekstrak

**Status.** Belum dikerjakan. Rencana awal memakai ElevenLabs, tapi dibatalkan
karena alasan hak cipta (lihat bagian 6).

**Cara yang benar: lokal, tanpa API.** Install `faster-whisper`, jalankan pada
stem `05_Vocals-lead.wav` saja (bukan mix — vokal lebih bersih tanpa instrumen).

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel
m = WhisperModel("medium", device="cpu", compute_type="int8")
segs, info = m.transcribe("05_Vocals-lead.wav", word_timestamps=True, language="id")
for s in segs:
    for w in s.words:
        print(f"{w.start:7.2f} {w.end:7.2f} {w.word}")
```

**Kegunaan.** Dengan timestamp per kata, bisa:

- menyelaraskan harmoni vokal background ke suku kata yang tepat
- menaruh aksen timpani di kata tertentu (mis. kata terakhir tiap frasa)
- membuat swell strings yang puncaknya pas di vokal panjang

**Kendala:** Whisper tidak dirancang untuk vokal yang dinyanyikan — akurasi kata
pada nyanyian lebih rendah daripada ucapan. Untuk kebutuhan penyelarasan musikal,
yang penting biasanya cukup batas frasa (bukan tiap kata), dan itu biasanya akurat.

**Alternatif tanpa Whisper:** deteksi onset vokal dari envelope RMS stem
Vocals-lead, lalu ambil batas frasa dari jeda > 300 ms. Lebih kasar tapi tidak
perlu install apa pun.

---

## 5. Struktur lagu: 76 atau 77 bar?

**Temuan.** Ada perbedaan kecil antar sumber:

- `chords.json` dan `analyze.py` memakai **77 bar**
- `stems_manifest.json` mencatat durasi **229,6169 s** = 76,54 bar pada 3,0 s/bar
- `orch_levels.json` memakai `N = 10126110` sampel (= 229,617 s)

**Artinya.** Lagu berakhir di tengah bar ke-77. Bar 77 tidak penuh — sisa 0,54 bar.
Ini bukan error, hanya outro yang fade sebelum bar selesai.

**Yang perlu diputuskan kalau lanjut:** apakah clip di FL sebaiknya 77 bar penuh
(29.568 tick) atau 76,54 bar (29.391 tick, yang dipakai sekarang). Yang sekarang
dipakai adalah panjang audio sebenarnya, jadi tidak ada ekor senyap. **Jangan ubah
tanpa alasan** — 29.391 sudah benar.

---

## 6. Hak cipta — belum selesai, dan ini penghalang untuk rilis

**Situasi.**

- Pencipta lagu: **Nana Harahap**, tercatat di **Aquarius Musicindo**
- Hak distribusi master: **Aquarius Musicindo**
- Yang dipakai proyek ini: rekaman master `CD FORMAT-1.wav`

Menambah orkestra di atas master = **derivative work**. Butuh izin pemegang master,
terpisah dari hak cipta komposisi.

**Yang perlu diminta ke Aquarius — empat hal spesifik, jangan kabur:**

1. Izin membuat *derivative work* dari master "UNTUKMU"
2. Izin *AI processing* (stem separation, sintesis)
3. Batas distribusi: portofolio/demo saja, atau komersial? Platform apa?
4. Pembagian kredit dan royalti antara penulis dan pemegang master

**Selama belum ada izin tertulis:** jangan unggah `mix_v3.wav` ke platform mana pun.
Repo ini aman karena hanya berisi kode dan metadata — rekaman asli tidak disertakan
(`.gitignore` memblokir semua audio).

**Catatan tentang ElevenLabs.** Sempat ditelusuri sebagai kemungkinan integrasi.
Kesimpulan: **tidak dipakai**, karena:

- ToS mereka mewajibkan pernyataan punya hak atas audio yang di-upload
- Endpoint inpainting **mengganti** bagian, bukan menambah lapisan — bukan yang dibutuhkan
- Inpainting hanya jalan di `music_v2` (model lama), tidak di `music_v2_5`
- `conditioning_ref` maksimal 30 s, lagu ini 229,6 s

Yang **aman** dari ElevenLabs: generate materi orkestra **baru dari prompt**
(tanpa upload lagu asli). Lisensi hasil generate diberikan ke pengguna.

---

## 7. Hal teknis yang mudah menjebak kalau lanjut

**pyflp butuh 2 patch** untuk Python 3.12+ (keduanya ada di `build_all.py` dan `verify.py`):

1. `EventEnum` kosong → suntik `_DUMMY = 255`, kalau tidak parsing gagal total
2. `PlaylistEvent.STRUCT` harus 32 byte, kalau tidak `position`/`length` salah baca

**FL Studio menulis ulang .flp saat membuka.** Ini pernah merusak hasil: 5 clip
tergeser (Cello pos 9216, Drums/Bass/Vocals-background pos 12288, Timpani pos 13824)
dengan panjang terpotong 15567–20175 tick.

**Penyebab yang tidak kelihatan:** selain `position`, FL juga mengisi
`start_offset` dan `end_offset` (72000/96000/108000 **milidetik**). Kalau hanya
`position` yang dinolkan, clip akan membaca dari tengah file — senyap di awal, lagu
baru bunyi mulai menit 1:12.

**Kalau ini terjadi lagi**, reset ketiganya. Record playlist 32 byte mulai offset
**9410**, berurutan tiap 32 byte:

```
+0   position      u32   -> 0
+4   pattern_base  u16   -> 20480
+6   item_index    u16   -> iid channel
+8   length        u32   -> 29391
+12  track_rvidx   u16   -> 499 = track 1, turun 1 per track
+14  group         u16
+16  _u1          2 byte
+18  item_flags    u16   -> 64
+20  _u2          4 byte
+24  start_offset  f32   -> -1.0
+28  end_offset    f32   -> -1.0
```

**Panjang clip** = `round(detik × BPM/60 × PPQ)` = `round(229,617 × 80/60 × 96)` = **29.391**.
Salah satu iterasi sempat menulis 293.913 (10× lipat = 38 menit). Selalu cek dengan
`verify.py` sebelum membuka di FL.

---

## 8. Urutan pengerjaan yang disarankan

1. **Dengar `mix_v3.wav` dulu** — kalau keseimbangan belum pas, perbaiki level di
   `orch_levels.json` (bagian `target_offset_db`), jalankan ulang `export_stems.py`.
   Jangan sentuh timbre dulu sebelum keseimbangan disetujui.
2. **Ganti timbre** (bagian 1) — ini yang paling terasa. Butuh sampel nyata.
3. **Word timestamps lokal** (bagian 4) — kalau mau aksen orkestra lebih presisi.
4. **Mastering terpisah** (bagian 2) — hanya kalau mau rilis.
5. **Urus izin Aquarius** (bagian 6) — sebelum unggah ke mana pun.

---

## Cara verifikasi ulang semuanya

```bash
python verify.py arrangement_v3.flp              # cek FLP
python verify.py arrangement_v3.flp mix_v3.wav   # cek FLP + mix (butuh soundfile)
```

Mencetak: tempo, PPQ, jumlah channel/clip, posisi & panjang tiap clip, kecocokan
nama track vs channel, keberadaan file sample, peak, RMS, jumlah klik. Keluar
dengan kode 1 kalau ada yang gagal.

Untuk memeriksa ulang klaim "stem asli tidak diubah", lihat
`stems_manifest.json` — tiap stem asli punya `"sample_identical": true` dan
`"maxdiff": 0.0` terhadap sumbernya.
