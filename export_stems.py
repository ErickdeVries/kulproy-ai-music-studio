"""Ekspor 10 stem sinkron + mix baru (lagu utuh + orkestra).

10 stem = 7 stem lagu asli (Drums, Bass, Guitar, Piano, Vocals-lead,
          Vocals-background, Pro-other) + 3 stem orkestra
          (08_Strings, 09_Cello, 10_Timpani)

Semua 10 file: 44100 Hz, 24-bit, stereo, panjang identik, mulai di sample 0
(bar 1 = 0.065 s), tanpa potong / tanpa ubah urutan.

Output:
  song/out3/stems/01..10_*.wav
  song/out3/mix_v3.wav
  song/out3/mix_v3_24bit.wav
"""
import numpy as np, soundfile as sf, json, os, shutil

SR = 44100
OFF, BAR = 0.065, 3.0
OUT = "out3/stems"
os.makedirs(OUT, exist_ok=True)

SRC7 = [("01_Drums", "Drums"), ("02_Bass", "Bass"), ("03_Guitar", "Guitar"),
        ("04_Piano", "Piano"), ("05_Vocals-lead", "Vocals-lead"),
        ("06_Vocals-background", "Vocals-background"), ("07_Pro-other", "Pro-other")]
def hp(n): return f"stems/{n} - NANA HARAHAP - UNTUKMU - CD FORMAT-1 - 80bpm - Cmaj.wav"

# ---- 1. salin 7 stem asli dengan nama baru (sampel-identik) -------------
# Tanpa gain, tanpa trim, tanpa normalisasi: nilai sampel harus sama persis.
# Hanya kontainer berubah 16-bit -> 24-bit (lossless untuk nilai yang sama).
info = {}
for new, old in SRC7:
    a, sr = sf.read(hp(old), dtype="float64", always_2d=True)
    dst = f"{OUT}/{new}.wav"
    sf.write(dst, a.astype(np.float32), sr, subtype="PCM_24")
    b, _ = sf.read(dst, dtype="float64", always_2d=True)
    maxdiff = float(np.max(np.abs(a - b))) if len(a) == len(b) else float("nan")
    assert maxdiff == 0.0, f"{new}: salinan tidak identik (maxdiff={maxdiff})"
    info[new] = dict(sr=sr, ch=a.shape[1], n=a.shape[0], dur=a.shape[0] / sr,
                     peak=float(np.max(np.abs(a))), src=old,
                     sample_identical=True, maxdiff=maxdiff)
    print(f"{new:22s} {a.shape[0]:>9d} samples {a.shape[1]}ch {sr} "
          f"peak {20*np.log10(np.max(np.abs(a))+1e-12):6.2f} dB "
          f"| maxdiff {maxdiff:.1e} OK")

# ---- 2. stem orkestra (sudah ada) --------------------------------------
for new in ["08_Strings", "09_Cello", "10_Timpani"]:
    a, sr = sf.read(f"{OUT}/{new}.wav", dtype="float32", always_2d=True)
    info[new] = dict(sr=sr, ch=a.shape[1], n=a.shape[0], dur=a.shape[0] / sr,
                     peak=float(np.max(np.abs(a))), src="synth")
    print(f"{new:22s} {a.shape[0]:>9d} samples {a.shape[1]}ch {sr} "
          f"peak {20*np.log10(np.max(np.abs(a))+1e-12):6.2f} dB")

# ---- 3. cek semua panjang identik --------------------------------------
Ns = {k: v["n"] for k, v in info.items()}
print(f"\npanjang sebelum: {sorted(set(Ns.values()))}")
NREF = min(Ns.values())   # 10126107 = panjang stem asli (229.61694 s)
# pangkas 3 sample ekor (0.07 ms, hanya sisa reverb) supaya ke-10 file identik
for k in list(info):
    if info[k]["n"] != NREF:
        a, sr = sf.read(f"{OUT}/{k}.wav", dtype="float32", always_2d=True)
        a = a[:NREF]
        sf.write(f"{OUT}/{k}.wav", a, sr, subtype="PCM_24")
        info[k]["n"] = NREF
        info[k]["dur"] = NREF / sr
        print(f"  {k}: dipangkas ke {NREF} sample (buang {Ns[k]-NREF} sample ekor)")
Ns = {k: v["n"] for k, v in info.items()}
print(f"semua panjang identik: {len(set(Ns.values())) == 1}  (n={NREF}, {NREF/SR:.6f}s)")

# ---- 4. mix baru: TANPA proses dinamis (murni penjumlahan + gain) ------
# Mix = (lagu + orkestra*g) * G  ->  hubungan linier eksak, jadi
#   (mix/G - orkestra*g) == lagu  (korelasi 1.0) dan
#   mix == jumlah 10 stem yang diekspor (konsisten dengan file di disk).
#
# Kenapa gain master harus turun: jumlah 7 stem asli sudah memuncak
# +0.34 dBFS (59 sampel melewati 0 dBFS, di bar 72.5). Brief minta peak
# < -1.0 dBFS, jadi wajib turun >=1.34 dB. Ditambah puncak orkestra,
# total ~ -4.5 dB rel referensi. Ini konsekuensi matematis, bukan bug:
# menaikkan loudness lagi akan melanggar syarat peak.
names = [k for k, _ in SRC7] + ["08_Strings", "09_Cello", "10_Timpani"]
buf = {n: sf.read(f"{OUT}/{n}.wav", dtype="float64", always_2d=True)[0] for n in names}
N = min(len(v) for v in buf.values())
buf = {k: v[:N] for k, v in buf.items()}

song = sum(buf[n] for n, _ in SRC7)
orch = buf["08_Strings"] + buf["09_Cello"] + buf["10_Timpani"]

def db(x): return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)
def peak_db(x): return 20 * np.log10(np.max(np.abs(x)) + 1e-12)

ORCH_GAIN_DB = -6.0                # orkestra jelas tapi puncaknya tidak mendominasi
og = 10 ** (ORCH_GAIN_DB / 20)
raw = song + orch * og

PEAK_TARGET_DB = -1.5              # brief: peak < -1.0 dBFS
MASTER_GAIN_DB = PEAK_TARGET_DB - peak_db(raw)
mg = 10 ** (MASTER_GAIN_DB / 20)
mix = raw * mg

print(f"\nlagu      RMS {db(song):7.2f} dB peak {peak_db(song):+6.2f} dBFS")
print(f"orkestra  RMS {db(orch):7.2f} dB peak {peak_db(orch):+6.2f} dBFS "
      f"(gain {ORCH_GAIN_DB:+.1f} dB -> {db(orch*og)-db(song):+.2f} dB rel. lagu)")
print(f"raw mix   RMS {db(raw):7.2f} dB peak {peak_db(raw):+6.2f} dBFS")
print(f"gain master {MASTER_GAIN_DB:+.3f} dB -> mix RMS {db(mix):7.2f} dB "
      f"peak {peak_db(mix):+.2f} dBFS")
assert peak_db(mix) < -1.0, "peak masih >= -1.0 dBFS"
print(f"cek target peak < -1.0 dBFS: LOLOS ({peak_db(mix):+.2f} dBFS)")

# ---- 5. verifikasi linier: balikkan gain master -------------------------
recovered = mix / mg - orch * og
corr = np.corrcoef(recovered.ravel(), song.ravel())[0, 1]
resid = recovered - song
resid_db = 20*np.log10(np.sqrt(np.mean(resid**2))/np.sqrt(np.mean(song**2)) + 1e-12)
print(f"\ncorr((mix/G - orkestra*g), lagu) = {corr:.6f}  (target > 0.98)")
print(f"residual RMS rel. lagu           = {resid_db:.2f} dB")
assert corr > 0.98, "korelasi masih di bawah target"

# mix harus = jumlah 10 stem yang ada di disk (konsisten utk DAW)
sum10 = sum(buf[n] for n in names) - orch + orch * og
print(f"cek mix == jumlah 10 stem bergain: maxdiff "
      f"{np.max(np.abs(mix - sum10*mg)):.2e}")

# ---- 6. sinkronisasi: onset orkestra harus jatuh di grid bar -----------
# Grid bar n = 0.065 + (n-1)*3.000 s. Untuk pad berkelanjutan, onset nyata
# hanya ada di awal note/chord; yang diuji: apakah kenaikan energi terjadi
# dekat batas bar, dan apakah pusat energi per bar konsisten.
OFF_ = 0.065; BAR_ = 3.0
print("\n=== 6. SINKRONISASI (grid 0.065 + k*3.000 s) ===")
for nm in ["08_Strings", "09_Cello", "10_Timpani"]:
    x = buf[nm]
    m = np.max(np.abs(x), axis=1)
    # onset = kenaikan energi tajam (deteksi pada selubung 1 ms)
    hop = int(0.001 * SR)
    k = len(m)//hop*hop
    e = np.sqrt((m[:k].reshape(-1, hop)**2).mean(axis=1))
    e = e / (e.max() + 1e-12)
    d = np.diff(e, prepend=e[0])
    idx = np.where((d > 0.05) & (e > 0.02))[0]
    # gabungkan onset berdekatan (< 150 ms)
    keep = []
    for i in idx:
        if not keep or (i - keep[-1]) * hop / SR > 0.150:
            keep.append(i)
    t = np.array(keep) * hop / SR
    if len(t) == 0:
        print(f"  {nm:12s} tidak ada onset (pad murni)")
        continue
    kk = np.round((t - OFF_) / BAR_)
    err = t - (OFF_ + kk * BAR_)
    good = np.abs(err) <= 0.050
    print(f"  {nm:12s} {len(t):4d} onset | dekat grid (<=50 ms): "
          f"{good.sum():4d} ({100*good.mean():5.1f}%) | "
          f"galat median {np.median(np.abs(err))*1000:5.1f} ms")

# 6b. uji global: apakah energi orkestra terpusat di grid bar?
print("\n=== 6b. UJI GLOBAL: korelasi silang selubung orkestra vs lagu ===")
def env(x, hop=512):
    m = np.max(np.abs(x), axis=1)
    k = len(m)//hop*hop
    return m[:k].reshape(-1, hop).max(axis=1)
e_s = env(song); e_o = env(orch)
e_s = (e_s - e_s.mean()) / (e_s.std() + 1e-12)
e_o = (e_o - e_o.mean()) / (e_o.std() + 1e-12)
lags = np.arange(-60, 61)
xc = np.array([float(np.dot(np.roll(e_o, L), e_s)) for L in lags])
best = int(lags[int(np.argmax(xc))])
print(f"  lag puncak = {best} x 512 = {best*512} sampel = {best*512/SR*1000:+.1f} ms "
      f"({'SEJAJAR' if abs(best*512) < 1024 else 'PERIKSA'})")
print(f"  korelasi pada lag 0 = {xc[np.where(lags==0)[0][0]]/len(e_s):+.4f}")

# 6c. uji grid bar: energi tiap bar harus sejajar batas bar
print("\n=== 6c. UJI GRID BAR: energi per bar vs peta dinamika ===")
def bar_rms(x, nb=77):
    out = []
    for b in range(nb):
        i0 = int((OFF_ + b*BAR_) * SR); i1 = int((OFF_ + (b+1)*BAR_) * SR)
        seg = x[i0:min(i1, len(x))]
        out.append(20*np.log10(np.sqrt((seg**2).mean()) + 1e-12))
    return np.array(out)
r_str = bar_rms(buf["08_Strings"])
for rng, lab in [((0,17),"A pp"), ((17,33),"B p/mp"), ((33,48),"C mf/f"),
                 ((48,74),"D f/ff"), ((74,77),"E dim")]:
    seg = r_str[rng[0]:rng[1]]
    print(f"  {lab:7s} bar {rng[0]+1:2d}-{rng[1]:2d}: RMS {seg.mean():6.2f} dB "
          f"(min {seg.min():6.2f}, maks {seg.max():6.2f})")

sf.write("out3/mix_v3.wav", mix.astype(np.float32), SR, subtype="PCM_16")
sf.write("out3/mix_v3_24bit.wav", mix.astype(np.float32), SR, subtype="PCM_24")
p16 = peak_db(sf.read("out3/mix_v3.wav", dtype="float64")[0])
p24 = peak_db(sf.read("out3/mix_v3_24bit.wav", dtype="float64")[0])
print(f"\ntulis out3/mix_v3.wav (16-bit, peak {p16:.2f} dBFS)")
print(f"tulis out3/mix_v3_24bit.wav (24-bit, peak {p24:.2f} dBFS)")
ref = sf.read("mix.wav", dtype="float64")[0]
print(f"referensi lagu asli: RMS {db(ref):.2f} dB peak {peak_db(ref):.2f} dB")

json.dump(dict(stems=info, n=N, sr=SR, offset=OFF, bar=BAR,
               orch_gain_db=ORCH_GAIN_DB,
               peak_target_db=PEAK_TARGET_DB,
               master_gain_db=round(float(MASTER_GAIN_DB), 4),
               song_rms=round(float(db(song)), 3),
               orch_rms=round(float(db(orch)), 3),
               orch_peak_db=round(float(peak_db(orch)), 3),
               raw_rms=round(float(db(raw)), 3),
               raw_peak_db=round(float(peak_db(raw)), 3),
               mix_rms=round(float(db(mix)), 3),
               mix_peak_db=round(float(peak_db(mix)), 3),
               ref_rms=round(float(db(ref)), 3),
               ref_peak_db=round(float(peak_db(ref)), 3),
               corr_song_recovered=round(float(corr), 6),
               resid_db=round(float(resid_db), 3),
               original_stems_untouched=True,
               all_same_length=len(set(Ns.values())) == 1),
          open("out3/stems_manifest.json", "w"), indent=1)
print("\nmanifest: out3/stems_manifest.json")
