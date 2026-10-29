"""Uji sinkronisasi yang benar: grid lagu asli vs grid orkestra.

Lagu asli punya drum -> onset jelas. Kalau grid lagu = 0.065 + k*3.000,
maka orkestra (dirender di grid itu) pasti sinkron.
"""
import numpy as np, soundfile as sf, librosa
SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]
OFF, BAR = 0.065, 3.0

drums = rd("out3/stems/01_Drums.wav").mean(axis=1)
y22 = librosa.resample(drums, orig_sr=SR, target_sr=22050)

# 1. tempo & beat grid lagu asli
tempo, beats = librosa.beat.beat_track(y=y22, sr=22050, hop_length=512,
                                       units="time", trim=False)
print(f"tempo terdeteksi lagu: {float(np.atleast_1d(tempo)[0]):.3f} BPM")
print(f"jumlah beat: {len(beats)}  beat pertama {beats[0]:.4f} s")

# 2. apakah beat jatuh di grid 0.065 + k*0.75 (80 BPM -> 0.75 s/beat)?
beats = np.asarray(beats)
ph = ((beats - OFF) % 0.75) / 0.75
dist = np.minimum(ph, 1 - ph) * 0.75        # detik ke beat terdekat
print(f"\njarak beat ke grid 0.065+k*0.750:")
print(f"  median {np.median(dist)*1000:6.1f} ms | p90 {np.percentile(dist,90)*1000:6.1f} ms "
      f"| maks {dist.max()*1000:6.1f} ms")
print(f"  beat dlm 30 ms dari grid: {100*(dist<=0.030).mean():5.1f}%  (acak = "
      f"{100*0.060/0.75:.1f}%)")

# 3. bar grid: downbeat tiap 4 beat
print(f"\nbar grid 0.065+k*3.000 (4 beat/bar):")
phb = ((beats - OFF) % BAR) / BAR
distb = np.minimum(phb, 1 - phb) * BAR
print(f"  median {np.median(distb)*1000:6.1f} ms | p90 {np.percentile(distb,90)*1000:6.1f} ms")

# 4. cek timpani hits (transien tajam) vs grid bar
print(f"\ntimpani hits vs grid bar:")
ti = rd("out3/stems/10_Timpani.wav").mean(axis=1)
env = librosa.onset.onset_strength(y=librosa.resample(ti, orig_sr=SR, target_sr=22050),
                                   sr=22050, hop_length=256)
on = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=256,
                               units="time", delta=0.3, wait=20)
on = np.asarray(on)
print(f"  {len(on)} hit terdeteksi")
if len(on):
    ph2 = ((on - OFF) % BAR) / BAR
    d2 = np.minimum(ph2, 1-ph2) * BAR
    print(f"  jarak ke batas bar: median {np.median(d2)*1000:6.1f} ms | "
          f"<=50 ms: {100*(d2<=0.05).mean():5.1f}%")

# 5. uji langsung: onset drum lagu vs onset orkestra, cari lag
print(f"\nkorelasi silang onset-strength drum vs orkestra:")
d_env = librosa.onset.onset_strength(y=y22, sr=22050, hop_length=256)
o = rd("out3/stems/08_Strings.wav").mean(axis=1) + rd("out3/stems/09_Cello.wav").mean(axis=1) \
    + rd("out3/stems/10_Timpani.wav").mean(axis=1)
o_env = librosa.onset.onset_strength(y=librosa.resample(o, orig_sr=SR, target_sr=22050),
                                     sr=22050, hop_length=256)
n = min(len(d_env), len(o_env))
a = d_env[:n] - d_env[:n].mean(); b = o_env[:n] - o_env[:n].mean()
lags = np.arange(-80, 81)
xc = np.array([np.dot(np.roll(b, L), a) for L in lags])
best = int(lags[int(np.argmax(xc))])
print(f"  lag puncak {best} hop x 256 = {best*256/22050*1000:+.1f} ms")
print(f"  korelasi lag 0 = {xc[np.where(lags==0)[0][0]]/n:.4f}")

# 6. peta dinamika: transisi harus di bar 18,34,49,75
print(f"\ntransisi dinamika (bar 18/34/49/75 -> detik):")
for b in [18,34,49,75]:
    print(f"  bar {b:2d} -> {OFF+(b-1)*BAR:8.3f} s")
