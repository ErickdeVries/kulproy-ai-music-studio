"""Ukur tempo & drift lagu asli dengan presisi tinggi.

Kalau lagu bukan tepat 80.000 BPM, orkestra yang dirender di grid 3.000 s
akan bergeser (drift) makin jauh di bagian akhir lagu.
"""
import numpy as np, soundfile as sf, librosa
SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]
OFF = 0.065

drums = rd("out3/stems/01_Drums.wav").mean(axis=1)
y = librosa.resample(drums, orig_sr=SR, target_sr=22050)
hop = 128                                   # ~5.8 ms resolusi
env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
fps = 22050 / hop
print(f"onset env: {len(env)} frame, {fps:.1f} fps, durasi {len(env)/fps:.1f} s")

def local_period(env, t0, t1, lo=0.60, hi=0.95):
    """Periode beat via autokorelasi di jendela waktu tertentu (detik)."""
    i0, i1 = int(t0*fps), int(t1*fps)
    seg = env[i0:i1].astype(np.float64)
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode="full")[len(seg)-1:]
    ac /= ac[0] + 1e-12
    l0, l1 = int(lo*fps), int(hi*fps)
    k = l0 + int(np.argmax(ac[l0:l1]))
    # interpolasi parabolik utk presisi sub-frame
    if 0 < k < len(ac)-1:
        a, b, cc = ac[k-1], ac[k], ac[k+1]
        d = 0.5*(a-cc)/(a-2*b+cc + 1e-12)
    else:
        d = 0.0
    return (k+d)/fps, ac[k]

print("\n=== PERIODE BEAT LOKAL (autokorelasi) ===")
wins = [(2,30),(30,60),(60,90),(90,120),(120,150),(150,180),(180,210),(210,229)]
pers = []
for t0,t1 in wins:
    P, conf = local_period(env, t0, t1)
    bpm = 60/P
    pers.append((0.5*(t0+t1), P))
    print(f"  {t0:5.0f}-{t1:5.0f}s: periode {P*1000:7.2f} ms -> {bpm:7.3f} BPM "
          f"(konfidensi {conf:.3f})")

# regresi linear periode vs waktu -> drift
ts = np.array([p[0] for p in pers]); Ps = np.array([p[1] for p in pers])
A = np.vstack([ts, np.ones_like(ts)]).T
slope, inter = np.linalg.lstsq(A, Ps, rcond=None)[0]
print(f"\nregresi periode: P(t) = {inter*1000:.2f} ms + {slope*1e6:.3f} us/s * t")
print(f"  periode di awal (t=0)  : {inter*1000:.2f} ms -> {60/inter:.3f} BPM")
print(f"  periode di akhir (229s): {(inter+slope*229)*1000:.2f} ms -> {60/(inter+slope*229):.3f} BPM")
print(f"  drift total 229 s      : {slope*229*1000:+.2f} ms per beat, "
      f"akumulasi {slope*229*229/2*1000:+.0f} ms")

# periode rata-rata seluruh lagu
P_all, conf_all = local_period(env, 2, 229)
print(f"\nperiode rata-rata seluruh lagu: {P_all*1000:.3f} ms -> {60/P_all:.3f} BPM "
      f"(konfidensi {conf_all:.3f})")

# uji: berapa galat akumulatif kalau pakai 0.7500 s vs periode terukur?
print("\n=== UJI DRIFT: galat posisi bar ke-n ===")
for P in [0.7500, P_all]:
    bpm = 60/P
    bar = 4*P
    print(f"  asumsi bar {bar:.4f} s ({bpm:.3f} BPM):", end="")
    for n in [10, 30, 50, 70, 77]:
        err = (OFF + (n-1)*3.0) - (OFF + (n-1)*bar)
        print(f"  bar{n}:{err*1000:+7.0f}ms", end="")
    print()

# cek langsung: onset kick di awal vs akhir, apakah bar 1 dan bar 77 sejajar?
print("\n=== ONSET KUAT (kick/snare) awal vs akhir ===")
on = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=hop,
                               units="time", delta=0.08, wait=4)
on = np.asarray(on)
print(f"total onset: {len(on)}")
for lo_t, hi_t in [(0,20),(200,229)]:
    sel = on[(on>=lo_t)&(on<hi_t)]
    if len(sel)==0: continue
    ph = ((sel - OFF) % 0.75)/0.75
    d = np.minimum(ph,1-ph)*0.75
    print(f"  {lo_t:3d}-{hi_t:3d}s: {len(sel):4d} onset, jarak ke grid 0.75s "
          f"median {np.median(d)*1000:6.1f} ms")
