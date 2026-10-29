"""Fit grid beat presisi: t_n = t0 + n*P, pakai onset kuat dari drum+bass."""
import numpy as np, soundfile as sf, librosa
SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]

# gabung drum + bass: onset paling tegas
d = rd("out3/stems/01_Drums.wav").mean(axis=1)
b = rd("out3/stems/02_Bass.wav").mean(axis=1)
y = librosa.resample(d + 0.5*b, orig_sr=SR, target_sr=22050)
hop = 128
env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
fps = 22050/hop

on = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=hop,
                               units="time", delta=0.05, wait=3)
on = np.sort(np.asarray(on))
print(f"{len(on)} onset, rentang {on.min():.2f}-{on.max():.2f} s")

def fit(P0, tol=0.28):
    """Iteratif: tetapkan onset ke indeks beat terdekat, lalu fit t0,P."""
    t0, P = on[0], P0
    for it in range(12):
        n = np.round((on - t0)/P)
        pred = t0 + n*P
        res = on - pred
        m = np.abs(res) < tol
        if m.sum() < 20: break
        # regresi least-squares pada onset yang cocok
        A = np.vstack([n[m], np.ones(m.sum())]).T
        P, t0 = np.linalg.lstsq(A, on[m], rcond=None)[0]
    return t0, P, m, res

for P0 in [0.75, 0.746, 0.751]:
    t0, P, m, res = fit(P0)
    r = res[m]
    print(f"\nP0={P0:.4f} -> P={P*1000:.4f} ms ({60/P:.4f} BPM), t0={t0:.4f} s, "
          f"cocok {m.sum()}/{len(on)}")
    print(f"   residual: RMS {r.std()*1000:.2f} ms, median |r| {np.median(np.abs(r))*1000:.2f} ms, "
          f"maks {np.abs(r).max()*1000:.1f} ms")
    # galat di awal vs akhir (uji drift)
    for lo,hi,lab in [(0,50,"awal"),(150,200,"tengah"),(200,230,"akhir")]:
        sel = (on>=lo)&(on<hi)
        nn = np.round((on[sel]-t0)/P); rr = on[sel]-(t0+nn*P)
        if len(rr):
            print(f"   {lab:6s} {lo:3d}-{hi:3d}s: median |galat| {np.median(np.abs(rr))*1000:6.2f} ms "
                  f"bias {np.median(rr)*1000:+7.2f} ms")

# uji hipotesis: apakah 80.000 BPM (P=0.75) cukup?
print("\n=== UJI HIPOTESIS P = 0.750000 s (80.000 BPM) ===")
for t0 in [0.065, on[0]]:
    n = np.round((on - t0)/0.75)
    r = on - (t0 + n*0.75)
    print(f"  t0={t0:.4f}: median |r| {np.median(np.abs(r))*1000:6.2f} ms, "
          f"p90 {np.percentile(np.abs(r),90)*1000:6.2f} ms")
    for lo,hi,lab in [(0,50,"awal"),(150,200,"tengah"),(200,230,"akhir")]:
        sel=(on>=lo)&(on<hi); rr=r[sel]
        if len(rr): print(f"     {lab:6s}: median {np.median(rr)*1000:+7.2f} ms")

# berapa banyak bar sampai grid 0.75 melenceng 100 ms?
print("\n=== DRIFT grid 0.7500 vs periode terfit ===")
t0f, Pf, mf, _ = fit(0.75)
print(f"periode terfit {Pf*1000:.4f} ms vs 750.0000 ms -> selisih {(Pf-0.75)*1000:+.4f} ms/beat")
if abs(Pf-0.75) > 1e-5:
    beats_per_100ms = 0.100/abs(Pf-0.75)
    print(f"  butuh {beats_per_100ms:.0f} beat ({beats_per_100ms/4:.0f} bar) utk melenceng 100 ms")
else:
    print("  periode = 80.000 BPM tepat, tidak ada drift")
