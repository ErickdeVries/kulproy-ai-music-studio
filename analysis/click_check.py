import numpy as np, soundfile as sf
SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]

for nm in ["08_Strings","09_Cello","10_Timpani"]:
    x = rd(f"out3/stems/{nm}.wav")
    m = np.max(np.abs(x), axis=1)
    d = np.abs(np.diff(x, axis=0)).max(axis=1)
    # lompatan sampel besar = klik
    med = np.median(d)
    thr = max(0.02, 50*med)
    idx = np.where(d > thr)[0]
    print(f"{nm}: crest {20*np.log10(m.max()+1e-12)-20*np.log10(np.sqrt((x**2).mean())+1e-12):5.2f} dB | "
          f"median |dx| {med:.2e} | lompatan > {thr:.3f}: {len(idx)}")
    if len(idx):
        print("   5 terbesar:", [(int(i), round(float(d[i]),4), round(i/SR,3)) for i in idx[np.argsort(d[idx])[::-1][:5]]])
    # lihat 1 attack: sekitar awal strings
    if nm == "08_Strings":
        i0 = 5523
        seg = m[i0-200:i0+2000]
        print(f"   attack: max 200 smp sebelum {m[i0-200:i0].max():.4f}, "
              f"200 sesudah {m[i0:i0+200].max():.4f}")
        # berapa lama naik ke puncak
        k = np.argmax(m[i0:i0+4410])
        print(f"   puncak lokal di +{k} smp ({k/SR*1000:.1f} ms) nilai {m[i0+k]:.4f}")
