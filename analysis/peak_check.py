import numpy as np, soundfile as sf

SR = 44100
def rd(p):
    a, _ = sf.read(p, dtype="float64", always_2d=True)
    return a

names = ["01_Drums","02_Bass","03_Guitar","04_Piano","05_Vocals-lead",
         "06_Vocals-background","07_Pro-other"]
song = sum(rd(f"out3/stems/{n}.wav") for n in names)
st = rd("out3/stems/08_Strings.wav")
ce = rd("out3/stems/09_Cello.wav")
ti = rd("out3/stems/10_Timpani.wav")
orch = st + ce + ti

def pk(x): return 20*np.log10(np.max(np.abs(x))+1e-12)
def rms(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)

print("stem        peak     RMS    crest")
for nm, x in [("song", song), ("strings", st), ("cello", ce), ("timpani", ti), ("orch", orch)]:
    print(f"{nm:9s} {pk(x):7.2f} {rms(x):7.2f} {pk(x)-rms(x):7.2f}")

# di mana puncak orkestra?
mono = np.max(np.abs(orch), axis=1)
i = int(np.argmax(mono))
print(f"\npuncak orkestra di sample {i} = {i/SR:.3f} s  ({i/SR/3.0:.2f} bar)")
print(f"nilai {mono[i]:.4f}  peak {pk(orch):.2f} dBFS")

# berapa banyak sampel yang melewati 0 dBFS?
for thr in [0.0, -0.5, -1.0, -3.0, -6.0]:
    n = int(np.sum(mono > 10**(thr/20)))
    print(f"  sampel > {thr:5.1f} dBFS : {n:9d}  ({n/SR*1000:.2f} ms)")

# distribusi puncak per 1 detik
sec = len(mono)//SR
blk = mono[:sec*SR].reshape(sec, SR).max(axis=1)
top = np.argsort(blk)[::-1][:8]
print("\n8 detik terkeras orkestra:")
for t in top:
    print(f"  detik {t:3d} (bar {t/3.0+1:5.1f}) peak {20*np.log10(blk[t]+1e-12):6.2f} dB")

# puncak song vs orch per detik: apakah spike orkestra menumpuk dgn song?
sblk = np.max(np.abs(song), axis=1)[:sec*SR].reshape(sec, SR).max(axis=1)
print("\ndi detik2 terkeras orkestra, peak lagu:")
for t in top[:5]:
    print(f"  detik {t:3d}: orch {20*np.log10(blk[t]+1e-12):6.2f}  song {20*np.log10(sblk[t]+1e-12):6.2f}  "
          f"sum {20*np.log10(blk[t]+sblk[t]+1e-12):6.2f}")

# awal bunyi tiap stem (offset check)
print("\n=== awal bunyi (sampel pertama > -60 dBFS) ===")
for nm, x in [("song", song), ("08_Strings", st), ("09_Cello", ce), ("10_Timpani", ti)]:
    m = np.max(np.abs(x), axis=1)
    idx = np.argmax(m > 10**(-60/20))
    print(f"  {nm:12s} sample {idx:8d} = {idx/SR*1000:8.2f} ms")
