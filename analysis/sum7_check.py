import numpy as np, soundfile as sf
SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]

n7 = ["01_Drums","02_Bass","03_Guitar","04_Piano","05_Vocals-lead",
      "06_Vocals-background","07_Pro-other"]
song = sum(rd(f"out3/stems/{n}.wav") for n in n7)
ref = rd("mix.wav")

print(f"sum7  peak {20*np.log10(np.abs(song).max()+1e-12):+.2f} dBFS  RMS "
      f"{20*np.log10(np.sqrt((song**2).mean())+1e-12):.2f}")
print(f"ref   peak {20*np.log10(np.abs(ref).max()+1e-12):+.2f} dBFS  RMS "
      f"{20*np.log10(np.sqrt((ref**2).mean())+1e-12):.2f}")
d = np.abs(song - ref)
print(f"selisih sum7 vs ref: maks {d.max():.4f}  RMS {np.sqrt((d**2).mean()):.4f}")
print(f"corr(sum7, ref) = {np.corrcoef(song.ravel(), ref.ravel())[0,1]:.5f}")

# apakah ref = sum7 * gain konstan?
g = np.sum(song*ref)/np.sum(song*song)
print(f"\ngain least-squares ref/sum7 = {g:.5f} ({20*np.log10(g):+.2f} dB)")
res = ref - song*g
print(f"residual setelah gain: RMS {np.sqrt((res**2).mean()):.5f} "
      f"({20*np.log10(np.sqrt((res**2).mean())/np.sqrt((ref**2).mean())):.1f} dB rel ref)")

# di mana sum7 melewati 0 dBFS?
mono = np.max(np.abs(song), axis=1)
for thr in [0.0, -0.3, -1.0]:
    n = int((mono > 10**(thr/20)).sum())
    print(f"  sum7 sampel > {thr:+.1f} dBFS: {n} ({n/SR*1000:.2f} ms)")
i = int(np.argmax(mono))
print(f"  puncak sum7 di {i/SR:.3f} s (bar {i/SR/3+1:.2f}), nilai {mono[i]:.4f}")
print(f"  di titik itu ref = {np.abs(ref[i]).max():.4f} "
      f"({20*np.log10(np.abs(ref[i]).max()+1e-12):+.2f} dBFS)")

# cek: apakah stem asli masing2 peak <=1?
print("\npeak tiap stem asli:")
for n in n7:
    x = rd(f"out3/stems/{n}.wav")
    print(f"  {n:22s} {20*np.log10(np.abs(x).max()+1e-12):+7.2f} dBFS")
