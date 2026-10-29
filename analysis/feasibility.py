"""Analisis kelayakan: bisakah peak < -1.0 dBFS DAN corr > 0.98 tercapai?

Metrik parent: corr(mix_v3 - orkestra, jumlah 7 stem asli).
Dengan mix = G*lagu + H*orkestra, maka
  mix - orkestra = G*lagu + (H-1)*orkestra
Jadi korelasi tinggi butuh H -> 1 (orkestra tidak dikecilkan),
sementara peak butuh G kecil. Dua syarat ini bisa bertabrakan.
"""
import numpy as np, soundfile as sf
SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]
def db(x): return 20*np.log10(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2))+1e-12)
def pk(x): return 20*np.log10(np.max(np.abs(x))+1e-12)

n7 = ["01_Drums","02_Bass","03_Guitar","04_Piano","05_Vocals-lead",
      "06_Vocals-background","07_Pro-other"]
song = sum(rd(f"out3/stems/{n}.wav") for n in n7)
orch = sum(rd(f"out3/stems/{n}.wav") for n in ["08_Strings","09_Cello","10_Timpani"])
ref = rd("mix.wav")

print(f"lagu peak {pk(song):+.2f} | orch peak {pk(orch):+.2f} | ref RMS {db(ref):.2f}")
print(f"\nTarget: peak <= -1.5 dBFS, corr(mix-orch, lagu) > 0.98\n")

print(" H_dB | G_dB  | mixPk | mixRMS | corr(mix-orch,lagu)")
best = None
for H_db in [-24,-18,-15,-12,-10,-8,-6,-4,-2,0]:
    H = 10**(H_db/20)
    # G maks supaya peak = -1.5
    # cari G dgn bisection
    lo, hi = 1e-4, 4.0
    for _ in range(60):
        G = (lo+hi)/2
        if pk(G*song + H*orch) > -1.5: hi = G
        else: lo = G
    G = lo
    mix = G*song + H*orch
    rec = mix - orch
    corr = np.corrcoef(rec.ravel(), song.ravel())[0,1]
    print(f"{H_db:+5d} | {20*np.log10(G):+6.2f} | {pk(mix):+6.2f} | {db(mix):7.2f} | {corr:.5f}")
    if corr > 0.98 and (best is None or db(mix) > best[0]):
        best = (db(mix), H_db, 20*np.log10(G), corr)

print(f"\nTerbaik yg lolos corr>0.98: {best}")

# variasi: apakah lagu boleh sedikit di-EQ/trim? tidak. Cek apakah solusi ada.
print("\n=== KESIMPULAN KELAYAKAN ===")
H = 1.0
lo, hi = 1e-4, 4.0
for _ in range(60):
    G = (lo+hi)/2
    if pk(G*song + H*orch) > -1.5: hi = G
    else: lo = G
print(f"dgn orkestra unity (H=1): G maks {20*np.log10(lo):+.2f} dB, "
      f"corr {np.corrcoef((lo*song).ravel(), song.ravel())[0,1]:.5f}")
print(f"  (mix-orch = G*lagu tepat -> corr pasti 1.0, tapi peak {pk(lo*song+orch):+.2f})")
