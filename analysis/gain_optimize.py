"""Cari struktur gain terbaik: crest orkestra ditekan, mix tetap linier."""
import numpy as np, soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

SR = 44100
def rd(p): return sf.read(p, dtype="float64", always_2d=True)[0]
def db(x): return 20*np.log10(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2))+1e-12)
def pk(x): return 20*np.log10(np.max(np.abs(x))+1e-12)

names7 = ["01_Drums","02_Bass","03_Guitar","04_Piano","05_Vocals-lead",
          "06_Vocals-background","07_Pro-other"]
song = sum(rd(f"out3/stems/{n}.wav") for n in names7)
st = rd("out3/stems/08_Strings.wav")
ce = rd("out3/stems/09_Cello.wav")
ti = rd("out3/stems/10_Timpani.wav")
orch = st + ce + ti
ref = rd("mix.wav")

print(f"lagu   RMS {db(song):7.2f} peak {pk(song):6.2f} crest {pk(song)-db(song):5.2f}")
print(f"orch   RMS {db(orch):7.2f} peak {pk(orch):6.2f} crest {pk(orch)-db(orch):5.2f}")
print(f"ref    RMS {db(ref):7.2f} peak {pk(ref):6.2f}")

def compressor(x, thr_db=-18.0, ratio=3.0, atk_ms=5.0, rel_ms=120.0, knee_db=6.0):
    """Kompresor stereo-linked, deteksi RMS pendek. Menekan crest, bukan puncak saja."""
    mono = np.max(np.abs(x), axis=1)
    # envelope RMS ~5 ms
    w = max(1, int(0.005*SR))
    e = np.sqrt(np.convolve(mono**2, np.ones(w)/w, mode="same"))
    edb = 20*np.log10(e + 1e-12)
    over = edb - thr_db
    # soft knee
    gr = np.zeros_like(over)
    lo, hi = -knee_db/2, knee_db/2
    m1 = over <= lo
    m2 = (over > lo) & (over <= hi)
    m3 = over > hi
    gr[m2] = (1/ratio - 1) * (over[m2] - lo)**2 / (2*knee_db)
    gr[m3] = (1/ratio - 1) * (over[m3] - knee_db/2)
    # smoothing attack/release
    atk = np.exp(-1.0/(atk_ms*SR/1000)); rel = np.exp(-1.0/(rel_ms*SR/1000))
    g = np.empty_like(gr); cur = 0.0
    for i in range(len(gr)):
        tgt = gr[i]
        cur = tgt*atk + cur*(1-atk) if tgt < cur else tgt*rel + cur*(1-rel)
        g[i] = cur
    return x * (10**(g/20))[:, None], g

def peak_limit(x, ceil_db, look_ms=6.0, rel_ms=90.0):
    ceil = 10**(ceil_db/20)
    mono = np.max(np.abs(x), axis=1)
    la = max(1, int(look_ms*SR/1000))
    need = np.minimum(1.0, ceil/(mono+1e-12))
    pad = np.pad(need, (0, la), constant_values=1.0)
    g = sliding_window_view(pad, la+1).min(axis=1)[:len(need)]
    rel = np.exp(-1.0/(rel_ms*SR/1000))
    out = np.empty_like(g); cur = 1.0
    for i in range(len(g)):
        cur = g[i] if g[i] < cur else cur*rel + g[i]*(1-rel)
        out[i] = cur
    return x*out[:, None], out

TARGET_PEAK = -1.5
print("\n=== A. TANPA PROSES (murni statis) ===")
for og_db in [-4, -2, 0, 2]:
    og = 10**(og_db/20)
    raw = song + orch*og
    mg = TARGET_PEAK - pk(raw)
    mix = raw*10**(mg/20)
    print(f"  og {og_db:+3.0f} dB: orch_rel {db(orch*og)-db(song):+6.2f} | "
          f"mixRMS {db(mix):7.2f} | dRef {db(mix)-db(ref):+5.2f}")

print("\n=== B. KOMPRESI CREST ORKESTRA (bus) ===")
best = None
for thr in [-24, -20, -16]:
    for ratio in [2.0, 3.0, 4.0]:
        c, g = compressor(orch, thr_db=thr, ratio=ratio)
        mk = -pk(c) - 0.001
        c = c * 10**(mk/20)                     # makeup ke peak 0 dB utk perbandingan
        for og_db in [-2, 0, 2, 4, 6]:
            og = 10**(og_db/20)
            raw = song + c*og
            mg = TARGET_PEAK - pk(raw)
            mix = raw*10**(mg/20)
            rel = db(c*og) - db(song)
            r = (thr, ratio, og_db, rel, db(mix))
            if best is None or r[4] > best[4]:
                best = r
            print(f"  thr{thr:4d} r{ratio:.0f} crest{pk(c)-db(c):5.2f} og{og_db:+3.0f} -> "
                  f"orch_rel {rel:+6.2f} mixRMS {db(mix):7.2f} dRef {db(mix)-db(ref):+5.2f}")

print(f"\nTERBAIK: thr={best[0]} ratio={best[1]} og={best[2]:+d} -> orch_rel {best[3]:+.2f} dB, "
      f"mixRMS {best[4]:.2f} dB (ref {db(ref):.2f})")
