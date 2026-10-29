"""Analisis harmoni final: chord per bar (77 bar), grid offset terverifikasi.

Grid: BPM 80.000, bar = 3.000 s, OFFSET = 0.065 s
  bukti: novelty chroma puncak di fase 0.04-0.07 s; onset bass pertama 99.068 s
         = 0.065 + 33*3.000  -> tepat downbeat bar 34.

Chord: chroma CQT dari Instrumental (harmoni) + chroma bass (root), prior C mayor,
       beat-weighted (downbeat & beat 3 berbobot lebih), median lalu cosine template.
Output: song/out3/chords.json
"""
import numpy as np, json, os
import librosa

BPM, BEAT = 80.0, 60.0 / 80.0
BAR = 4 * BEAT
OFF = 0.065
NBARS = 77
SR = 22050
os.makedirs("out3", exist_ok=True)
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def hp(name):
    return f"stems/{name} - NANA HARAHAP - UNTUKMU - CD FORMAT-1 - 80bpm - Cmaj.wav"

y, _ = librosa.load(hp("Instrumental"), sr=SR, mono=True)
dur = len(y) / SR
yh = librosa.effects.harmonic(y, margin=3.0)

C = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=256, bins_per_octave=36)
C = librosa.util.normalize(C, axis=0, norm=np.inf)
T = librosa.times_like(C, sr=SR, hop_length=256)

Cb = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=256, bins_per_octave=36,
                                fmin=librosa.note_to_hz("C1"), n_octaves=2)
Cb = librosa.util.normalize(Cb, axis=0, norm=np.inf)

QUAL = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10],
        "maj7": [0, 4, 7, 11], "sus4": [0, 5, 7], "sus2": [0, 2, 7],
        "add9": [0, 4, 7, 2], "m7b5": [0, 3, 6, 10], "dim": [0, 3, 6]}
W = {0: 1.0, 3: 0.95, 4: 0.95, 7: 0.9, 10: 0.6, 11: 0.55, 2: 0.5, 5: 0.5, 6: 0.5}
TP = []
for r in range(12):
    for q, ivs in QUAL.items():
        v = np.zeros(12)
        for i in ivs: v[i] = W.get(i, 0.45)
        TP.append((NOTES[r] + q, r, q, np.roll(v / np.linalg.norm(v), r)))

DIAT = {0: ["", "maj7", "add9", "sus2", "sus4"], 2: ["m", "m7"],
        4: ["m", "m7"], 5: ["", "maj7", "add9", "sus2"], 7: ["", "7", "sus4", "maj7"],
        9: ["m", "m7"], 11: ["dim", "m7b5"]}

# bobot dalam bar: downbeat, beat2, beat3, beat4 (16th sub-frames)
def seg_slice(arr, t0, t1, beatw=(1.0, 0.55, 0.85, 0.55)):
    i0, i1 = np.searchsorted(T, [t0, t1])
    i1 = min(i1, arr.shape[1])
    if i1 - i0 < 4: return None
    s = arr[:, i0:i1]
    n = s.shape[1]
    bnd = np.linspace(0, n, 5).astype(int)
    out = []
    for k in range(4):
        a, b = bnd[k], bnd[k + 1]
        if b > a: out.append(s[:, a:b].mean(1) * beatw[k])
    return np.sum(np.stack(out, 0), 0) / sum(beatw)

rows = []
for b in range(1, NBARS + 1):
    t0, t1 = OFF + (b - 1) * BAR, min(OFF + b * BAR, dur)
    vs, vbs = seg_slice(C, t0, t1), seg_slice(Cb, t0, t1)
    if vs is None or np.max(vs) < 1e-6:
        rows.append(dict(bar=b, t0=round(t0, 3), chord="n/a", root=None,
                         conf=0.0, silent=True, bass="-", alts=[]))
        continue
    v = vs / (np.linalg.norm(vs) + 1e-9)
    vb = vbs / (vbs.sum() + 1e-9) if vbs is not None else np.zeros(12)
    sc = []
    for name, r, q, tv in TP:
        s = float(v @ tv) + 0.40 * float(vb[r])
        if r in DIAT and q in DIAT[r]: s += 0.07
        sc.append((s, name, r, q))
    sc.sort(reverse=True)
    best = sc[0]
    rows.append(dict(bar=b, t0=round(t0, 3), chord=best[1], root=best[2], qual=best[3],
                     conf=round(best[0], 4), bass=NOTES[int(np.argmax(vb))],
                     alts=[x[1] for x in sc[1:4]],
                     chroma=[round(float(x), 3) for x in v], silent=False))

prof = np.sum([r["chroma"] for r in rows if not r["silent"]], 0)
prof /= prof.sum() + 1e-9
MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
kk = []
for r in range(12):
    kk.append((float(np.corrcoef(prof, np.roll(MAJ, r))[0,1]), NOTES[r]+" major"))
    kk.append((float(np.corrcoef(prof, np.roll(MIN, r))[0,1]), NOTES[r]+" minor"))
kk.sort(reverse=True)

json.dump(dict(offset=OFF, bar_sec=BAR, bpm=BPM, nbars=NBARS, audio_dur=round(dur,4),
               key=kk[0][1], key_corr=round(kk[0][0],4),
               key_cands=[[k[1], round(k[0],4)] for k in kk[:5]], bars=rows),
          open("out3/chords.json","w"), indent=1)

print(f"dur {dur:.4f}s | {dur/BAR:.2f} bar | key {kk[0][1]} r={kk[0][0]:.3f}")
print(f"next: {kk[1][1]} {kk[1][0]:.3f} | {kk[2][1]} {kk[2][0]:.3f}")
print(f"\n{'bar':>3} {'t0':>7} {'chord':<8} {'bass':<4} {'conf':>5}  alts")
for r in rows:
    print(f"{r['bar']:3d} {r['t0']:7.2f} {r['chord']:<8} {r['bass']:<4} {r['conf']:5.3f}  {' '.join(r['alts'])}")
