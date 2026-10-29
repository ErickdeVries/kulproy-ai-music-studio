"""Analisis harmoni v2 - lebih akurat.

Perbaikan vs v1:
  * chroma dari stem Instrumental (harmoni tanpa vokal) -> tidak kabur
  * chroma bass (C1..C3) untuk constraint root
  * prior diatonik C mayor (lagu ini Cmaj)
  * skor = cosine(chroma, template) + 0.35*bass_root + prior
Output: song/out3/chords.json
"""
import numpy as np, json, os
import librosa

BPM = 80.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
NBARS = 77
os.makedirs("out3", exist_ok=True)
SR = 22050

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --- grid: offset dari onset drum (kick/snare) ---------------------------
y_dr, sr = librosa.load("stems/Drums - NANA HARAHAP - UNTUKMU - CD FORMAT-1 - 80bpm - Cmaj.wav",
                        sr=SR, mono=True)
oe = librosa.onset.onset_strength(y=y_dr, sr=sr, hop_length=512)
hop_t = 512 / SR
grid = np.arange(0.0, BAR, 0.005)
sc = np.array([np.mean([oe[int(round((o + k * BAR) / hop_t))]
                        for k in range(NBARS)
                        if 0 <= int(round((o + k * BAR) / hop_t)) < len(oe)])
               for o in grid])
OFF = float(grid[int(np.argmax(sc))])
print(f"offset downbeat (drum onsets) = {OFF:.3f}s   score {sc.max():.3f} / median {np.median(sc):.3f}")

# --- chroma harmonik dari Instrumental ----------------------------------
y, _ = librosa.load("stems/Instrumental - NANA HARAHAP - UNTUKMU - CD FORMAT-1 - 80bpm - Cmaj.wav",
                    sr=SR, mono=True)
dur = len(y) / SR
yh = librosa.effects.harmonic(y, margin=3.0)
C = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=512, bins_per_octave=36)
C = librosa.util.normalize(C, axis=0, norm=np.inf)
T = librosa.times_like(C, sr=SR, hop_length=512)

# chroma bass: C1..C3
Cb = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=512, bins_per_octave=36,
                                fmin=librosa.note_to_hz("C1"), n_octaves=2)
Cb = librosa.util.normalize(Cb, axis=0, norm=np.inf)

QUAL = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10],
        "maj7": [0, 4, 7, 11], "sus4": [0, 5, 7], "sus2": [0, 2, 7],
        "add9": [0, 4, 7, 2], "m7b5": [0, 3, 6, 10], "dim": [0, 3, 6]}
W = {0: 1.0, 3: 0.95, 4: 0.95, 7: 0.9, 10: 0.65, 11: 0.6, 2: 0.55, 5: 0.55, 6: 0.55}

def vec(ivs):
    v = np.zeros(12)
    for i in ivs:
        v[(0 + i) % 12] = W.get(i, 0.5)
    return v / np.linalg.norm(v)

TP = []
for r in range(12):
    for q, ivs in QUAL.items():
        TP.append((NOTES[r] + q, r, q, np.roll(vec(ivs), r)))

# diatonik C mayor: derajat -> kualitas yang lazim
DIAT = {0: ["", "maj7", "add9", "sus2", "sus4"],
        2: ["m", "m7"],
        4: ["m", "m7"],
        5: ["", "maj7", "add9", "sus2", "sus4"],
        7: ["", "7", "sus4", "sus2", "maj7"],
        9: ["m", "m7"],
        11: ["dim", "m7b5"]}

rows = []
for b in range(1, NBARS + 1):
    t0, t1 = OFF + (b - 1) * BAR, OFF + b * BAR
    i0, i1 = np.searchsorted(T, [t0, t1])
    i1 = min(i1, C.shape[1])
    if i1 <= i0 + 2:
        rows.append(dict(bar=b, chord="n/a", root=None, conf=0.0, silent=True))
        continue
    seg, segb = C[:, i0:i1], Cb[:, i0:i1]
    e = seg.sum(0)
    if e.max() < 1e-6:
        rows.append(dict(bar=b, chord="n/a", root=None, conf=0.0, silent=True))
        continue
    keep = e > 0.30 * np.percentile(e, 90)
    if keep.sum() >= 4:
        seg, segb = seg[:, keep], segb[:, keep]
    v = np.median(seg, 1); v /= np.linalg.norm(v) + 1e-9
    vb = np.median(segb, 1); vb /= vb.sum() + 1e-9

    scored = []
    for name, r, q, tv in TP:
        s = float(v @ tv)
        s += 0.35 * float(vb[r])                 # root harus terdengar di bass
        if r in DIAT and q in DIAT[r]:
            s += 0.06                            # prior diatonik C mayor
        scored.append((s, name, r, q))
    scored.sort(reverse=True)
    best = scored[0]
    rows.append(dict(bar=b, chord=best[1], root=best[2], qual=best[3],
                     conf=round(best[0], 4),
                     alt=[s[1] for s in scored[1:4]],
                     chroma=[round(float(x), 3) for x in v],
                     bass=NOTES[int(np.argmax(vb))], silent=False))

# key
prof = np.sum([r["chroma"] for r in rows if not r["silent"]], 0)
prof /= prof.sum() + 1e-9
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
kk = []
for r in range(12):
    kk.append((float(np.corrcoef(prof, np.roll(MAJ, r))[0, 1]), NOTES[r] + " major"))
    kk.append((float(np.corrcoef(prof, np.roll(MIN, r))[0, 1]), NOTES[r] + " minor"))
kk.sort(reverse=True)

json.dump(dict(offset=OFF, bar_sec=BAR, bpm=BPM, nbars=NBARS, audio_dur=dur,
               key=kk[0][1], key_cands=[[k[1], round(k[0], 4)] for k in kk[:5]],
               bars=rows), open("out3/chords.json", "w"), indent=1)

print("key:", kk[0][1], round(kk[0][0], 3), "| next:", kk[1][1], round(kk[1][0], 3))
print(f"\n{'bar':>3} {'t0':>7} {'chord':<8} {'bass':<4} conf  alts")
for r in rows:
    print(f"{r['bar']:3d} {OFF+(r['bar']-1)*BAR:7.2f} {r['chord']:<8} "
          f"{r.get('bass','-'):<4} {r['conf']:.3f}  {' '.join(r.get('alt',[]))}")
