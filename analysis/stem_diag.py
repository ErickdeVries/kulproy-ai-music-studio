"""Diagnostik stem: energi per bar per stem, onset pertama, dan uji chord dari Piano saja."""
import numpy as np, librosa, glob, os

SR = 22050
BAR = 3.0
NB = 77
files = sorted(glob.glob("stems/*.wav"))
prof = {}
for f in files:
    name = os.path.basename(f).split(" - ")[0]
    y, _ = librosa.load(f, sr=SR, mono=True)
    r = librosa.feature.rms(y=y, hop_length=512)[0]
    dt = 512 / SR
    prof[name] = np.array([float(np.mean(r[int(k*BAR/dt):int((k+1)*BAR/dt)])) if int((k+1)*BAR/dt) <= len(r) else 0.0
                           for k in range(NB)])

names = list(prof)
print("stem".ljust(18), "".join(f"{b:>5}" for b in range(1, 78, 4)))
for n in names:
    v = prof[n]
    print(n.ljust(18), "".join(f"{v[b-1]*1000:5.1f}" for b in range(1, 78, 4)))

# onset pertama tiap stem
print("\nfirst onset (s):")
for f in files:
    name = os.path.basename(f).split(" - ")[0]
    y, _ = librosa.load(f, sr=SR, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=256, units="time",
                                    backtrack=False, delta=0.15)
    print(f"  {name:<18} {on[0]:.3f}" if len(on) else f"  {name:<18} -", "  n=", len(on))

# beat phase
y, _ = librosa.load("mix.wav", sr=SR, mono=True)
tempo, beats = librosa.beat.beat_track(y=y, sr=SR, units="time", start_bpm=80, tightness=200)
ph = np.sort((beats % BAR))
print(f"\ntempo {float(np.atleast_1d(tempo)[0]):.2f}  nbeats {len(beats)}")
print("beat phase mod 3s: median %.3f  mean %.3f" % (np.median(ph), ph.mean()))
