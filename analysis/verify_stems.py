"""Verifikasi: 7 stem diskret == mix?  + energi RMS per section dinamika."""
import numpy as np, librosa, json

SR = 22050
BAR, OFF = 3.0, 0.065
NAMES7 = ["Drums", "Bass", "Guitar", "Piano", "Vocals-lead", "Vocals-background", "Pro-other"]
def hp(n): return f"stems/{n} - NANA HARAHAP - UNTUKMU - CD FORMAT-1 - 80bpm - Cmaj.wav"

mix, _ = librosa.load("mix.wav", sr=SR, mono=True)
S = {}
for n in NAMES7:
    y, _ = librosa.load(hp(n), sr=SR, mono=True)
    S[n] = y
n = min(len(mix), *(len(v) for v in S.values()))
mix, S = mix[:n], {k: v[:n] for k, v in S.items()}
tot = sum(S.values())
r = lambda x: 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)
print(f"mix {r(mix):7.2f} dB | sum7 {r(tot):7.2f} dB | selisih {r(mix)-r(tot):+.2f} dB")
print(f"korelasi mix vs sum7: {np.corrcoef(mix, tot)[0,1]:.4f}")
resid = mix - tot
print(f"residual (mix-sum7) {r(resid):7.2f} dB  -> energi sisa {(np.sum(resid**2)/np.sum(mix**2))*100:.2f}%")

SECS = [("A", 1, 17), ("B", 18, 33), ("C", 34, 48), ("D", 49, 74), ("E", 75, 77)]
print(f"\n{'sec':>4} {'bars':>8} {'mix':>8} {'vox':>8} {'drum':>8} {'bass':>8} {'gtr':>8} {'pno':>8} {'oth':>8}")
sec_rms = {}
for tag, b0, b1 in SECS:
    i0 = int((OFF + (b0 - 1) * BAR) * SR); i1 = int((OFF + b1 * BAR) * SR); i1 = min(i1, n)
    row = {k: r(S[k][i0:i1]) for k in NAMES7}
    sec_rms[tag] = dict(bars=[b0, b1], t0=round(OFF + (b0-1)*BAR, 3), t1=round(OFF + b1*BAR, 3),
                        dur=round((i1-i0)/SR, 3), mix=round(r(mix[i0:i1]), 2),
                        **{k: round(v, 2) for k, v in row.items()})
    print(f"{tag:>4} {b0:3d}-{b1:<3d} {r(mix[i0:i1]):8.2f} " +
          " ".join(f"{row[k]:8.2f}" for k in NAMES7))

json.dump(dict(sections=sec_rms, names=NAMES7,
               mix_minus_sum7_db=round(r(mix)-r(tot), 3),
               corr=round(float(np.corrcoef(mix, tot)[0, 1]), 5),
               resid_pct=round(float(np.sum(resid**2)/np.sum(mix**2)*100), 3)),
          open("out3/section_rms.json", "w"), indent=1)
