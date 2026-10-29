"""Verifikasi lengkap stem orkestra v3: dinamika per section + harmoni + sinkron."""
import numpy as np, librosa, json, soundfile as sf

SR = 22050
BAR, OFF = 3.0, 0.065
N = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
SECTIONS = dict(A=(1,17), B=(18,33), C=(34,48), D=(49,74), E=(75,77))
ch = json.load(open("out3/chords.json")); BARS = {b["bar"]: b for b in ch["bars"]}

def L(p): return librosa.load(p, sr=SR, mono=True)[0]
orch = {n: L(f"out3/stems/{n}.wav") for n in ["08_Strings","09_Cello","10_Timpani"]}
mix = L("mix.wav"); inst = L("stems/Instrumental - NANA HARAHAP - UNTUKMU - CD FORMAT-1 - 80bpm - Cmaj.wav")

def rms(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)

print("=== 1. PANJANG & SINKRON ===")
for n, y in orch.items():
    info = sf.info(f"out3/stems/{n}.wav")
    print(f"  {n}: {len(y)/SR:.3f}s  sr={info.samplerate} ch={info.channels} "
          f"sub={info.subtype} | mix {len(mix)/SR:.3f}s | delta {len(y)/SR-len(mix)/SR:+.4f}s")

print("\n=== 2. PETA DINAMIKA (RMS dB per section) ===")
print(f"{'sec':>4} {'bars':>8} {'lagu':>7} {'Strings':>8} {'d':>6} {'Cello':>7} {'d':>6} {'Timpani':>8} {'d':>6}")
tab = {}
for tag,(b0,b1) in SECTIONS.items():
    i0,i1 = int((OFF+(b0-1)*BAR)*SR), int((OFF+b1*BAR)*SR)
    r_m = rms(mix[i0:i1]); row = {}
    line = f"{tag:>4} {b0:3d}-{b1:<3d} {r_m:7.2f}"
    for n in orch:
        v = rms(orch[n][i0:i1]); row[n] = round(float(v),2)
        line += f" {v:8.2f} {v-r_m:+6.2f}" if n!="09_Cello" else f" {v:7.2f} {v-r_m:+6.2f}"
    print(line); tab[tag] = dict(song=round(float(r_m),2), **row)

print("\n=== 3. KESESUAIAN HARMONI (cosine chroma vs lagu, per bar) ===")
Cs = librosa.util.normalize(librosa.feature.chroma_cqt(y=orch["08_Strings"],sr=SR,hop_length=512,bins_per_octave=36),axis=0)
Ci = librosa.util.normalize(librosa.feature.chroma_cqt(y=librosa.effects.harmonic(inst,margin=3),sr=SR,hop_length=512,bins_per_octave=36),axis=0)
Cc = librosa.util.normalize(librosa.feature.chroma_cqt(y=orch["09_Cello"],sr=SR,hop_length=512,bins_per_octave=36,fmin=librosa.note_to_hz("C1"),n_octaves=2),axis=0)
T = librosa.times_like(Cs, sr=SR, hop_length=512)
cos = []
for b in range(20,75):
    i0,i1 = np.searchsorted(T,[OFF+(b-1)*BAR, OFF+b*BAR])
    if i1-i0 < 4: continue
    a=Cs[:,i0:i1].mean(1); b2=Ci[:,i0:i1].mean(1)
    cos.append(float(a@b2/(np.linalg.norm(a)*np.linalg.norm(b2)+1e-9)))
cos=np.array(cos)
print(f"  n={len(cos)} bar | median {np.median(cos):.3f} | mean {cos.mean():.3f} | min {cos.min():.3f} "
      f"| >0.5: {(cos>0.5).sum()}/{len(cos)} | >0.6: {(cos>0.6).sum()}/{len(cos)}")

# root match: bass cello vs chord root
ok=0; tot=0
for b in range(26,75):
    r = BARS[b]["root"]
    if r is None: continue
    i0,i1 = np.searchsorted(T,[OFF+(b-1)*BAR, OFF+b*BAR])
    if i1-i0<4: continue
    v=Cc[:,i0:i1].mean(1); tot+=1
    if int(np.argmax(v))==r: ok+=1
print(f"  cello root benar: {ok}/{tot} bar ({100*ok/max(tot,1):.1f}%)")

print("\n=== 4. BENTUK BUILD (monoton naik?) ===")
secs = ["A","B","C","D","E"]
sv = [tab[t]["08_Strings"] for t in secs]
print("  Strings per section:", " -> ".join(f"{t}:{v:.1f}" for t,v in zip(secs,sv)))
print("  naik monoton A<B<C<D:", sv[0]<sv[1]<sv[2]<sv[3], "| turun di E:", sv[4]<sv[3])
cv = [tab[t]["09_Cello"] for t in secs]
print("  Cello  per section:", " -> ".join(f"{t}:{v:.1f}" for t,v in zip(secs,cv)))
tv = [tab[t]["10_Timpani"] for t in secs]
print("  Timpani per section:", " -> ".join(f"{t}:{v:.1f}" for t,v in zip(secs,tv)))

print("\n=== 5. PEAK / HEADROOM ===")
for n,y in orch.items():
    print(f"  {n}: peak {20*np.log10(np.max(np.abs(y))+1e-12):.2f} dBFS | rms {rms(y):.2f} dB")

json.dump(dict(sections=tab, cos_median=round(float(np.median(cos)),4),
               cos_mean=round(float(cos.mean()),4), cos_min=round(float(cos.min()),4),
               root_ok=f"{ok}/{tot}"), open("out3/orch_verify.json","w"), indent=1)
print("\nwrote out3/orch_verify.json")
