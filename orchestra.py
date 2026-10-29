"""Sintesis 3 stem orkestra sinematik di atas 'NANA HARAHAP - UNTUKMU'.

Stem:
  08_Strings  : string pad / ensemble (violin+viola, divisi 3-4 suara, bowed)
  09_Cello    : cello sustained (register rendah, arco)
  10_Timpani  : timpani (modal synthesis) - hanya di section besar

Dinamika mengikuti peta Sound of Silence (Disturbed):
  bars  1-17 : pp   (pad tipis, drone)
  bars 18-33 : p->mp (string masuk penuh, cello masuk bar 26)
  bars 34-48 : mf->f (cello penuh, timpani roll masuk bar 40, crescendo)
  bars 49-74 : f->ff (klimaks, octave doubling, timpani hits)
  bars 75-77 : dim  (fade ke pp, tail)

Semua panjang = panjang lagu asli (tidak memotong apa pun).
Output: song/out3/stems/08_Strings.wav, 09_Cello.wav, 10_Timpani.wav
"""
import numpy as np, json, os, soundfile as sf

SR = 44100
BPM, BAR = 80.0, 3.0
OFF = 0.065
OUT = "out3/stems"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20240607)

ch = json.load(open("out3/chords.json"))
BARS = {b["bar"]: b for b in ch["bars"]}
DUR = ch["audio_dur"]
N = int(round(DUR * SR))
print(f"dur {DUR:.4f}s -> N {N} samples")

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUAL = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10],
        "maj7": [0, 4, 7, 11], "sus4": [0, 5, 7], "sus2": [0, 2, 7],
        "add9": [0, 4, 7, 2], "m7b5": [0, 3, 6, 10], "dim": [0, 3, 6]}

def mtof(m): return 440.0 * 2 ** ((m - 69) / 12.0)

# ---------------------------------------------------------------- wavetables
def saw_table(nharm=28, roll=1.15, size=4096):
    ph = np.arange(size) / size * 2 * np.pi
    w = np.zeros(size)
    for k in range(1, nharm + 1):
        w += np.sin(k * ph) / k ** roll
    w -= w.mean()
    return (w / np.max(np.abs(w))).astype(np.float32)

TBL_BRIGHT = saw_table(30, 1.05)
TBL_MID    = saw_table(16, 1.10)
TBL_SOFT   = saw_table(7,  1.30)
TBL = np.stack([TBL_SOFT, TBL_MID, TBL_BRIGHT])   # index 0..2 = gelap..terang
TSIZE = TBL.shape[1]

def wav_render(freq, n, bright=0.5, detunes=(-7.0, 0.0, 7.0), vib_hz=5.2,
               vib_cents=6.0, vib_ramp=0.9, attack=0.45, release=0.9,
               seed=0, gain=1.0, drift=0.0):
    """Satu nada bowed: unison detune + vibrato + swell. Mono float32."""
    if n <= 8: return np.zeros(max(n, 0), np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    r = np.random.default_rng(seed)
    out = np.zeros(n, np.float32)
    vib_env = np.clip(t / max(vib_ramp, 1e-3), 0, 1)
    for i, c in enumerate(detunes):
        cents = c + drift * r.standard_normal()
        f = freq * 2 ** (cents / 1200.0)
        # phase = 2pi*f*t + vibrato term (integral of sin)
        vib = (vib_cents / 1200.0) * np.log(2) * freq * vib_env * \
              np.sin(2 * np.pi * vib_hz * t + r.uniform(0, 2 * np.pi)) / (2 * np.pi * vib_hz)
        phase = 2 * np.pi * f * t + vib
        idx = (phase * (TSIZE / (2 * np.pi))) % TSIZE
        i0 = idx.astype(np.int32)
        frac = (idx - i0).astype(np.float32)
        i1 = (i0 + 1) % TSIZE
        lo, hi = np.floor(bright * 2).astype(int), np.ceil(bright * 2).astype(int)
        lo, hi = int(np.clip(lo, 0, 2)), int(np.clip(hi, 0, 2))
        fw = float(bright * 2 - lo)
        a = TBL[lo][i0] * (1 - frac) + TBL[lo][i1] * frac
        b = TBL[hi][i0] * (1 - frac) + TBL[hi][i1] * frac
        out += (a * (1 - fw) + b * fw)
    out /= len(detunes)
    # envelope
    na, nr = min(int(attack * SR), n // 2), min(int(release * SR), n // 2)
    env = np.ones(n, np.float32)
    if na > 1: env[:na] = np.sin(np.linspace(0, np.pi / 2, na)) ** 2
    if nr > 1: env[-nr:] *= np.cos(np.linspace(0, np.pi / 2, nr)) ** 2
    return (out * env * gain).astype(np.float32)

def timpani_hit(freq, n, gain=1.0, seed=0):
    """Modal synthesis timpani: membran modes + transient noise."""
    if n <= 16: return np.zeros(max(n, 0), np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    r = np.random.default_rng(seed)
    # pitch drop kecil di awal
    bend = 1.0 + 0.025 * np.exp(-t / 0.045)
    out = np.zeros(n, np.float32)
    modes = [(1.00, 1.00, 1.55), (1.504, 0.62, 1.05), (1.742, 0.42, 0.80),
             (2.000, 0.34, 0.62), (2.245, 0.22, 0.48), (2.494, 0.16, 0.38),
             (2.800, 0.10, 0.28)]
    for ratio, amp, tau in modes:
        out += amp * np.exp(-t / tau) * np.sin(2 * np.pi * freq * ratio * bend * t
                                              + r.uniform(0, 6.28))
    # transient: filtered noise burst
    nb = min(n, int(0.09 * SR))
    nz = r.standard_normal(nb).astype(np.float32)
    k = np.exp(-np.arange(nb) / (0.012 * SR))
    nz = np.convolve(nz, np.ones(6, np.float32) / 6, "same") * k
    out[:nb] += 0.5 * nz
    out /= max(np.max(np.abs(out)), 1e-9)
    # soft attack supaya tidak klik
    na = min(int(0.004 * SR), n)
    out[:na] *= np.linspace(0, 1, na)
    return (out * gain).astype(np.float32)

# ---------------------------------------------------------------- peta dinamika
# gain (dB) per stem per bar; None = tidak aktif
def env_points(sec, kind):
    """(bar, dB) kontrol; diinterpolasi linear per bar."""
    if kind == "strings":
        if sec <= 17:  return [(1, -34), (6, -30), (12, -28), (17, -26)]
        if sec <= 33:  return [(18, -24), (24, -20), (30, -17), (33, -15)]
        if sec <= 48:  return [(34, -14), (40, -11), (46, -8), (48, -7)]
        if sec <= 74:  return [(49, -6), (57, -4), (64, -4.5), (70, -6), (74, -7)]
        return [(75, -9), (76, -16), (77, -26), (78, -40)]
    if kind == "cello":
        if sec <= 25:  return []
        if sec <= 33:  return [(26, -30), (30, -24), (33, -21)]
        if sec <= 48:  return [(34, -19), (40, -15), (48, -11)]
        if sec <= 74:  return [(49, -9), (57, -7), (64, -7.5), (74, -10)]
        return [(75, -13), (77, -30), (78, -45)]
    if kind == "timpani":
        if sec <= 39:  return []
        if sec <= 48:  return [(40, -34), (44, -26), (48, -18)]
        if sec <= 74:  return [(49, -14), (57, -11), (64, -12), (74, -16)]
        return [(75, -20), (76, -30), (77, -45)]
    return []

SECTIONS = dict(A=(1, 17), B=(18, 33), C=(34, 48), D=(49, 74), E=(75, 77))
SECS = [(1, 17), (18, 33), (34, 48), (49, 74), (75, 77)]
def sec_of(bar):
    for i, (a, b) in enumerate(SECS):
        if a <= bar <= b: return i
    return 0

def gain_curve(kind):
    """array panjang N: gain linear per sample (dB diinterpolasi per bar)."""
    pts = []
    for (a, b) in SECS:
        pts += env_points(b, kind)
    if not pts: return np.zeros(N, np.float32), np.full(N, -120.0, np.float32)
    pts = sorted(set(pts))
    bx = np.array([p[0] for p in pts], float)          # bar (boleh 78)
    by = np.array([p[1] for p in pts], float)
    bar_idx = (np.arange(N) / SR - OFF) / BAR + 1.0
    db = np.interp(bar_idx, bx, by, left=by[0], right=by[-1])
    return (10 ** (db / 20.0)).astype(np.float32), db

# ---------------------------------------------------------------- harmoni
def chord_pcs(bar):
    b = BARS.get(bar)
    if not b or b.get("silent") or b.get("root") is None: return None, None
    root = b["root"]; q = b.get("qual", "")
    ivs = QUAL.get(q, [0, 4, 7])
    return root, [(root + i) % 12 for i in ivs]

def merge_spans(b0, b1, min_len=2):
    """Gabungkan bar ber-chord sama jadi span."""
    spans, cur = [], None
    for b in range(b0, b1 + 1):
        r, pcs = chord_pcs(b)
        key = (r, tuple(pcs)) if pcs else None
        if cur and key == cur["key"]:
            cur["b1"] = b
        else:
            if cur: spans.append(cur)
            cur = dict(key=key, b0=b, b1=b, root=r, pcs=pcs)
    if cur: spans.append(cur)
    # HANYA gabungkan span kosong ke tetangga; JANGAN cascade span pendek
    # (chord berubah tiap bar - itu memang benar, jangan dihapus)
    out = []
    for s in spans:
        if s["key"] is None:
            if out: out[-1]["b1"] = s["b1"]
            continue
        out.append(s)
    return out

def voice_strings(pcs, prev=None, base=60):
    """Voicing pad 3-4 suara di C4..A5 (base 60 = C4), dekat dgn voicing sebelumnya."""
    if prev:
        cand = []
        for pc in pcs:
            for octv in range(4):
                m = base - 12 + pc + 12 * octv
                if 55 <= m <= 84: cand.append(m)
        cand = sorted(set(cand))
        # satu nada per pitch-class, dipilih terdekat dengan voicing sebelumnya
        sel = []
        for pc in pcs:
            pool = [x for x in cand if x % 12 == pc]
            if not pool: continue
            ref = min(prev) + (max(prev) - min(prev)) / 2 if prev else base
            near = min(pool, key=lambda x: (min(abs(x - p) for p in prev), x))
            sel.append(near)
        sel = sorted(set(sel))
        # hindari tabrakan: geser oktaf, JANGAN ubah pitch-class
        out = []
        for m in sel:
            while out and any(abs(m - o) < 2 for o in out):
                m += 12
            out.append(m)
        out = sorted(out)
        # kalau melebar, rapatkan oktaf bawah ke atas
        while out and out[-1] - out[0] > 24:
            out[0] += 12
            out = sorted(out)
        return out[:4]
    sel = []
    for i, pc in enumerate(pcs):
        m = base + pc
        while m < 55: m += 12
        while m > 74: m -= 12
        sel.append(m + (12 if i == 1 and m < 62 else 0))
    return sorted(set(sel))

def voice_cello(root, pcs):
    """Cello: root (+ fifth bila ruang), register C2..C3 (36..50)."""
    m = 36 + (root % 12)
    while m < 36: m += 12
    while m > 50: m -= 12
    fifth = (root + 7) % 12
    m2 = 36 + fifth
    while m2 <= m: m2 += 12
    while m2 > m + 7: m2 -= 12
    return [m, m2] if m2 != m else [m]

# ---------------------------------------------------------------- render
def place(buf, sig, t0):
    i0 = int(round(t0 * SR))
    if i0 >= len(buf): return
    n = min(len(sig), len(buf) - i0)
    if n > 0: buf[i0:i0 + n] += sig[:n]

strings = np.zeros(N, np.float32)
cello = np.zeros(N, np.float32)
timp = np.zeros(N, np.float32)

g_str, db_str = gain_curve("strings")
g_cel, db_cel = gain_curve("cello")
g_tim, db_tim = gain_curve("timpani")

def gain_at(g, t): return float(g[min(int(t * SR), N - 1)])

def bright_at(db):
    """dinamika -> kecerahan 0..1 (pp gelap, ff terang)."""
    return float(np.clip((db + 34) / 28.0, 0.0, 1.0))

# ---- STRINGS
prev_voi = None
for s in merge_spans(1, 77):
    t0 = OFF + (s["b0"] - 1) * BAR
    t1 = min(OFF + s["b1"] * BAR, DUR)
    dur = t1 - t0
    n = int(dur * SR)
    if n < 64: continue
    voi = voice_strings(s["pcs"], prev_voi)
    prev_voi = voi
    # dinamika: rata-rata dB di tengah span (untuk timbre), gain per-sample
    db = float(np.mean([db_str[min(int((t0 + dur * q) * SR), N - 1)]
                        for q in (0.25, 0.5, 0.75)]))
    br = bright_at(db)
    i0 = int(round(t0 * SR))
    # octave doubling di section klimaks
    oct_double = s["b0"] >= 49 and s["b1"] <= 74
    for j, m in enumerate(voi):
        f = mtof(m)
        sig = wav_render(f, n, bright=br, attack=0.55 if db < -20 else 0.35,
                         release=0.85, vib_cents=4.5 + 3 * br,
                         seed=1000 + j + s["b0"] * 7, gain=1.0 / max(len(voi), 1) ** 0.5,
                         drift=1.2)
        place(strings, sig, t0)
        if oct_double and j <= 1:
            place(strings, wav_render(f * 2, n, bright=br, attack=0.4,
                                      release=0.8, vib_cents=5.0,
                                      seed=2000 + j + s["b0"] * 7,
                                      gain=0.45 / max(len(voi), 1) ** 0.5), t0)

# ---- CELLO
for s in merge_spans(1, 77):
    if s["b1"] < 26: continue
    t0 = OFF + (s["b0"] - 1) * BAR
    t1 = min(OFF + s["b1"] * BAR, DUR)
    n = int((t1 - t0) * SR)
    if n < 64: continue
    voi = voice_cello(s["root"], s["pcs"])
    db = float(np.mean([db_cel[min(int((t0 + (t1 - t0) * q) * SR), N - 1)]
                        for q in (0.25, 0.5, 0.75)]))
    br = bright_at(db)
    for j, m in enumerate(voi):
        sig = wav_render(mtof(m), n, bright=0.25 + 0.5 * br, attack=0.30,
                         release=1.0, vib_cents=3.5 + 2.5 * br,
                         detunes=(-4.0, 4.0), vib_hz=4.6,
                         seed=3000 + j + s["b0"] * 11, gain=0.8, drift=0.8)
        place(cello, sig, t0)

# ---- TIMPANI
for b in range(40, 78):
    r, pcs = chord_pcs(b)
    if r is None: continue
    t_bar = OFF + (b - 1) * BAR
    if gain_at(g_tim, t_bar + 0.1) < 1e-4: continue
    base = 36 + (r % 12)
    while base < 36: base += 12
    while base > 45: base -= 12
    f = mtof(base)
    if b <= 48:
        # roll crescendo (tremolo 16th)
        step = BAR / 16
        for k in range(16):
            tt = t_bar + k * step
            if tt >= DUR: break
            place(timp, timpani_hit(f, int(0.22 * SR), 0.5, seed=b * 40 + k)
                  * (0.35 + 0.65 * k / 15.0), tt)
    else:
        hits = ([(0.0, 1.0), (1.5, 0.55), (2.25, 0.42)] if b % 2 == 0
                else [(0.0, 1.0), (1.5, 0.5)])
        for k, (pos, amp) in enumerate(hits):
            tt = t_bar + pos * BAR / 4
            if tt >= DUR: break
            place(timp, timpani_hit(f, int(1.1 * SR), 0.9, seed=b * 50 + k) * amp, tt)
# final hit bar 75 (tonik F -> akhir lagu di F)
t75 = OFF + 74 * BAR
place(timp, timpani_hit(mtof(41), int(2.0 * SR), 0.9, seed=9999) * gain_at(g_tim, t75) * 1.1, t75)

# ---------------------------------------------------------------- reverb (hall)
def make_ir(decay=2.4, pre=0.022, seed=5):
    n = int(decay * SR)
    r = np.random.default_rng(seed)
    ir = r.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / (decay * SR / 3.2))
    # pre-delay
    p = np.zeros(int(pre * SR), np.float32)
    ir = np.concatenate([p, ir])
    # lowpass supaya tidak tajam
    b = np.ones(6, np.float32) / 6
    ir = np.convolve(ir, b, "same")
    return ir / np.max(np.abs(ir))

IR = make_ir()
def verb(x, wet=0.34):
    n = len(x) + len(IR) - 1
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x.astype(np.float64), nfft)
    H = np.fft.rfft(IR.astype(np.float64), nfft)
    y = np.fft.irfft(X * H, nfft)[:len(x)]
    y /= max(np.max(np.abs(y)), 1e-9)
    y *= np.max(np.abs(x))
    return (x * (1 - wet) + y.astype(np.float32) * wet).astype(np.float32)

print("reverb...")
strings = verb(strings, 0.38)
cello = verb(cello, 0.30)
timp = verb(timp, 0.26)

# ---- TERAPKAN DINAMIKA (per-sample) SETELAH reverb ----------------------
# (kalau sebelum reverb, ekor reverb section keras bocor ke section pelan)
strings *= g_str
cello *= g_cel
timp *= g_tim
print("dinamika diterapkan: strings/cello/timpani")

# ---------------------------------------------------------------- stereo + tulis
def stereo(x, width=0.0, haas_ms=0.0, tilt=1.0):
    """Mono -> stereo. width: 0..1 (detune/phase beda), haas: delay kanan."""
    if haas_ms > 0:
        d = int(haas_ms * SR / 1000)
        R = np.concatenate([np.zeros(d, np.float32), x[:-d] if d else x])
    else:
        R = x
    L = x
    if width > 0:
        # dekorrelasi halus: allpass-ish via short comb berbeda
        def comb(sig, delay, g):
            d = int(delay * SR / 1000)
            out = sig.copy()
            out[d:] += g * sig[:-d]
            return out
        L = comb(L, 11.0, 0.28 * width)
        R = comb(R, 13.7, -0.26 * width)
    return np.stack([L, R], 1)

S_out = stereo(strings, width=0.9, haas_ms=7.0)
C_out = stereo(cello, width=0.45, haas_ms=2.5)
T_out = stereo(timp, width=0.55, haas_ms=1.5)

# ---- kalibrasi level: orkestra harus TERDENGAR di atas lagu -----------
# acuan: RMS lagu asli per section. Target = lagu + offset (dB) per section,
# lalu diterapkan sebagai gain keseluruhan (bentuk dinamika tetap utuh).
mix, _ = __import__("librosa").load("mix.wav", sr=SR, mono=True)
if mix.ndim > 1: mix = mix.mean(1)
mix = mix[:N]
sec_rms = {}
for tag, (b0, b1) in SECTIONS.items():
    i0, i1 = int((OFF + (b0 - 1) * BAR) * SR), min(int((OFF + b1 * BAR) * SR), N)
    sec_rms[tag] = float(np.sqrt(np.mean(mix[i0:i1] ** 2)))

TARGET = {  # offset dB orkestra terhadap lagu (positif = di atas lagu)
    "08_Strings": dict(A=-19.0, B=-12.0, C=-8.0, D=-5.5, E=-9.0),
    "09_Cello":   dict(A=None,  B=-14.0, C=-9.0, D=-6.5, E=-10.0),
    "10_Timpani": dict(A=None,  B=None,  C=-14.0, D=-8.0, E=-12.0),
}
SEC_OF_BAR = {}
for tag, (b0, b1) in SECTIONS.items():
    for b in range(b0, b1 + 1): SEC_OF_BAR[b] = tag

def soft_limit(x, ceiling_db=-1.0, knee=0.75):
    """Limiter lembut: hanya menyentuh puncak, tidak mengubah dinamika."""
    ceil = 10 ** (ceiling_db / 20)
    a = np.abs(x)
    over = a > ceil * knee
    if not np.any(over): return x
    # soft clip tanh di atas knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (ceil * knee + (ceil - ceil * knee) *
                                  np.tanh((a[over] - ceil * knee) / (ceil - ceil * knee)))
    return y.astype(np.float32)

def calibrate(arr, name):
    """Koreksi gain per section supaya RMS = lagu + target; crossfade 40 ms."""
    out = arr.copy()
    gains = []
    for tag, tgt in TARGET[name].items():
        if tgt is None: continue
        b0, b1 = SECTIONS[tag]
        i0, i1 = int((OFF + (b0 - 1) * BAR) * SR), min(int((OFF + b1 * BAR) * SR), N)
        cur = float(np.sqrt(np.mean(arr[i0:i1] ** 2))) + 1e-12
        want = sec_rms[tag] * (10 ** (tgt / 20))
        gains.append(20 * np.log10(want / cur))
    g_med = float(np.median(gains)) if gains else 0.0
    # kurva gain per-sample: konstanta per section, crossfade linear di batas
    gdb = np.full(N, -120.0, np.float64)
    for tag, (b0, b1) in SECTIONS.items():
        tgt = TARGET[name].get(tag)
        if tgt is None: continue
        i0, i1 = int((OFF + (b0 - 1) * BAR) * SR), min(int((OFF + b1 * BAR) * SR), N)
        cur = float(np.sqrt(np.mean(arr[i0:i1] ** 2))) + 1e-12
        gdb[i0:i1] = 20 * np.log10((sec_rms[tag] * 10 ** (tgt / 20)) / cur)
    # isi section kosong dengan nilai tetangga (agar tidak -120 di ekor)
    valid = gdb > -119
    if valid.any():
        idx = np.arange(N)
        gdb = np.interp(idx, idx[valid], gdb[valid])
    # haluskan transisi (moving average 40 ms)
    k = int(0.040 * SR)
    gdb = np.convolve(gdb, np.ones(k, np.float64) / k, "same")
    g2 = (10 ** (gdb / 20))[:, None] if out.ndim == 2 else (10 ** (gdb / 20))
    out = (out * g2).astype(np.float32)
    out = soft_limit(out, -1.0)
    return out, g_med, gains

report = {}
for name, arr in (("08_Strings", S_out), ("09_Cello", C_out), ("10_Timpani", T_out)):
    a, g, per = calibrate(arr, name)
    sf.write(f"{OUT}/{name}.wav", a, SR, subtype="PCM_24")
    # RMS akhir per section
    fin = {}
    for tag, (b0, b1) in SECTIONS.items():
        i0, i1 = int((OFF + (b0 - 1) * BAR) * SR), min(int((OFF + b1 * BAR) * SR), N)
        fin[tag] = round(20 * np.log10(np.sqrt(np.mean(a[i0:i1] ** 2)) + 1e-12), 2)
    report[name] = dict(trim_db=round(float(g), 2),
                        per_section_wanted=[round(float(x), 2) for x in per],
                        final_rms={k: round(float(v), 2) for k, v in fin.items()},
                        peak_db=round(float(20*np.log10(np.max(np.abs(a))+1e-12)), 2),
                        rms_db=round(float(20*np.log10(np.sqrt(np.mean(a**2))+1e-12)), 2))
    print(f"{name}: trim {float(g):+.2f} dB | peak {report[name]['peak_db']:.2f} | "
          f"rms {report[name]['rms_db']:.2f} | per-sec " +
          " ".join(f"{k}={v}" for k, v in fin.items()))

json.dump(dict(N=N, dur=DUR, sr=SR, offset=OFF, bar=BAR, song_sec_rms={k: round(float(v), 6) for k, v in sec_rms.items()},
               target_offset_db=TARGET, report=report),
          open("out3/orch_levels.json", "w"), indent=1)
print("selesai.")
