"""
Mesin produksi musik sederhana: komposisi -> audio WAV + MIDI (untuk import ke FL Studio).
Semua parameter di bagian CONFIG bisa diubah.
"""
import numpy as np
import wave, os

# ============ CONFIG ============
SR      = 44100
BPM     = 90
BARS    = 8
OUT     = os.path.dirname(os.path.abspath(__file__))
# ================================

BEAT = 60.0 / BPM
BAR  = 4 * BEAT
TOTAL = BARS * BAR
N = int(TOTAL * SR) + SR  # +1s tail untuk reverb

rng = np.random.default_rng(7)


def hz(m):
    """MIDI note -> frekuensi."""
    return 440.0 * 2 ** ((m - 69) / 12.0)


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    """Envelope ADSR dengan panjang n sample (detik)."""
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    s = max(1, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, max(1, a), endpoint=False),
        np.linspace(1, s, max(1, d), endpoint=False),
        np.full(s, s),
        np.linspace(s, 0, max(1, r)),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def additive(freq, dur, harmonics=30, detune=0.0, shape="saw"):
    """Sintesis aditif (bebas aliasing). shape: saw / tri / square."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    kmax = min(harmonics, int(20000 / max(freq, 20)))
    for k in range(1, kmax + 1):
        if shape == "saw":
            amp = 1.0 / k
        elif shape == "tri":
            amp = (1.0 / k ** 2) * ((-1) ** ((k - 1) // 2)) if k % 2 else 0.0
        else:  # square
            amp = (1.0 / k) if k % 2 else 0.0
        if amp == 0:
            continue
        f = freq * k * (1 + detune)
        out += amp * np.sin(2 * np.pi * f * t)
    return out / (np.max(np.abs(out)) + 1e-9)


class Mix:
    def __init__(self, n):
        self.L = np.zeros(n)
        self.R = np.zeros(n)
        self.buses = {}          # nama bus -> [L, R] untuk export stem

    def add(self, sig, at, gain=1.0, pan=0.0, bus=None):
        """pan -1..1 ; bus = nama track untuk export stem terpisah"""
        i = int(at * SR)
        if i < 0:
            sig = sig[-i:]
            i = 0
        end = min(len(self.L), i + len(sig))
        if end <= i:
            return
        s = sig[: end - i] * gain
        # equal-power pan
        l = np.cos((pan + 1) * np.pi / 4)
        r = np.sin((pan + 1) * np.pi / 4)
        self.L[i:end] += s * l * 1.414
        self.R[i:end] += s * r * 1.414
        if bus:
            b = self.buses.setdefault(bus, [np.zeros(len(self.L)), np.zeros(len(self.R))])
            b[0][i:end] += s * l * 1.414
            b[1][i:end] += s * r * 1.414


def kick(dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 22) + 42          # pitch sweep 120Hz -> 42Hz
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7)
    click = rng.normal(0, 1, n) * np.exp(-t * 220) * 0.35
    return np.tanh((body + click) * 1.8) * 0.9


def snare(dur=0.25):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.normal(0, 1, n) * np.exp(-t * 22)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30)
    return (noise * 0.8 + tone * 0.5) * 0.55


def hat(dur=0.08):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.normal(0, 1, n)
    # high-pass kasar: beda antar sampel
    noise = np.diff(np.concatenate([[0], noise]))
    return noise * np.exp(-t * 90) * 0.28


def reverb_ir(dur=1.1, decay=3.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, n) * np.exp(-t * decay)
    ir[:200] *= np.linspace(0, 1, 200)   # pre-delay lembut
    return ir / np.max(np.abs(ir))


def convolve(x, ir):
    """Convolusi FFT (fast)."""
    n = len(x) + len(ir) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:n]


# ---------- ARANSEMEN ----------
# Progresi: Am - F - C - G  (i - VI - III - VII di A minor)
CHORDS = {
    "Am": [57, 60, 64],   # A3 C4 E4
    "F":  [53, 57, 60],   # F3 A3 C4
    "C":  [48, 55, 64],   # C3 G3 E4
    "G":  [55, 59, 62],   # G3 B3 D4
}
ROOT = {"Am": 45, "F": 41, "C": 36, "G": 43}   # bass (A2 F2 C2 G2)
PROG = ["Am", "F", "C", "G"] * 2

# melodi: (bar, beat_mulai, durasi_beat, midi_note)
MELODY = [
    (0, 0.0, 1.5, 72), (0, 1.5, 0.5, 76), (0, 2.0, 1.0, 74), (0, 3.0, 1.0, 72),
    (1, 0.0, 1.0, 69), (1, 1.0, 1.0, 72), (1, 2.0, 2.0, 77),
    (2, 0.0, 1.5, 76), (2, 1.5, 0.5, 79), (2, 2.0, 1.0, 76), (2, 3.0, 1.0, 74),
    (3, 0.0, 2.0, 71), (3, 2.0, 1.0, 74), (3, 3.0, 1.0, 79),
    (4, 0.0, 1.5, 81), (4, 1.5, 0.5, 79), (4, 2.0, 1.0, 76), (4, 3.0, 1.0, 72),
    (5, 0.0, 1.0, 69), (5, 1.0, 1.0, 72), (5, 2.0, 2.0, 77),
    (6, 0.0, 1.5, 76), (6, 1.5, 0.5, 74), (6, 2.0, 1.0, 72), (6, 3.0, 1.0, 71),
    (7, 0.0, 4.0, 69),
]

mix = Mix(N)
midi_events = []   # (track, time_detik, durasi, note, velocity)

# PAD
for b, ch in enumerate(PROG):
    for i, note in enumerate(CHORDS[ch]):
        dur = BAR * 0.98
        sig = additive(hz(note), dur, harmonics=26, detune=0.0015, shape="saw")
        sig *= adsr(len(sig), a=0.35, d=0.3, s=0.75, r=0.5)
        pan = -0.35 + i * 0.35
        mix.add(sig, b * BAR, gain=0.10, pan=pan, bus="PAD")
        midi_events.append((1, b * BAR, dur, note, 62))

# BASS
for b, ch in enumerate(PROG):
    root = ROOT[ch]
    pattern = [(0.0, 0.45), (0.75, 0.25), (1.5, 0.45), (2.0, 0.45), (3.0, 0.45), (3.5, 0.4)]
    for bt, d in pattern:
        dur = d * BEAT
        f = hz(root)
        n = int(dur * SR)
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * f * t) * 1.0
        sig += 0.35 * additive(f * 2, dur, harmonics=12, shape="saw")
        sig *= adsr(n, a=0.005, d=0.12, s=0.55, r=0.06)
        mix.add(sig, b * BAR + bt * BEAT, gain=0.30, pan=0.0, bus="BASS")
        midi_events.append((2, b * BAR + bt * BEAT, dur, root, 92))

# MELODY (pluck)
for bar, bt, dbeats, note in MELODY:
    dur = dbeats * BEAT
    sig = additive(hz(note), dur, harmonics=22, shape="tri")
    sig *= adsr(len(sig), a=0.008, d=0.18, s=0.35, r=0.18)
    mix.add(sig, bar * BAR + bt * BEAT, gain=0.22, pan=0.18, bus="LEAD")
    midi_events.append((3, bar * BAR + bt * BEAT, dur, note, 100))

# DRUMS
for b in range(BARS):
    t0 = b * BAR
    for bt in (0.0, 2.5):
        mix.add(kick(), t0 + bt * BEAT, gain=0.62, bus="KICK")
    for bt in (1.0, 3.0):
        mix.add(snare(), t0 + bt * BEAT, gain=0.42, pan=0.05, bus="SNARE")
    for i in range(8):
        g = 0.16 if i % 2 == 0 else 0.10
        mix.add(hat(), t0 + i * 0.5 * BEAT, gain=g, pan=0.30, bus="HAT")

# ---------- REVERB + MASTER ----------
ir = reverb_ir()
wetL = convolve(mix.L, ir)[:N] * 0.16
wetR = convolve(mix.R, np.roll(ir, 300))[:N] * 0.16
L = mix.L + wetL
R = mix.R + wetR

# ---- gain staging yang benar ----
L -= np.mean(L); R -= np.mean(R)
pre = max(np.max(np.abs(L)), np.max(np.abs(R)))
L /= pre; R /= pre                      # normalisasi dulu ke 1.0
L = np.tanh(L * 1.15) / np.tanh(1.15)   # soft-clip ringan saja
R = np.tanh(R * 1.15) / np.tanh(1.15)
# loudness target ~ -14 LUFS (kira-kira RMS 0.16-0.22 untuk musik)
rms = np.sqrt(np.mean(np.concatenate([L, R]) ** 2))
target_rms = 0.18
gain = target_rms / (rms + 1e-9)
L *= gain; R *= gain
peak = max(np.max(np.abs(L)), np.max(np.abs(R)))
if peak > 0.95:                          # jaga headroom, tanpa merusak RMS
    L = L / peak * 0.95; R = R / peak * 0.95
peak = max(np.max(np.abs(L)), np.max(np.abs(R)))

# fade out 0.8s
fo = int(0.8 * SR)
L[-fo:] *= np.linspace(1, 0, fo); R[-fo:] *= np.linspace(1, 0, fo)

stereo = np.stack([L, R], axis=1)
pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
wav_path = os.path.join(OUT, "demo_loop.wav")
with wave.open(wav_path, "w") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())

print(f"WAV  : {wav_path}")
print(f"Durasi: {len(L)/SR:.1f}s | peak={peak:.2f} | RMS={np.sqrt(np.mean(L**2)):.3f}")

# ---------- EXPORT STEM (untuk import ke FL Studio per-track) ----------
stem_dir = os.path.join(OUT, "stems")
os.makedirs(stem_dir, exist_ok=True)
for name, (bl, br) in mix.buses.items():
    m = max(np.max(np.abs(bl)), np.max(np.abs(br)), 1e-9)
    st = np.stack([bl / m * 0.9, br / m * 0.9], axis=1)
    pcm = (np.clip(st, -1, 1) * 32767).astype("<i2")
    fp = os.path.join(stem_dir, f"{name}.wav")
    with wave.open(fp, "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"STEM : {fp}")

# ---------- MIDI (untuk import ke FL Studio) ----------
try:
    import mido
    mf = mido.MidiFile(ticks_per_beat=480)
    names = {1: "PAD", 2: "BASS", 3: "LEAD"}
    for tr_i, tr_name in names.items():
        track = mido.MidiTrack(); mf.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=tr_name, time=0))
        ch = {1: 0, 2: 1, 3: 2}[tr_i]
        track.append(mido.Message("program_change", channel=ch, program=81, time=0))
        evs = sorted([e for e in midi_events if e[0] == tr_i], key=lambda e: e[1])
        abs_t = 0
        for _, st, dur, note, vel in evs:
            on_t = int(st / BEAT * 480)
            track.append(mido.Message("note_on", channel=ch, note=note,
                                      velocity=vel, time=max(0, on_t - abs_t)))
            abs_t = on_t
            off = int(dur / BEAT * 480)
            track.append(mido.Message("note_off", channel=ch, note=note,
                                      velocity=0, time=off))
            abs_t += off
        track.append(mido.MetaMessage("end_of_track", time=0))
    mid_path = os.path.join(OUT, "demo_loop.mid")
    mf.save(mid_path)
    print(f"MIDI : {mid_path}  (drag ke FL Studio -> otomatis jadi channel)")
except Exception as e:
    print("MIDI skip:", e)
