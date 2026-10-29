"""Verifikasi arrangement_v3.flp + mix: semua angka, bukan klaim.

Pakai: python verify.py [path_flp] [path_mix]

Butuh pyflp. Untuk Python 3.12+ perlu 2 patch (lihat build_all.py):
  1. EventEnum._DUMMY = 255   -> tanpa ini parsing gagal
  2. PlaylistEvent.STRUCT 32B -> tanpa ini position/length salah baca
"""
import os
import sys
import struct

# --- patch pyflp (harus sebelum import pyflp) -------------------------------
import pyflp._events as _ev

_E = _ev.EventEnum
if "_DUMMY" not in _E._member_map_:
    _d = int.__new__(_E, 255)
    _d._name_ = "_DUMMY"
    _d._value_ = 255
    _E._member_map_["_DUMMY"] = _d
    _E._value2member_map_[255] = _d

import construct as c
from pyflp.arrangement import PlaylistEvent, ArrangementID

PlaylistEvent.STRUCT = c.GreedyRange(c.Struct(
    "position" / c.Int32ul,
    "pattern_base" / c.Int16ul,
    "item_index" / c.Int16ul,
    "length" / c.Int32ul,
    "track_rvidx" / c.Int16ul,
    "group" / c.Int16ul,
    "_u1" / c.Bytes(2),
    "item_flags" / c.Int16ul,
    "_u2" / c.Bytes(4),
    "start_offset" / c.Float32l,
    "end_offset" / c.Float32l,
))

import pyflp

FLP = sys.argv[1] if len(sys.argv) > 1 else "arrangement_v3.flp"
MIX = sys.argv[2] if len(sys.argv) > 2 else None
TRACKS_BASE = 500          # FL menomori 500 track; rvidx 499 = track 1


def sec(ticks, ppq, bpm):
    return ticks / ppq / (bpm / 60.0)


def main():
    fails = []
    p = pyflp.parse(FLP)
    arr = p.arrangements[0]
    pl = list(arr.events.first(ArrangementID.Playlist))
    tracks = list(arr.tracks)
    chname = {ch.iid: ch.name for ch in p.channels}

    print("=" * 74)
    print("FLP :", os.path.abspath(FLP))
    print("     tempo %s | ppq %s | channel %s | clip %d"
          % (p.tempo, p.ppq, p.channel_count, len(pl)))
    print("=" * 74)

    # --- header ------------------------------------------------------------
    if p.tempo != 80.0:
        fails.append("tempo %s != 80" % p.tempo)
    if p.ppq != 96:
        fails.append("ppq %s != 96" % p.ppq)

    # --- channel + sample path --------------------------------------------
    print("\nCHANNEL")
    for ch in p.channels:
        sp = getattr(ch, "sample_path", "") or ""
        path = os.path.expandvars(sp)
        ada = os.path.exists(path) if sp else False
        if not sp:
            fails.append("channel iid %d tanpa sample_path" % ch.iid)
        elif not ada:
            fails.append("sample tidak ada: %s" % path)
        print("  iid %-2d %-20s %-26s %s"
              % (ch.iid, str(ch.name), os.path.basename(path), "ada" if ada else "HILANG"))

    # --- playlist ----------------------------------------------------------
    print("\nPLAYLIST")
    print("  %-8s %-20s %-20s %8s %9s %8s" %
          ("track", "nama track", "nama channel", "pos", "length", "detik"))
    for it in pl:
        t = TRACKS_BASE - it["track_rvidx"]
        nm = tracks[t - 1].name
        cn = chname.get(it["item_index"], "?")
        dur = sec(it["length"], p.ppq, p.tempo)
        print("  %-8s %-20s %-20s %8d %9d %8.2f"
              % (t, str(nm), cn, it["position"], it["length"], dur))
        if nm != cn:
            fails.append("track %d: nama track '%s' != nama channel '%s'" % (t, nm, cn))
        if it["position"] != 0:
            fails.append("track %d: position %d != 0" % (t, it["position"]))

    # --- panjang clip harus identik & sesuai durasi lagu -------------------
    lens = {it["length"] for it in pl}
    if len(lens) != 1:
        fails.append("panjang clip tidak seragam: %s" % sorted(lens))
    else:
        L = lens.pop()
        d = sec(L, p.ppq, p.tempo)
        print("\n  panjang clip seragam: %d tick = %.3f s" % (L, d))
        if not (225.0 <= d <= 235.0):
            fails.append("durasi clip %.2f s di luar rentang wajar 225-235 s" % d)

    # --- mix ---------------------------------------------------------------
    if MIX and os.path.exists(MIX):
        try:
            import soundfile as sf
            import numpy as np

            x, sr = sf.read(MIX, always_2d=True)
            peak = 20 * np.log10(np.abs(x).max() + 1e-12)
            rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
            jump = np.abs(np.diff(x.mean(1)))
            nclick = int((jump > 0.5).sum())
            print("\nMIX : %s" % os.path.abspath(MIX))
            print("  durasi %.2f s | SR %d | ch %d" % (len(x) / sr, sr, x.shape[1]))
            print("  peak %.2f dBFS | RMS %.2f dBFS | sampel loncat >0.5: %d"
                  % (peak, rms, nclick))
            if peak > -1.0:
                fails.append("peak %.2f dBFS > -1.0" % peak)
            if nclick > 0:
                fails.append("terdeteksi %d klik" % nclick)
        except ImportError:
            print("\n(mix dilewati: soundfile/numpy tidak terpasang)")

    # --- ringkasan ---------------------------------------------------------
    print("\n" + "=" * 74)
    if fails:
        print("GAGAL — %d masalah:" % len(fails))
        for f in fails:
            print("  -", f)
        return 1
    print("LULUS — semua pemeriksaan lolos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
