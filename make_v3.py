"""Buat song/arrangement_v3.flp: 10 channel + 10 clip @ pos 0, track bernama.

Template : song/arrangement_clean.flp
Pola     : build_all.py (patch pyflp + workaround EventEnum)
"""
import copy

import pyflp._events as _ev

# --- workaround bug pyflp 2.2.1 di Python 3.12+ (EventEnum kosong)
_E = _ev.EventEnum
_d = int.__new__(_E, 255)
_d._name_ = "_DUMMY"
_d._value_ = 255
_E._member_map_["_DUMMY"] = _d
if isinstance(_E._member_names_, dict):
    _E._member_names_["_DUMMY"] = None
else:
    _E._member_names_.append("_DUMMY")
_E._value2member_map_[255] = _d

import construct as c
import pyflp
from pyflp._events import UnicodeEvent, IndexedEvent
from pyflp.arrangement import ArrangementID, PlaylistEvent, TrackID
from pyflp.channel import ChannelID
from pyflp.plugin import PluginID

NAME_IDS = (int(PluginID.Name), int(ChannelID._Name))

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

SRC = r"C:\Users\User\hermes-music\song\arrangement_clean.flp"
DST = r"C:\Users\User\hermes-music\song\arrangement_v3.flp"

STEMS = [
    ("Drums", "01_Drums.wav"),
    ("Bass", "02_Bass.wav"),
    ("Guitar", "03_Guitar.wav"),
    ("Piano", "04_Piano.wav"),
    ("Vocals-lead", "05_Vocals-lead.wav"),
    ("Vocals-background", "06_Vocals-background.wav"),
    ("Pro-other", "07_Pro-other.wav"),
    ("Strings", "08_Strings.wav"),
    ("Cello", "09_Cello.wav"),
    ("Timpani", "10_Timpani.wav"),
]
WANT = [n for n, _ in STEMS]
LEN_TICKS = 293913
SAMPLE_DIR = "%USERPROFILE%\\hermes-music\\song\\out3\\stems\\"

p = pyflp.parse(SRC)
root = p.events.root
print("template: tempo=%s ppq=%s channel_count=%s events=%d"
      % (p.tempo, p.ppq, p.channel_count, len(root.lst)))

lst = sorted(root.lst, key=lambda ie: ie.r)
starts = [(ie.r, ie) for ie in lst if int(ie.e.id) == ChannelID.New]
arr_new_r = min(ie.r for ie in lst if int(ie.e.id) == ArrangementID.New)
spans = []
for i, (r0, ie) in enumerate(starts):
    iid = int.from_bytes(bytes(ie.e)[1:3], "little")
    r1 = starts[i + 1][0] if i + 1 < len(starts) else arr_new_r
    spans.append((iid, r0, r1))

header = [ie.e for ie in lst if ie.r < starts[0][0]]
arrangement = [ie.e for ie in lst if ie.r >= arr_new_r]

# prototipe: blok channel yang punya Name + SamplePath (sampler stem)
proto = None
for iid, r0, r1 in spans:
    evs = [ie.e for ie in lst if r0 <= ie.r < r1]
    ids = {int(e.id) for e in evs}
    if int(PluginID.Name) in ids and int(ChannelID.SamplePath) in ids:
        proto = evs
        print("prototipe iid=%d, %d event" % (iid, len(proto)))
        break
assert proto is not None

# --- 1. 10 channel baru: clone prototipe, iid 0..9, nama + sample path
final = list(header)
for idx, (name, stem) in enumerate(STEMS):
    for ev in proto:
        e = copy.deepcopy(ev)
        eid = int(e.id)
        if eid == int(ChannelID.New):
            e.value = idx
        elif eid == int(PluginID.Name):
            e.value = name
        elif eid == int(ChannelID.SamplePath):
            e.value = SAMPLE_DIR + stem
        final.append(e)

# --- 2. playlist: 10 clip, position 0, length 293913 tick
a = p.arrangements[0]
pl = a.events.first(ArrangementID.Playlist)
pl.clear()
for idx in range(len(STEMS)):
    pl.append(dict(
        position=0, pattern_base=20480, item_index=idx,
        length=LEN_TICKS, track_rvidx=499 - idx, group=0,
        _u1=b"\x78\x00", item_flags=64, _u2=b"\x40\x64\x80\x80",
        start_offset=-1.0, end_offset=-1.0,
    ))
print("playlist item:", len(pl))

# --- 3. track 0..9 diberi nama (track_idx = 499 - track_rvidx)
trk_idx = -1
for i, e in enumerate(arrangement):
    eid = int(e.id)
    if eid == TrackID.Data:
        trk_idx += 1
        final.append(e)
        nxt = arrangement[i + 1] if i + 1 < len(arrangement) else None
        # template sudah punya event Name untuk track 0..6: jangan sisip duplikat
        if trk_idx < len(WANT) and (nxt is None or int(nxt.id) != TrackID.Name):
            final.append(UnicodeEvent(TrackID.Name, WANT[trk_idx].encode("utf-16-le")))
    elif eid == TrackID.Name:
        if trk_idx < len(WANT):
            e.value = WANT[trk_idx]
        final.append(e)
    else:
        final.append(e)

print("total event baru:", len(final))

# --- 4. tulis ulang root tree dengan r berurutan
root.lst.clear()
for r, e in enumerate(final):
    root.lst.add(IndexedEvent(r, e))

print("channel_count baru:", len(p.channels))
pyflp.save(p, DST)
print("tersimpan:", DST)
