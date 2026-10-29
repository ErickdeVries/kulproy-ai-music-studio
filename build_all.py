"""Rapikan project FL Studio: buang channel duplikat, 7 clip di bar 1, track bernama.

Output: song/arrangement_final.flp
"""
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

SRC = r"C:\Users\User\hermes-music\song\arrangement.flp"
DST = r"C:\Users\User\hermes-music\song\arrangement_clean.flp"

WANT = ["Drums", "Bass", "Guitar", "Piano",
        "Vocals-lead", "Vocals-background", "Pro-other"]
DROP_IID = {0, 1, 2, 3,              # 4 channel drum bawaan (tidak dipakai)
            11, 12, 13, 14, 15, 16}  # duplikat 01_..07_ dari drag gagal


def insert_event_after(tree, after_ie, new_event):
    """EventTree.insert yang aman untuk tree berisi 1 event."""
    root = tree.root
    ridx = after_ie.r + 1
    for ie in root.lst:
        if ie.r >= ridx:
            ie.r += 1
    new_ie = IndexedEvent(ridx, new_event)
    tree._recursive(lambda et: et.lst.add(new_ie))
    return new_ie


p = pyflp.parse(SRC)
root = p.events.root
print("channel sebelum:", p.channel_count, "| root events:", len(root.lst))

lst = sorted(root.lst, key=lambda ie: ie.r)
starts = [(ie.r, ie) for ie in lst if ie.e.id == 64]   # ChannelID.New
arr_new_r = min(ie.r for ie in lst if int(ie.e.id) == 99)  # ArrangementID.New

spans = []
for i, (r0, ie) in enumerate(starts):
    iid = int.from_bytes(bytes(ie.e)[1:3], "little")
    r1 = starts[i + 1][0] if i + 1 < len(starts) else arr_new_r
    spans.append((iid, r0, r1))
print("channel:", [(s[0], s[1], s[2]) for s in spans])

# --- 1. buang channel duplikat
kill = set()
for iid, r0, r1 in spans:
    if iid in DROP_IID:
        kill.update(ie.r for ie in root.lst if r0 <= ie.r < r1)
print("event channel duplikat akan dibuang:", len(kill))

n0 = len(root.lst)
for ie in [ie for ie in root.lst if ie.r in kill]:
    root.lst.remove(ie)
print("root events: %d -> %d" % (n0, len(root.lst)))
for new_r, ie in enumerate(sorted(root.lst, key=lambda x: x.r)):
    ie.r = new_r
print("channel sesudah:", len([ie for ie in root.lst if ie.e.id == 64]))

# --- 2. playlist: 7 clip di pos 0
BPM = float(p.tempo)
PPQ = int(p.ppq)
ch_by_name = {}
for ch in p.channels:
    ch_by_name.setdefault(ch.name, ch)
missing = [n for n in WANT if n not in ch_by_name]
if missing:
    raise SystemExit("channel hilang: %s" % missing)
print("channel terpakai:", {n: ch_by_name[n].iid for n in WANT})

DUR_S = 129.270
LEN_TICKS = int(round(DUR_S * BPM / 60.0 * PPQ))

a = p.arrangements[0]
pl = a.events.first(ArrangementID.Playlist)
pl.clear()
for idx, name in enumerate(WANT):
    ch = ch_by_name[name]
    pl.append(dict(
        position=0, pattern_base=20480, item_index=ch.iid,
        length=LEN_TICKS, track_rvidx=499 - idx, group=0,
        _u1=b"\x78\x00", item_flags=64, _u2=b"\x40\x64\x80\x80",
        start_offset=-1.0, end_offset=-1.0,
    ))

# --- 3. nama track 1..7
tracks = list(a.tracks)
for idx, name in enumerate(WANT):
    t = tracks[idx]
    if TrackID.Name in t.events.ids:
        t.name = name
    else:
        data_ie = next(ie for ie in t.events.lst if ie.e.id == TrackID.Data)
        insert_event_after(t.events, data_ie,
                           UnicodeEvent(TrackID.Name, name.encode("utf-16-le")))

pyflp.save(p, DST)
print("tersimpan:", DST)
