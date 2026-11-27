# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""The choir nodes against the package: every parameter of choir() reachable, the lists and ranges
the package's, the samples the same bit for bit in stereo, and files only from the input folder."""

import asyncio
import inspect
import json

import folder_paths
import numpy as np
import pytest
import torch

import lipservice

PIECE = "Barbershop tag"


@pytest.fixture(scope="session")
def nodes(pack):
    found = asyncio.run(asyncio.run(pack.comfy_entrypoint()).get_node_list())
    return {n.define_schema().node_id: n for n in found}


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    return tmp_path


def every(node):
    t = node.INPUT_TYPES()
    return {**t["required"], **t.get("optional", {})}


def stereo(out):
    a = out.result[0]
    assert set(a) == {"waveform", "sample_rate"} and a["sample_rate"] == lipservice.RATE
    w = a["waveform"]
    assert isinstance(w, torch.Tensor) and w.dtype == torch.float32 and w.shape[:2] == (1, 2) and w.shape[2] > 0
    return w[0].numpy()


def same(out, clip):
    assert stereo(out).tobytes() == np.asarray(clip.samples, dtype=np.float32).tobytes()
    assert out.result[1] == clip.timing


def test_every_parameter_of_choir(pack, nodes):
    got = every(nodes["LipServiceChoir"])
    params = inspect.signature(lipservice.choir).parameters
    # songs come in a loaded piece; seed_from_piece and loaded are the node's own
    assert set(params) - set(got) == {"songs"}
    assert set(got) - set(params) == {"seed_from_piece", "loaded"}
    assert all(p.default is None for n, p in params.items() if n not in ("piece", "quality"))
    assert got["quality"][1]["default"] == params["quality"].default


def test_choices_and_ranges_from_the_package(pack, nodes):
    got = every(nodes["LipServiceChoir"])
    own = pack.nodes.OWN
    assert got["piece"][1]["options"] == lipservice.choirs()
    assert got["quality"][1]["options"] == list(lipservice.QUALITIES)
    assert got["room"][1]["options"] == [own, *lipservice.ROOMS] and got["room"][1]["default"] == own
    assert got["motion"][1]["options"] == [own, pack.nodes.NO_MOTION, *lipservice.MOTIONS] and got["motion"][1]["default"] == own
    optional = nodes["LipServiceChoir"].INPUT_TYPES()["optional"]
    for name, (lo, hi) in lipservice.HOW.items():
        kind, opts = optional[name]
        assert kind == ("INT" if isinstance(lo, int) else "FLOAT") and (opts["min"], opts["max"]) == (lo, hi) and opts["forceInput"], name


def ask(pack, **kw):
    own = pack.nodes.OWN
    return {"piece": PIECE, "quality": "hifi", "seed_from_piece": True, "seed": 0, "room": own, "motion": own, "voices": "", **kw}


def test_the_piece_as_it_is(pack, nodes):
    same(nodes["LipServiceChoir"].execute(**ask(pack)), lipservice.choir(PIECE))


def test_everything_asked(pack, nodes):
    out = nodes["LipServiceChoir"].execute(**ask(pack, quality="fast", seed_from_piece=False, seed=5, room="bare", motion=pack.nodes.NO_MOTION, voices="\nleo\n\n",
                                                 sway=0.5, vibrato=1.5, loose=2.0, glide=12, dynamics=0.7, breath=0.2, breathiness=0.6))
    same(out, lipservice.choir(PIECE, quality="fast", seed=5, room="bare", motion="none", voices=[None, "leo"],
                               sway=0.5, vibrato=1.5, loose=2.0, glide=12, dynamics=0.7, breath=0.2, breathiness=0.6))


def test_a_seed_is_another_take(pack, nodes):
    a = stereo(nodes["LipServiceChoir"].execute(**ask(pack, seed_from_piece=False, seed=1)))
    b = stereo(nodes["LipServiceChoir"].execute(**ask(pack, seed_from_piece=False, seed=2)))
    assert a.tobytes() != b.tobytes()


def piece():
    return {"choir": PIECE, "seed": 3, "room": "drier", "how": {"sway": 0.8}}


def test_a_file_from_the_input_folder(pack, nodes, input_dir):
    (input_dir / "tag.json").write_text(json.dumps(piece()))
    (input_dir / "notes.txt").write_text("not a piece")
    assert nodes["LipServiceLoadChoir"].INPUT_TYPES()["required"]["file"][1]["options"] == ["tag.json"]
    loaded = nodes["LipServiceLoadChoir"].execute(file="tag.json").result[0]
    assert loaded == piece()
    same(nodes["LipServiceChoir"].execute(**ask(pack, piece="Ode to Joy", loaded=loaded)), lipservice.choir(piece()))


@pytest.mark.parametrize("name", ["../outside.json", "/etc/hosts", "sub/../../outside.json"])
def test_a_file_anywhere_else_is_refused(nodes, input_dir, name):
    (input_dir.parent / "outside.json").write_text(json.dumps(piece()))
    with pytest.raises(ValueError, match="not in the input folder"):
        nodes["LipServiceLoadChoir"].execute(file=name)


def test_a_link_out_of_the_input_folder_is_refused(nodes, input_dir):
    (input_dir.parent / "outside.json").write_text(json.dumps(piece()))
    (input_dir / "link.json").symlink_to(input_dir.parent / "outside.json")
    with pytest.raises(ValueError, match="not in the input folder"):
        nodes["LipServiceLoadChoir"].execute(file="link.json")


def test_a_piece_from_text(pack, nodes):
    loaded = nodes["LipServiceChoirFromText"].execute(text=json.dumps(piece())).result[0]
    same(nodes["LipServiceChoir"].execute(**ask(pack, loaded=loaded)), lipservice.choir(piece()))


def test_refusals_come_through(pack, nodes):
    with pytest.raises(lipservice.Error, match="a choir has parts"):
        nodes["LipServiceChoirFromText"].execute(text="{}")
    with pytest.raises(lipservice.Error, match="no voice"):
        nodes["LipServiceChoir"].execute(**ask(pack, voices="nobody"))


HIGH = "%bpm 90\nsing C5:1\nwith B4:1\nme C5:2\n"
LOW = "%bpm 90\nsing E4:1\nwith G4:1\nme E4:2\n"


def duet(nodes, **kw):
    one = nodes["LipServiceSinger"].execute(song=HIGH, voice="cora", name="High").result[0]
    two = nodes["LipServiceSinger"].execute(song=LOW, voice="ben", name="", singers=one).result[0]
    return nodes["LipServiceChoirFromSingers"].execute(**{"singers": two, "room": "drier", "glide": 12, "dynamics": 1.0, "breathiness": 0.6,
                                                             "breath": 0.5, "diction": "%diction spoken\n\n# a note\n", "motion": "straight", "pan": 0.5, **kw}).result[0]


def test_singers_typed_on_the_canvas(pack, nodes):
    written = {"choir": {"room": "drier", "pan": 0.5, "style": {"glide": 12, "dynamics": 1.0, "breathiness": 0.6, "breath": 0.5, "diction": ["spoken"], "motion": "straight"},
                         "parts": [{"name": "High", "voice": "cora", "song": HIGH}, {"name": "ben", "voice": "ben", "song": LOW}]},
               "voices": ["cora", "ben"]}
    loaded = duet(nodes)
    assert loaded == written
    out = nodes["LipServiceChoir"].execute(**ask(pack, loaded=loaded))
    same(out, lipservice.choir(written))
    assert [(p["name"], p["voice"]) for p in out.result[1]["parts"]] == [("High", "cora"), ("ben", "ben")]


def test_a_singer_and_a_room_from_the_package(pack, nodes):
    sing = inspect.signature(lipservice.sing).parameters
    singer = every(nodes["LipServiceSinger"])
    assert singer["voice"][1]["options"] == [v["name"] for v in lipservice.voices()] and singer["voice"][1]["default"] == sing["voice"].default
    got = every(nodes["LipServiceChoirFromSingers"])
    assert got["room"][1]["options"] == list(lipservice.ROOMS)
    for name in ("glide", "dynamics", "breathiness"):
        opts = got[name][1]
        assert opts["default"] == sing[name].default and (opts["min"], opts["max"]) == lipservice.HOW[name], name
    assert (got["breath"][1]["min"], got["breath"][1]["max"]) == lipservice.HOW["breath"] and got["diction"][1]["default"] == "" and sing["diction"].default is None
    assert got["motion"][1]["options"] == [pack.nodes.NO_MOTION, *lipservice.MOTIONS] and got["motion"][1]["default"] == pack.nodes.NO_MOTION


def test_the_plain_piece_is_the_engine_s_own(pack, nodes):
    # breath 0, no rules of diction, no motion and pan 0.85 are what the engine takes a bare piece to mean
    one = nodes["LipServiceSinger"].execute(song=HIGH, voice="cora", name="High").result[0]
    got = every(nodes["LipServiceChoirFromSingers"])
    plain = {k: got[k][1]["default"] for k in ("glide", "dynamics", "breathiness", "breath", "diction", "motion", "pan")}
    loaded = nodes["LipServiceChoirFromSingers"].execute(singers=one, room="card", **plain).result[0]
    bare = {"choir": {"room": "card", "style": {k: plain[k] for k in ("glide", "dynamics", "breathiness")}, "parts": one}, "voices": ["cora"]}
    same(nodes["LipServiceChoir"].execute(**ask(pack, loaded=loaded)), lipservice.choir(bare))


def test_a_singer_does_not_change_the_wire_it_reads(nodes):
    one = nodes["LipServiceSinger"].execute(song=HIGH, voice="cora", name="High").result[0]
    nodes["LipServiceSinger"].execute(song=LOW, voice="ben", name="Low", singers=one)
    assert len(one) == 1


def test_a_song_the_engine_refuses(pack, nodes):
    one = nodes["LipServiceSinger"].execute(song="%bpm 90\namazing C5:1\n", voice="cora", name="High").result[0]
    loaded = nodes["LipServiceChoirFromSingers"].execute(singers=one, room="card", glide=9, dynamics=0.0, breathiness=1.0, breath=0.0, diction="", motion="none", pan=0.85).result[0]
    with pytest.raises(lipservice.Error, match="amazing: 3 syllables"):
        nodes["LipServiceChoir"].execute(**ask(pack, loaded=loaded))
