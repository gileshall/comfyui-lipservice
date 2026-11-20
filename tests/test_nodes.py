# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""The nodes against the package: every parameter there with its default, and the samples the same
bit for bit."""

import asyncio
import inspect

import numpy as np
import pytest
import torch

import lipservice


@pytest.fixture(scope="session")
def nodes(pack):
    found = asyncio.run(asyncio.run(pack.comfy_entrypoint()).get_node_list())
    return {n.define_schema().node_id: n for n in found}


def inputs(node):
    return node.INPUT_TYPES()["required"]


def package(pack, kw):
    """The node's inputs as the package's arguments."""
    if kw.get("motion") == pack.nodes.NO_MOTION:
        return {**kw, "motion": None}
    return kw


def test_the_nodes(nodes):
    assert set(nodes) == {"LipServiceSay", "LipServiceSing"}


@pytest.mark.parametrize("node_id, fn", [("LipServiceSay", lipservice.say), ("LipServiceSing", lipservice.sing)])
def test_every_parameter_in_order_with_its_default(pack, nodes, node_id, fn):
    got = inputs(nodes[node_id])
    params = inspect.signature(fn).parameters
    assert list(got) == list(params)
    for name, p in params.items():
        if p.default is inspect.Parameter.empty:
            assert "default" not in got[name][1]
        else:
            assert package(pack, {name: got[name][1]["default"]})[name] == p.default, name
        if got[name][0] in ("INT", "FLOAT"):
            assert got[name][1]["min"] <= got[name][1]["default"] <= got[name][1]["max"], name


def test_choices_from_the_package(pack, nodes):
    for node in nodes.values():
        assert inputs(node)["voice"][1]["options"] == [v["name"] for v in lipservice.voices()]
        assert inputs(node)["quality"][1]["options"] == list(lipservice.QUALITIES)
    assert inputs(nodes["LipServiceSing"])["motion"][1]["options"] == [pack.nodes.NO_MOTION, *lipservice.MOTIONS]


def cases(pack):
    song = pack.nodes.SCORE
    return [
        ("LipServiceSay", lipservice.say, dict(text="Talk is cheap. Singing is free.", voice="ada", quality="hifi", rate=1.0, shift=0.0, seed=0, ssml=False)),
        ("LipServiceSay", lipservice.say, dict(text="And now, a little faster.\n\nThen a pause.", voice="ben", quality="fast", rate=1.15, shift=-2.5, seed=7, ssml=False)),
        ("LipServiceSay", lipservice.say, dict(text='<speak>Hello <break time="300ms"/> <emphasis level="strong">there</emphasis>.</speak>', voice="opal",
                                               quality="hifi", rate=0.9, shift=1.0, seed=2**32 - 1, ssml=True)),
        ("LipServiceSing", lipservice.sing, dict(song=song, voice="ada", quality="hifi", shift=0.0, seed=0, vibrato=0.25, vibrato_hz=5.5, vibrato_delay=0.25,
                                                 motion=pack.nodes.NO_MOTION, motion_scale=1.0, glide=9, dynamics=0.0, breathiness=1.0, diction=False)),
        ("LipServiceSing", lipservice.sing, dict(song=song, voice="hal", quality="fast", shift=-12.0, seed=3, vibrato=0.5, vibrato_hz=6.0, vibrato_delay=0.1,
                                                 motion=pack.nodes.NO_MOTION, motion_scale=1.0, glide=20, dynamics=1.0, breathiness=0.5, diction=True)),
        ("LipServiceSing", lipservice.sing, dict(song=song, voice="cora", quality="hifi", shift=0.0, seed=11, vibrato=0.25, vibrato_hz=5.5, vibrato_delay=0.25,
                                                 motion="wide", motion_scale=1.5, glide=4, dynamics=0.5, breathiness=1.5, diction=False)),
    ]


@pytest.mark.parametrize("i", range(6))
def test_samples_are_the_package(pack, nodes, i):
    node_id, fn, kw = cases(pack)[i]
    out = nodes[node_id].execute(**kw)
    assert isinstance(out.result, tuple) and len(out.result) == 1
    a = out.result[0]
    assert set(a) == {"waveform", "sample_rate"} and a["sample_rate"] == lipservice.RATE == 24000
    w = a["waveform"]
    assert isinstance(w, torch.Tensor) and w.dtype == torch.float32 and w.device.type == "cpu"
    want = np.frombuffer(fn(**package(pack, kw)).samples, dtype=np.float32)
    assert w.shape == (1, 1, len(want)) and len(want) > 0
    assert w.numpy().tobytes() == want.tobytes()


def test_a_refusal_comes_through(pack, nodes):
    _, _, kw = cases(pack)[3]
    with pytest.raises(lipservice.Error, match="a note is NOTE:BEATS"):
        nodes["LipServiceSing"].execute(**{**kw, "song": "this is not a score"})
