# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""Lip Service in ComfyUI: Say and Sing, each giving AUDIO.

Every parameter of lipservice.say and lipservice.sing is an input, with the package's own default.
The voices, qualities and motions are the package's. The samples are handed on as the package
returns them: mono, 24 kHz, float32.
"""

import inspect

import numpy as np
import torch
from comfy_api.latest import io

import lipservice

SAY = inspect.signature(lipservice.say).parameters
SING = inspect.signature(lipservice.sing).parameters
# sing's motion=None, the plain vibrato
NO_MOTION = "none"
if NO_MOTION in lipservice.MOTIONS:
    raise RuntimeError(f"lipservice has a motion named {NO_MOTION!r}")

SCORE = """bpm 90
amazing D4:1 G4:2 B4:0.5+G4:0.5
grace B4:2
rest 1
how A4:1
sweet G4:2"""


def _audio(clip):
    """A Clip as ComfyUI AUDIO, [batch 1, channel 1, samples], its samples as they are."""
    samples = torch.from_numpy(np.frombuffer(clip.samples, dtype=np.float32))
    return {"waveform": samples.reshape(1, 1, -1), "sample_rate": lipservice.RATE}


def _float(params, name, lo, hi, step, tooltip):
    return io.Float.Input(name, default=params[name].default, min=lo, max=hi, step=step, round=0.01, tooltip=tooltip)


def _voice(params):
    vs = lipservice.voices()
    return io.Combo.Input("voice", options=[v["name"] for v in vs], default=params["voice"].default,
                          tooltip="Who: " + ", ".join(f"{v['name']} ({v['sex']})" for v in vs) + ".")


def _quality(params):
    return io.Combo.Input("quality", options=list(lipservice.QUALITIES), default=params["quality"].default, tooltip="hifi sounds best; fast is quicker.")


def _seed(params, tooltip):
    return io.Int.Input("seed", default=params["seed"].default, min=0, max=2**32 - 1, control_after_generate=io.ControlAfterGenerate.fixed, tooltip=tooltip)


class Say(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceSay",
            display_name="Lip Service Say",
            category="audio/Lip Service",
            description="Text said by a Lip Service voice: mono, 24 kHz, made on the CPU.",
            search_aliases=["text to speech", "tts", "speech", "speak", "voice"],
            inputs=[
                io.String.Input("text", multiline=True, placeholder="What to say. Paragraphs are said one at a time with a pause between.",
                                tooltip="What to say. With ssml on, one <speak> element."),
                _voice(SAY),
                _quality(SAY),
                _float(SAY, "rate", 0.25, 4.0, 0.05, "Speed, times: 1.15 is 15% faster."),
                _float(SAY, "shift", -24.0, 24.0, 1.0, "Semitones up or down from the voice's own register."),
                _seed(SAY, "The same seed says it the same way; another gives another take."),
                io.Boolean.Input("ssml", default=SAY["ssml"].default, tooltip="The text is SSML."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, **kw) -> io.NodeOutput:
        return io.NodeOutput(_audio(lipservice.say(**kw)))


class Sing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        motion = SING["motion"].default
        return io.Schema(
            node_id="LipServiceSing",
            display_name="Lip Service Sing",
            category="audio/Lip Service",
            description="A score sung by a Lip Service voice: mono, 24 kHz, made on the CPU.",
            search_aliases=["sing", "singing", "song", "score", "vocal", "melody"],
            inputs=[
                io.String.Input("song", multiline=True, placeholder=SCORE,
                                tooltip="The score: a word a line, a NOTE:BEATS for each syllable (C4, F#3, Bb2 or a MIDI number), + joining notes on one syllable, "
                                        "`rest BEATS`, `bpm N` once, # for a comment."),
                _voice(SING),
                _quality(SING),
                _float(SING, "shift", -24.0, 24.0, 1.0, "Semitones, the whole song: -12 puts a song written for a woman in a man's range."),
                _seed(SING, "The same seed sings it the same way; with a motion, it picks the singer."),
                _float(SING, "vibrato", 0.0, 3.0, 0.05, "The vibrato's peak, semitones. 0 for none."),
                _float(SING, "vibrato_hz", 0.1, 12.0, 0.1, "The vibrato's speed."),
                _float(SING, "vibrato_delay", 0.0, 4.0, 0.05, "Seconds into a vowel before the vibrato starts."),
                io.Combo.Input("motion", options=[NO_MOTION, *lipservice.MOTIONS], default=NO_MOTION if motion is None else motion,
                               tooltip="A trained singer's pitch motion in place of the plain vibrato; none keeps the vibrato."),
                _float(SING, "motion_scale", 0.0, 3.0, 0.05, "The motion's size, times."),
                io.Int.Input("glide", default=SING["glide"].default, min=1, max=100, step=1, tooltip="Frames (10 ms) the pitch takes between notes."),
                _float(SING, "dynamics", 0.0, 3.0, 0.05, "How far loudness and brightness follow the notes. 0 is none."),
                _float(SING, "breathiness", 0.0, 3.0, 0.05, "The sung vowels' breath noise, times."),
                io.Boolean.Input("diction", default=SING["diction"].default, tooltip="Every vowel sung as a stressed one."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, motion, **kw) -> io.NodeOutput:
        return io.NodeOutput(_audio(lipservice.sing(motion=None if motion == NO_MOTION else motion, **kw)))
