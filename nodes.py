# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""Lip Service in ComfyUI: Say, Sing and Choir, each giving AUDIO.

Every parameter of lipservice.say, lipservice.sing and lipservice.choir is an input, with the
package's own default. The voices, qualities, motions, rooms, pieces and ranges are the package's.
The samples are handed on as the package returns them: 24 kHz float32, mono from Say and Sing,
stereo from Choir.
"""

import inspect
import json
import os

import folder_paths
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

CHOIR = inspect.signature(lipservice.choir).parameters
# choir's None for room, motion and seed: what the piece itself says
OWN = "the piece's own"
if OWN in lipservice.ROOMS or OWN in lipservice.MOTIONS or OWN == NO_MOTION:
    raise RuntimeError(f"lipservice has a room or a motion named {OWN!r}")
# a piece on a wire (what lipservice.load gives) and how a choir was sung (Clip.timing)
Piece = io.Custom("LIPSERVICE_PIECE")
Timing = io.Custom("LIPSERVICE_TIMING")
# singers on a wire, left to right: each a part of a choir (name, voice, song)
Singers = io.Custom("LIPSERVICE_SINGERS")

SCORE = """%bpm 90
amazing D4:1 G4:2 B4:0.5+G4:0.5
grace B4:2
%rest 1
how A4:1
sweet G4:2"""


def _audio(clip):
    """A Clip as ComfyUI AUDIO, [batch 1, channels, samples], its samples as they are."""
    samples = torch.from_numpy(np.asarray(clip.samples, dtype=np.float32))
    return {"waveform": samples.reshape(1, clip.channels, -1), "sample_rate": lipservice.RATE}


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


def _diction():
    """sing's diction as text: rules, a line each. Empty is the package's None, sung by the default rules."""
    if SING["diction"].default is not None:
        raise RuntimeError(f"lipservice.sing's diction defaults to {SING['diction'].default!r}, not None")
    return io.String.Input("diction", multiline=True, default="", placeholder="Rules for how sounds are sung, a line each. Empty: a singer's vowels.",
                           tooltip="Empty: every vowel stressed, a held \"eye\" held open, as every song is sung. `spoken`: vowels as in speech. "
                                   "A rule is [for=WHO] [word=WORD] [SOUND when..] -> WHAT, as a score's %diction lines.")


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
                                        "`%rest BEATS`, `%bpm N` once before the first word, # for a comment."),
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
                _diction(),
                io.String.Input("tags", default="", tooltip="Names this singer goes by besides the voice's, for a rule's for=: separated by commas."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, motion, tags, **kw) -> io.NodeOutput:
        names = [t.strip() for t in tags.split(",") if t.strip()]
        return io.NodeOutput(_audio(lipservice.sing(motion=None if motion == NO_MOTION else motion, tags=names, **kw)))


def _how(name, tooltip):
    """One of choir's ways of singing as a socket: left unconnected, the piece's own."""
    lo, hi = lipservice.HOW[name]
    kind = io.Int if isinstance(lo, int) else io.Float
    return kind.Input(name, optional=True, force_input=True, min=lo, max=hi, tooltip=f"{tooltip} {lo} to {hi}. Unconnected: the piece's own.")


def _how_set(name, tooltip):
    """One of a piece's style values as a number: sing()'s default, choir's range."""
    lo, hi = lipservice.HOW[name]
    return io.Float.Input(name, default=SING[name].default, min=lo, max=hi, step=0.05, round=0.01, tooltip=f"{tooltip} {lo} to {hi}.")


def _inside_input(name):
    """The path of a file in ComfyUI's input folder, refused if it is anywhere else."""
    root = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"{name}: not in the input folder")
    return path


class Choir(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceChoir",
            display_name="Lip Service Choir",
            category="audio/Lip Service",
            description="A piece for several voices, sung as lipservice.dev sings its choirs: each singer a little off the beat and the pitch, breaths in the rests, "
                        "seated left to right, in a room. Stereo, 24 kHz, made on the CPU.",
            search_aliases=["choir", "chorus", "harmony", "round", "canon", "ensemble", "singers"],
            inputs=[
                io.Combo.Input("piece", options=lipservice.choirs(), tooltip="One of lipservice.dev's choirs."),
                _quality(CHOIR),
                io.Boolean.Input("seed_from_piece", default=True, tooltip="The piece's own seed; off, the seed below."),
                io.Int.Input("seed", default=0, min=0, max=2**32 - 1, control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="The singers' draws: the same seed sings it the same way; another gives another take."),
                io.Combo.Input("room", options=[OWN, *lipservice.ROOMS], default=OWN, tooltip="The room the choir sings in."),
                io.Combo.Input("motion", options=[OWN, NO_MOTION, *lipservice.MOTIONS], default=OWN,
                               tooltip="Trained singers' pitch motion for every part; none is each singer's plain vibrato."),
                io.String.Input("voices", multiline=True, default="", placeholder="A voice a line, by part; an empty line keeps the part's own.",
                                tooltip="Who sings each part, in the piece's order: " + ", ".join(v["name"] for v in lipservice.voices()) + "."),
                Piece.Input("loaded", optional=True, tooltip="A piece from Load Choir or Choir From Text, sung in place of the one named."),
                _how("sway", "The pitch's movement, times its own size."),
                _how("vibrato", "The written vibrato's size, times."),
                _how("loose", "How far off the beat and the pitch each singer is, times."),
                _how("glide", "Frames (10 ms) the pitch takes between notes."),
                _how("dynamics", "How far loudness and brightness follow the notes."),
                _how("breath", "The chance of a breath before a phrase."),
                _how("breathiness", "The sung vowels' breath noise, times."),
            ],
            outputs=[io.Audio.Output(), Timing.Output(display_name="timing", tooltip="How each part was sung: its singer, where it starts, its breaths.")],
        )

    @classmethod
    def execute(cls, piece, quality, seed_from_piece, seed, room, motion, voices, loaded=None, **how) -> io.NodeOutput:
        names = [line.strip() or None for line in voices.split("\n")]
        clip = lipservice.choir(
            piece if loaded is None else loaded, quality=quality, seed=None if seed_from_piece else seed,
            room=None if room == OWN else room, motion=None if motion == OWN else motion,
            voices=names if any(names) else None, **{k: v for k, v in how.items() if v is not None})
        return io.NodeOutput(_audio(clip), clip.timing)


class LoadChoir(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        root = folder_paths.get_input_directory()
        files = sorted(f for f in os.listdir(root) if f.lower().endswith(".json") and os.path.isfile(os.path.join(root, f)))
        return io.Schema(
            node_id="LipServiceLoadChoir",
            display_name="Lip Service Load Choir",
            category="audio/Lip Service",
            description="A piece for Lip Service Choir from a JSON file in the input folder: a choir, or a choir with how it is sung, as lipservice.dev saves them.",
            search_aliases=["choir", "load", "piece", "score"],
            inputs=[io.Combo.Input("file", options=files, tooltip="A .json file in ComfyUI's input folder.")],
            outputs=[Piece.Output(display_name="piece")],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        return io.NodeOutput(lipservice.load(_inside_input(file)))

    @classmethod
    def fingerprint_inputs(cls, file):
        s = os.stat(_inside_input(file))
        return f"{s.st_mtime_ns}:{s.st_size}"


class ChoirFromText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceChoirFromText",
            display_name="Lip Service Choir From Text",
            category="audio/Lip Service",
            description="A piece for Lip Service Choir from its JSON text, typed here or arriving on a wire.",
            search_aliases=["choir", "json", "piece", "score"],
            inputs=[io.String.Input("text", multiline=True, placeholder='{"room": "card", "pan": 0.85, "style": {...}, "parts": [{"name": "Alto", "voice": "cora", "song": "%bpm 90\\n..."}]}',
                                    tooltip="A choir as JSON: room, pan, style and parts (name, voice, song), or {\"choir\": ...} with how it is sung.")],
            outputs=[Piece.Output(display_name="piece")],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        return io.NodeOutput(lipservice.loads(text))


class Singer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceSinger",
            display_name="Lip Service Singer",
            category="audio/Lip Service",
            description="One singer of a choir: a voice and the song it sings. Chain singers left to right, then Lip Service Choir From Singers.",
            search_aliases=["choir", "part", "voice", "singer", "score"],
            inputs=[
                io.String.Input("song", multiline=True, placeholder=SCORE,
                                tooltip="This singer's part. %bpm N, %rest BEATS, then a word a line with a NOTE:BEATS for each syllable; + joins notes on one syllable. "
                                        "Written where a woman sings it: a man's voice sings an octave down."),
                _voice(SING),
                io.String.Input("name", default="", tooltip="The singer's name in the timing. Empty: the voice's name."),
                Singers.Input("singers", optional=True, tooltip="The singers seated to this one's left."),
            ],
            outputs=[Singers.Output(display_name="singers")],
        )

    @classmethod
    def execute(cls, song, voice, name, singers=None) -> io.NodeOutput:
        return io.NodeOutput([*(singers or []), {"name": name.strip() or voice, "voice": voice, "song": song}])


class ChoirFromSingers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        lo, hi = lipservice.HOW["glide"]
        return io.Schema(
            node_id="LipServiceChoirFromSingers",
            display_name="Lip Service Choir From Singers",
            category="audio/Lip Service",
            description="A piece for Lip Service Choir from the singers on the wire, seated left to right in a room.",
            search_aliases=["choir", "piece", "singers", "ensemble", "score"],
            inputs=[
                Singers.Input("singers", tooltip="The last Lip Service Singer of the chain."),
                io.Combo.Input("room", options=list(lipservice.ROOMS), tooltip="The room the choir sings in, from most room to none."),
                io.Int.Input("glide", default=SING["glide"].default, min=lo, max=hi, step=1, tooltip="Frames (10 ms) the pitch takes between notes."),
                _how_set("dynamics", "How far loudness and brightness follow the notes."),
                _how_set("breathiness", "The sung vowels' breath noise, times."),
                io.Float.Input("breath", default=0.0, min=lipservice.HOW["breath"][0], max=lipservice.HOW["breath"][1], step=0.05, round=0.01,
                               tooltip="The chance of a breath in each rest."),
                _diction(),
                io.Combo.Input("motion", options=[NO_MOTION, *lipservice.MOTIONS], default=NO_MOTION,
                               tooltip="Trained singers' pitch motion for every part; none is each singer's plain vibrato."),
                io.Float.Input("pan", default=0.85, min=0.0, max=1.0, step=0.05, round=0.01, tooltip="How wide the singers are seated, left to right."),
            ],
            outputs=[Piece.Output(display_name="piece")],
        )

    @classmethod
    def execute(cls, singers, room, glide, dynamics, breathiness, breath, diction, motion, pan) -> io.NodeOutput:
        rules = [line.strip().removeprefix("%diction ").strip() for line in diction.splitlines()]      # as lipservice.sing reads a text of rules
        style = {"glide": glide, "dynamics": dynamics, "breathiness": breathiness, "breath": breath, "diction": [r for r in rules if r and not r.startswith("#")]}
        if motion != NO_MOTION:
            style["motion"] = motion
        choir = {"room": room, "pan": pan, "style": style, "parts": singers}
        piece = {"choir": choir, "voices": [s["voice"] for s in singers]}
        lipservice.loads(json.dumps(piece))
        return io.NodeOutput(piece)
