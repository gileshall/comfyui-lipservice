# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""A cast in one picture, each character sung by itself and put back.

The scene is one image with every character in it, so they share its light. Cast Panels cuts a
panel round each character, for a video model to animate one at a time. Lay Cast lifts each
animated character off its panel with a matte and lays it on the scene's empty plate where its
panel came from. Check Plate refuses a plate that is not the scene with the cast gone.
"""

import math

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image
from scipy import ndimage

# where each panel was cut from the scene, and where its character stands
Crops = io.Custom("LIPSERVICE_CROPS")
CATEGORY = "image/Lip Service"


def _resize(plane, width, height):
    return np.asarray(Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), "F").resize((width, height), Image.LANCZOS))


def _one(batch, name):
    if batch.shape[0] != 1:
        raise ValueError(f"{name}: one picture, not a batch of {batch.shape[0]}")
    return batch[0].cpu().numpy().astype(np.float32)


def panels(scene, cast, width, height, margin, smallest, singers):
    """scene [H, W, 3], cast [H, W]: the panels [N, height, width, 3] left to right, and their crops."""
    sh, sw = cast.shape
    if scene.shape[:2] != (sh, sw):
        raise ValueError(f"the scene is {scene.shape[1]}x{scene.shape[0]}, the cast's mask {sw}x{sh}")
    lab, _ = ndimage.label(cast > 0.5)
    found = [(i + 1, s) for i, s in enumerate(ndimage.find_objects(lab)) if s[0].stop - s[0].start >= smallest * sh]
    found.sort(key=lambda f: f[1][1].start)
    if not found:
        raise ValueError("no character found in the cast's mask")
    if singers and len(found) != singers:
        raise ValueError(f"{len(found)} characters found in the scene, {singers} singers in the piece")
    out, crops = [], []
    for k, (label, (ys, xs)) in enumerate(found):
        h = min((ys.stop - ys.start) * (1 + 2 * margin), sh)
        w = h * width / height
        if w > sw:
            raise ValueError(f"character {k + 1}: a {width}:{height} panel round it is wider than the scene")
        cx = (xs.start + xs.stop) / 2
        top = min(max((ys.start + ys.stop) / 2 - h / 2, 0), sh - h)
        left = min(max(cx - w / 2, 0), sw - w)
        x0, y0, x1, y1 = round(left), round(top), round(left + w), round(top + h)
        others = (lab[y0:y1, x0:x1] != label) & (cast[y0:y1, x0:x1] > 0.5)
        if others.any():
            raise ValueError(f"character {k + 1}: its panel takes in part of another character; they stand too close for a panel each")
        out.append(np.clip(np.stack([_resize(scene[y0:y1, x0:x1, c], width, height) for c in range(3)], axis=2), 0, 1))
        crops.append({"crop": [x0, y0, x1, y1], "feet": [float(cx), float(ys.stop)], "wide": float(xs.stop - xs.start)})
    return np.stack(out), crops


def lift(seen, matte, solid, reach):
    """seen [H, W, 3], matte [H, W]: the character's colour times alpha, and alpha.

    A rim pixel is part character, part backdrop, and nothing says what the character's own colour
    is there. The rim takes it from just inside: the solid part's colour spread outwards over reach."""
    core = ndimage.binary_erosion(matte > solid).astype(np.float32)
    weight = ndimage.gaussian_filter(core, reach)
    inside = np.stack([ndimage.gaussian_filter(seen[..., c] * core, reach) for c in range(3)], axis=2) / (weight[..., None] + 1e-6)
    a = matte * (weight > 1e-3)
    return np.where(core[..., None] > 0, seen, inside) * a[..., None], a


def shadows(plate, crops, strength):
    """The plate with a soft dark patch where each character stands: a tight one under the feet, a broad one round it."""
    out = plate.copy()
    yy, xx = np.mgrid[0:plate.shape[0], 0:plate.shape[1]].astype(np.float32)
    for c in crops:
        (x, y), r = c["feet"], c["wide"] * 0.5
        tight = ((xx - x) / (r * 0.95)) ** 2 + ((yy - y + 2) / (r * 0.12)) ** 2
        broad = ((xx - x) / (r * 1.7)) ** 2 + ((yy - y) / (r * 0.4)) ** 2
        out *= ((1 - strength * np.exp(-tight ** 2)) * (1 - 0.5 * strength * np.exp(-broad)))[..., None]
    return out


def lay(canvas, panel, matte, crop, solid, reach):
    """One panel's character laid on canvas where the panel was cut from."""
    pre, a = lift(panel, matte, solid, reach)
    x0, y0, x1, y1 = crop
    small = np.clip(np.stack([_resize(p, x1 - x0, y1 - y0) for p in (pre[..., 0], pre[..., 1], pre[..., 2], a)], axis=2), 0, 1)
    canvas[y0:y1, x0:x1] = small[..., :3] + (1 - small[..., 3:]) * canvas[y0:y1, x0:x1]


def outside(cast, grow):
    """Where the scene should be untouched: everywhere but the cast and the pixels round it."""
    near = ndimage.binary_dilation(cast > 0.5, iterations=grow) if grow else cast > 0.5
    if near.all():
        raise ValueError("the cast's mask covers the whole scene")
    return ~near


def toned(scene, plate, out):
    """The plate with its tone brought back to the scene's: a gain and an offset a channel, fitted outside the cast."""
    fixed = np.empty_like(plate)
    for c in range(plate.shape[-1]):
        s, p = scene[..., c][out].astype(np.float64), plate[..., c][out].astype(np.float64)
        spread = p.var()
        gain = ((s - s.mean()) * (p - p.mean())).mean() / spread if spread > 1e-8 else 1.0  # a flat plate has an offset only
        fixed[..., c] = np.clip((plate[..., c] - p.mean()) * gain + s.mean(), 0.0, 1.0)
    return fixed


def changed(scene, plate, cast, left, out):
    """How far the plate is from the scene outside the cast (mean absolute difference), and how much of the cast's area is still found in it."""
    return float(np.abs(scene - plate)[out].mean()), float((left > 0.5).sum() / max((cast > 0.5).sum(), 1))


class CastPanels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceCastPanels",
            display_name="Lip Service Cast Panels",
            category=CATEGORY,
            description="A panel round each character of a scene, left to right, for a video model to animate one at a time. "
                        "The characters are the separate shapes of the mask.",
            search_aliases=["crop", "characters", "cast", "panels", "bounding box"],
            inputs=[
                io.Image.Input("scene", tooltip="One picture with every character in it."),
                io.Mask.Input("cast", tooltip="The characters, from a background remover run on the scene."),
                io.Int.Input("width", default=384, min=64, max=4096, step=16, tooltip="A panel's width: what the video model is given."),
                io.Int.Input("height", default=640, min=64, max=4096, step=16, tooltip="A panel's height."),
                io.Float.Input("margin", default=0.10, min=0.0, max=1.0, step=0.01, round=0.01, tooltip="Room above and below the character, as a share of its height."),
                io.Float.Input("smallest", default=0.06, min=0.0, max=1.0, step=0.01, round=0.01,
                               tooltip="A shape shorter than this share of the scene's height is not a character."),
                io.Int.Input("singers", default=0, min=0, max=64, tooltip="How many characters there must be; another count is an error. 0 takes what is found."),
            ],
            outputs=[
                io.Image.Output(display_name="panels", is_output_list=True, tooltip="One panel a character: what follows runs once for each."),
                Crops.Output(display_name="crops", tooltip="Where each panel was cut from, for Lay Cast."),
            ],
        )

    @classmethod
    def execute(cls, scene, cast, width, height, margin, smallest, singers) -> io.NodeOutput:
        out, crops = panels(_one(scene, "scene")[..., :3], _one(cast, "cast"), width, height, margin, smallest, singers)
        return io.NodeOutput([torch.from_numpy(p)[None] for p in out], crops)


class LayCast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceLayCast",
            display_name="Lip Service Lay Cast",
            category=CATEGORY,
            description="Each animated character lifted off its panel with its matte and laid on the plate where the panel was cut from, frame by frame.",
            search_aliases=["composite", "layers", "matte", "alpha", "plate"],
            is_input_list=True,
            inputs=[
                io.Image.Input("plate", tooltip="The scene with the characters gone."),
                Crops.Input("crops", tooltip="From Cast Panels."),
                io.Image.Input("frames", tooltip="Each panel's frames, in the panels' order."),
                io.Mask.Input("mattes", tooltip="Each panel's character, frame by frame, from a background remover."),
                io.Int.Input("length", default=0, min=0, max=100000, tooltip="How many frames there must be; another count is an error. 0 takes what comes."),
                io.Float.Input("shadow", default=0.5, min=0.0, max=1.0, step=0.05, round=0.01, tooltip="How dark the patch under each character's feet is. 0 for none."),
                io.Float.Input("solid", default=0.9, min=0.5, max=1.0, step=0.01, round=0.01, tooltip="Matte above this is the character itself; its colour is trusted there."),
                io.Float.Input("reach", default=2.5, min=0.5, max=16.0, step=0.5, round=0.1, tooltip="Pixels the character's own colour is carried outwards into its rim."),
            ],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, plate, crops, frames, mattes, length, shadow, solid, reach) -> io.NodeOutput:
        for name, v in (("plate", plate), ("crops", crops), ("length", length), ("shadow", shadow), ("solid", solid), ("reach", reach)):
            if len(v) != 1:
                raise ValueError(f"{name}: one, not {len(v)}")
        crops = crops[0]
        if not len(frames) == len(mattes) == len(crops):
            raise ValueError(f"{len(crops)} panels were cut, {len(frames)} came back animated, with {len(mattes)} mattes")
        n = frames[0].shape[0]
        if length[0] and n != length[0]:
            raise ValueError(f"{n} frames came back animated, {length[0]} are wanted: the video is shorter than the sound, or was not cut to it")
        for k, (f, m) in enumerate(zip(frames, mattes)):
            if f.shape[0] != n or tuple(m.shape) != tuple(f.shape[:3]):
                raise ValueError(f"panel {k + 1}: frames {tuple(f.shape)}, mattes {tuple(m.shape)}, the first panel has {n} frames")
        ground = shadows(_one(plate[0], "plate")[..., :3], crops, shadow[0])
        frames = [f.cpu().numpy().astype(np.float32)[..., :3] for f in frames]
        mattes = [m.cpu().numpy().astype(np.float32) for m in mattes]
        out = np.empty((n, *ground.shape), dtype=np.float32)
        for t in range(n):
            out[t] = ground
            for f, m, c in zip(frames, mattes, crops):
                lay(out[t], f[t], m[t], c["crop"], solid[0], reach[0])
        return io.NodeOutput(torch.from_numpy(np.clip(out, 0, 1)))


class CheckPlate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceCheckPlate",
            display_name="Lip Service Check Plate",
            category=CATEGORY,
            description="The plate handed on if it is the scene with the cast gone; an error if it changed the scene elsewhere or left a character standing. "
                        "An edit model often brightens or tints the whole picture: match_tone takes that back out before the check.",
            search_aliases=["plate", "inpaint", "check", "verify"],
            inputs=[
                io.Image.Input("scene", tooltip="The picture with the characters in it."),
                io.Image.Input("plate", tooltip="The same picture after an edit model took them out."),
                io.Mask.Input("cast", tooltip="The characters in the scene."),
                io.Mask.Input("left", tooltip="A background remover run on the plate: what it still finds."),
                io.Int.Input("grow", default=24, min=0, max=512, tooltip="Pixels round the cast that may change: their shadows and rims."),
                io.Boolean.Input("match_tone", default=True,
                                 tooltip="Bring the plate's brightness and colour back to the scene's, a gain and an offset a channel, fitted outside the cast."),
                io.Float.Input("most_changed", default=0.03, min=0.0, max=1.0, step=0.005, round=0.001,
                               tooltip="The most the plate handed on may differ from the scene outside the cast: mean difference, 0 to 1."),
                io.Float.Input("most_left", default=0.05, min=0.0, max=1.0, step=0.01, round=0.01,
                               tooltip="The most of the cast's area that may still be found in the plate."),
            ],
            outputs=[
                io.Image.Output(display_name="plate"),
                io.Image.Output(display_name="scene", tooltip="The scene, handed on once the plate is good: feed what is costly from here, so a bad plate stops it first."),
            ],
        )

    @classmethod
    def execute(cls, scene, plate, cast, left, grow, match_tone, most_changed, most_left) -> io.NodeOutput:
        was, now, shape, still = _one(scene, "scene")[..., :3], _one(plate, "plate")[..., :3], _one(cast, "cast"), _one(left, "left")
        if was.shape != now.shape or shape.shape != still.shape or shape.shape != was.shape[:2]:
            raise ValueError("the scene, the plate and their masks differ in size")
        out = outside(shape, grow)
        if match_tone:
            now = toned(was, now, out)
            plate = torch.from_numpy(now)[None].to(plate)
        diff, share = changed(was, now, shape, still, out)
        if diff > most_changed:
            raise ValueError(f"the plate differs from the scene by {diff:.3f} outside the cast (most allowed {most_changed}): the edit changed the scene; try another seed")
        if share > most_left:
            raise ValueError(f"{share:.0%} of the cast's area is still found in the plate (most allowed {most_left:.0%}): a character was left; try another seed")
        return io.NodeOutput(plate, scene)


class FrameCount(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LipServiceFrameCount",
            display_name="Lip Service Frame Count",
            category=CATEGORY,
            description="How many video frames cover a sound at a frame rate.",
            search_aliases=["length", "duration", "frames", "audio length"],
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input("fps", default=16.0, min=1.0, max=240.0, step=1.0, round=0.01, tooltip="The video's frames a second."),
            ],
            outputs=[io.Int.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, audio, fps) -> io.NodeOutput:
        return io.NodeOutput(math.ceil(audio["waveform"].shape[-1] * fps / audio["sample_rate"] - 1e-6))
