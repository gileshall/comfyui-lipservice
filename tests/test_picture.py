# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""The picture nodes on drawn scenes: panels found and ordered, a character laid back where it was
cut from with no backdrop on its rim, and every wrong count or bad plate an error."""

import asyncio

import numpy as np
import pytest
import torch

W, H = 600, 200
BACK, PLATE = (0.1, 0.6, 0.2), (0.5, 0.5, 0.5)
# x0, x1, y0, y1, colour: three characters of different heights
CAST = [(60, 100, 60, 180, (0.9, 0.1, 0.1)), (280, 330, 40, 180, (0.1, 0.1, 0.9)), (480, 520, 90, 180, (0.9, 0.9, 0.1))]


@pytest.fixture(scope="session")
def nodes(pack):
    found = asyncio.run(asyncio.run(pack.comfy_entrypoint()).get_node_list())
    return {n.define_schema().node_id: n for n in found}


def scene(cast=CAST, back=BACK):
    image = torch.tensor(back).repeat(1, H, W, 1)
    mask = torch.zeros(1, H, W)
    for x0, x1, y0, y1, colour in cast:
        image[0, y0:y1, x0:x1] = torch.tensor(colour)
        mask[0, y0:y1, x0:x1] = 1
    return image, mask


def cut(nodes, image, mask, singers=0, width=96, height=160):
    return nodes["LipServiceCastPanels"].execute(image, mask, width, height, 0.10, 0.06, singers).result


def laid(nodes, plate, crops, frames, mattes, length=0, shadow=0.0):
    return nodes["LipServiceLayCast"].execute([plate], [crops], frames, mattes, [length], [shadow], [0.9], [2.5]).result[0]


def test_panels_left_to_right_one_a_character(nodes):
    image, mask = scene(list(reversed(CAST)))
    panels, crops = cut(nodes, image, mask)
    assert [tuple(p.shape) for p in panels] == [(1, 160, 96, 3)] * 3
    assert [c["feet"][0] for c in crops] == [80.0, 305.0, 500.0]
    for (x0, x1, y0, y1, colour), panel, c in zip(CAST, panels, crops):
        a, top, b, bottom = c["crop"]
        assert a <= x0 and x1 <= b and top <= y0 and y1 <= bottom
        assert (b - a) * 160 == pytest.approx((bottom - top) * 96, abs=160)
        assert torch.allclose(panel[0, 80, 48], torch.tensor(colour), atol=0.02)


def test_specks_are_not_characters(nodes):
    image, mask = scene()
    mask[0, 5:9, 5:9] = 1
    assert len(cut(nodes, image, mask)[0]) == 3


def test_wrong_count_is_an_error(nodes):
    image, mask = scene()
    assert len(cut(nodes, image, mask, singers=3)[0]) == 3
    with pytest.raises(ValueError, match="3 characters found in the scene, 4 singers"):
        cut(nodes, image, mask, singers=4)
    with pytest.raises(ValueError, match="no character"):
        cut(nodes, image, torch.zeros(1, H, W))


def test_characters_too_close_is_an_error(nodes):
    image, mask = scene([CAST[0], (110, 150, 60, 180, (0.1, 0.1, 0.9))])
    with pytest.raises(ValueError, match="too close"):
        cut(nodes, image, mask)


def test_laid_back_where_it_was_cut_with_no_backdrop_on_the_rim(nodes):
    image, mask = scene()
    panels, crops = cut(nodes, image, mask)
    mattes = [cut(nodes, mask[..., None].repeat(1, 1, 1, 3), mask)[0][k][..., 0].clamp(0, 1) for k in range(3)]
    plate = torch.tensor(PLATE).repeat(1, H, W, 1)
    out = laid(nodes, plate, crops, [p.repeat(2, 1, 1, 1) for p in panels], [m.repeat(2, 1, 1) for m in mattes], length=2)
    assert tuple(out.shape) == (2, H, W, 3)
    for x0, x1, y0, y1, colour in CAST:
        assert torch.allclose(out[0, y0 + 4:y1 - 4, x0 + 4:x1 - 4], torch.tensor(colour), atol=0.03)
    assert torch.allclose(out[0, 5, 5], torch.tensor(PLATE), atol=1e-4)
    # the scene's green is nowhere in the result: every pixel lies between the plate and a character
    green = out[..., 1] - torch.maximum(out[..., 0], out[..., 2])
    assert float(green.max()) < 0.03


def test_shadow_darkens_the_plate_under_the_feet(nodes):
    image, mask = scene()
    panels, crops = cut(nodes, image, mask)
    plate = torch.tensor(PLATE).repeat(1, H, W, 1)
    none = [torch.zeros(1, 160, 96)] * 3
    out = laid(nodes, plate, crops, panels, none, shadow=0.5)
    x, y = (int(v) for v in crops[0]["feet"])
    assert float(out[0, min(y, H - 1), x, 0]) < PLATE[0] - 0.1
    assert torch.allclose(laid(nodes, plate, crops, panels, none, shadow=0.0), plate)


def test_wrong_counts_and_lengths_are_errors(nodes):
    image, mask = scene()
    panels, crops = cut(nodes, image, mask)
    plate = torch.tensor(PLATE).repeat(1, H, W, 1)
    mattes = [torch.zeros(1, 160, 96)] * 3
    with pytest.raises(ValueError, match="3 panels were cut, 2 came back"):
        laid(nodes, plate, crops, panels[:2], mattes[:2])
    with pytest.raises(ValueError, match="1 frames came back animated, 5 are wanted"):
        laid(nodes, plate, crops, panels, mattes, length=5)
    with pytest.raises(ValueError, match="panel 2"):
        laid(nodes, plate, crops, [panels[0], panels[1].repeat(2, 1, 1, 1), panels[2]], mattes)


def test_check_plate(nodes):
    image, mask = scene()
    check = lambda plate, left: nodes["LipServiceCheckPlate"].execute(image, plate, mask, left, 4, False, 0.03, 0.05).result[0]
    good = torch.tensor(BACK).repeat(1, H, W, 1)
    assert check(good, torch.zeros(1, H, W)) is good
    assert nodes["LipServiceCheckPlate"].execute(image, good, mask, torch.zeros(1, H, W), 4, False, 0.03, 0.05).result[1] is image
    with pytest.raises(ValueError, match="the edit changed the scene"):
        check(torch.tensor(PLATE).repeat(1, H, W, 1), torch.zeros(1, H, W))
    with pytest.raises(ValueError, match="a character was left"):
        check(good, mask)


def test_check_plate_tone(nodes):
    _, mask = scene()
    grain = torch.rand(1, H, W, 3, generator=torch.Generator().manual_seed(0))
    none = torch.zeros(1, H, W)
    check = lambda plate, match: nodes["LipServiceCheckPlate"].execute(grain, plate, mask, none, 4, match, 0.03, 0.05).result[0]
    tinted = grain * torch.tensor([0.8, 0.85, 0.9]) + torch.tensor([0.1, 0.08, 0.02])
    with pytest.raises(ValueError, match="the edit changed the scene"):
        check(tinted, False)
    out = (mask[0] < 0.5)
    assert (check(tinted, True)[0][out] - grain[0][out]).abs().max() < 1e-4
    with pytest.raises(ValueError, match="the edit changed the scene"):
        check(torch.rand(1, H, W, 3, generator=torch.Generator().manual_seed(1)), True)


def test_frame_count(nodes):
    count = lambda samples, fps: nodes["LipServiceFrameCount"].execute({"waveform": torch.zeros(1, 2, samples), "sample_rate": 24000}, fps).result[0]
    assert count(24000 * 12, 16.0) == 192
    assert count(24000 * 12 + 1, 16.0) == 193
    assert count(24000, 23.976) == 24
