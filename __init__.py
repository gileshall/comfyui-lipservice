# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""Lip Service for ComfyUI: speech, singing and choirs on the CPU, as AUDIO."""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes import Choir, ChoirFromSingers, ChoirFromText, LoadChoir, Say, Sing, Singer
from .picture import CastPanels, CheckPlate, FrameCount, LayCast


class LipService(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Say, Sing, Choir, LoadChoir, ChoirFromText, Singer, ChoirFromSingers, CastPanels, CheckPlate, FrameCount, LayCast]


async def comfy_entrypoint() -> LipService:
    return LipService()
