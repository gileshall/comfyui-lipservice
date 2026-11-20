# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""Lip Service for ComfyUI: speech and singing on the CPU, as AUDIO."""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes import Say, Sing


class LipService(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Say, Sing]


async def comfy_entrypoint() -> LipService:
    return LipService()
