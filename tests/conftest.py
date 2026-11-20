# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""The pack, imported as ComfyUI imports it: a package from its folder. ComfyUI's own folder must be
importable (comfy_api), and lipservice installed."""

import importlib.util
import os
import sys

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def pack():
    name = "comfyui_lipservice_under_test"
    spec = importlib.util.spec_from_file_location(name, os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
