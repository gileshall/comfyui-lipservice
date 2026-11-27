# Lip Service for ComfyUI

Voices that say text and sing scores, made on the CPU by [Lip Service](https://lipservice.dev),
as ComfyUI `AUDIO`: mono, 24 kHz. They go straight into Preview Audio, Save Audio or any node
that takes audio.

## Install

Search for Lip Service in ComfyUI's Manager, or from the command line:

```bash
comfy node install comfyui-lipservice
```

By hand: clone this repository into ComfyUI's `custom_nodes`, install its one requirement into
ComfyUI's Python, and restart ComfyUI:

```bash
pip install -r requirements.txt
```

The first run fetches the models, about 56 MB for `hifi`, from models.lipservice.dev into the
`lipservice` package's own cache (`LIPSERVICE_CACHE` moves it). The voice list is read from the
package when ComfyUI starts. macOS and Linux.

Coming from `lipservice` 0.1 with its models already fetched: run `lipservice fetch` once with
ComfyUI's Python before starting ComfyUI. The cache's list of models is older than the choirs, and
the nodes do not load until it is read again.

## Nodes

**Lip Service Say** says text. **Lip Service Sing** sings a score. Each is the package's `say` or
`sing`, every parameter with its default; hover an input for what it does. **Lip Service Choir**
sings a piece for several voices in stereo, the package's `choir`: one of lipservice.dev's, one
loaded from a file or typed as JSON, or one built on the canvas from **Lip Service Singer** nodes
chained left to right into **Lip Service Choir From Singers**. The seed stays fixed
after a run, so a change is heard on the same take; set its control to `randomize` for a new take
each run.

What the engine refuses, a miscounted word or a note it cannot read, stops the run with the
engine's own words. The samples are handed on exactly as the package returns them.

## A score

A word a line, and a note for each of its syllables:

```
%bpm 90
amazing D4:1 G4:2 B4:0.5+G4:0.5
grace B4:2
%rest 1
how A4:1
sweet G4:2
```

`NOTE:BEATS` with the note as `C4`, `F#3`, `Bb2` or a MIDI number; `+` joins notes on one
syllable; `%rest BEATS` is silence; `%bpm N` is the tempo, once, before the first word. The whole
format is
[SCORE.md](https://github.com/gileshall/lipservice/blob/main/SCORE.md).

## Examples

`example_workflows/` has a say, a sing and a choir, each saving its audio. ComfyUI lists them
among its templates.

## Tests

From a Python that imports ComfyUI's folder (`comfy_api`) and has `lipservice` installed:

```bash
python -m pytest tests
```

```bash
python tests/e2e.py --comfyui COMFYUI_DIR --base SCRATCH_DIR
```

The first checks the nodes' samples bit for bit against the package's; the second runs the
examples on a headless ComfyUI (see its docstring).

## License

AGPL-3.0-only, as Lip Service.
