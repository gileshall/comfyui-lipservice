# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Giles Hall

"""End to end: a headless ComfyUI with this pack, the example workflows run through its HTTP API,
and the saved audio checked against lipservice's own samples.

    python tests/e2e.py --comfyui COMFYUI_DIR --base SCRATCH_DIR [--python PY] [--port N]

COMFYUI_DIR is a ComfyUI checkout (its main.py); it is run, not changed, and writes no bytecode.
SCRATCH_DIR is its base directory (custom_nodes, input, output, temp, user), with this pack linked
into custom_nodes. PY runs ComfyUI and must import its requirements and lipservice; it defaults to
this Python, which must import lipservice, numpy and av.

The saved files are FLAC from SaveAudioAdvanced and PreviewAudio. ComfyUI hands the encoder float
samples; FLAC holds 16-bit integers here, so each sample is checked to be the nearest 16-bit value
to x * 32768 (within half a step; ties go either way), or the end of the range past full scale:
the format's loss and nothing else.
"""

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import av
import numpy as np

import lipservice

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(PACK, "example_workflows")
NAME = "comfyui-lipservice"
# frontend-only nodes: never sent to the server
NOTES = {"MarkdownNote", "Note"}
# inputs the frontend keeps in widgets_values
WIDGETS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, comfyui, base, python, port):
        self.url = f"http://127.0.0.1:{port}"
        self.base = base
        nodes = os.path.join(base, "custom_nodes")
        os.makedirs(nodes, exist_ok=True)
        link = os.path.join(nodes, NAME)
        if os.path.islink(link):
            if os.path.realpath(link) != PACK:
                raise RuntimeError(f"{link} links to {os.path.realpath(link)}, not {PACK}")
        else:
            os.symlink(PACK, link)
        self.log_path = os.path.join(base, "server.log")
        self.log = open(self.log_path, "w")
        cmd = [python, "-B", "-s", os.path.join(comfyui, "main.py"), "--base-directory", base, "--listen", "127.0.0.1", "--port", str(port), "--cpu",
               "--disable-auto-launch", "--disable-api-nodes"]
        print("server:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, cwd=base, stdout=self.log, stderr=subprocess.STDOUT)

    def tail(self, n=40):
        with open(self.log_path) as f:
            return "".join(f.readlines()[-n:])

    def alive(self):
        if self.proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with {self.proc.returncode}:\n{self.tail()}")

    def request(self, method, route, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + route, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def get(self, route):
        code, body = self.request("GET", route)
        if code != 200:
            raise RuntimeError(f"GET {route}: {code} {body}")
        return body

    def wait_ready(self, seconds=300):
        deadline = time.monotonic() + seconds
        while True:
            self.alive()
            try:
                self.get("/system_stats")
                return
            except (urllib.error.URLError, ConnectionError):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"ComfyUI not answering at {self.url} after {seconds} s:\n{self.tail()}")
                time.sleep(0.5)

    def queue(self, prompt):
        return self.request("POST", "/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})

    def run(self, prompt, seconds=900):
        code, body = self.queue(prompt)
        if code != 200:
            raise RuntimeError(f"POST /prompt: {code} {json.dumps(body)}")
        pid = body["prompt_id"]
        deadline = time.monotonic() + seconds
        while True:
            self.alive()
            h = self.get(f"/history/{pid}")
            if pid in h and h[pid]["status"].get("completed") is not None:
                return h[pid]
            if time.monotonic() > deadline:
                raise RuntimeError(f"prompt {pid} not done after {seconds} s:\n{self.tail()}")
            time.sleep(0.25)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise RuntimeError("ComfyUI did not stop on SIGTERM within 60 s; killed")
        self.log.close()
        print(f"server stopped ({self.proc.returncode})")


def api_prompt(workflow, info):
    """A saved workflow as the frontend queues it: each node's widgets_values given out in its
    input order, a seed's control value passed over, links as [node, slot]."""
    links = {link[0]: link for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        if node["type"] in NOTES:
            continue
        if node["mode"] != 0:
            raise ValueError(f"node {node['id']} ({node['type']}) is muted or bypassed")
        spec = info[node["type"]]
        linked = {i["name"]: links[i["link"]] for i in node["inputs"] if i.get("link") is not None}
        values = list(node["widgets_values"])
        inputs = {}
        for name in spec["input_order"]["required"]:
            kind, *rest = spec["input"]["required"][name]
            if name in linked:
                inputs[name] = [str(linked[name][1]), linked[name][2]]
            elif kind in WIDGETS:
                if not values:
                    raise ValueError(f"node {node['id']} ({node['type']}): no value left for {name}")
                inputs[name] = values.pop(0)
                if rest and rest[0].get("control_after_generate"):
                    values.pop(0)
            else:
                raise ValueError(f"node {node['id']} ({node['type']}): {name} is not linked")
        if values:
            raise ValueError(f"node {node['id']} ({node['type']}): widget values left over: {values}")
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt


def made(node):
    """What the package makes for a node's inputs, called directly."""
    i = node["inputs"]
    if i.get("motion") == "none":
        i = {**i, "motion": None}
    if "tags" in i:
        i = {**i, "tags": [t.strip() for t in i["tags"].split(",") if t.strip()]}
    return {"LipServiceSay": lipservice.say, "LipServiceSing": lipservice.sing}[node["class_type"]](**i)


def decode(path):
    with av.open(path) as f:
        (s,) = f.streams.audio
        frames = [fr.to_ndarray() for fr in f.decode(s)]
        return s.codec_context.sample_rate, s.channels, frames[0].dtype, np.concatenate(frames, axis=1)


def compare(path, want):
    """The file against the package's samples: each the nearest 16-bit value to it (a tie either
    way), or the end of the range where it is past full scale. Anything else fails."""
    rate, channels, dtype, got = decode(path)
    assert rate == lipservice.RATE, f"{path}: {rate} Hz"
    assert channels == 1 and got.shape == (1, len(want)), f"{path}: {channels} channels, {got.shape} samples, want {len(want)}"
    assert dtype == np.int16, f"{path}: decoded as {dtype}, not 16-bit"
    scaled = want.astype(np.float64) * 32768
    g = got[0].astype(np.float64)
    wrong = int(np.count_nonzero(np.abs(g - np.clip(scaled, -32768, 32767)) > 0.5))
    assert wrong == 0, f"{path}: {wrong} of {len(want)} samples are not the nearest 16-bit value to the package's"
    return dict(samples=len(want), peak=round(float(np.abs(want).max()), 4), max_error=float(np.abs(g / 32768 - want.astype(np.float64)).max()),
                clipped=int(np.count_nonzero(np.abs(np.clip(scaled, -32768, 32767) - scaled) > 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", required=True, help="the ComfyUI checkout: its main.py is run")
    p.add_argument("--base", required=True, help="ComfyUI's base directory for this run")
    p.add_argument("--python", default=sys.executable, help="the Python that runs ComfyUI")
    p.add_argument("--port", type=int, default=0, help="0 picks a free one")
    a = p.parse_args()
    base = os.path.abspath(a.base)
    server = Server(os.path.abspath(a.comfyui), base, a.python, a.port or free_port())
    print("ComfyUI:", server.url)
    try:
        server.wait_ready()
        info = server.get("/object_info")
        for node_id in ("LipServiceSay", "LipServiceSing"):
            if node_id not in info:
                raise RuntimeError(f"{node_id} not loaded:\n{server.tail(80)}")
        prompts = {}
        for name in sorted(os.listdir(WORKFLOWS)):
            with open(os.path.join(WORKFLOWS, name)) as f:
                prompts[name] = prompt = api_prompt(json.load(f), info)
            t = time.monotonic()
            entry = server.run(prompt)
            took = time.monotonic() - t
            assert entry["status"]["status_str"] == "success", json.dumps(entry["status"])
            print(f"{name}: success in {took:.1f} s")
            saved = [(v, f) for k, v in prompt.items() for f in entry["outputs"].get(k, {}).get("audio", [])]
            assert saved, f"{name}: no audio saved"
            if prompt["1"]["class_type"] not in ("LipServiceSay", "LipServiceSing"):
                # a choir is built over several nodes: its samples are held to the package's in tests/test_choir.py; here it has to run and save
                for v, f in saved:
                    path = os.path.join(base, f["type"], f["subfolder"], f["filename"])
                    assert os.path.getsize(path) > 10000, path
                    print(f"  {v['class_type']}: {os.path.relpath(path, base)}: saved, {os.path.getsize(path)} bytes")
                continue
            want = np.frombuffer(made(prompt["1"]).samples, dtype=np.float32)
            for v, f in saved:
                path = os.path.join(base, f["type"], f["subfolder"], f["filename"])
                print(f"  {v['class_type']}: {os.path.relpath(path, base)}: {compare(path, want)}")
        say, sing = prompts["lipservice_say.json"], prompts["lipservice_sing.json"]

        bad = copy.deepcopy(say)
        bad["1"]["inputs"]["voice"] = "nobody"
        code, body = server.queue(bad)
        errors = body.get("node_errors", {}).get("1", {}).get("errors", [])
        assert code == 400 and errors and errors[0]["type"] == "value_not_in_list", (code, body)
        print(f"unknown voice: refused at POST /prompt, {code}: {errors[0]['message']}: {errors[0]['details']}")

        for what, prompt, change, words in [
            ("malformed score", sing, {"song": "this is not a score"}, "a note is NOTE:BEATS"),
            ("unsupported SSML", say, {"text": "<speak><bogus/></speak>", "ssml": True}, "<bogus> is not supported"),
        ]:
            bad = copy.deepcopy(prompt)
            bad["1"]["inputs"].update(change)
            entry = server.run(bad)
            err = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
            assert entry["status"]["status_str"] == "error" and len(err) == 1, json.dumps(entry["status"])
            e = err[0]
            assert e["node_id"] == "1" and words in e["exception_message"], e
            assert not entry["outputs"], entry["outputs"]
            print(f"{what}: execution_error at node 1 ({e['node_type']}): {e['exception_type']}: {e['exception_message'].strip()}; no audio saved")
    finally:
        server.stop()
    print("all passed")


if __name__ == "__main__":
    main()
