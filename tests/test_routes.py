"""The ComfyUI glue, with stand-ins for ComfyUI's folder_paths and server modules."""
import asyncio
import importlib.util
import json
import sys
import types

import pytest
from aiohttp import web

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "loras" / "deep").mkdir(parents=True)
    (models / "loras" / "have.safetensors").write_bytes(b"x")
    (models / "loras" / "deep" / "nested.safetensors").write_bytes(b"x")
    fp = types.ModuleType("folder_paths")
    fp.folder_names_and_paths = {"loras": ([str(models / "loras")], {".safetensors"})}
    fp.get_folder_paths = lambda d: fp.folder_names_and_paths[d][0]
    fp.get_full_path = lambda d, n: str(models / d / n) if (models / d / n).exists() else None
    fp.get_filename_list = lambda d: ["have.safetensors", "deep/nested.safetensors"]
    srv = types.ModuleType("server")
    sent = []

    class PromptServer:
        instance = types.SimpleNamespace(routes=web.RouteTableDef(), send_sync=lambda ev, data: sent.append((ev, data)))
    srv.PromptServer = PromptServer
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setitem(sys.modules, "server", srv)
    spec = importlib.util.spec_from_file_location("missing_models_pkg", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "missing_models_pkg", mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.core, "size", lambda url, tok=None: 42)
    mod.sent, mod.models = sent, models
    return mod


class Req:
    def __init__(self, body):
        self.body = body

    async def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def call(handler, body):
    resp = asyncio.run(handler(Req(body)))
    return resp.status, json.loads(resp.text)


HF = "https://huggingface.co/o/r/resolve/main/"


def test_check_lists_only_missing(pkg):
    wf = {"nodes": [{"type": "X", "properties": {"models": [
        {"name": "have.safetensors", "url": HF + "have.safetensors", "directory": "loras"},
        {"name": "nested.safetensors", "url": HF + "nested.safetensors", "directory": "loras"},
        {"name": "get.safetensors", "url": HF + "get.safetensors", "directory": "loras"}]}}]}
    status, body = call(pkg.check, {"workflow": wf})
    assert status == 200
    assert [(r["name"], r["downloadable"], r["size"]) for r in body["missing"]] == [("get.safetensors", True, 42)]


def test_check_odd_input_is_empty(pkg):
    for body in ({}, {"workflow": None}, {"workflow": []}, ValueError("not json")):
        assert call(pkg.check, body) == (200, {"missing": []})


def test_dest_for(pkg):
    assert pkg.dest_for("loras", "sub/x.safetensors") == pkg.models / "loras" / "sub" / "x.safetensors"
    with pytest.raises(ValueError, match="unknown folder"):
        pkg.dest_for("nope", "x.safetensors")


def test_download_busy_and_cancel(pkg, monkeypatch):
    monkeypatch.setattr(pkg.downloader, "start", lambda items: False)
    assert call(pkg.download, {"items": []})[0] == 409
    monkeypatch.setattr(pkg.downloader, "start", lambda items: True)
    assert call(pkg.download, {"items": [{"name": "a.safetensors"}]}) == (200, {"started": True})
    assert call(pkg.download, {"items": "nope"})[0] == 400
    assert call(pkg.cancel, {}) == (200, {"cancelling": True}) and pkg.downloader.cancel.is_set()
