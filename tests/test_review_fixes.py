"""Findings from the final review: each test reproduces one."""
import io
import json
import urllib.error
import urllib.request

import pytest

import core
import service

HF = "https://huggingface.co/o/r/resolve/main/"


class Resp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# 1. a dropped connection must not become an "installed" model
def test_truncated_download_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_open", lambda req, timeout=60: Resp(b"x" * 100, {"Content-Length": "500"}))
    with pytest.raises(RuntimeError, match="incomplete"):
        core.download(HF + "m.safetensors", tmp_path / "m.safetensors")
    assert list(tmp_path.iterdir()) == []


# 2. the token goes to huggingface.co only, and only https redirects are followed
def redirect(url, newurl, tok="secret"):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {tok}")
    return core._SafeRedirect().redirect_request(req, io.BytesIO(b""), 302, "Found", {}, newurl)


def test_redirect_to_cdn_drops_token():
    new = redirect(HF + "m.safetensors", "https://cdn-lfs.hf.co/repos/x/y?sig=1")
    assert new.full_url.startswith("https://cdn-lfs.hf.co/") and not new.has_header("Authorization")


def test_redirect_within_huggingface_keeps_token():
    new = redirect(HF + "m.safetensors", "https://huggingface.co/o/r2/resolve/main/m.safetensors")
    assert new.get_header("Authorization") == "Bearer secret"


def test_redirect_to_plain_http_refused():
    with pytest.raises(urllib.error.HTTPError):
        redirect(HF + "m.safetensors", "http://evil.example/m.safetensors")


# 3. malformed workflows: no exceptions
@pytest.mark.parametrize("wf", [
    {"nodes": 5},
    {"nodes": [{"type": "VAELoader", "widgets_values": 5}]},
    {"nodes": [{"type": "X", "properties": {"models": [{"name": ["a"], "directory": "loras", "url": HF}]}}]},
    {"nodes": [{"type": "X", "properties": {"models": [{"name": 5, "directory": "loras", "url": HF}]}}]},
    {"nodes": [{"type": "X", "properties": {"models": 7}}]},
    {"nodes": [{"type": "X", "properties": 3}]},
    {"definitions": 4},
    {"definitions": {"subgraphs": [{"nodes": 9}]}},
])
def test_collect_malformed_is_empty(wf):
    assert core.collect(wf) == []


# 4. the same file written with \ and / is one entry
def test_backslash_and_slash_names_are_one_entry():
    url = HF + "x.safetensors"
    wf = {"nodes": [{"type": "LoraLoaderModelOnly", "widgets_values": [r"ltxv\ltx2\x.safetensors", 1.0]},
                    {"type": "X", "properties": {"models": [{"name": "ltxv/ltx2/x.safetensors", "url": url,
                                                             "directory": "loras"}]}}]}
    assert core.collect(wf) == [{"name": "ltxv/ltx2/x.safetensors", "directory": "loras", "url": url}]


# 5. the batch always reports "done", whatever goes wrong
def run(items, token, tmp_path):
    events = []
    d = service.Downloader(lambda ev, data: events.append((ev, data)), lambda di, n: tmp_path / n, token=token)
    assert d.start(items)
    d.thread.join(10)
    return events


def test_token_failure_still_reports_done(tmp_path):
    def broken():
        raise AttributeError("'list' object has no attribute 'get'")
    events = run([{"name": "a.safetensors", "directory": "loras", "url": "ftp://x"}], broken, tmp_path)
    assert events[-1][0] == "missing_models.done"
    assert events[-1][1]["results"][0]["status"] == "failed"


def test_non_dict_item_fails_that_row(tmp_path):
    events = run(["oops"], lambda: None, tmp_path)
    assert events[-1] == ("missing_models.done", {"results": [{"name": None, "directory": None, "status": "failed",
                                                               "error": "bad item"}]})


def test_token_ignores_non_dict_config(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    core.token(cfg)            # no exception
