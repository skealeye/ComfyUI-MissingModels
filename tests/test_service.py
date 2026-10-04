import threading

import pytest

import core
import service

HF = "https://huggingface.co/o/r/resolve/main/"


def run(items, fake_download, dest_for=None, tmp=None):
    events = []
    d = service.Downloader(lambda ev, data: events.append((ev, data)),
                           dest_for or (lambda directory, name: tmp / directory / name), token=lambda: "t")
    mp = pytest.MonkeyPatch()
    mp.setattr(core, "download", fake_download)
    try:
        assert d.start(items)
        d.thread.join(10)
    finally:
        mp.undo()
    return d, events


def test_batch_installs_and_reports(tmp_path):
    def fake(url, dest, tok, progress, cancel):
        progress(5, 10)
        progress(10, 10)
        return dest
    items = [{"name": "a.safetensors", "directory": "loras", "url": HF + "a.safetensors"},
             {"name": "../b.safetensors", "directory": "loras", "url": HF + "b.safetensors"},
             {"name": "c.safetensors", "directory": "loras", "url": "https://evil.example/c.safetensors"}]
    _, events = run(items, fake, tmp=tmp_path)
    done = events[-1]
    assert done[0] == "missing_models.done"
    assert [(r["name"], r["status"]) for r in done[1]["results"]] == [
        ("a.safetensors", "installed"), ("../b.safetensors", "failed"), ("c.safetensors", "failed")]
    assert done[1]["results"][1]["error"] == "unsafe file name"
    assert done[1]["results"][2]["error"] == "unsupported source"
    prog = [e[1] for e in events if e[0] == "missing_models.progress"]
    assert prog[-1] == {"index": 0, "count": 3, "name": "a.safetensors", "done": 10, "total": 10}


def test_one_batch_at_a_time(tmp_path):
    gate = threading.Event()

    def slow(url, dest, tok, progress, cancel):
        gate.wait(5)
        return dest
    events = []
    d = service.Downloader(lambda ev, data: events.append(ev), lambda di, n: tmp_path / n, token=lambda: None)
    mp = pytest.MonkeyPatch()
    mp.setattr(core, "download", slow)
    try:
        item = [{"name": "a.safetensors", "directory": "loras", "url": HF + "a.safetensors"}]
        assert d.start(item) and d.busy()
        assert not d.start(item)                                   # second click while busy: refused
        gate.set()
        d.thread.join(5)
        assert not d.busy() and events[-1] == "missing_models.done"
    finally:
        mp.undo()


def test_cancel_marks_the_rest(tmp_path):
    def cancelling(url, dest, tok, progress, cancel):
        cancel.set()
        raise core.Cancelled("cancelled")
    items = [{"name": f"{c}.safetensors", "directory": "loras", "url": HF + f"{c}.safetensors"} for c in "ab"]
    _, events = run(items, cancelling, tmp=tmp_path)
    assert [r["status"] for r in events[-1][1]["results"]] == ["cancelled", "cancelled"]


def test_unknown_folder_fails_that_row(tmp_path):
    def dest_for(directory, name):
        raise ValueError("unknown folder")
    _, events = run([{"name": "a.safetensors", "directory": "nope", "url": HF + "a.safetensors"}],
                    lambda *a: None, dest_for=dest_for)
    assert events[-1][1]["results"][0] == {"name": "a.safetensors", "directory": "nope", "status": "failed",
                                           "error": "unknown folder"}
