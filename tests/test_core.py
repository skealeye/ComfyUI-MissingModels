import io
import json
import threading
import urllib.error

import pytest

import core
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_collect_real_template():
    wf = json.loads((FIXTURES / "video_ltx2_3_i2v.json").read_text(encoding="utf-8"))
    got = {(e["directory"], e["name"]): e["url"] for e in core.collect(wf)}
    assert set(got) == {
        ("checkpoints", "ltx-2.3-22b-dev-fp8.safetensors"),
        ("loras", "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"),
        ("latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors"),
        ("loras", "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"),
    }
    assert all(u and u.startswith("https://huggingface.co/") for u in got.values())


def test_collect_loader_widgets_and_subgraphs():
    wf = {"nodes": [{"type": "UNETLoader", "widgets_values": ["wan.safetensors", "default"]},
                    {"type": "LoraLoader", "widgets_values": ["sub/style.safetensors", 1.0, 1.0]},
                    {"type": "KSampler", "widgets_values": [1, "fixed", 20]},
                    {"type": "DualCLIPLoader", "widgets_values": ["a.safetensors", "b.gguf", "flux"]}],
          "definitions": {"subgraphs": [{"nodes": [{"type": "VAELoader", "widgets_values": ["v.safetensors"]}]}]}}
    got = {(e["directory"], e["name"], e["url"]) for e in core.collect(wf)}
    assert got == {("diffusion_models", "wan.safetensors", None), ("loras", "sub/style.safetensors", None),
                   ("text_encoders", "a.safetensors", None), ("text_encoders", "b.gguf", None),
                   ("vae", "v.safetensors", None)}


def test_collect_linked_entry_wins_over_widget():
    url = "https://huggingface.co/o/r/resolve/main/x.safetensors"
    for order in (1, -1):
        nodes = [{"type": "LoraLoaderModelOnly", "widgets_values": ["x.safetensors", 1.0]},
                 {"type": "Other", "properties": {"models": [{"name": "x.safetensors", "url": url,
                                                              "directory": "loras"}]}}][::order]
        assert core.collect({"nodes": nodes}) == [{"name": "x.safetensors", "directory": "loras", "url": url}]


def test_collect_tolerates_odd_input():
    assert core.collect({}) == []
    assert core.collect({"nodes": [None, "x", {"type": "VAELoader", "widgets_values": {"vae_name": "v.safetensors"}}]}) == [
        {"name": "v.safetensors", "directory": "vae", "url": None}]


HF = "https://huggingface.co/o/r/resolve/main/"


def test_allowed():
    assert core.allowed(HF + "x.safetensors")
    for bad in (None, "", "http://huggingface.co/x", "https://huggingface.co.evil.com/x", "https://civitai.com/x"):
        assert not core.allowed(bad)


def test_safe_name():
    assert core.safe_name("x.safetensors") == "x.safetensors"
    assert core.safe_name(r"ltxv\ltx2\x.safetensors") == "ltxv/ltx2/x.safetensors"   # subfolder kept
    for bad in ("../x.safetensors", "a/../../x.safetensors", "/etc/x.safetensors", "C:/x.safetensors",
                "C:x.safetensors", "x.exe", "", "a//x.safetensors"):
        with pytest.raises(ValueError, match="unsafe"):
            core.safe_name(bad)


def test_missing_rows():
    entries = [{"name": "have.safetensors", "directory": "loras", "url": HF + "have.safetensors"},
               {"name": "get.safetensors", "directory": "loras", "url": HF + "get.safetensors"},
               {"name": "nolink.safetensors", "directory": "vae", "url": None},
               {"name": "odd.safetensors", "directory": "mystery", "url": HF + "odd.safetensors"},
               {"name": "civ.safetensors", "directory": "loras", "url": "https://civitai.com/api/download/1"},
               {"name": "../x.safetensors", "directory": "loras", "url": HF + "x.safetensors"}]
    rows = core.missing(entries, lambda d, n: n == "have.safetensors", {"loras", "vae"})
    assert [(r["name"], r["downloadable"], r["reason"]) for r in rows] == [
        ("get.safetensors", True, ""), ("nolink.safetensors", False, "not found"),
        ("odd.safetensors", False, "unknown folder"), ("civ.safetensors", False, "unsupported source"),
        ("../x.safetensors", False, "unsafe file name")]


class Resp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_token_env_then_config_file(tmp_path, monkeypatch):
    tok = tmp_path / "t.txt"
    tok.write_text(" secret \n", encoding="utf-8")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hf_token_file": str(tok)}), encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert core.token(cfg) == "secret"
    assert core.token(tmp_path / "none.json") == core.token(tmp_path / "none.json")  # default path, no crash
    monkeypatch.setenv("HF_TOKEN", "env")
    assert core.token(cfg) == "env"


def test_download_part_then_rename_with_progress_and_token(tmp_path, monkeypatch):
    seen = {}

    def fake(req, timeout=60):
        seen["auth"] = req.get_header("Authorization")
        return Resp(b"x" * 3_000_000)
    monkeypatch.setattr(core, "_open", fake)
    got = []
    dest = tmp_path / "sub" / "m.safetensors"
    assert core.download(HF + "m.safetensors", dest, "tok", lambda d, t: got.append((d, t))) == dest
    assert dest.stat().st_size == 3_000_000 and not (tmp_path / "sub" / "m.safetensors.part").exists()
    assert seen["auth"] == "Bearer tok" and got[-1] == (3_000_000, 3_000_000)
    core.download(HF + "n.safetensors", tmp_path / "n.safetensors")
    assert seen["auth"] is None                                  # no token, no header


def test_download_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_open", lambda req, timeout=60: Resp(b"new"))
    dest = tmp_path / "m.safetensors"
    dest.write_bytes(b"old")
    with pytest.raises(FileExistsError, match="already exists"):
        core.download(HF + "m.safetensors", dest)
    assert dest.read_bytes() == b"old"


def test_download_keeps_a_file_that_appeared_meanwhile(tmp_path, monkeypatch):
    dest = tmp_path / "m.safetensors"

    class Racing(Resp):
        def read(self, n=-1):
            chunk = super().read(n)
            if chunk and not dest.exists():
                dest.write_bytes(b"copied in by hand")     # user copies the file in mid-download
            return chunk
    monkeypatch.setattr(core, "_open", lambda req, timeout=60: Racing(b"x" * 100))
    with pytest.raises(FileExistsError):
        core.download(HF + "m.safetensors", dest)
    assert dest.read_bytes() == b"copied in by hand" and not (tmp_path / "m.safetensors.part").exists()


def test_download_cancel_removes_part(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_open", lambda req, timeout=60: Resp(b"x" * 5_000_000))
    stop = threading.Event()
    dest = tmp_path / "m.safetensors"
    with pytest.raises(core.Cancelled):
        core.download(HF + "m.safetensors", dest, progress=lambda d, t: stop.set(), cancel=stop)
    assert not dest.exists() and not (tmp_path / "m.safetensors.part").exists()


def test_download_gated(tmp_path, monkeypatch):
    def refuse(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b""))
    monkeypatch.setattr(core, "_open", refuse)
    with pytest.raises(core.GatedError, match=r"Accept the license at https://huggingface.co/o/r, then try again"):
        core.download(HF + "m.safetensors", tmp_path / "m.safetensors")
    assert list(tmp_path.iterdir()) == []


def test_size(monkeypatch):
    monkeypatch.setattr(core, "_open",
                        lambda req, timeout=30: Resp(b"", {"X-Linked-Size": "123", "Content-Length": "9"}))
    assert core.size(HF + "m.safetensors") == 123

    def boom(req, timeout=30):
        raise OSError("offline")
    monkeypatch.setattr(core, "_open", boom)
    assert core.size(HF + "m.safetensors") is None
