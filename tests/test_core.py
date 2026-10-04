import json

import pytest

import core
from conftest import FIXTURES


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
