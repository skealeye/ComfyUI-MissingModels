# ComfyUI Missing Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ComfyUI extension with a "Missing models" button that downloads a loaded workflow's missing models
straight into the right `models/` folders.

**Architecture:** `core.py` (pure: collect model entries from a workflow, decide what is missing and downloadable,
safe names, Hugging Face download with `.part` files), `service.py` (one background download batch with progress
events and cancel), `__init__.py` (three aiohttp routes on ComfyUI's `PromptServer`, using `folder_paths`), and
`web/missing_models.js` (toolbar button, menu command, load-time check, dialog, progress, list refresh).

**Tech Stack:** Python 3.14 for tests (system), ComfyUI v0.38 embedded Python at runtime, aiohttp, urllib, plain
JavaScript ES module (ComfyUI frontend 1.53), pytest 9.

**Spec:** `docs/superpowers/specs/2026-10-03-missing-models-design.md`

## Global Constraints

- Repo `Z:\Git\ComfyUI-MissingModels`, branch `main` (new repo; nothing to protect yet), private GitHub repo `skealeye/ComfyUI-MissingModels`.
- Install: a directory **symlink** `Y:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MissingModels` → `Z:\Git\ComfyUI-MissingModels` (the spec says junction; junctions can't point at a network share like Z:, symlinks can; Developer Mode is on).
- Downloads only from URLs starting `https://huggingface.co/`.
- Model file extensions: `.safetensors .ckpt .pt .pth .bin .gguf .sft`.
- Never overwrite an existing model file; write to `<name>.part`, rename when complete, remove `.part` on any failure or cancel.
- Token: env `HF_TOKEN`, else the file named by `config.json` key `hf_token_file`, default `Z:\Git\SlopFactory\.hf_token`. Never log it or send it to the browser.
- Gated message: `Accept the license at https://huggingface.co/<owner>/<repo>, then try again`.
- Websocket events: `missing_models.progress`, `missing_models.done`. Routes: `POST /missing_models/check`, `/missing_models/download`, `/missing_models/cancel`.
- Tests: `python -m pytest -q` from the repo root (system Python has pytest 9 and aiohttp 3.13).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Restarting ComfyUI interrupts the user's work: ask before restarting it.

## Review Focus

1. **A workflow with the same file in two nodes, one linked and one not**: listed once, downloadable (test in Task 1).
2. **A name with a subfolder** (`ltxv/ltx2/x.safetensors`, as in Lightricks' examples): allowed, saved into that subfolder (tests in Tasks 2 and 5).
3. **Clicking Download twice / a second request while busy**: refused (409), the first batch carries on (test in Task 4).
4. **A file that appears in the folder while the batch runs** (user copied it in): not overwritten, row says it already exists (test in Task 3: `test_download_keeps_a_file_that_appeared_meanwhile`).
5. **The check route given an empty or malformed workflow** (blank canvas, old-format JSON): returns an empty list, no 500 (test in Task 5).

---

### Task 1: Repo scaffold and `collect`

**Files:**
- Create: `core.py`, `tests/conftest.py`, `tests/test_core.py`, `tests/fixtures/video_ltx2_3_i2v.json`, `.gitignore`, `pytest.ini`

**Interfaces:**
- Produces: `core.MODEL_EXTS: tuple[str, ...]`, `core.LOADER_FOLDERS: dict[str, str]`,
  `core.collect(workflow: dict) -> list[dict]` with entries `{"name": str, "directory": str, "url": str | None}`.

- [ ] **Step 1: Scaffold**

```bash
cd /z/Git/ComfyUI-MissingModels
mkdir -p tests/fixtures web
cp "/y/ComfyUI_windows_portable/python_embeded/Lib/site-packages/comfyui_workflow_templates_json/templates/video_ltx2_3_i2v.json" tests/fixtures/
```

`.gitignore`:

```
__pycache__/
*.pyc
config.json
.superpowers/
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -p no:cacheprovider
```

`tests/conftest.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # import core / service as plain modules (the package __init__ needs ComfyUI)
FIXTURES = Path(__file__).resolve().parent / "fixtures"
```

- [ ] **Step 2: Write the failing tests** (`tests/test_core.py`)

```python
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
```

- [ ] **Step 3: Run them to see them fail**

Run: `python -m pytest -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'core'`)

- [ ] **Step 4: Implement** (`core.py`)

```python
"""Pure logic for the Missing models extension (no ComfyUI imports, so it can be unit-tested)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
# loader node type -> models folder, for workflows without download links
LOADER_FOLDERS = {
    "CheckpointLoaderSimple": "checkpoints", "ImageOnlyCheckpointLoader": "checkpoints",
    "LoraLoader": "loras", "LoraLoaderModelOnly": "loras", "LTXICLoRALoaderModelOnly": "loras",
    "UNETLoader": "diffusion_models", "VAELoader": "vae",
    "CLIPLoader": "text_encoders", "DualCLIPLoader": "text_encoders", "TripleCLIPLoader": "text_encoders",
    "CLIPVisionLoader": "clip_vision", "UpscaleModelLoader": "upscale_models", "ControlNetLoader": "controlnet",
}


def _nodes(workflow: dict):
    yield from workflow.get("nodes") or []
    for sg in (workflow.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sg, dict):
            yield from sg.get("nodes") or []


def collect(workflow: dict) -> list[dict]:
    """Model files a workflow uses: templates' per-node download links (properties.models) and the file names in
    known loader nodes. One entry per (directory, name); an entry with a link wins over one without."""
    found: dict[tuple[str, str], dict] = {}
    for n in _nodes(workflow if isinstance(workflow, dict) else {}):
        if not isinstance(n, dict):
            continue
        for m in (n.get("properties") or {}).get("models") or []:
            if isinstance(m, dict) and m.get("name") and m.get("directory"):
                found[(m["directory"], m["name"])] = {"name": m["name"], "directory": m["directory"],
                                                      "url": m.get("url") or None}
        folder = LOADER_FOLDERS.get(n.get("type"))
        if folder:
            values = n.get("widgets_values")
            values = values.values() if isinstance(values, dict) else values or []
            for v in values:
                if isinstance(v, str) and v.lower().endswith(MODEL_EXTS):
                    found.setdefault((folder, v), {"name": v, "directory": folder, "url": None})
    return list(found.values())
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest -q`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add .gitignore pytest.ini core.py tests
git commit -m "core: collect a workflow's model files (template links + loader widgets)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: What is missing and downloadable (`core.py`)

**Files:**
- Modify: `core.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `core.collect` entries.
- Produces: `core.allowed(url) -> bool`, `core.safe_name(name: str) -> str` (raises `ValueError("unsafe file name")`),
  `core.missing(entries, exists: Callable[[str, str], bool], known_dirs: set[str]) -> list[dict]`: entries not
  present, each plus `"downloadable": bool, "reason": str` (`""`, `"not found"`, `"unknown folder"`,
  `"unsupported source"`, `"unsafe file name"`).

- [ ] **Step 1: Write the failing tests**

```python
HF = "https://huggingface.co/o/r/resolve/main/"


def test_allowed():
    assert core.allowed(HF + "x.safetensors")
    for bad in (None, "", "http://huggingface.co/x", "https://huggingface.co.evil.com/x", "https://civitai.com/x"):
        assert not core.allowed(bad)


def test_safe_name():
    assert core.safe_name("x.safetensors") == "x.safetensors"
    assert core.safe_name("ltxv\\ltx2\\x.safetensors") == "ltxv/ltx2/x.safetensors"   # subfolder kept
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q`
Expected: FAIL (`AttributeError: module 'core' has no attribute 'allowed'`)

- [ ] **Step 3: Implement** (append to `core.py`)

```python
HF_PREFIX = "https://huggingface.co/"


def allowed(url) -> bool:
    return isinstance(url, str) and url.startswith(HF_PREFIX)


def safe_name(name: str) -> str:
    """A model file name relative to its folder (subfolders allowed), or ValueError."""
    n = (name or "").replace("\\", "/")
    parts = n.split("/")
    if (not n or n.startswith("/") or ":" in n or any(p in ("", ".", "..") for p in parts)
            or not n.lower().endswith(MODEL_EXTS)):
        raise ValueError("unsafe file name")
    return n


def missing(entries: list[dict], exists, known_dirs: set[str]) -> list[dict]:
    """The entries ComfyUI doesn't have, each with downloadable + reason."""
    out = []
    for e in entries:
        known = e["directory"] in known_dirs
        if known and exists(e["directory"], e["name"]):
            continue
        row = dict(e, downloadable=False, reason="")
        if not known:
            row["reason"] = "unknown folder"
        elif not e.get("url"):
            row["reason"] = "not found"
        elif not allowed(e["url"]):
            row["reason"] = "unsupported source"
        else:
            try:
                safe_name(e["name"])
                row["downloadable"] = True
            except ValueError:
                row["reason"] = "unsafe file name"
        out.append(row)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add core.py tests/test_core.py
git commit -m "core: missing / downloadable rows, Hugging Face only, safe file names

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Token, size and download (`core.py`)

**Files:**
- Modify: `core.py`
- Create: `config.example.json`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `core.GatedError(RuntimeError)`, `core.Cancelled(RuntimeError)`, `core.token(config_path=ROOT/"config.json") -> str | None`,
  `core.repo_page(url) -> str`, `core.size(url, tok=None) -> int | None`,
  `core.download(url, dest, tok=None, progress=None, cancel=None) -> Path` (raises `FileExistsError`, `GatedError`,
  `Cancelled`, `RuntimeError("HTTP <code> <reason>")`).

- [ ] **Step 1: Write the failing tests**

```python
import io
import threading
import urllib.error


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
    monkeypatch.setattr(core.urllib.request, "urlopen", fake)
    got = []
    dest = tmp_path / "sub" / "m.safetensors"
    assert core.download(HF + "m.safetensors", dest, "tok", lambda d, t: got.append((d, t))) == dest
    assert dest.stat().st_size == 3_000_000 and not (tmp_path / "sub" / "m.safetensors.part").exists()
    assert seen["auth"] == "Bearer tok" and got[-1] == (3_000_000, 3_000_000)
    core.download(HF + "n.safetensors", tmp_path / "n.safetensors")
    assert seen["auth"] is None                                  # no token, no header


def test_download_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(core.urllib.request, "urlopen", lambda req, timeout=60: Resp(b"new"))
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
    monkeypatch.setattr(core.urllib.request, "urlopen", lambda req, timeout=60: Racing(b"x" * 100))
    with pytest.raises(FileExistsError):
        core.download(HF + "m.safetensors", dest)
    assert dest.read_bytes() == b"copied in by hand" and not (tmp_path / "m.safetensors.part").exists()


def test_download_cancel_removes_part(tmp_path, monkeypatch):
    monkeypatch.setattr(core.urllib.request, "urlopen", lambda req, timeout=60: Resp(b"x" * 5_000_000))
    stop = threading.Event()
    dest = tmp_path / "m.safetensors"
    with pytest.raises(core.Cancelled):
        core.download(HF + "m.safetensors", dest, progress=lambda d, t: stop.set(), cancel=stop)
    assert not dest.exists() and not (tmp_path / "m.safetensors.part").exists()


def test_download_gated(tmp_path, monkeypatch):
    def refuse(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b""))
    monkeypatch.setattr(core.urllib.request, "urlopen", refuse)
    with pytest.raises(core.GatedError, match=r"Accept the license at https://huggingface.co/o/r, then try again"):
        core.download(HF + "m.safetensors", tmp_path / "m.safetensors")
    assert list(tmp_path.iterdir()) == []


def test_size(monkeypatch):
    monkeypatch.setattr(core.urllib.request, "urlopen",
                        lambda req, timeout=30: Resp(b"", {"X-Linked-Size": "123", "Content-Length": "9"}))
    assert core.size(HF + "m.safetensors") == 123

    def boom(req, timeout=30):
        raise OSError("offline")
    monkeypatch.setattr(core.urllib.request, "urlopen", boom)
    assert core.size(HF + "m.safetensors") is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q`
Expected: FAIL (`AttributeError: module 'core' has no attribute 'token'`)

- [ ] **Step 3: Implement**

Add to the imports at the top of `core.py`:

```python
import json
import os
import urllib.error
import urllib.request
```

Append:

```python
DEFAULT_TOKEN_FILE = r"Z:\Git\SlopFactory\.hf_token"


class GatedError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def token(config_path=ROOT / "config.json") -> str | None:
    """Hugging Face token: env HF_TOKEN, else the file config.json names (never logged or sent to the browser)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    try:
        return Path(cfg.get("hf_token_file") or DEFAULT_TOKEN_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def repo_page(url: str) -> str:
    parts = url.split("/")
    return "/".join(parts[:5])          # https://huggingface.co/<owner>/<repo>


def _request(url: str, tok: str | None, method: str = "GET") -> urllib.request.Request:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "ComfyUI-MissingModels"})
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    return req


def size(url: str, tok: str | None = None) -> int | None:
    """File size from a HEAD request (Hugging Face's X-Linked-Size, else Content-Length); None if unknown."""
    try:
        with urllib.request.urlopen(_request(url, tok, "HEAD"), timeout=30) as r:
            v = r.headers.get("X-Linked-Size") or r.headers.get("Content-Length")
            return int(v) if v else None
    except (OSError, ValueError):
        return None


def download(url: str, dest, tok: str | None = None, progress=None, cancel=None) -> Path:
    """url -> dest via dest.part. progress(done, total); cancel: threading.Event. Never overwrites dest."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest.name} already exists")
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(_request(url, tok), timeout=60) as r:
            total, done = int(r.headers.get("Content-Length") or 0), 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled("cancelled")
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        if e.code in (401, 403):
            raise GatedError(f"Accept the license at {repo_page(url)}, then try again") from None
        raise RuntimeError(f"HTTP {e.code} {e.reason}") from None
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if dest.exists():                    # appeared while downloading (copied in by hand): keep it
        part.unlink(missing_ok=True)
        raise FileExistsError(f"{dest.name} already exists")
    part.replace(dest)
    return dest
```

`config.example.json`:

```json
{"hf_token_file": "Z:\\Git\\SlopFactory\\.hf_token"}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add core.py config.example.json tests/test_core.py
git commit -m "core: Hugging Face token, size and download (.part files, cancel, gated, no overwrite)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Background download batch (`service.py`)

**Files:**
- Create: `service.py`, `tests/test_service.py`

**Interfaces:**
- Consumes: `core.allowed`, `core.safe_name`, `core.download`, `core.token`, `core.Cancelled`.
- Produces: `service.Downloader(emit: Callable[[str, dict], None], dest_for: Callable[[str, str], Path], token=core.token)`
  with `.start(items: list[dict]) -> bool` (False when busy), `.busy() -> bool`, `.cancel` (threading.Event),
  `.thread`. Events: `("missing_models.progress", {"index", "count", "name", "done", "total"})` (at most ~4/s per
  file plus the last), `("missing_models.done", {"results": [{"name", "directory", "status": "installed"|"failed"|"cancelled", "path"?, "error"?}]})`.

- [ ] **Step 1: Write the failing tests** (`tests/test_service.py`)

```python
import threading
import time

import core
import service

HF = "https://huggingface.co/o/r/resolve/main/"


def run(items, fake_download, dest_for=None, tmp=None):
    events = []
    d = service.Downloader(lambda ev, data: events.append((ev, data)),
                           dest_for or (lambda directory, name: tmp / directory / name), token=lambda: "t")
    import pytest
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
        progress(5, 10); progress(10, 10)
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
    import pytest
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_service.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'service'`)

- [ ] **Step 3: Implement** (`service.py`)

```python
"""One download batch at a time, in a background thread; progress and results go out as events."""
import threading
import time

try:
    from . import core            # inside ComfyUI (package)
except ImportError:
    import core                   # tests (plain module)


class Downloader:
    def __init__(self, emit, dest_for, token=core.token):
        self.emit, self.dest_for, self.token = emit, dest_for, token
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None

    def busy(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def start(self, items: list[dict]) -> bool:
        if self.busy():
            return False
        self.cancel.clear()
        self.thread = threading.Thread(target=self._run, args=(list(items),), daemon=True)
        self.thread.start()
        return True

    def _run(self, items: list[dict]):
        tok, results = self.token(), []
        for i, it in enumerate(items):
            r = {"name": it.get("name"), "directory": it.get("directory")}
            if self.cancel.is_set():
                results.append(dict(r, status="cancelled"))
                continue
            try:
                if not core.allowed(it.get("url")):
                    raise ValueError("unsupported source")
                name = core.safe_name(it.get("name") or "")
                dest = self.dest_for(it.get("directory"), name)
                last = [0.0]

                def progress(done, total, i=i, name=name):
                    now = time.monotonic()
                    if done >= total or now - last[0] >= 0.25:   # at most ~4 updates a second
                        last[0] = now
                        self.emit("missing_models.progress",
                                  {"index": i, "count": len(items), "name": name, "done": done, "total": total})
                core.download(it["url"], dest, tok, progress, self.cancel)
                results.append(dict(r, status="installed", path=str(dest)))
            except core.Cancelled:
                results.append(dict(r, status="cancelled"))
            except Exception as e:
                results.append(dict(r, status="failed", error=str(e)))
        self.emit("missing_models.done", {"results": results})
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add service.py tests/test_service.py
git commit -m "service: one background download batch with throttled progress, cancel, per-row results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: ComfyUI routes (`__init__.py`)

**Files:**
- Create: `__init__.py`, `tests/test_routes.py`

**Interfaces:**
- Consumes: `core.collect`, `core.missing`, `core.size`, `core.token`, `service.Downloader`.
- Produces: package exports `WEB_DIRECTORY = "./web"`, `NODE_CLASS_MAPPINGS = {}`, `NODE_DISPLAY_NAME_MAPPINGS = {}`;
  module functions `check(request)`, `download(request)`, `cancel(request)` (aiohttp handlers), `exists(directory, name) -> bool`,
  `dest_for(directory, name) -> Path`, `downloader` (a `service.Downloader`).

- [ ] **Step 1: Write the failing tests** (`tests/test_routes.py`)

```python
"""The ComfyUI glue, with stand-ins for ComfyUI's folder_paths and server modules."""
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from aiohttp import web

from conftest import ROOT


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "loras").mkdir(parents=True)
    (models / "loras" / "have.safetensors").write_bytes(b"x")
    (models / "loras" / "deep").mkdir()
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_routes.py`
Expected: FAIL (`FileNotFoundError` / no `__init__.py`)

- [ ] **Step 3: Implement** (`__init__.py`)

```python
"""ComfyUI Missing models: download a workflow's missing models straight into the right models folders."""
import asyncio
from pathlib import Path

import folder_paths
from aiohttp import web
from server import PromptServer

from . import core, service

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def exists(directory: str, name: str) -> bool:
    """ComfyUI's own lookup, plus the same file name in any subfolder of that models folder."""
    if folder_paths.get_full_path(directory, name):
        return True
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return any(f.replace("\\", "/").rsplit("/", 1)[-1] == base for f in folder_paths.get_filename_list(directory))


def dest_for(directory: str, name: str) -> Path:
    if directory not in folder_paths.folder_names_and_paths:
        raise ValueError("unknown folder")
    return Path(folder_paths.get_folder_paths(directory)[0]) / name


downloader = service.Downloader(lambda ev, data: PromptServer.instance.send_sync(ev, data), dest_for)
routes = PromptServer.instance.routes


async def _body(request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


@routes.post("/missing_models/check")
async def check(request):
    wf = (await _body(request)).get("workflow")
    rows = core.missing(core.collect(wf if isinstance(wf, dict) else {}), exists,
                        set(folder_paths.folder_names_and_paths))
    tok, loop = core.token(), asyncio.get_running_loop()
    for r in rows:
        r["size"] = await loop.run_in_executor(None, core.size, r["url"], tok) if r["downloadable"] else None
    return web.json_response({"missing": rows})


@routes.post("/missing_models/download")
async def download(request):
    items = (await _body(request)).get("items")
    if not isinstance(items, list):
        return web.json_response({"error": "items must be a list"}, status=400)
    if not downloader.start(items):
        return web.json_response({"error": "a download is already running"}, status=409)
    return web.json_response({"started": True})


@routes.post("/missing_models/cancel")
async def cancel(request):
    downloader.cancel.set()
    return web.json_response({"cancelling": True})
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q`
Expected: 22 passed

- [ ] **Step 5: Commit**

```bash
git add __init__.py tests/test_routes.py
git commit -m "ComfyUI routes: check / download / cancel on PromptServer, folder_paths lookups

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Frontend (`web/missing_models.js`), README, install, end-to-end

**Files:**
- Create: `web/missing_models.js`, `README.md`

**Interfaces:**
- Consumes: the three routes and two events of Task 5.

- [ ] **Step 1: Write `web/missing_models.js`**

```javascript
// Missing models: download a workflow's missing models into the right ComfyUI/models folders.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TITLE = "Missing models";
const fmt = (b) => (b == null ? "?" : b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);
let dlg = null;

function el(tag, props = {}, ...kids) {
  const e = Object.assign(document.createElement(tag), props);
  for (const k of kids) e.append(k);
  return e;
}

function toast(detail, severity = "info") {
  const t = app.extensionManager?.toast;
  if (t) t.add({ severity, summary: TITLE, detail, life: 8000 });
  else console.log(`[${TITLE}]`, detail);
}

async function post(path, body) {
  const r = await api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify(body) });
  return { status: r.status, data: await r.json() };
}

async function check() {
  const workflow = (app.rootGraph ?? app.graph).serialize();
  return (await post("/missing_models/check", { workflow })).data.missing ?? [];
}

function close() {
  dlg?.remove();
  dlg = null;
}

function show(rows) {
  close();
  const status = el("div", { style: "margin-top:8px;min-height:1.4em" });
  const bar = el("progress", { max: 1, value: 0, style: "width:100%;display:none" });
  const boxes = [];
  const table = el("table", { style: "width:100%;border-collapse:collapse;font-size:13px" });
  table.append(el("tr", {}, ...["", "File", "Folder", "Size", ""].map((h) => el("th", { textContent: h,
    style: "text-align:left;padding:4px 6px;border-bottom:1px solid var(--border-color)" }))));
  for (const r of rows) {
    const box = el("input", { type: "checkbox", checked: r.downloadable, disabled: !r.downloadable });
    boxes.push([box, r]);
    const stem = r.name.split("/").pop().replace(/\.[^.]+$/, "");
    const note = r.downloadable ? el("span", { textContent: new URL(r.url).host, style: "opacity:.7" })
      : el("span", {}, `${r.reason}  `,
          el("a", { href: `https://huggingface.co/search/full-text?q=${encodeURIComponent(r.name)}`, target: "_blank",
                    textContent: "Search Hugging Face" }), "  ",
          el("a", { href: `https://civitai.com/search/models?query=${encodeURIComponent(stem)}`, target: "_blank",
                    textContent: "Search Civitai" }));
    const row = el("tr", { style: r.downloadable ? "" : "opacity:.6" },
      el("td", {}, box), el("td", { textContent: r.name }), el("td", { textContent: r.directory }),
      el("td", { textContent: fmt(r.size) }), el("td", {}, note));
    row.dataset.key = `${r.directory}/${r.name}`;
    for (const td of row.children) td.style.padding = "4px 6px";
    table.append(row);
  }
  const go = el("button", { textContent: "Download", disabled: !rows.some((r) => r.downloadable) });
  const stop = el("button", { textContent: "Cancel", disabled: true });
  const shut = el("button", { textContent: "Close" });
  shut.onclick = close;
  stop.onclick = () => post("/missing_models/cancel", {});
  go.onclick = async () => {
    const items = boxes.filter(([b]) => b.checked).map(([, r]) => ({ name: r.name, directory: r.directory, url: r.url }));
    if (!items.length) return;
    const res = await post("/missing_models/download", { items });
    if (res.status !== 200) { status.textContent = res.data.error; return; }
    go.disabled = true; stop.disabled = false; bar.style.display = "";
    for (const [b] of boxes) b.disabled = true;
  };
  dlg = el("div", { style: "position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:10000;display:flex;" +
                           "align-items:center;justify-content:center" },
    el("div", { style: "background:var(--comfy-menu-bg);color:var(--fg-color);border:1px solid var(--border-color);" +
                       "border-radius:8px;padding:16px;min-width:640px;max-width:90vw;max-height:80vh;overflow:auto" },
      el("h3", { textContent: rows.length ? `${TITLE} (${rows.length})` : `${TITLE}: none`, style: "margin:0 0 8px" }),
      rows.length ? table : el("div", { textContent: "Everything this workflow needs is installed." }),
      bar, status, el("div", { style: "margin-top:12px;display:flex;gap:8px;justify-content:flex-end" }, go, stop, shut)));
  dlg.ui = { status, bar, go, stop };
  document.body.append(dlg);
}

async function open() {
  try { show(await check()); } catch (e) { toast(`Check failed: ${e}`, "error"); }
}

api.addEventListener("missing_models.progress", ({ detail: d }) => {
  if (!dlg) return;
  dlg.ui.bar.max = d.total || 1; dlg.ui.bar.value = d.done;
  dlg.ui.status.textContent = `${d.index + 1} of ${d.count}: ${d.name}  ${fmt(d.done)} / ${fmt(d.total)}`;
});

api.addEventListener("missing_models.done", async ({ detail }) => {
  const res = detail.results ?? [];
  const ok = res.filter((r) => r.status === "installed").length;
  await app.refreshComboInNodes?.();
  toast(`${ok} of ${res.length} installed`, ok === res.length ? "success" : "warn");
  if (!dlg) return;
  dlg.ui.stop.disabled = true; dlg.ui.bar.style.display = "none";
  dlg.ui.status.replaceChildren(...res.map((r) => el("div", {
    textContent: `${r.status === "installed" ? "✓" : "✗"} ${r.name}: ${r.status === "installed" ? r.path : r.error ?? r.status}`,
  })));
});

app.registerExtension({
  name: "MissingModels",
  commands: [{ id: "MissingModels.Open", label: TITLE, icon: "pi pi-download", function: open }],
  menuCommands: [{ path: ["Extensions"], commands: ["MissingModels.Open"] }],
  actionBarButtons: [{ icon: "pi pi-download", label: TITLE, tooltip: "Download this workflow's missing models",
                       onClick: open }],
  async afterConfigureGraph() {
    try {
      const rows = await check();
      const n = rows.filter((r) => r.downloadable).length;
      if (rows.length) toast(`${rows.length} missing (${n} downloadable): use the ${TITLE} button`, "warn");
    } catch (e) { console.warn(`[${TITLE}]`, e); }
  },
});
```

- [ ] **Step 2: Write `README.md`**

```markdown
# ComfyUI Missing Models

A **Missing models** button (top bar, and Extensions menu) that downloads the loaded workflow's missing models and
LoRAs straight into the right `ComfyUI/models/<folder>`. Works with the download links ComfyUI's templates carry
(Hugging Face); files without a link are listed with search links.

- Gated Hugging Face repos need your token: env `HF_TOKEN`, or copy `config.example.json` to `config.json` and point
  `hf_token_file` at a file holding it. Accept the model's license on its Hugging Face page first.
- Only `https://huggingface.co/` links and model files (`.safetensors .ckpt .pt .pth .bin .gguf .sft`) are
  downloaded; existing files are never overwritten; downloads go to `<name>.part` until complete.

Install (Windows, Developer Mode on):

    mklink /D "<ComfyUI>\custom_nodes\ComfyUI-MissingModels" "Z:\Git\ComfyUI-MissingModels"

Tests: `python -m pytest -q`
```

- [ ] **Step 3: Commit, then install the symlink**

```bash
git add web README.md
git commit -m "Frontend: Missing models button, load-time check, dialog with progress and cancel; README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```powershell
cmd /c mklink /D "Y:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MissingModels" "Z:\Git\ComfyUI-MissingModels"
```

Expected: `symbolic link created for ...`

- [ ] **Step 4: Restart ComfyUI (ask the user first), then check the routes**

After the user agrees and ComfyUI is back up:

```bash
grep -i "missingmodels\|missing_models\|Traceback" "/y/ComfyUI_windows_portable/ComfyUI/user/comfyui.log" | tail -5
curl -s -X POST -H "Content-Type: application/json" -d '{"workflow": {}}' http://127.0.0.1:8188/missing_models/check
curl -s http://127.0.0.1:8188/extensions | grep -o "ComfyUI-MissingModels[^\"]*"
```

Expected: no traceback for the extension; `{"missing": []}`; `ComfyUI-MissingModels/missing_models.js`.

- [ ] **Step 5: End to end in Chrome** (claude-in-chrome; load its tools first)

1. Open `http://127.0.0.1:8188`, load the template **LTX-2.3 Image to Video** (`video_ltx2_3_i2v`).
2. Expect a toast "N missing (… downloadable)". Click **Missing models** in the top bar: the dialog lists the files
   the install lacks (per the fixture: e.g. `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` in
   `latent_upscale_models`), with sizes.
3. Untick everything except the smallest row; **Download**; watch progress; expect "✓ <name>: <path>" and the toast.
4. Confirm the file exists with the expected size and the loader node shows the file without a red marker.
5. Take a screenshot for the user.

- [ ] **Step 6: Private GitHub repo and push**

```bash
gh repo create skealeye/ComfyUI-MissingModels --private --source . --push
git status -sb
```

Expected: `## main...origin/main`.
