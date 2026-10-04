"""Pure logic for the Missing models extension (no ComfyUI imports, so it can be unit-tested)."""
import json
import os
import urllib.error
import urllib.request
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
