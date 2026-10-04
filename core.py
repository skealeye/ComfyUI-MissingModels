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
