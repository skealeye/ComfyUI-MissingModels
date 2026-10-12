"""Pure logic for the Missing models extension (no ComfyUI imports, so it can be unit-tested)."""
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
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


def _list(v) -> list:
    return v if isinstance(v, list) else []


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _nodes(workflow: dict):
    yield from _list(workflow.get("nodes"))
    for sg in _list(_dict(workflow.get("definitions")).get("subgraphs")):
        yield from _list(_dict(sg).get("nodes"))


def collect(workflow: dict) -> list[dict]:
    """Model files a workflow uses: templates' per-node download links (properties.models) and the file names in
    known loader nodes. One entry per (directory, name); an entry with a link wins over one without. Names use /
    (Windows widgets store subfolders with \\). Anything malformed is skipped."""
    found: dict[tuple[str, str], dict] = {}
    for n in _nodes(_dict(workflow)):
        if not isinstance(n, dict):
            continue
        for m in _list(_dict(n.get("properties")).get("models")):
            m = _dict(m)
            name, directory, url = m.get("name"), m.get("directory"), m.get("url")
            if isinstance(name, str) and name and isinstance(directory, str) and directory:
                name = name.replace("\\", "/")
                found[(directory, name)] = {"name": name, "directory": directory,
                                            "url": url if isinstance(url, str) and url else None}
        folder = LOADER_FOLDERS.get(n.get("type")) if isinstance(n.get("type"), str) else None
        if folder:
            values = n.get("widgets_values")
            values = list(values.values()) if isinstance(values, dict) else _list(values)
            for v in values:
                if isinstance(v, str) and v.lower().endswith(MODEL_EXTS):
                    v = v.replace("\\", "/")
                    found.setdefault((folder, v), {"name": v, "directory": folder, "url": None})
    return list(found.values())


HF_PREFIX = "https://huggingface.co/"


CIVITAI = "https://civitai.com/"
CIVITAI_DOWNLOAD = CIVITAI + "api/download/models/"
CIVITAI_API = CIVITAI + "api/v1/"


def allowed(url) -> bool:
    return isinstance(url, str) and url.startswith((HF_PREFIX, CIVITAI_DOWNLOAD))


def is_civitai(url) -> bool:
    return isinstance(url, str) and url.startswith(CIVITAI_DOWNLOAD)


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


DEFAULT_TOKEN_FILE = ROOT / "hf_token.txt"   # install.bat can write it; gitignored


class GatedError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


DEFAULT_CIVITAI_TOKEN_FILE = ROOT / "civitai_token.txt"


def _read_token(env_names, cfg_key, default_file, config_path) -> str | None:
    for env in env_names:
        if os.environ.get(env):
            return os.environ[env].strip()
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    cfg = _dict(cfg)
    try:
        return Path(cfg.get(cfg_key) or default_file).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def token(config_path=ROOT / "config.json") -> str | None:
    """Hugging Face token: env HF_TOKEN, else the file config.json names, else hf_token.txt next to this file
    (never logged or sent to the browser)."""
    return _read_token(("HF_TOKEN",), "hf_token_file", DEFAULT_TOKEN_FILE, config_path)


def civitai_token(config_path=ROOT / "config.json") -> str | None:
    """Civitai API key: env CIVITAI_API_TOKEN (or CIVITAI_TOKEN), else config.json's civitai_token_file, else
    civitai_token.txt next to this file (never logged or sent to the browser)."""
    return _read_token(("CIVITAI_API_TOKEN", "CIVITAI_TOKEN"), "civitai_token_file", DEFAULT_CIVITAI_TOKEN_FILE,
                       config_path)


def token_for(url, hf: str | None, civitai: str | None) -> str | None:
    """Each site gets only its own token."""
    if is_civitai(url) or (isinstance(url, str) and url.startswith(CIVITAI_API)):
        return civitai
    return hf if isinstance(url, str) and url.startswith(HF_PREFIX) else None


def repo_page(url: str) -> str:
    parts = url.split("/")
    return "/".join(parts[:5])          # https://huggingface.co/<owner>/<repo>


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow https redirects only, and keep the token only while the redirect stays on the same site (not to
    Hugging Face's or Civitai's CDN or anywhere else): urllib's default handler copies the Authorization header to
    every redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https":
            raise urllib.error.HTTPError(newurl, code, f"refused redirect to {target.scheme}", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and target.hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def _tls() -> ssl.SSLContext:
    """The system's CAs plus certifi's when installed (ComfyUI ships it): Windows' store can lack a root that a
    site's chain needs (civitai.com: ISRG Root X2), and OpenSSL then fails on an expired cross-signed copy."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except (ImportError, OSError, ssl.SSLError):
        pass
    return ctx


_OPENER = urllib.request.build_opener(_SafeRedirect, urllib.request.HTTPSHandler(context=_tls()))


def _open(req, timeout=60):
    return _OPENER.open(req, timeout=timeout)


def _request(url: str, tok: str | None, method: str = "GET") -> urllib.request.Request:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "ComfyUI-MissingModels"})
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    return req


def _gated_message(url: str) -> str:
    if is_civitai(url):
        return "Civitai needs your API key for this file: add one (see the README), then try again"
    return f"Accept the license at {repo_page(url)}, then try again"


def size(url: str, tok: str | None = None) -> int | None:
    """File size from a HEAD request (Hugging Face's X-Linked-Size, else Content-Length); None if unknown."""
    try:
        with _open(_request(url, tok, "HEAD"), timeout=30) as r:
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
        with _open(_request(url, tok), timeout=60) as r:
            if (r.headers.get("Content-Type") or "").startswith("text/html"):   # a sign-in page, not a model
                raise GatedError(_gated_message(url))
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
            if total and done != total:   # urllib stops quietly when the connection drops early
                raise RuntimeError(f"incomplete download ({done:,} of {total:,} bytes); try again")
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        if e.code in (401, 403):
            raise GatedError(_gated_message(url)) from None
        raise RuntimeError(f"HTTP {e.code} {e.reason}") from None
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if dest.exists():                    # appeared while downloading (copied in by hand): keep it
        part.unlink(missing_ok=True)
        raise FileExistsError(f"{dest.name} already exists")
    part.replace(dest)
    return dest


# ---- Civitai: find a missing file by exact name, or resolve a link the user pasted --------------------------
# models folder -> Civitai model types, so a search isn't crowded out by other kinds of models
CIVITAI_TYPES = {"loras": ["LORA", "LoCon", "DoRA"], "checkpoints": ["Checkpoint"], "vae": ["VAE"],
                 "upscale_models": ["Upscaler"], "controlnet": ["Controlnet"], "embeddings": ["TextualInversion"]}
_LINK = re.compile(r"https://(?:www\.)?civitai\.(?:com|green)/(?:api/download/models/(\d+)|models/(\d+))")


def _get_json(url: str, tok: str | None = None):
    try:
        with _open(_request(url, tok), timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Civitai: HTTP {e.code} {e.reason}") from None
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Civitai: {e}") from None


def _base(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _civitai_file(f, model_id, version_id) -> dict | None:
    """A Civitai file entry as a download: url, size, Civitai's file name, and the model page."""
    f = _dict(f)
    url, fname = f.get("downloadUrl"), f.get("name")
    if not (is_civitai(url) and isinstance(fname, str) and fname.lower().endswith(MODEL_EXTS)):
        return None
    kb = f.get("sizeKB")
    page = f"{CIVITAI}models/{model_id}" + (f"?modelVersionId={version_id}" if version_id else "")
    return {"url": url, "size": int(kb * 1024) if isinstance(kb, (int, float)) else None, "file": fname,
            "page": page}


def _queries(name: str) -> list[str]:
    """Search words from a file name, most to least specific (Civitai searches model names, not file names):
    juggernautXL_ragnarokBy_v2.safetensors -> "juggernaut xl ragnarok by v2", "juggernaut xl ragnarok by",
    "juggernaut"."""
    stem = name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem).lower()          # camelCase -> camel case
    words = re.sub(r"[_\-.\s]+", " ", stem).strip()
    short = re.sub(r"(\s+(v|ver|version)?\s*\d[\w]*)+$", "", words).strip()
    first = short.split(" ")[0] if short else ""
    return [q for q in dict.fromkeys([words, short, first if len(first) >= 4 else ""]) if q]


def civitai_find(name: str, directory: str, tok: str | None = None) -> dict | None:
    """The Civitai file whose name is exactly this file's name (case aside), or None. Never a near miss: the
    workflow needs that exact file."""
    want = _base(name)
    for q in _queries(name):
        params = [("query", q), ("limit", "20")] + [("types", t) for t in CIVITAI_TYPES.get(directory, [])]
        data = _get_json(CIVITAI_API + "models?" + urllib.parse.urlencode(params), tok)
        for item in _list(_dict(data).get("items")):
            item = _dict(item)
            for v in _list(item.get("modelVersions")):
                v = _dict(v)
                for f in _list(v.get("files")):
                    if isinstance(_dict(f).get("name"), str) and f["name"].lower() == want:
                        hit = _civitai_file(f, item.get("id"), v.get("id"))
                        if hit:
                            return hit
    return None


def civitai_resolve(link: str, name: str, tok: str | None = None) -> dict:
    """A Civitai model page or download link -> the file to save as `name`: the file with that exact name in it,
    else the version's primary model file (newest version when the link names none). ValueError if it has none."""
    link = (link or "").strip()
    m = _LINK.match(link)
    if not m:
        raise ValueError("not a Civitai model link (https://civitai.com/models/...)")
    version = m.group(1) or urllib.parse.parse_qs(urllib.parse.urlsplit(link).query).get("modelVersionId", [None])[0]
    if version:
        if not str(version).isdigit():
            raise ValueError("not a Civitai model link")
        v = _dict(_get_json(f"{CIVITAI_API}model-versions/{version}", tok))
        model_id, versions = v.get("modelId"), [v]
    else:
        model_id = m.group(2)
        versions = [_dict(v) for v in _list(_dict(_get_json(f"{CIVITAI_API}models/{model_id}", tok))
                                            .get("modelVersions"))]
    want = _base(name)
    for v in versions:                       # the exact file, in any version the link covers
        for f in _list(v.get("files")):
            if isinstance(_dict(f).get("name"), str) and f["name"].lower() == want:
                hit = _civitai_file(f, model_id, v.get("id"))
                if hit:
                    return hit
    if versions:                             # else the version's main file
        files = _list(versions[0].get("files"))
        for f in sorted(files, key=lambda f: not _dict(f).get("primary")):
            hit = _civitai_file(f, model_id, versions[0].get("id"))
            if hit:
                return hit
    raise ValueError("no model file at that link")
