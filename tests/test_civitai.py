"""Civitai: links, tokens, the exact-name lookup, pasted links, and the routes."""
import io
import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

import core
import service
from test_routes import call, pkg  # noqa: F401  (pkg is a fixture)

HF = "https://huggingface.co/o/r/resolve/main/"
CIV = "https://civitai.com/api/download/models/"


class Resp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers if headers is not None else {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def f(name, vid, kb=1024.0, primary=True):
    return {"name": name, "sizeKB": kb, "primary": primary, "downloadUrl": f"{CIV}{vid}"}


# a search result: model 7, versions 71 (newest) and 70
SEARCH = {"items": [
    {"id": 6, "modelVersions": [{"id": 60, "files": [f("add_detail_xl.safetensors", 60)]}]},
    {"id": 7, "modelVersions": [{"id": 71, "files": [f("add_detail_v2.safetensors", 71)]},
                                {"id": 70, "files": [f("Add_Detail.safetensors", 70, kb=2.0)]}]}]}


# ---- links and tokens -------------------------------------------------------------------------------------
def test_civitai_download_links_are_allowed_and_nothing_else_there():
    assert core.allowed(CIV + "123") and core.allowed(CIV + "123?type=Model&format=SafeTensor")
    assert not core.allowed("https://civitai.com/models/123")
    assert not core.allowed("https://civitai.com.evil.example/api/download/models/1")
    assert not core.allowed("http://civitai.com/api/download/models/1")


def test_each_site_gets_only_its_own_token():
    assert core.token_for(CIV + "1", "hf", "civ") == "civ"
    assert core.token_for(core.CIVITAI_API + "models?query=x", "hf", "civ") == "civ"
    assert core.token_for(HF + "m.safetensors", "hf", "civ") == "hf"
    assert core.token_for("https://example.com/m.safetensors", "hf", "civ") is None


def test_civitai_token_env_then_config_then_file_next_to_extension(tmp_path, monkeypatch):
    for env in ("CIVITAI_API_TOKEN", "CIVITAI_TOKEN"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(core, "DEFAULT_CIVITAI_TOKEN_FILE", tmp_path / "civitai_token.txt")
    none = tmp_path / "none.json"
    assert core.civitai_token(none) is None
    (tmp_path / "civitai_token.txt").write_text("abc\r\n", encoding="utf-8")
    assert core.civitai_token(none) == "abc"
    other = tmp_path / "k.txt"
    other.write_text("fromcfg", encoding="utf-8")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"civitai_token_file": str(other)}), encoding="utf-8")
    assert core.civitai_token(cfg) == "fromcfg"
    monkeypatch.setenv("CIVITAI_TOKEN", "env2")
    assert core.civitai_token(cfg) == "env2"
    monkeypatch.setenv("CIVITAI_API_TOKEN", "env1")
    assert core.civitai_token(cfg) == "env1"


def redirect(url, newurl):
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Bearer secret")
    return core._SafeRedirect().redirect_request(req, io.BytesIO(b""), 307, "Temporary", {}, newurl)


def test_civitai_redirect_to_its_cdn_drops_token():
    new = redirect(CIV + "1", "https://b2.civitai.com/file/x.safetensors?Authorization=signed")
    assert not new.has_header("Authorization")


def test_civitai_redirect_within_civitai_keeps_token_and_hf_token_never_reaches_civitai():
    assert redirect(CIV + "1", CIV + "2").get_header("Authorization") == "Bearer secret"
    assert not redirect(HF + "m.safetensors", CIV + "2").has_header("Authorization")


# ---- downloads --------------------------------------------------------------------------------------------
def test_civitai_401_asks_for_an_api_key(tmp_path, monkeypatch):
    def deny(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(core, "_open", deny)
    with pytest.raises(core.GatedError, match="Civitai needs your API key"):
        core.download(CIV + "1", tmp_path / "m.safetensors")


def test_a_sign_in_page_is_never_saved_as_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_open", lambda req, timeout=60: Resp(b"<html>log in</html>",
                                                                    {"Content-Type": "text/html; charset=utf-8"}))
    with pytest.raises(core.GatedError, match="Civitai"):
        core.download(CIV + "1", tmp_path / "m.safetensors")
    assert list(tmp_path.iterdir()) == []


def test_downloader_sends_each_url_its_own_token(tmp_path, monkeypatch):
    seen = {}

    def fake(url, dest, tok=None, progress=None, cancel=None):
        seen[url] = tok
        return dest
    monkeypatch.setattr(core, "download", fake)
    events = []
    d = service.Downloader(lambda ev, data: events.append(ev), lambda di, n: tmp_path / n,
                           token=lambda: "hf", civitai_token=lambda: "civ")
    d.start([{"name": "a.safetensors", "directory": "loras", "url": HF + "a.safetensors"},
             {"name": "b.safetensors", "directory": "loras", "url": CIV + "5"}])
    d.thread.join(5)
    assert seen == {HF + "a.safetensors": "hf", CIV + "5": "civ"} and events[-1] == "missing_models.done"


def test_downloader_survives_a_broken_civitai_token_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "download", lambda url, dest, tok=None, progress=None, cancel=None: dest)
    events = []

    def boom():
        raise OSError("unreadable")
    d = service.Downloader(lambda ev, data: events.append((ev, data)), lambda di, n: tmp_path / n,
                           token=lambda: None, civitai_token=boom)
    d.start([{"name": "b.safetensors", "directory": "loras", "url": CIV + "5"}])
    d.thread.join(5)
    assert events[-1][1]["results"][0]["status"] == "installed"


# ---- exact-name lookup ------------------------------------------------------------------------------------
def test_queries_from_file_names():
    assert core._queries("loras/add_detail_v2.safetensors") == ["add detail v2", "add detail"]
    assert core._queries("Detail-Tweaker.XL.safetensors") == ["detail tweaker xl", "detail"]
    assert core._queries("eyes-000010.safetensors") == ["eyes 000010", "eyes"]
    assert core._queries(r"sdxl\juggernautXL_ragnarok.safetensors") == ["juggernaut xl ragnarok", "juggernaut"]
    assert core._queries("abc_v2.safetensors") == ["abc v2", "abc"]   # short first words aren't searched alone


def fake_api(monkeypatch, answers):
    """answers: url-substring -> json; records the urls asked for."""
    asked = []

    def get(url, tok=None):
        asked.append((url, tok))
        for k, v in answers.items():
            if k in url:
                return v
        return {"items": []}
    monkeypatch.setattr(core, "_get_json", get)
    return asked


def test_find_takes_only_the_exact_file_name(monkeypatch):
    asked = fake_api(monkeypatch, {"models?": SEARCH})
    hit = core.civitai_find("sub/add_detail.safetensors", "loras", "civ")
    assert hit == {"url": CIV + "70", "size": 2048, "file": "Add_Detail.safetensors",
                   "page": "https://civitai.com/models/7?modelVersionId=70"}
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(asked[0][0]).query)
    assert q["query"] == ["add detail"] and q["types"] == ["LORA", "LoCon", "DoRA"] and asked[0][1] == "civ"


def test_find_no_near_misses(monkeypatch):
    fake_api(monkeypatch, {"models?": SEARCH})
    assert core.civitai_find("add_detail_v3.safetensors", "loras") is None


def test_find_tries_the_short_query_too(monkeypatch):
    asked = fake_api(monkeypatch, {"query=add+detail&": SEARCH})
    assert core.civitai_find("add_detail_v2.safetensors", "checkpoints")["url"] == CIV + "71"
    assert len(asked) == 2 and "types=Checkpoint" in asked[0][0]


def test_find_skips_files_that_are_not_models_or_not_civitai_downloads(monkeypatch):
    bad = {"items": [{"id": 1, "modelVersions": [{"id": 2, "files": [
        {"name": "x.safetensors", "downloadUrl": "https://evil.example/x"},
        {"name": "x.safetensors", "downloadUrl": None}]}]}, "junk", {"modelVersions": "junk"}]}
    fake_api(monkeypatch, {"models?": bad})
    assert core.civitai_find("x.safetensors", "loras") is None


# ---- pasted links -----------------------------------------------------------------------------------------
VERSION_70 = {"id": 70, "modelId": 7, "files": [f("notes.zip", 70), f("other.safetensors", 70, primary=False),
                                                 f("Main.safetensors", 70)]}
MODEL_7 = {"id": 7, "modelVersions": [{"id": 71, "files": [f("new.safetensors", 71)]},
                                      {"id": 70, "files": [f("wanted.safetensors", 70)]}]}


@pytest.mark.parametrize("link", ["https://civitai.com/models/7/some-slug?modelVersionId=70",
                                  "https://civitai.com/api/download/models/70",
                                  " https://www.civitai.com/models/7?modelVersionId=70 "])
def test_resolve_version_links_take_the_primary_model_file(monkeypatch, link):
    asked = fake_api(monkeypatch, {"model-versions/70": VERSION_70})
    hit = core.civitai_resolve(link, "my_name.safetensors", "civ")
    assert (hit["url"], hit["file"]) == (CIV + "70", "Main.safetensors") and asked == [
        (core.CIVITAI_API + "model-versions/70", "civ")]


def test_resolve_prefers_the_exact_name_in_the_link(monkeypatch):
    fake_api(monkeypatch, {"model-versions/70": VERSION_70})
    assert core.civitai_resolve(CIV + "70", "OTHER.safetensors")["file"] == "other.safetensors"


def test_resolve_model_page_finds_the_name_in_any_version_else_the_newest(monkeypatch):
    fake_api(monkeypatch, {"models/7": MODEL_7})
    assert core.civitai_resolve("https://civitai.com/models/7/slug", "wanted.safetensors")["url"] == CIV + "70"
    assert core.civitai_resolve("https://civitai.com/models/7", "else.safetensors")["url"] == CIV + "71"


@pytest.mark.parametrize("link", ["", "https://evil.example/models/7", "civitai.com/models/7",
                                  "https://civitai.com/user/x", "https://civitai.com/models/7?modelVersionId=x"])
def test_resolve_refuses_other_links(monkeypatch, link):
    fake_api(monkeypatch, {})
    with pytest.raises(ValueError):
        core.civitai_resolve(link, "a.safetensors")


def test_resolve_without_a_model_file(monkeypatch):
    fake_api(monkeypatch, {"model-versions/9": {"id": 9, "modelId": 1, "files": [f("x.zip", 9)]}})
    with pytest.raises(ValueError, match="no model file"):
        core.civitai_resolve(CIV + "9", "a.safetensors")


def test_api_errors_become_readable(monkeypatch):
    def deny(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 503, "Busy", {}, None)
    monkeypatch.setattr(core, "_open", deny)
    with pytest.raises(RuntimeError, match="Civitai: HTTP 503"):
        core.civitai_find("a.safetensors", "loras")


# ---- routes -----------------------------------------------------------------------------------------------
def test_check_marks_civitai_links_downloadable(pkg):  # noqa: F811
    wf = {"nodes": [{"type": "X", "properties": {"models": [
        {"name": "c.safetensors", "url": CIV + "5", "directory": "loras"}]}}]}
    rows = call(pkg.check, {"workflow": wf})[1]["missing"]
    assert [(r["name"], r["downloadable"]) for r in rows] == [("c.safetensors", True)]


def test_find_route(pkg, monkeypatch):  # noqa: F811
    def find(name, directory, tok=None):
        if name == "boom.safetensors":
            raise RuntimeError("Civitai: down")
        return {"url": CIV + "1"} if name == "a.safetensors" else None
    monkeypatch.setattr(pkg.core, "civitai_find", find)
    status, body = call(pkg.civitai_find, {"items": [{"name": "a.safetensors", "directory": "loras"},
                                                     {"name": "b.safetensors", "directory": "loras"},
                                                     {"name": "boom.safetensors", "directory": "loras"}, 5]})
    assert status == 200
    assert [r.get("found") for r in body["results"][:2]] == [{"url": CIV + "1"}, None]
    assert body["results"][2]["error"] == "Civitai: down" and body["results"][3]["error"] == "bad item"
    assert call(pkg.civitai_find, {"items": "x"})[0] == 400


def test_resolve_route(pkg, monkeypatch):  # noqa: F811
    def resolve(link, name, tok=None):
        if "bad" in link:
            raise ValueError("not a Civitai model link")
        return {"url": CIV + "2", "file": name}
    monkeypatch.setattr(pkg.core, "civitai_resolve", resolve)
    assert call(pkg.civitai_resolve, {"link": "https://civitai.com/models/2", "name": "a.safetensors"}) == (
        200, {"found": {"url": CIV + "2", "file": "a.safetensors"}})
    assert call(pkg.civitai_resolve, {"link": "bad", "name": "a.safetensors"}) == (
        400, {"error": "not a Civitai model link"})
    assert call(pkg.civitai_resolve, {"link": 5})[0] == 400
