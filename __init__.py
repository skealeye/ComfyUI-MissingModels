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
    try:
        rows = core.missing(core.collect(wf if isinstance(wf, dict) else {}), exists,
                            set(folder_paths.folder_names_and_paths))
    except Exception as e:   # an odd workflow must never break loading it in ComfyUI
        print(f"[MissingModels] check failed: {e}")
        rows = []
    tokens, loop = (core.token(), core.civitai_token()), asyncio.get_running_loop()
    for r in rows:
        r["size"] = (await loop.run_in_executor(None, core.size, r["url"], core.token_for(r["url"], *tokens))
                     if r["downloadable"] else None)
    return web.json_response({"missing": rows})


@routes.post("/missing_models/civitai/find")
async def civitai_find(request):
    """Look each file up on Civitai by its exact name: {found: {...}} or {found: null}, or {error}."""
    items = (await _body(request)).get("items")
    if not isinstance(items, list):
        return web.json_response({"error": "items must be a list"}, status=400)
    tok, loop = core.civitai_token(), asyncio.get_running_loop()

    async def one(it):
        it = it if isinstance(it, dict) else {}
        name, directory = it.get("name"), it.get("directory")
        if not (isinstance(name, str) and isinstance(directory, str)):
            return {"name": name, "directory": directory, "error": "bad item"}
        try:
            found = await loop.run_in_executor(None, core.civitai_find, name, directory, tok)
            return {"name": name, "directory": directory, "found": found}
        except Exception as e:
            return {"name": name, "directory": directory, "error": str(e)}
    return web.json_response({"results": await asyncio.gather(*(one(it) for it in items[:50]))})


@routes.post("/missing_models/civitai/resolve")
async def civitai_resolve(request):
    """A Civitai link the user pasted for one file -> {found: {...}}, or {error} (status 400)."""
    body = await _body(request)
    link, name = body.get("link"), body.get("name")
    if not (isinstance(link, str) and isinstance(name, str)):
        return web.json_response({"error": "link and name are needed"}, status=400)
    try:
        found = await asyncio.get_running_loop().run_in_executor(None, core.civitai_resolve, link, name,
                                                                 core.civitai_token())
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"found": found})


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
