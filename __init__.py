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
