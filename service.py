"""One download batch at a time, in a background thread; progress and results go out as events."""
import threading
import time

try:
    from . import core            # inside ComfyUI (package)
except ImportError:
    import core                   # tests (plain module)


class Downloader:
    def __init__(self, emit, dest_for, token=core.token, civitai_token=core.civitai_token):
        self.emit, self.dest_for, self.token, self.civitai_token = emit, dest_for, token, civitai_token
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
        results = []
        tokens = []
        for read in (self.token, self.civitai_token):
            try:
                tokens.append(read())
            except Exception:   # an unreadable config must not kill the batch (the dialog waits for "done")
                tokens.append(None)
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                results.append({"name": None, "directory": None, "status": "failed", "error": "bad item"})
                continue
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
                core.download(it["url"], dest, core.token_for(it["url"], *tokens), progress, self.cancel)
                results.append(dict(r, status="installed", path=str(dest)))
            except core.Cancelled:
                results.append(dict(r, status="cancelled"))
            except Exception as e:
                results.append(dict(r, status="failed", error=str(e)))
        self.emit("missing_models.done", {"results": results})
