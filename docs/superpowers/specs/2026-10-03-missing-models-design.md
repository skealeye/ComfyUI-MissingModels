# ComfyUI Missing Models: design

Date: 2026-10-03. A ComfyUI custom node (extension) in its own repo, `Z:\Git\ComfyUI-MissingModels`, linked into
`Y:\ComfyUI_windows_portable\ComfyUI\custom_nodes` with a directory junction. Private GitHub repo
`skealeye/ComfyUI-MissingModels`.

## Intent

Loading a ComfyUI template often shows missing models and LoRAs. ComfyUI's own dialog only downloads them to the
browser's Downloads folder, so each file has to be moved into the right `models/` subfolder by hand. This extension
downloads them straight into the right folder from inside ComfyUI.

What the user asked for: auto download and install of a loaded workflow's missing models and LoRAs; as a button
inside ComfyUI (option A); files without a download link are listed with search links, not guessed (option A).

Context: the user is moving work from SlopFactory into ComfyUI workflows (build in ComfyUI, export API JSON). This
tool is ComfyUI-side on purpose and does not touch SlopFactory.

Success: load a template with missing files, click once, the files land in the right folders, and the workflow's
loader nodes stop showing them as missing without restarting ComfyUI.

## Facts this rests on (checked 2026-10-03 on this install)

- ComfyUI v0.38.0, frontend `comfyui_frontend_package` 1.53.6: has `actionBarButtons`, `menuCommands`,
  `app.refreshComboInNodes`.
- 243 of the 594 installed templates list their models per node in `properties.models` as
  `{name, url, directory}` (also inside subgraphs, `definitions.subgraphs[].nodes`); all 1,135 URLs are
  `https://huggingface.co/...`. Folders used: text_encoders, diffusion_models, vae, loras, checkpoints, clip_vision,
  latent_upscale_models, geometry_estimation, model_patches, controlnet, audio_encoders, background_removal,
  upscale_models, embeddings.
- `folder_paths.folder_names_and_paths` maps folder names to one or more directories (e.g. `diffusion_models` =
  `models/unet` + `models/diffusion_models`; `text_encoders` = `models/text_encoders` + `models/clip`).
- Some Hugging Face repos are gated (e.g. `Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients`): 401/403 until the user
  accepts the license; then a token works. The user's token is in `Z:\Git\SlopFactory\.hf_token`.

## What the user sees

- **Toolbar button "Missing models"** (and a menu command of the same name).
- **On workflow load**, a quiet check: if anything is missing, a toast "N models missing" with a button to open
  the dialog. Nothing missing: no toast.
- **Dialog**: one row per missing file: checkbox (ticked), file name, target folder, size (from a HEAD request;
  "?" if unknown), source host.
  - Files with no link: greyed, not downloadable, "not found" with **Search Hugging Face** / **Search Civitai**
    links (open in a new browser tab).
  - Files already present are not listed. "Present" uses ComfyUI's lookup, so a file in a subfolder or in either
    of a folder's directories counts.
  - **Download** fetches ticked files one at a time; per-file and total progress (GB done / total); **Cancel**.
  - Errors per row: gated repo → "Accept the license at <repo page link>, then try again"; other HTTP or network
    errors → the message; the rest continue.
  - When done: model lists refresh (`app.refreshComboInNodes()`), the dialog shows what was installed and where.

## Architecture

**`core.py`** (no ComfyUI imports; unit-tested):
- `collect(workflow: dict) -> list[dict]`: model entries `{name, directory, url | None}` from (1) every node's
  `properties.models` (top level and subgraphs), (2) file-name widget values of known loader nodes, mapped to a
  folder: `CheckpointLoaderSimple`/`ImageOnlyCheckpointLoader` → checkpoints; `LoraLoader`/`LoraLoaderModelOnly`/
  `LTXICLoRALoaderModelOnly` → loras; `UNETLoader` → diffusion_models; `VAELoader` → vae; `CLIPLoader` /
  `DualCLIPLoader` / `TripleCLIPLoader` → text_encoders; `CLIPVisionLoader` → clip_vision; `UpscaleModelLoader` →
  upscale_models; `ControlNetLoader` → controlnet. Deduplicated by (directory, name); a linked entry wins over an
  unlinked one.
- `allowed(url) -> bool`: `https://huggingface.co/` only.
- `safe_name(name) -> str`: rejects absolute paths, drive letters, `..`, and extensions other than
  `.safetensors .ckpt .pt .pth .bin .gguf .sft`; keeps a relative subfolder (`subdir/file.safetensors`).
- `download(url, dest, token=None, progress=None, cancel=None) -> Path`: streams to `dest.part`, renames when
  complete; never overwrites an existing `dest`; removes the `.part` on any failure or cancel; 401/403 →
  `GatedError` with the repo page URL; sends `Authorization: Bearer` only when a token is set.
- `token() -> str | None`: env `HF_TOKEN`, else the file named by `config.json` `hf_token_file` (default
  `Z:\Git\SlopFactory\.hf_token`). The token is never logged or sent to the browser.

**`__init__.py`** (ComfyUI glue; `WEB_DIRECTORY = "./web"`, `NODE_CLASS_MAPPINGS = {}`):
- `POST /missing_models/check` body `{workflow}` → `[{name, directory, url, size, downloadable, reason}]` for the
  missing ones only. Present = `folder_paths.get_full_path(directory, name)` is not None or the basename appears in
  `folder_paths.get_filename_list(directory)`. Unknown `directory` → listed as not downloadable ("unknown folder").
- `POST /missing_models/download` body `{items: [{name, directory, url}]}` → starts one background thread (a second
  request while busy → 409). Re-validates every item with `allowed` / `safe_name`. Destination: the first directory
  of `folder_paths.get_folder_paths(directory)` / `name`. Progress via
  `PromptServer.instance.send_sync("missing_models.progress", {...})`; final event `missing_models.done` with
  per-item results.
- `POST /missing_models/cancel`.

**`web/missing_models.js`**: registers the extension, the toolbar button and menu command, the load-time check
(after a graph is configured), the dialog (plain DOM, ComfyUI's CSS variables for theme), progress listener,
refresh when done.

## Errors

- Disallowed URL or unsafe name → row not downloadable ("unsupported source" / "unsafe file name"), and the
  download route refuses it even if sent.
- Gated → license message with link. Network/HTTP error → message on that row; other rows continue.
- Cancel → current `.part` removed; remaining rows marked "cancelled".
- Disk full or write error → that row fails with the OS message; `.part` removed.

## Testing

- `tests/test_core.py` (pytest, no ComfyUI): `collect` on a real installed template (`video_ltx2_3_*`, copied as a
  fixture) and on a synthetic workflow with loader nodes and a subgraph; dedupe; `allowed`; `safe_name` (`..`,
  `C:\x`, `/etc/x`, `.exe`, subfolder OK); `download` with a fake `urlopen`: progress, `.part` then rename, no
  overwrite, cancel removes `.part`, 403 → `GatedError`, token header only when set; `token()` env vs file.
- End to end in the user's Chrome (claude-in-chrome): ComfyUI running, load a template whose smallest model is
  temporarily renamed away, click, see it download, the node no longer red; restore.

## Out of scope

- Finding files without links (Hugging Face / Civitai search and auto-match).
- Civitai downloads, installing missing custom nodes (ComfyUI-Manager does that), deleting models.
