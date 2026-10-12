# ComfyUI Missing Models

Opened a workflow and got a wall of missing-model errors? This extension adds a **Missing models** button to
ComfyUI that downloads them, straight into the right `models/<folder>`, with one click.

- Lists every model and LoRA the loaded workflow needs that you don't have, with sizes.
- Downloads the ones that have a link (ComfyUI's built-in templates carry Hugging Face links), with progress and
  cancel.
- Files without a link: **Find on Civitai** looks them up by their exact file name, or paste a Civitai link for
  one yourself. "Search Hugging Face" and "Search Civitai" links help you find the rest.
- Checks each workflow when you open it and tells you if something is missing.

## Install

### Windows: installer (easiest)

1. Download [`install.bat`](https://github.com/skealeye/ComfyUI-MissingModels/raw/main/install.bat)
   (if your browser opens it as text, right-click the link and choose **Save link as**).
2. Double-click it. It finds your ComfyUI (portable, Desktop app or git install), or asks where it is, then
   installs the extension. Git is used if you have it; otherwise it downloads a zip.
3. Restart ComfyUI.

Run `install.bat` again any time to update. It keeps your saved token and key.

### Any OS: git

```
cd <your ComfyUI folder>/custom_nodes
git clone https://github.com/skealeye/ComfyUI-MissingModels
```

Restart ComfyUI. To update later: `git pull` in that folder.

For the ComfyUI Desktop app, `custom_nodes` is inside the base folder you picked at setup
(by default `Documents\ComfyUI`).

### Any OS: zip

Download the [zip](https://github.com/skealeye/ComfyUI-MissingModels/archive/refs/heads/main.zip), unzip it into
`ComfyUI/custom_nodes`, rename the folder to `ComfyUI-MissingModels`, and restart ComfyUI.

No extra Python packages are needed.

## Use

Open a workflow and click **Missing models** in the top bar (also under **Extensions** in the menu). Tick the
files you want and click **Download**.

### Files from Civitai

Workflows from Civitai usually carry no download links, so those files show up as "not found". Two ways to get
them:

- **Find on Civitai** (button at the bottom) searches Civitai for every such file. Only a file with **exactly** the
  same name counts as found, so it never downloads a look-alike. Civitai searches by model name, not file name, so
  files named after their model are found best; for the rest:
- **Paste Civitai link** (on the file's row): paste the model's page (`https://civitai.com/models/...`, ideally with
  the version you want selected) or its download link. The file is saved under the name the workflow expects, even
  if Civitai calls it something else (the row tells you when it does).

## Tokens and API keys (optional)

### Hugging Face token

Most models download without an account. Some, like FLUX.1-dev, are *gated*: you need a free Hugging Face
account, then click **Agree** on the model's page, then give the extension a token:

1. Create a **Read** token at https://huggingface.co/settings/tokens.
2. Give it to the extension in one of these ways:
   - Run `install.bat` again and paste the token when asked, **or**
   - save the token in a file named `hf_token.txt` in the extension's folder, **or**
   - set the `HF_TOKEN` environment variable (the one other Hugging Face tools use), **or**
   - keep your token in a file somewhere else: copy `config.example.json` to `config.json` and point
     `hf_token_file` at that file.

### Civitai API key

Many Civitai models download without an account. Some need you to be signed in, and then the download fails with
"Civitai needs your API key". To add one:

1. Sign in to Civitai, open https://civitai.com/user/account, and under **API Keys** click **Add API key**.
2. Give it to the extension in one of these ways:
   - Run `install.bat` again and paste the key when asked, **or**
   - save the key in a file named `civitai_token.txt` in the extension's folder, **or**
   - set the `CIVITAI_API_TOKEN` environment variable, **or**
   - in `config.json` (see above), point `civitai_token_file` at a file holding it.

Tokens and keys stay on your computer. Each is sent only to its own site (`huggingface.co` or `civitai.com`, not
even to their download servers), never to the browser, and never logged.

## Safety

- Only `https://huggingface.co/` and `https://civitai.com/api/download/models/` links are downloaded, and only
  model files (`.safetensors .ckpt .pt .pth .bin .gguf .sft`).
- Existing files are never overwritten.
- Downloads go to `<name>.part` and are renamed only once complete, so an interrupted download never leaves a
  broken model behind. A sign-in page is never saved as a model.

## Development

```
python -m pytest -q tests
```

## License

[MIT](LICENSE)
