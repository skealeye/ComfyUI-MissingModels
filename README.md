# ComfyUI Missing Models

Opened a workflow and got a wall of missing-model errors? This extension adds a **Missing models** button to
ComfyUI that downloads them, straight into the right `models/<folder>`, with one click.

- Lists every model and LoRA the loaded workflow needs that you don't have, with sizes.
- Downloads the ones that have a link (ComfyUI's built-in templates carry Hugging Face links), with progress and
  cancel. Files without a link get "Search Hugging Face" and "Search Civitai" links instead.
- Checks each workflow when you open it and tells you if something is missing.

## Install

### Windows: installer (easiest)

1. Download [`install.bat`](https://github.com/skealeye/ComfyUI-MissingModels/raw/main/install.bat)
   (if your browser opens it as text, right-click the link and choose **Save link as**).
2. Double-click it. It finds your ComfyUI (portable, Desktop app or git install), or asks where it is, then
   installs the extension. Git is used if you have it; otherwise it downloads a zip.
3. Restart ComfyUI.

Run `install.bat` again any time to update. It keeps your saved token.

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

## Gated models (Hugging Face token)

Most models download without an account. Some, like FLUX.1-dev, are *gated*: you need a free Hugging Face
account, then click **Agree** on the model's page, then give the extension a token:

1. Create a **Read** token at https://huggingface.co/settings/tokens.
2. Give it to the extension in one of these ways:
   - Run `install.bat` again and paste the token when asked, **or**
   - save the token in a file named `hf_token.txt` in the extension's folder, **or**
   - set the `HF_TOKEN` environment variable (the one other Hugging Face tools use), **or**
   - keep your token in a file somewhere else: copy `config.example.json` to `config.json` and point
     `hf_token_file` at that file.

The token stays on your computer. It is sent only to `huggingface.co`, never to the browser, and never logged.

## Safety

- Only `https://huggingface.co/` links are downloaded, and only model files
  (`.safetensors .ckpt .pt .pth .bin .gguf .sft`).
- Existing files are never overwritten.
- Downloads go to `<name>.part` and are renamed only once complete, so an interrupted download never leaves a
  broken model behind.

## Development

```
python -m pytest -q tests
```

## License

[MIT](LICENSE)
