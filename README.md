# ComfyUI Missing Models

A **Missing models** button (top bar, and Extensions menu) that downloads the loaded workflow's missing models and
LoRAs straight into the right `ComfyUI/models/<folder>`. Works with the download links ComfyUI's templates carry
(Hugging Face); files without a link are listed with search links.

- Gated Hugging Face repos need your token: env `HF_TOKEN`, or copy `config.example.json` to `config.json` and point
  `hf_token_file` at a file holding it. Accept the model's license on its Hugging Face page first.
- Only `https://huggingface.co/` links and model files (`.safetensors .ckpt .pt .pth .bin .gguf .sft`) are
  downloaded; existing files are never overwritten; downloads go to `<name>.part` until complete.

Install (Windows, Developer Mode on):

    mklink /D "<ComfyUI>\custom_nodes\ComfyUI-MissingModels" "Z:\Git\ComfyUI-MissingModels"

Tests: `python -m pytest -q tests`
