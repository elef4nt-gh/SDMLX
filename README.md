# 🍏 SDMLX - ComfyUI Node Suite for Faster Image Generation on macOS

SDMLX is an alpha-stage ComfyUI custom node suite for running SDXL workflows on Apple Silicon with Apple's MLX framework.

The goal is straightforward: make SDXL on the Mac feel less like a compromise. SDMLX ports the core SDXL workflow from the usual PyTorch-MPS path to an MLX-native runtime, with easy checkpoint conversion, `.sdmlx` package caching, Speed Patches, macOS-aware memory handling, and workflow nodes that try to stay close to familiar ComfyUI patterns.

In current local tests, SDMLX is typically about 25-30% faster than comparable PyTorch-MPS SDXL workflows on the same Mac, depending on the checkpoint, sampler, resolution, speed patch, ControlNet/IP-Adapter usage, and preview settings. It is still alpha software; results and APIs can change.

## Alpha Compatibility Notice

SDMLX is currently confirmed on the development test system: Mac Studio with Apple M1 Max and 64GB unified memory. Reports for other Apple Silicon chips and memory sizes are still needed.

If SDMLX generates images that ignore the prompt, look effectively unconditional, or produce obvious garbage output, please run the diagnostics mode and share the relevant terminal log in a GitHub issue or community thread:

```bash
SDMLX_CONDITIONING_DIAGNOSTICS=full /Applications/ComfyUI.app/Contents/MacOS/ComfyUI
```

Then run the included `sdmlx_txt2img.json` workflow once. The useful log lines start with:

```text
SDMLX Conditioning Diagnostics: mlx=...
SDMLX Conditioning Diagnostics: status=...
```

If ComfyUI Desktop crashes or hangs with the full diagnostics mode, use the lighter diagnostics path:

```bash
SDMLX_CONDITIONING_DIAGNOSTICS=1 /Applications/ComfyUI.app/Contents/MacOS/ComfyUI
```

Diagnostics are opt-in for that terminal launch only. They do not change your normal ComfyUI environment.

## What Works Today

- SDXL checkpoint loading, conversion and `.sdmlx` package caching
- Single-package `.sdmlx` model cache in ComfyUI's `models/SDMLX` folder
- SDXL sampling with MLX-native UNet, CLIP and VAE paths
- Fast SDXL sampler with image and latent outputs
- Built-in Speed Patch support for DMD2, Lightning, LCM and Hyper-SD style SDXL speed LoRAs
- Optional Spectrum Acceleration for selected Euler-based SDXL workflows
- LoRA loading, multi-LoRA loading, and scheduled LoRA strength curves
- IP-Adapter SDXL and FaceID / FaceID PlusV2 support
- MLX CLIP Vision encode path for IP-Adapter workflows
- ControlNet Union ProMax support with schedule curves
- Classic inpaint conditioning and SDMLX Inpaint Detailer
- Hires Fix and Tiled Upscale nodes
- ComfyUI preview toggle where useful
- macOS-oriented Memory Assist with optional preload / keep-warm behavior
- Auto-download helpers for common companion models

## Why SDMLX

PyTorch-MPS works, but SDXL workflows on the Mac can feel heavy, especially when adding ControlNet, IP-Adapter, FaceID, inpaint, hires fix, or tiled upscale. SDMLX is an attempt to build a Mac-first SDXL stack instead of adapting a CUDA-first one.

The main design ideas are:

- convert once, then load fast from `.sdmlx` packages
- keep workflows compact and familiar
- use MLX arrays and Apple Silicon memory behavior directly
- reduce avoidable data movement between PyTorch, NumPy, MLX and ComfyUI
- expose speed patches as first-class workflow controls
- use simple names for creative controls where possible

## Performance Snapshot

These are local alpha measurements from the current development machine and workflows. Treat them as orientation, not a benchmark promise.

Test system: Mac Studio with Apple M1 Max and 64GB unified memory.

| Workflow | PyTorch-MPS | SDMLX | Notes |
| --- | ---: | ---: | --- |
| SDXL, 20 steps, Euler/Karras, 768x1280 | 57.21s | 42.55s | about 26% faster |
| SDXL, 8 steps, speed patch, 768x1280 | 13.6-13.7s | 10.2-10.3s | about 24-25% faster |
| FaceID PlusV2, 20 steps, 832x1152 | 70.83s | 47.55s | about 33% faster |
| FaceID PlusV2, 8 steps, DMD2, 832x1152 | 18.55s | 14.02s | about 24% faster |
| ControlNet, 20 steps, 768x1280 | 82.31s | 60.05s | about 27% faster |
| ControlNet, 8 steps, speed patch, 768x1280 | 18.9-19.1s | 14.4-14.5s | about 24% faster |
| Various inpaint/detail/upscale workflows | varies by crop and tile size | usually faster | still being documented for 0.1b examples |

Preview mode, screen sharing, cold vs warm runs, model preload, memory pressure and disk location can all change these numbers.

### Cold-Start Snapshot

Cold runs are less dramatic than the warm-run speedups, but SDMLX still keeps a consistent advantage when a model or workflow is started fresh.

| Workflow, 832x1152 | PyTorch-MPS cold | SDMLX cold | Notes |
| --- | ---: | ---: | --- |
| FaceID PlusV2, 20 steps | 69.92s | 51.67s | about 26% faster |
| FaceID PlusV2, 8 steps, DMD2 | 32.06s | 20.39s | about 36% faster |
| SDXL txt2img, 20 steps | 58.36s | 48.82s | about 16% faster |
| SDXL txt2img, 8 steps, DMD2 | 19.11s | 16.40s | about 14% faster |
| Model switch: model 1 -> model 2 -> model 1 | 61.12s -> 63.50s -> 58.51s | 50.02s -> 50.52s -> 49.29s | about 16-20% faster |

### Conversion Snapshot

Initial checkpoint conversion is a one-time step. On the test system, typical cold conversion times are about 16-18s for FP16 SDXL checkpoints around 7GB and about 25-26s for FP32 checkpoints around 13GB. Converted packages are stored as FP16 `.sdmlx` packages, so FP32 checkpoints usually shrink to roughly half their original size.

## Installation

Clone this repository into ComfyUI's `custom_nodes` folder:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/elef4nt-gh/SDMLX.git SDMLX
```

Install dependencies into the Python environment used by ComfyUI:

```bash
pip install -r SDMLX/requirements.txt
```

Restart ComfyUI after installing or updating.

### ComfyUI Desktop on macOS

ComfyUI Desktop keeps its Python environment in `~/Documents/ComfyUI/.venv`. If SDMLX was installed from a ZIP or manual clone, install dependencies into that environment:

```bash
~/Documents/ComfyUI/.venv/bin/pip3 install -r ~/Documents/ComfyUI/custom_nodes/SDMLX/requirements.txt
```

If the folder is named `SDMLX-main`, adjust the path accordingly.

## Basic Usage

For a first workflow:

1. Add `SDMLX Loader Universal`.
2. Select a normal SDXL checkpoint from `models/checkpoints`.
3. Run once. SDMLX converts the checkpoint and writes a `.sdmlx` package into `models/SDMLX`.
4. Use `SDMLX CLIP Text Encode` for positive and negative prompts.
5. Use `Empty Latent Image`, `SDMLX KSampler`, and connect `mlx_vae` to the sampler.
6. The sampler outputs both `image` and `latent`, so a separate VAE Decode node is not needed for standard workflows.

`SDMLX Loader Universal` is the easiest starting point: it checks whether a checkpoint has already been converted and automatically prefers the cached `.sdmlx` package when available. `SDMLX Loader` is the cache-only loader; it lists only existing `.sdmlx` packages.

For a short explanation of SDMLX-specific controls such as Memory Assist, speed patches, scheduler curves, FaceID names, ControlNet curves and inpaint detailer settings, see [SDMLX Mini Guide](docs/SDMLX_GUIDE.md).

Example workflows are included in [resources/workflows](resources/workflows). Preview PNGs for quick inspection are in [resources/workflow_png](resources/workflow_png), and DMD2/Lightning variants are grouped under [resources/workflows/speed_workflows](resources/workflows/speed_workflows).

## Model Cache

Converted checkpoints are stored as macOS-style `.sdmlx` packages under:

```text
ComfyUI/models/SDMLX
```

Speed Patches are stored under:

```text
ComfyUI/models/SDMLX/SpeedPatches
```

The package format is intentionally simple: one visible model package per converted checkpoint, instead of loose UNet/CLIP/VAE files scattered through the custom node folder.

## Speed Patches

SDMLX can use MLX-mapped Speed Patches derived from common SDXL speed LoRAs. Supported patch families currently include:

- DMD2
- SDXL Lightning
- LCM LoRA SDXL
- Hyper-SD SDXL

The sampler can list downloaded patches directly. The Speed Patch Converter node remains available for users who want to convert supported local speed LoRAs themselves.

Important: Speed Patches are model-derived assets. Their licenses are not replaced by the SDMLX code license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Spectrum Acceleration

The sampler and Hires Fix node have a simple `spectrum_acceleration` switch with two user-facing modes. Spectrum is a training-free forecasting technique that skips selected UNet evaluations by predicting internal UNet features and running the final output projection normally.

- `fast`: Spectrum scheduling with three final real steps. This is the speed-first mode.
- `standard`: balanced Spectrum mode. It uses a guarded short-run policy below 35 steps and a more conservative schedule with five warmup steps and three final real steps from 35 steps upward.

The `fast` mode is based on the public Spectrum scheduling approach and ComfyUI reference implementations. `standard` keeps that foundation but adds SDMLX-specific policy choices for short SDXL runs, where overly aggressive forecasting can create more rejects than saved seconds. Hires Fix defaults to `fast`, because low-denoise upscale tests showed large speedups with no visible image difference.

Use `SDMLX Spectrum Advanced` only when you want to override the sampler or Hires Fix switch. It plugs into `spectrum_acceleration_advanced` and exposes the raw Spectrum parameters without presets. If you connect it, the simple switch is ignored.

Spectrum is not the same thing as a speed patch. Speed patches change model behavior; Spectrum changes which sampling steps run the full UNet. Keeping them separate makes it easier to understand which part of the workflow changed the image.

Current observations:

- Normal Euler SDXL runs from roughly 20 steps upward can become much faster while keeping high image quality.
- The simple Spectrum modes automatically stay off when a Speed Patch is selected. Power users can still combine Spectrum with Speed Patches through the advanced node.
- Dense prompts with hands, tiny objects, text or many overlapping details still need visual checking.

Credits: SDMLX's Spectrum support is an MLX adaptation inspired by the official [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) project and the paper [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623). The ComfyUI implementations [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum) and [ruwwww/ComfyUI-Spectrum-sdxl](https://github.com/ruwwww/ComfyUI-Spectrum-sdxl) were important practical references for ComfyUI-facing behavior and SDXL-specific use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license notes.

## Companion Models

SDMLX loaders can download common companion models into ComfyUI's standard model folders using the usual filenames:

- Xinsir ControlNet Union ProMax into `models/controlnet`
- IP-Adapter SDXL ViT-H variants into `models/ipadapter`
- FaceID / FaceID PlusV2 SDXL into `models/ipadapter`
- FaceID LoRA files into `models/loras`
- CLIP-ViT-H-14 into `models/clip_vision`
- InsightFace models into `models/insightface`

If you already keep models on another drive through ComfyUI's `extra_model_paths.yaml`, SDMLX uses ComfyUI's folder lookup where possible.

## Main Node Groups

- Loaders: `SDMLX Loader Universal`, `SDMLX Loader`
- Prompting: `SDMLX CLIP Text Encode`
- Sampling: `SDMLX KSampler`
- LoRA: `SDMLX LoRA Loader`, `SDMLX Multi LoRA Loader`, `SDMLX Scheduler`
- IP-Adapter: `SDMLX IP-Adapter Loader`, `SDMLX CLIP Vision Loader`, `SDMLX CLIP Vision Encode`, `SDMLX Apply IP-Adapter`
- FaceID: `SDMLX InsightFace Loader`, `SDMLX Apply IP-Adapter FaceID`, `SDMLX FaceID AIO`
- ControlNet: `SDMLX ControlNet Union ProMax Loader`, `SDMLX Apply ControlNet`
- Inpaint: `SDMLX Inpaint Conditioning`, `SDMLX Inpaint Detailer`, `SDMLX Differential Diffusion`
- Upscale: `SDMLX Hires Fix`, `SDMLX Tiled Upscale`

## Known Limits

SDMLX is focused on SDXL base-style checkpoints. The converter is expected to reject or fail on architectures that are not SDXL-compatible.

Currently out of scope or not supported:

- SD 1.x and SD 2.x checkpoints
- SD3 / SD3.5 style checkpoints
- Flux, Qwen-Image, Z-Image and other non-SDXL transformer architectures
- SDXL refiner checkpoints
- native 9-channel SDXL inpaint checkpoints
- arbitrary ControlNet models other than the supported Union ProMax path
- arbitrary IP-Adapter families outside the supported SDXL/FaceID variants
- text-encoder LoRA scheduling; scheduled LoRA currently targets UNet modules

Some SDXL finetunes and merges can still behave differently from PyTorch-MPS, especially if they depend on unusual sampler/scheduler assumptions.

## Known Issues

- This is alpha software. Node names, defaults and package internals may still change.
- First conversion and first warm run can be slower than later runs.
- Scheduled LoRA disables some fast-path module fusions for that run, so it can cost performance.
- Spectrum Acceleration is experimental. Complex prompts with many small objects, hands or text may show small artifacts; set it to `off` when quality matters more than speed.
- FaceID Portrait variants can be sensitive to weights, CFG and source images; FaceID PlusV2 is usually the more stable starting point.
- Very large tiled upscale jobs can be memory-heavy and slow even on high-end Macs.
- ComfyUI Desktop and Easy Install use different Python environments; dependencies must be installed into the environment that actually runs ComfyUI.
- Apple Silicon coverage is still being mapped. M1 Max is confirmed; other chips may expose MLX/Metal fast-path differences. Please use the diagnostics mode above if prompt adherence fails.

## Roadmap

Next priorities:

- SEGS-style automation for mask/detail workflows
- SAM and Ultralytics integration for detection, segmentation and automated detail passes
- Example workflows for txt2img, FaceID, ControlNet, inpaint, hires fix and tiled upscale
- More testing across smaller and newer Apple Silicon Macs
- More sampler/scheduler validation against common SDXL finetunes

Important follow-ups:

- Broader ControlNet support
- Tile-ControlNet based upscale refinements
- Better diagnostics when unsupported checkpoints are selected
- More polish for release packaging and documentation

Later experiments:

- FreeU and PAG experiments
- Advanced prompt / dual CLIP controls
- More permanent conversion utilities for LoRAs and companion models
- Optional deeper node/module split once the public API settles

## For Technical Readers

ComfyUI on macOS normally runs SDXL through PyTorch with the MPS backend. That path is powerful, but it still carries the shape of a PyTorch/CUDA-oriented ecosystem: model weights, scheduler state, VAE paths, preview paths and custom node integrations often move through several layers that were not designed around Apple Silicon as the primary target.

SDMLX takes a different route for SDXL. Checkpoints are converted into `.sdmlx` packages and loaded into an MLX runtime. The UNet, CLIP text encoders, VAE decode path, IP-Adapter attention integration, ControlNet Union path, Speed Patches and memory behavior are handled as directly as possible in MLX. The fast path uses float16 compute for the diffusion model, keeps VAE decode conservative by default, applies SDXL speed-LoRA derivatives as mapped MLX Speed Patches, and uses a Mac-specific Memory Assist layer to balance cache reuse against system memory pressure.

That does not make SDMLX a universal replacement for every ComfyUI model family. It is deliberately narrower: SDXL on Apple Silicon, with a strong bias toward practical Mac workflows.

## Project Note

SDMLX is a spare-time project by a non-programmer, built with help from Codex. Bug reports, pull requests and updates are welcome, but responses will depend on available free time.

## License

SDMLX source code is licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

This code license does not relicense checkpoints, LoRAs, ControlNet models, IP-Adapter models, CLIP Vision models, InsightFace models, or SDMLX Speed Patches. Those assets remain subject to their upstream licenses and model-card terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current practical summary.
