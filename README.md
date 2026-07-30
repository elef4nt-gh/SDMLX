# 🍏 SDMLX - ComfyUI Node Suite for Faster Image Generation on macOS

SDMLX is an alpha-stage ComfyUI custom node suite for running SDXL workflows on Apple Silicon with Apple's MLX framework. It also includes selected FLUX.1, FLUX.2 Klein, Qwen Image and Qwen Image Edit MLX paths.

The goal is straightforward: make image generation on the Mac feel less like a compromise. For SDXL, SDMLX ports the core workflow from the usual PyTorch-MPS path to an MLX-native runtime, with easy checkpoint conversion, `.sdmlx` package caching, Speed Patches, macOS-aware memory handling, and workflow nodes that try to stay close to familiar ComfyUI patterns. FLUX and Qwen support follow the same Mac-first idea, but use separate model-family nodes where the model architecture needs different conditioning, latent, cache, or edit behavior.

In current local tests, SDMLX is typically about 25-30% faster than comparable PyTorch-MPS SDXL workflows on the same Mac, depending on the checkpoint, sampler, resolution, speed patch, ControlNet/IP-Adapter usage, and preview settings. It is still alpha software; results and APIs can change.

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
- FLUX.1 nodes for diffusion model loading, dual CLIP/T5 loading, sampling, VAE encode/decode and Kontext reference workflows
- Optional SeaCache acceleration for FLUX.1 runs
- FLUX.1 LUA latent upscale adapter for FLUX/SD3-style 16-channel latents
- FLUX.2 Klein 4B/9B package loading, txt2img, image edit, multi-image edit and KV-cache support where the model supports it
- FLUX.2 Klein Enhanced Edit sampler for multi-reference identity/reference steering
- Qwen Image 2512 txt2img and Qwen Image Edit 2511 paths with native MLX package loading
- Qwen Image Edit Conditioning Plus with phr00t-style multi-image prompt
  framing and reference-latent scaling
- Qwen Lightning 4-step LoRAs / patches and merged Lightning checkpoints
- Image loading/scaling helpers that keep image and mask geometry aligned
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

### FLUX / Qwen Snapshot

The transformer-model paths are newer and more workflow-sensitive than the SDXL path. The rows below are local alpha anchors from the M1 Max test system unless noted otherwise. `yet to be evaluated` means a matching PyTorch-MPS comparison has not been captured with a clean enough contract for a README benchmark.

| Workflow | PyTorch-MPS | SDMLX | Notes |
| --- | ---: | ---: | --- |
| FLUX.2 Klein 4B txt2img, 4 steps | cold 63.41s, warm 47.94s | cold 51.28s, warm 25.51s | same class of workflow, strong warm-run gain |
| FLUX.2 Klein 4B multi-image edit, 4 steps | cold 182.47s, warm 165.86s | cold 131.52s, warm 109.89s | same class of workflow |
| FLUX.2 Klein 9B one-image edit BF16, 4 steps | cold 299.69s, warm 203.14s | switch 179.33s, warm 135.00s | packed-BF16 MLX path |
| FLUX.2 Klein 9B KV/scaled-FP8 one-image edit, 4 steps | not directly comparable | switch 137.39s, warm 101.59s | fastest practical 9B edit route when a KV/scaled-FP8 package is used |
| Qwen Image 2512 txt2img, 4 steps | yet to be evaluated | warm 67.52s | Wuli Turbo 4-step path, 832x1248 workflow anchor |
| Qwen Image Edit 2511, 4 steps | yet to be evaluated | warm 137.00s | Lightning 4-step path, 832x1248 workflow anchor |

A community report on an Apple M5 Max compared FLUX.1 Dev txt2img against the default ComfyUI FLUX.1 Dev workflow with CFG 1, 20 steps and the same "gold watch" prompt: at `1024x1024`, ComfyUI took `42.87s` and SDMLX took `39.71s`; at `2048x2048`, ComfyUI took `308s` and SDMLX took `229s`. See the [GitHub issue comment](https://github.com/elef4nt-gh/SDMLX/issues/1#issuecomment-4672060291).

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

ComfyUI Desktop paths vary by app version and installation. Find the
`ComfyUI Base Folder Path` in the startup log and use the `.venv` inside that
folder. If SDMLX was installed from a ZIP or manual clone, the general form is:

```bash
COMFYUI_ROOT="/path/shown/as/ComfyUI Base Folder Path"
"$COMFYUI_ROOT/.venv/bin/python3" -m pip install \
  -r "$COMFYUI_ROOT/custom_nodes/SDMLX/requirements.txt"
```

If the folder is named `SDMLX-main`, adjust the path accordingly.

### Hugging Face access for gated models

`huggingface_hub` is installed with SDMLX. Some official model repositories,
including selected Black Forest Labs FLUX models, additionally require the
repository terms to be accepted and a Hugging Face token to be available to
the same Python environment that runs ComfyUI.

After accepting the model terms on Hugging Face, set `COMFY_PYTHON` to
ComfyUI's Python executable and run:

```bash
COMFY_PYTHON="/path/to/comfy/python3"
"$COMFY_PYTHON" -c 'from getpass import getpass; from huggingface_hub import login; login(token=getpass("HF token: "), add_to_git_credential=False)'
```

The prompt keeps the token out of shell history and stores it only in the
normal Hugging Face user cache. Git credentials, Xcode, and Apple developer
tools are not required. Prebuilt `.sdmlx` packages that already contain their
tokenizer and scheduler assets do not need this download step.

## Basic Usage

For a first workflow:

1. Add `SDMLX Loader Universal`.
2. Select a normal SDXL checkpoint from `models/checkpoints`.
3. Run once. SDMLX converts the checkpoint and writes a `.sdmlx` package into `models/SDMLX`.
4. Use `SDMLX CLIP Text Encode` for positive and negative prompts.
5. Use `Empty Latent Image`, `SDMLX KSampler`, and connect `mlx_vae` to the sampler.
6. The sampler outputs both `image` and `latent`, so a separate VAE Decode node is not needed for standard workflows.

`SDMLX Loader Universal` is the easiest starting point: it checks whether a checkpoint has already been converted and automatically prefers the cached `.sdmlx` package when available. `SDMLX Loader` is the cache-only loader; it lists only existing `.sdmlx` packages.

For FLUX.1 workflows, do not use `SDMLX Loader Universal`. Use `SDMLX Load Diffusion Model`, `SDMLX Dual CLIP Loader` with `type=flux`, `SDMLX KSampler (FLUX.1)`, and `SDMLX VAE Loader` with `ae.safetensors` instead.

For FLUX.2 Klein workflows, use a FLUX.2 Klein `.sdmlx` package or supported FLUX.2 Klein checkpoint through the SDMLX loader path, load the matching Qwen3 text encoder with `SDMLX CLIP Loader` and `type=flux2`, load a FLUX.2 VAE with `SDMLX VAE Loader`, then use `SDMLX Empty Latent Image FLUX.2`, `SDMLX KSampler (FLUX.2-klein)`, and `SDMLX VAE Decode`. Use `SDMLX Reference Latent` for standard edit/reference workflows, or the Enhanced Edit sampler when you want the separate multi-reference steering path.

For Qwen Image 2512 txt2img workflows, use a Qwen Image `.sdmlx` package, load the Qwen Image/Edit text encoder with `SDMLX CLIP Loader` and `type=qwen_image`, then use `SDMLX CLIP Text Encode` and the existing `SDMLX KSampler`.

For Qwen Image Edit 2511 workflows, use an `.sdmlx` Qwen package in `SDMLX Loader`, load the Qwen Image/Edit text encoder with `SDMLX CLIP Loader` and `type=qwen_image`, connect the `mlx_clip` output to `SDMLX Qwen Image Edit Conditioning`, then run the existing `SDMLX KSampler`. The sampler detects Qwen packages and can apply supported Qwen Lightning 4-step acceleration patches.

`SDMLX Qwen Image Edit Conditioning Plus` is the multi-image alternative. It follows the phr00t-style Plus conditioning recipe: up to four image slots, `Picture N` vision framing, a small Qwen-VL image path, and separately scaled VAE reference latents with `target_size` defaulting to `896`. This avoids patching ComfyUI files while keeping the recognizable behavior of phr00t's fixed Qwen text-encode node. See the [phr00t Qwen Rapid AIO reference](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main/fixed-textencode-node) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For a short explanation of SDMLX-specific controls such as Memory Assist, speed patches, scheduler curves, FaceID names, ControlNet curves and inpaint detailer settings, see [SDMLX Mini Guide](docs/SDMLX_GUIDE.md).

Example workflows are included in [resources/workflows](resources/workflows). Preview PNGs for quick inspection are in [resources/workflow_png](resources/workflow_png), DMD2/Lightning variants are grouped under [resources/workflows/speed_workflows](resources/workflows/speed_workflows), basic FLUX workflows are grouped under [resources/workflows/flux](resources/workflows/flux), and the current FLUX.1 / FLUX.2 Klein / Qwen example set is grouped under [resources/workflows/qwen_flux](resources/workflows/qwen_flux).

## Model Cache

Converted checkpoints are stored as macOS-style `.sdmlx` packages under:

```text
ComfyUI/models/SDMLX
```

SDXL Speed Patches plus FLUX and Qwen acceleration patches are stored under:

```text
ComfyUI/models/SDMLX/AccelerationPatches
```

Generated FLUX acceleration caches are stored under:

```text
ComfyUI/models/SDMLX/cache/acceleration-patches
```

The package format is intentionally simple: one visible model package or patch package, instead of loose files scattered through the custom node folder.

## Speed Patches

SDMLX can use MLX-mapped Speed Patches derived from common SDXL speed LoRAs and selected transformer speed LoRAs. Supported SDXL patch families currently include:

- DMD2
- SDXL Lightning
- LCM LoRA SDXL
- Hyper-SD SDXL

The sampler can list downloaded patches directly. The Speed Patch Converter node remains available for users who want to convert supported local speed LoRAs themselves.

Qwen Lightning LoRAs and merged Lightning checkpoints are supported in the Qwen paths. When a clean merged Lightning checkpoint is available, it is usually the simplest fast route; separate Lightning LoRAs also work but can cost a little extra runtime.

Important: Speed Patches are model-derived assets. Their licenses are not replaced by the SDMLX code license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Spectrum Acceleration

The sampler and Hires Fix node have a simple `spectrum_acceleration` switch with two user-facing modes. Spectrum is a training-free forecasting technique that skips selected UNet evaluations by predicting internal UNet features and running the final output projection normally.

- `fast`: Spectrum scheduling with three final real steps. This is the speed-first mode.
- `standard`: balanced Spectrum mode. Below 35 steps it uses a cleaner Spectrum-Proper-style short-run path; from 35 steps upward it uses the more conservative schedule with five warmup steps and three final real steps.

The `fast` mode is based on the public Spectrum scheduling approach and ComfyUI reference implementations. `standard` keeps that foundation but uses separate SDMLX policy choices for fragile short SDXL runs and higher-step runs. Hires Fix defaults to `fast`, because low-denoise upscale tests showed large speedups with no visible image difference.

Use `SDMLX Spectrum Advanced` only when you want to override the sampler or Hires Fix switch. It plugs into `spectrum_acceleration_advanced` and exposes the raw Spectrum parameters without presets. If you connect it, the simple switch is ignored.

Spectrum is not the same thing as a speed patch. Speed patches change model behavior; Spectrum changes which sampling steps run the full UNet. Keeping them separate makes it easier to understand which part of the workflow changed the image.

Current observations:

- Normal Euler SDXL runs from roughly 20 steps upward can become much faster while keeping high image quality.
- The simple Spectrum modes automatically stay off when a Speed Patch is selected. Power users can still combine Spectrum with Speed Patches through the advanced node.
- Dense prompts with hands, tiny objects, text or many overlapping details still need visual checking.

Credits: SDMLX's Spectrum support is an MLX adaptation inspired by the official [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) project and the paper [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623). The ComfyUI implementations [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum) and [ruwwww/ComfyUI-Spectrum-sdxl](https://github.com/ruwwww/ComfyUI-Spectrum-sdxl) were important practical references for ComfyUI-facing behavior and SDXL-specific use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license notes.

## FLUX.1 Nodes

SDMLX also includes a FLUX.1 MLX path. It is separate from the SDXL checkpoint loader because FLUX workflows normally keep the diffusion model, text encoders and VAE as separate components.

Typical FLUX txt2img wiring:

1. `SDMLX Load Diffusion Model` for the FLUX diffusion model.
2. `SDMLX Dual CLIP Loader` with `type=flux` for `clip_l` plus a T5XXL encoder such as `t5xxl_fp8_e4m3fn.safetensors`, `t5xxl_fp16.safetensors`, or compatible T5XXL variants.
3. A normal FLUX conditioning node.
4. `EmptySD3LatentImage`.
5. `SDMLX KSampler (FLUX.1)`.
6. `SDMLX VAE Loader` with `ae.safetensors`.
7. `SDMLX VAE Decode`.

For FLUX Kontext, scale reference images with `SDMLX FLUX Kontext Scale`, load `ae.safetensors` with `SDMLX VAE Loader`, encode reference images with `SDMLX VAE Encode`, pass them through ComfyUI's `ReferenceLatent` node, and feed the resulting conditioning into the sampler. Use `SDMLX Empty Latent Image FLUX.1` when you want the target latent to use the same FLUX-friendly dimension family. The sampler uses the validated `offset` reference method internally, so a separate reference-method node is not needed for the normal SDMLX workflow. Very large source images should be scaled before VAE encode. Raw multi-megapixel reference images can create huge MLX allocations without improving the edit.

FLUX speed in SDMLX currently comes from three different layers:

- **MLX runtime work:** SDMLX uses its own FLUX transformer execution path, prepared weights, fp16 compute, memory guardrails and MLX-side VAE encode/decode. These are SDMLX implementation choices for running FLUX on Apple Silicon without relying on the normal PyTorch-MPS sampler path.
- **Acceleration patches:** the sampler can apply supported FLUX-dev speed patches as first-class sampling options. For canonical base models, SDMLX can build and reuse a local baked cache. Other compatible models may fall back to a runtime low-rank patch path.
- **SeaCache acceleration:** the sampler has an optional `seacache_acceleration` switch. SeaCache is a separate training-free cache method; SDMLX integrates and tunes FLUX-oriented SeaCache paths for MLX, but the core idea is not an SDMLX invention.

SeaCache is disabled by default. Turn it on when you want the extra speed/quality trade-off and visually check difficult prompts. The sampler chooses the SeaCache policy from the detected FLUX family:

- FLUX-schnell uses a short 4-step SeaCache path with token pooling on the late steps.
- FLUX-dev and FLUX Kontext use resolution-aware rising-threshold SeaCache starting points for longer runs. Squarer, high-token outputs use a gentler profile; other outputs use a more open profile.
- FLUX-dev with a 4-step acceleration patch uses a narrow one-step SeaCache path. This is useful for fast prompt tests, but still prompt-dependent.

FLUX acceleration patches and SeaCache are separate mechanisms: an acceleration patch changes the model trajectory, while SeaCache reuses part of the sampler computation. For Kontext image-conditioning workflows, acceleration patches are disabled automatically; SeaCache can still be used.

For FLUX-dev txt2img, SeaCache should be treated as a practical starting point, not a universal quality-preserving shortcut. Detail-heavy prompts and square `1024x1024` style outputs may prefer a few extra steps with SeaCache rather than the same step count with SeaCache. In local tests, `25` SeaCache steps can land near the runtime of `20` all-real steps while preserving more fine detail. If a prompt needs a different trade-off, connect `SDMLX FLUX SeaCache Advanced` and tune `threshold_start`, `threshold_end`, `start_at` and `final_guard`.

A recent local FLUX.1 Kontext stress test with a scaled-FP8 checkpoint, `20` steps and a same-image PyTorch-MPS reference landed at `325.65s` all-real in SDMLX and `193.37s` with SeaCache, versus `00:17:54` for the PyTorch-MPS reference on the same M1 Max system. This is a stress-test anchor, not a general benchmark table; FLUX Kontext timing depends heavily on reference-image handling, target size, checkpoint format and memory state.

For the practical control matrix, see [SDMLX Mini Guide](docs/SDMLX_GUIDE.md).

Credits: the FLUX SeaCache path is an MLX adaptation inspired by [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) and the paper [SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](https://arxiv.org/abs/2602.18993). SDMLX's implementation and defaults are practical Apple Silicon tuning choices layered on top of that idea.

SDMLX's FLUX sampler uses an SDMLX MLX sampling loop with Euler-style flow updates and sigma/runtime scheduling adapted from [mflux](https://github.com/filipstrand/mflux). The surrounding FLUX runtime, cache paths, acceleration-patch handling, Kontext path and VAE integration are SDMLX implementation work.

## FLUX.2 Klein Nodes

FLUX.2 Klein uses its own SDMLX path and stays separate from FLUX.1. A typical txt2img or native edit workflow uses the package/checkpoint loader, `SDMLX CLIP Loader` with `type=flux2` for the Qwen3 text encoder, `SDMLX VAE Loader`, `SDMLX Empty Latent Image FLUX.2`, `SDMLX KSampler (FLUX.2-klein)`, and `SDMLX VAE Decode`.

The current validated FLUX.2 Klein path covers the official BFL 4B/9B dense-BF16 packages and the official 9B KV/scaled-FP8 package. Community FLUX.2 Klein finetunes are detected through a quantization contract so SDMLX can distinguish dense BF16, scaled FP8, Comfy-quant/MXFP8 and raw unscaled FP8 packages instead of treating every FP8 file as the same format.

For multi-image identity/reference steering, use `SDMLX KSampler (FLUX.2-klein Enhanced Edit)`. It accepts the normal SDMLX text conditioning plus direct image and mask inputs, or existing `SDMLX Reference Latent` conditioning. `SDMLX FLUX.2 Klein Enhancer Advanced` is optional and only supplies deeper steering settings; the standard FLUX.2 sampler remains the validated default path for ordinary generation.

When Enhanced Edit uses reference-token steering, it may disable the KV-cache by design and prints that in the terminal. Native-only, text-only, and color-anchor-only modes can still use the normal cache path where applicable.

The Enhanced Edit method in SDMLX is an independent clean-room implementation inspired by the workflow behavior of the external Flux2Klein Enhancer reference project. SDMLX does not bundle, vendor, or copy source code or assets from that project. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and license context.

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

- Loaders: `SDMLX Loader Universal`, `SDMLX Loader`, `SDMLX CLIP Loader`, `SDMLX Dual CLIP Loader`, `SDMLX VAE Loader`, `SDMLX Load Diffusion Model`
- Image: `SDMLX LoadImage Scale`, `SDMLX Load Image Advanced`, `SDMLX FLUX Kontext Scale`, `SDMLX Scale To Megapixels`
- Conditioning: `SDMLX CLIP Text Encode`, `SDMLX Reference Latent`, `SDMLX Qwen Image Edit Conditioning`, `SDMLX Qwen Image Edit Conditioning Plus`, `SDMLX Qwen Token Budget`
- Latent: `SDMLX Empty Latent Image FLUX.1`, `SDMLX Empty Latent Image FLUX.2`
- Sampling: `SDMLX KSampler`, `SDMLX KSampler Advanced`, `SDMLX KSampler (FLUX.1)`, `SDMLX KSampler (FLUX.2-klein)`, `SDMLX KSampler (FLUX.2-klein Enhanced Edit)`
- VAE: `SDMLX VAE Encode`, `SDMLX VAE Decode`
- LoRA: `SDMLX LoRA Loader`, `SDMLX Multi LoRA Loader`
- Advanced: `SDMLX Spectrum Advanced`, `SDMLX FLUX SeaCache Advanced`, `SDMLX Scheduler`, `SDMLX Qwen ModelSampling AuraFlow`, `SDMLX FLUX.2 Klein Enhancer Advanced`
- Utilities: `SDMLX Number Picker`, `SDMLX Speed Patch Converter`
- IP-Adapter: `SDMLX IP-Adapter Loader`, `SDMLX CLIP Vision Loader`, `SDMLX CLIP Vision Encode`, `SDMLX Apply IP-Adapter`
- FaceID: `SDMLX InsightFace Loader`, `SDMLX Apply IP-Adapter FaceID`, `SDMLX FaceID AIO`
- ControlNet: `SDMLX ControlNet Union ProMax Loader`, `SDMLX Apply ControlNet`
- Inpaint: `SDMLX Inpaint Conditioning`, `SDMLX Inpaint Detailer`, `SDMLX Differential Diffusion`
- Upscale: `SDMLX Hires Fix`, `SDMLX Tiled Upscale`, `SDMLX FLUX LUA Adapter`

## Known Limits

The SDXL checkpoint converter is focused on SDXL base-style checkpoints. It is expected to reject or fail on architectures that are not SDXL-compatible. FLUX.1 uses a separate diffusion-model loader and is not converted into `.sdmlx` checkpoint packages. FLUX.2 Klein and Qwen Image/Edit use their own package/runtime contracts.

Currently out of scope or not supported:

- SD 1.x and SD 2.x checkpoints
- SD3 / SD3.5 style checkpoints
- Z-Image and other transformer architectures outside the listed FLUX.1, FLUX.2 Klein and Qwen Image/Edit paths
- Qwen variants outside the currently supported Qwen Image 2512 and Qwen Image Edit 2511 contracts
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
- FLUX.1 support is newer than the SDXL path. SeaCache acceleration, Kontext, FP8 checkpoints, LoRAs and acceleration patches should be treated as alpha features and checked visually.
- FLUX.2 Klein and Qwen support are alpha paths. The validated contracts are narrower than the full ComfyUI ecosystem, and unsupported partial-denoise or model-family mixes should fail clearly rather than silently falling back.
- FaceID Portrait variants can be sensitive to weights, CFG and source images; FaceID PlusV2 is usually the more stable starting point.
- Very large tiled upscale jobs can be memory-heavy and slow even on high-end Macs.
- ComfyUI Desktop and Easy Install use different Python environments; dependencies must be installed into the environment that actually runs ComfyUI.

## Roadmap

Next priorities:

- SEGS-style automation for mask/detail workflows
- SAM and Ultralytics integration for detection, segmentation and automated detail passes
- ~~Example workflows for txt2img, FaceID, ControlNet, inpaint, hires fix and tiled upscale~~
- More testing across smaller and newer Apple Silicon Macs
- More sampler/scheduler validation against common SDXL finetunes

Important follow-ups:

- Broader ControlNet support
- Tile-ControlNet based upscale refinements
- ~~Better diagnostics when unsupported checkpoints are selected~~
- More polish for release packaging and documentation

The crossed-out roadmap items are not "done forever"; they are completed enough for the current alpha release and will keep receiving smaller fixes.

Later experiments:

- FreeU and PAG experiments
- Advanced prompt / dual CLIP controls
- More permanent conversion utilities for LoRAs and companion models
- Optional deeper node/module split once the public API settles

## For Technical Readers

ComfyUI on macOS normally runs SDXL through PyTorch with the MPS backend. That path is powerful, but it still carries the shape of a PyTorch/CUDA-oriented ecosystem: model weights, scheduler state, VAE paths, preview paths and custom node integrations often move through several layers that were not designed around Apple Silicon as the primary target.

SDMLX takes a different route for SDXL. Checkpoints are converted into `.sdmlx` packages and loaded into an MLX runtime. The UNet, CLIP text encoders, VAE decode path, IP-Adapter attention integration, ControlNet Union path, Speed Patches and memory behavior are handled as directly as possible in MLX. The fast path uses float16 compute for the diffusion model, keeps VAE decode conservative by default, applies SDXL speed-LoRA derivatives as mapped MLX Speed Patches, and uses a Mac-specific Memory Assist layer to balance cache reuse against system memory pressure.

The FLUX.1 path is separate: it loads diffusion-model files directly, uses dual CLIP/T5 and VAE nodes, and runs a native MLX transformer path. SeaCache is an optional cache/reuse layer on top of that FLUX runtime, not the same thing as the underlying MLX port.

FLUX.2 Klein and Qwen Image/Edit are separate again. They use `.sdmlx` package contracts, explicit text-encoder type selection, family-specific conditioning objects, and model-family samplers or sampler branches where the architecture requires it. That separation is intentional: SDMLX should not silently treat SDXL, FLUX.1, FLUX.2 and Qwen as interchangeable just because they all end in an image.

That does not make SDMLX a universal replacement for every ComfyUI model family. It is deliberately narrower: practical Apple Silicon workflows for SDXL and selected FLUX.1, FLUX.2 Klein and Qwen Image/Edit models.

## Project Note

SDMLX is a spare-time project by a non-programmer, built with help from Codex. Bug reports, pull requests and updates are welcome, but responses will depend on available free time.

## Third-Party References

Several SDMLX features are MLX adaptations, clean-room implementations, or behavior-compatible workflow ports inspired by public projects and papers. Important references include `mflux`, Spectrum, SeaCache, the LUA latent upscale adapter, phr00t-style Qwen Image Edit Plus conditioning, and the Flux2Klein Enhancer workflow pattern. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, license context and model-asset notes.

## License

SDMLX source code is licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

This code license does not relicense checkpoints, LoRAs, ControlNet models, IP-Adapter models, CLIP Vision models, InsightFace models, or SDMLX Speed Patches. Those assets remain subject to their upstream licenses and model-card terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current practical summary.
