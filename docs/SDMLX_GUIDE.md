# SDMLX Mini Guide

SDMLX tries to keep ComfyUI familiar while making SDXL on Apple Silicon feel more direct. A few controls are named differently from older diffusion nodes on purpose: the goal is to describe what the control does in the image or workflow, not the implementation detail behind it.

Example workflows are shipped with the node suite in `resources/workflows`. The `speed_workflows` subfolder contains DMD2 and Lightning variants, the `flux` subfolder contains basic FLUX workflows, and `resources/workflow_png` contains PNG previews without embedded workflow JSON.

## Loaders And Memory Assist

`SDMLX Loader Universal` is the easiest starting point. It loads normal SDXL checkpoints from `models/checkpoints`, converts them once, and then automatically prefers the cached `.sdmlx` package.

`SDMLX Loader` only lists already converted `.sdmlx` packages from `models/SDMLX`.

`memory_assist` controls how aggressively SDMLX keeps models warm:

- `auto`: recommended default. Balanced cache behavior for normal Mac use.
- `max_performance`: keeps more model data warm when memory allows it.
- `low_memory`: frees more aggressively for smaller Macs or heavy workflows.
- `off`: leaves Comfy/MLX closer to their default behavior.

## Speed Patches

Speed patches are MLX-converted SDXL speed LoRAs. They live in the sampler as first-class speed options because they are usually used like part of the sampling setup, not like a creative style LoRA.

They can reduce step counts dramatically, for example with DMD2, Lightning, LCM or Hyper-SD style workflows. Use the settings recommended by the patch or the model author as the starting point.

## Spectrum Acceleration

The sampler and Hires Fix node have a simple `spectrum_acceleration` switch:

- `off`: normal SDMLX sampling.
- `fast`: Spectrum scheduling plus three final real steps. Best when speed matters most.
- `standard`: balanced Spectrum mode. Below 35 steps it uses a cleaner Spectrum-Proper-style short-run path; from 35 steps upward it uses five warmup steps and three final real steps.

Spectrum is different from a speed patch: a speed patch changes the model behavior, while Spectrum predicts selected internal UNet features and skips some real UNet evaluations.

The `fast` mode follows the public Spectrum scheduling idea used by ComfyUI Spectrum implementations: run real steps to build a cache, forecast selected internal features, and keep a small real-step safety zone at the end.

The `standard` mode keeps that foundation but splits the policy by step count. Shorter runs use the cleaner short-run path found in the SDXL Spectrum audit; higher-step runs keep the conservative schedule that tested better from roughly 35 steps upward.

Hires Fix uses the same switch and defaults to `fast`. In low-denoise upscale tests this gave the strongest speed win while staying visually indistinguishable from the full run.

`SDMLX Spectrum Advanced` is the advanced override node. Plug it into `spectrum_acceleration_advanced` when you want to set the Spectrum parameters yourself. If this input is connected, it overrides the simple switch.

The simple switch handles:

- no patch, Euler, 20+ steps: uses the selected `fast` or `standard` policy in KSampler or Hires Fix
- selected Speed Patch: stays off and prints a short reason
- LCM, non-Euler samplers, ControlNet, or unknown combinations: stays off and prints a short reason

Advanced Spectrum can still be combined manually with Speed Patches. The simple modes avoid that by default because speed-patch behavior is checkpoint-dependent and can turn experimental quickly.

Advanced parameters:

- `weight` and `degree`: the strongest character controls. Change these first.
- `ridge`: fit stabilization.
- `window_size` and `flex_window`: the forecast rhythm.
- `warmup_steps`: real steps before forecasting can begin.
- `final_real_steps`: the final safety zone; `0` means no forced final zone, `3` means the last three steps are always real.

The advanced node intentionally has no presets or manual step plan. It is for experiments and power users; most workflows should use the sampler's `fast` or `standard` switch.

Spectrum can be excellent on portraits and simpler scenes, but it can expose small artifacts in very dense prompts with hands, tiny objects or lots of overlapping details. If a prompt is already pushing SDXL hard, set `spectrum_acceleration` to `off` or use the advanced node to experiment.

Credits: Spectrum support in SDMLX is an MLX adaptation inspired by the official [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) project and the paper [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) by Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo and Stefano Ermon. The ComfyUI implementations [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum) and [ruwwww/ComfyUI-Spectrum-sdxl](https://github.com/ruwwww/ComfyUI-Spectrum-sdxl) were important practical references for ComfyUI-facing behavior and SDXL-specific use. The short-run `standard` policy is our own MLX/SDXL tuning on top of that foundation.

## FLUX Acceleration

FLUX workflows use a separate node family: `SDMLX Load Diffusion Model`, `SDMLX FLUX CLIP Loader`, `SDMLX FLUX MLX Sampler`, `SDMLX FLUX VAE Encode`, and `SDMLX FLUX VAE Decode`.

The FLUX sampler has two user-facing speed controls:

- `acceleration_patch`: optional FLUX-dev speed patch. Supported patches are downloaded or loaded as SDMLX acceleration-patch packages and cached locally when needed.
- `seacache_acceleration`: optional training-free cache/reuse acceleration. It is off by default.

These controls are separate. An acceleration patch changes the model trajectory. SeaCache tries to avoid recomputing selected sampler work. Turning both on can be useful, but it is still a speed/quality trade-off.

Current sampler behavior:

| Model path | `acceleration_patch` | `seacache_acceleration` | What runs |
| --- | --- | --- | --- |
| FLUX-schnell | ignored / not applicable | `false` | all real SDMLX FLUX-schnell steps |
| FLUX-schnell | ignored / not applicable | `true` | short 4-step SeaCache path plus late `pad_pool128` token pooling |
| FLUX-dev, no patch | `None` | `false` | all real SDMLX FLUX-dev steps |
| FLUX-dev, no patch | `None` | `true` | resolution-aware rising-threshold SeaCache starting point for longer Dev runs |
| FLUX-dev, 4-step patch | selected | `false` | all real speed-patch steps |
| FLUX-dev, 4-step patch | selected | `true` | speed patch plus one SeaCache reuse step for fast prompt tests |
| FLUX Kontext | automatically off | `false` | all real Kontext path with SDMLX `offset` reference method |
| FLUX Kontext | automatically off | `true` | Kontext path plus resolution-aware rising-threshold SeaCache |

For FLUX Kontext, acceleration patches are turned off automatically when reference image conditioning is present. This avoids mixing a speed-patch trajectory with image-reference behavior that has not been validated. SeaCache is still allowed.

Scale very large reference images before `SDMLX FLUX VAE Encode`. A 1024px or 768px long edge is usually the useful range for SDMLX Kontext workflows; raw multi-megapixel inputs can create very large MLX allocations without adding useful edit fidelity.

SeaCache can give very useful speedups, especially for prompt testing. It can also change small details. The built-in FLUX-dev settings are starting points or sweet spots for many prompts, not a promise that every image will match an all-real run. For detail-heavy FLUX-dev txt2img prompts, adding a few steps with SeaCache can be better than trying to make the original step count faster. If exact detail fidelity matters, compare once with `seacache_acceleration=false` or connect `SDMLX FLUX SeaCache Advanced` to tune `threshold_start`, `threshold_end`, `start_at` and `final_guard`.

SDXL Speed Patches and FLUX acceleration patches live in:

```text
ComfyUI/models/SDMLX/AccelerationPatches
```

Generated FLUX acceleration caches live in:

```text
ComfyUI/models/SDMLX/cache/acceleration-patches
```

Both folders are intentionally user-visible so generated caches can be removed if needed.

## Scheduler Curves

`SDMLX Scheduler` can drive LoRA strength, IP-Adapter strength and ControlNet strength over the sampling run.

Core controls:

- `mode`: the general movement.
- `minimum_strength`: the low point of the effect.
- `maximum_strength`: the high point of the effect.
- `curve`: how the effect moves between those values.

Modes:

- `blend in`: starts low and moves toward the maximum.
- `blend out`: starts high and moves toward the minimum.
- `bell`: rises and falls during the run.

Curves:

- `linear`: even movement.
- `progressive`: starts gently, becomes stronger later.
- `progressive fast`: stronger version of progressive.
- `degressive`: moves strongly at first, then settles.
- `degressive fast`: stronger version of degressive.
- `s-curve`: gentle start, stronger middle, gentle end.

In fast 4-8 step workflows, small changes can matter. Start with simple curves and moderate strengths.

## LoRA Controls

`SDMLX LoRA Loader` is the simple one-LoRA loader with an `enabled` toggle and optional scheduler input.

`SDMLX Multi LoRA Loader` is for stacking several LoRAs quickly. It intentionally stays simple; complex per-LoRA curve setups are better built with several single loaders and schedulers.

When a scheduler is connected, the scheduler controls the strength over time. The static strength value becomes the base amount that the schedule scales.

## FaceID Names

FaceID nodes use a few more image-oriented names:

- `identity_bias`: raises or lowers how strongly the generated face follows the identity.
- `img_details_v2_only`: PlusV2-only detail transfer from the source image. Higher values can pull in concrete image details such as jewelry, skin highlights or photo-specific traits.
- `auto_lora`: automatically applies the matching FaceID LoRA when the adapter uses one.
- `lora_strength`: strength of that FaceID LoRA path.

`identity_bias` is about who the person is. `img_details_v2_only` is about how much visual detail from this exact input image is transferred.

Portrait FaceID variants behave more like style-transfer models. FaceID PlusV2 is usually the safer default.

## IP-Adapter Controls

`crop center` keeps the center of the reference image.

`crop top` keeps the top of the reference image and crops more from the bottom. This is useful for portraits or upper-body references where cutting off the head would be bad.

`stretch` forces the image into the target size and can distort it. Use it only when distortion is acceptable.

Schedulers can be connected to IP-Adapter nodes when the image influence should fade in, fade out or pulse during the run.

## ControlNet Controls

`SDMLX Apply ControlNet` follows familiar ControlNet wiring with positive and negative conditioning.

The `control_type` names are simplified to describe the visual input:

- `line to canny`
- `soft edge to scribble`
- `depth`
- `normal`
- `openpose`
- `tile`
- `repaint`

The node supports classic `start_percent` / `end_percent` behavior and can also accept an `SDMLX Scheduler` for curve-based strength changes.

## Inpaint Conditioning

`SDMLX Inpaint Conditioning` is the classic inpaint-style node. It takes image, mask, VAE and conditioning and returns conditioning plus latent data for the sampler.

For soft transitions, use a mask blur node before it. SDMLX includes a Gaussian blur mask node for that purpose.

## Inpaint Detailer

`SDMLX Inpaint Detailer` is a compact mask-detail workflow in one node. It crops the masked area, optionally renders it at a larger working size, samples it, decodes it, then inserts the full crop back into the original image.

Important controls:

- `crop`: how much surrounding context the crop includes.
- `guide_size`: target render size for the crop/detail pass.
- `guide_size_for`: whether `guide_size` follows the mask box or the full crop region.
- `max_size`: upper limit for the render size.
- `soft_mask`: blur amount for the mask used during sampling.
- `soft_mask_strength`: how strongly the soft mask affects differential diffusion.
- `crop_blend`: final blend at the outside edge of the crop when inserting it back.

`soft_mask` and `soft_mask_strength` shape what the model is allowed to repaint during sampling. `crop_blend` is only the final paste edge for the returned crop.

## Hires Fix And Tiled Upscale

`SDMLX Hires Fix` is for moderate second-pass upscales. It returns both image and latent.

`SDMLX Tiled Upscale` is for larger upscale jobs. It works tile by tile and can use tile ControlNet to reduce hallucination.

For tiled upscale, a safe starting point is usually:

- `scale`: `2x`
- `tile_size`: `1024`
- `tile_control_strength`: around `0.45`
- `denoise`: around `0.5-0.6`

Larger scales are possible, but memory use and render time rise quickly.

## Preview

The sampler `preview` toggle can help decide early whether a run is going in the right direction. It costs time, so it defaults to off in heavier nodes.

SDMLX follows ComfyUI's global preview setting instead of forcing a specific preview method. If ComfyUI is set to no previews, the SDMLX toggle will not override that. If ComfyUI is set to TAESD or another preview mode, SDMLX uses that mode.

For crop-based workflows, preview modes can focus on the crop area instead of the whole image.
