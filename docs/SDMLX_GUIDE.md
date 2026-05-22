# SDMLX Mini Guide

SDMLX tries to keep ComfyUI familiar while making SDXL on Apple Silicon feel more direct. A few controls are named differently from older diffusion nodes on purpose: the goal is to describe what the control does in the image or workflow, not the implementation detail behind it.

Example workflows are shipped with the node suite in `resources/workflows`. The `speed_workflows` subfolder contains DMD2 and Lightning variants, and `resources/workflow_png` contains PNG previews without embedded workflow JSON.

## Loaders And Memory Assist

`SDMLX Loader Universal` is the easiest starting point. It loads normal SDXL checkpoints from `models/checkpoints`, converts them once, and then automatically prefers the cached `.sdmlx` package.

`SDMLX Loader` only lists already converted `.sdmlx` packages from `models/SDMLX`.

`memory_assist` controls how aggressively SDMLX keeps models warm:

- `auto`: recommended default. Balanced cache behavior for normal Mac use.
- `max_performance`: keeps more model data warm when memory allows it.
- `low_memory`: frees more aggressively for smaller Macs or heavy workflows.
- `off`: leaves Comfy/MLX closer to their default behavior.

## Speed Patches

Speed patches are MLX-converted SDXL speed LoRAs. They live in the sampler as first-class acceleration options because they are usually used like part of the sampling setup, not like a creative style LoRA.

They can reduce step counts dramatically, for example with DMD2, Lightning, LCM or Hyper-SD style workflows. Use the settings recommended by the patch or the model author as the starting point.

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

TAESD preview can help decide early whether a run is going in the right direction. It costs time, so it defaults to off in heavier nodes.

TAESD preview needs a TAESDXL decoder model in ComfyUI's `models/vae_approx` folder, usually named like `taesdxl_decoder.*`.

For crop-based workflows, preview modes can focus on the crop area instead of the whole image.
