# SDMLX Acceleration Patches

Pre-mapped SDXL acceleration patches for SDMLX / ComfyUI on Apple Silicon.

These packages are **not fully expanded UNet deltas**. They contain SDMLX-mapped low-rank LoRA factors:

- target keys already match the SDMLX MLX UNet
- linear factors use `[out, rank]` and `[rank, in]`
- Conv2D down factors use MLX layout `[rank, kernel_h, kernel_w, in]`
- runtime scale is `strength * alpha / rank`

Each `.sdmlxpatch` package contains:

- `patch.safetensors`
- `manifest.json`
- optional `source_metadata.json`

## Packages

| Package | Recommended Steps | CFG | Sampler | Scheduler | Force No CFG |
| --- | ---: | ---: | --- | --- | --- |
| `dmd2_sdxl_4step_lora_fp16.sdmlxpatch` | 4 | 1.0 | `lcm` | `normal` | true |
| `dmd2-lighting8step_cfg1.5.sdmlxpatch` | 8 | 1.5 | `lcm` | `normal` | false |
| `sdxl_lightning_4step_lora.sdmlxpatch` | 4 | 1.0 | `euler` | `sgm_uniform` | true |
| `sdxl_lightning_8step_lora.sdmlxpatch` | 8 | 1.0 | `lcm` | `normal` | true |
| `lcm-lora-sdxl.sdmlxpatch` | 8 | 1.0 | `lcm` | `normal` | true |
| `Hyper-SDXL-8steps-CFG-lora.sdmlxpatch` | 8 | 5.0 | `lcm` | `normal` | false |
| `Hyper-SDXL-12steps-CFG-lora.sdmlxpatch` | 12 | 7.5 | `lcm` | `normal` | false |

The recommendations are metadata only. SDMLX should keep sampler settings explicit in the UI.

## Source Licenses

These packages are transformed LoRA factors. They are provided with source attribution and remain subject to their upstream licenses:

- `dmd2_*`: source `tianweiy/DMD2`, Hugging Face metadata lists `cc-by-nc-4.0`; the model card text names Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- `sdxl_lightning_*`: source `ByteDance/SDXL-Lightning`, license `openrail++`.
- `lcm-lora-sdxl`: source `latent-consistency/lcm-lora-sdxl`, license `openrail++`.
- `Hyper-SDXL-*`: source `ByteDance/Hyper-SD`; SD-related Hyper-SD models are covered by the CreativeML Open RAIL++-M license in the upstream `LICENSE.md`.

Do not publish or redistribute packages whose upstream license does not fit your use case. In particular, DMD2 is non-commercial/share-alike. OpenRAIL++ and OpenRAIL++-M packages include use-based restrictions that downstream users must follow.

The SDMLX code license does not relicense these packages. For the repository-level notice, see `THIRD_PARTY_NOTICES.md` in the SDMLX code repository.

## Hugging Face Upload

Use Git LFS for package tensors:

```bash
git lfs track "*.safetensors"
git lfs track "*.sdmlxpatch/patch.safetensors"
git add .gitattributes README.md *.sdmlxpatch
git commit -m "Add SDMLX acceleration patches"
git push
```

Suggested repo type: model.
