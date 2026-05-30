# Third-Party Notices

SDMLX source code is licensed separately under the GNU General Public License v3.0 or later. This notice file covers third-party model assets, model-derived Speed Patches, and optional companion models that SDMLX can use or download.

SDMLX does not claim ownership of third-party model weights, LoRAs, ControlNet weights, IP-Adapter weights, CLIP Vision weights, InsightFace models, or any derivatives of those assets. Those assets remain governed by their upstream licenses and terms.

This file is a practical attribution and license summary, not legal advice. Always check the upstream repository or model card before redistributing, modifying, or using these assets in a commercial context.

## SDMLX Speed Patches

SDMLX Speed Patches are transformed LoRA/model factors mapped for the SDMLX MLX runtime. They are not relicensed by SDMLX. They remain subject to the upstream licenses of the source LoRAs or models.

| SDMLX patch | Upstream source | Upstream license / terms | Notes |
| --- | --- | --- | --- |
| `dmd2_sdxl_4step_lora_fp16.sdmlxpatch` | `tianweiy/DMD2` | Hugging Face metadata lists `cc-by-nc-4.0`; related DMD2 materials also reference Creative Commons non-commercial/share-alike terms | Treat as non-commercial unless the upstream project grants broader rights. |
| `dmd2-lighting8step_cfg1.5.sdmlxpatch` | DMD2-derived / Lightning-related source material | Same caution as DMD2 plus the relevant Lightning/OpenRAIL terms where applicable | Treat as non-commercial unless the upstream project grants broader rights. |
| `sdxl_lightning_4step_lora.sdmlxpatch` | `ByteDance/SDXL-Lightning` | `openrail++` | OpenRAIL licenses include use-based restrictions. |
| `sdxl_lightning_8step_lora.sdmlxpatch` | `ByteDance/SDXL-Lightning` | `openrail++` | OpenRAIL licenses include use-based restrictions. |
| `lcm-lora-sdxl.sdmlxpatch` | `latent-consistency/lcm-lora-sdxl` | `openrail++` | The related code repository may use a different software license; the model weights are governed by the model-card license. |
| `Hyper-SDXL-8steps-CFG-lora.sdmlxpatch` | `ByteDance/Hyper-SD` | CreativeML Open RAIL++-M / upstream model-card terms | OpenRAIL licenses include use-based restrictions. |
| `Hyper-SDXL-12steps-CFG-lora.sdmlxpatch` | `ByteDance/Hyper-SD` | CreativeML Open RAIL++-M / upstream model-card terms | OpenRAIL licenses include use-based restrictions. |

## Optional Auto-Downloaded Models

SDMLX can download common companion models into ComfyUI's normal model folders. These downloads are provided for convenience only; users remain responsible for complying with the upstream licenses.

| Asset family | Upstream source | Used for |
| --- | --- | --- |
| Xinsir ControlNet Union ProMax | `xinsir/controlnet-union-sdxl-1.0` | SDXL ControlNet workflows |
| IP-Adapter SDXL ViT-H variants | `h94/IP-Adapter` | Standard IP-Adapter workflows |
| IP-Adapter FaceID / FaceID PlusV2 SDXL | `h94/IP-Adapter-FaceID` | FaceID workflows |
| CLIP-ViT-H-14 image encoder | `h94/IP-Adapter` image encoder files | IP-Adapter and FaceID PlusV2 CLIP Vision encoding |
| InsightFace models | InsightFace model zoo | Face detection and face embeddings for FaceID |

## Practical Guidance

- The GPL license for SDMLX code does not override model, LoRA, or dataset licenses.
- Do not redistribute SDMLX Speed Patches unless the upstream source license permits redistribution of transformed weights.
- DMD2-derived patches should be treated as non-commercial.
- OpenRAIL and OpenRAIL++-M assets may allow broad use but include use-based restrictions that downstream users must follow.
- Keep attribution to upstream model authors when publishing workflows, derivative patches, or packaged releases.

## Spectrum Acceleration

SDMLX includes an MLX adaptation of Spectrum-style feature forecasting for SDXL sampling. Spectrum is described in the paper [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) by Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo and Stefano Ermon.

The implementation in SDMLX is written for MLX and SDXL, but it is inspired by and cross-checked against the official Spectrum project and community ComfyUI implementations. The normal sampler and Hires Fix `fast` mode use public Spectrum scheduling behavior with three final real steps. The `standard` mode uses the same scheduling family from 35 steps upward and an SDMLX-specific guarded short-run policy below 35 steps.

| Component | Upstream source | License | Notes |
| --- | --- | --- | --- |
| Spectrum paper | [arXiv:2603.01623](https://arxiv.org/abs/2603.01623) | Paper license as listed by arXiv | Original Spectrum method and terminology. |
| Official Spectrum code | [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) | MIT License, copyright (c) 2026 Jiaqi Han | Official implementation and reference for Chebyshev/ridge feature forecasting. |
| ComfyUI Spectrum reference | [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum) | MIT License; license file preserves Jiaqi Han copyright notice | Reference for ComfyUI-oriented scheduling behavior and broad model integration. |
| ComfyUI SDXL Spectrum reference | [ruwwww/ComfyUI-Spectrum-sdxl](https://github.com/ruwwww/ComfyUI-Spectrum-sdxl) | MIT License, copyright (c) 2026 A. Izzuddin Al Faruq | Reference for SDXL-oriented ComfyUI integration, manual parameters and final real-step guard behavior. |

SDMLX's default `fast` and `standard` policies are practical SDXL/MLX presets, not upstream defaults. They should be treated as SDMLX tuning choices layered on top of the Spectrum idea.
