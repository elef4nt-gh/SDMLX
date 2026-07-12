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
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.sdmlxpatch` | `lightx2v/Qwen-Image-Edit-2511-Lightning` | Check upstream model card before redistribution or commercial use | SDMLX treats this as a LoRA-backed acceleration patch for Qwen Image Edit 2511. |

## Optional Auto-Downloaded Models

SDMLX can download common companion models into ComfyUI's normal model folders. These downloads are provided for convenience only; users remain responsible for complying with the upstream licenses.

| Asset family | Upstream source | Used for |
| --- | --- | --- |
| Xinsir ControlNet Union ProMax | `xinsir/controlnet-union-sdxl-1.0` | SDXL ControlNet workflows |
| IP-Adapter SDXL ViT-H variants | `h94/IP-Adapter` | Standard IP-Adapter workflows |
| IP-Adapter FaceID / FaceID PlusV2 SDXL | `h94/IP-Adapter-FaceID` | FaceID workflows |
| CLIP-ViT-H-14 image encoder | `h94/IP-Adapter` image encoder files | IP-Adapter and FaceID PlusV2 CLIP Vision encoding |
| InsightFace models | InsightFace model zoo | Face detection and face embeddings for FaceID |
| Qwen Image Edit 2511 8-bit MLX package | `mlx-community/qwen-image-edit-2511-8bit` | Qwen Image Edit / upstream MLX package terms | Optional Qwen Image Edit model package used through SDMLX `.sdmlx` packages. |
| Qwen Image Edit 2511 Lightning LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` | Check upstream model card before use or redistribution | Optional first-class Qwen acceleration patch. |

## FLUX.2 Klein Enhanced Edit Reference

SDMLX includes an independent clean-room Enhanced Edit sampler for FLUX.2 Klein multi-reference identity/reference steering. The SDMLX implementation does not bundle source code, assets, or model files from the external reference project below.

| Component | Upstream source | License / terms | Notes |
| --- | --- | --- | --- |
| ComfyUI Flux2Klein Enhancer reference | [capitan01R/ComfyUI-Flux2Klein-Enhancer](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer) | [PolyForm Noncommercial License 1.0.0](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer/blob/main/LICENSE), also see [polyformproject.org](https://polyformproject.org/licenses/noncommercial/1.0.0/) | Behavioral reference only. SDMLX does not vendor or copy this code. Treat the upstream project as non-commercial unless its author grants broader rights. |

## Practical Guidance

- The GPL license for SDMLX code does not override model, LoRA, or dataset licenses.
- Do not redistribute SDMLX Speed Patches unless the upstream source license permits redistribution of transformed weights.
- DMD2-derived patches should be treated as non-commercial.
- OpenRAIL and OpenRAIL++-M assets may allow broad use but include use-based restrictions that downstream users must follow.
- Keep attribution to upstream model authors when publishing workflows, derivative patches, or packaged releases.

## Spectrum Acceleration

SDMLX includes an MLX adaptation of Spectrum-style feature forecasting for SDXL sampling. Spectrum is described in the paper [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) by Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo and Stefano Ermon.

The implementation in SDMLX is written for MLX and SDXL, but it is inspired by and cross-checked against the official Spectrum project and community ComfyUI implementations. The normal sampler and Hires Fix `fast` mode use public Spectrum scheduling behavior with three final real steps. The `standard` mode uses a cleaner Spectrum-Proper-style short-run path below 35 steps and the more conservative scheduling family from 35 steps upward.

| Component | Upstream source | License | Notes |
| --- | --- | --- | --- |
| Spectrum paper | [arXiv:2603.01623](https://arxiv.org/abs/2603.01623) | Paper license as listed by arXiv | Original Spectrum method and terminology. |
| Official Spectrum code | [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) | MIT License, copyright (c) 2026 Jiaqi Han | Official implementation and reference for Chebyshev/ridge feature forecasting. |
| ComfyUI Spectrum reference | [judian17/ComfyUI-Spectrum](https://github.com/judian17/ComfyUI-Spectrum) | MIT License; license file preserves Jiaqi Han copyright notice | Reference for ComfyUI-oriented scheduling behavior and broad model integration. |
| ComfyUI SDXL Spectrum reference | [ruwwww/ComfyUI-Spectrum-sdxl](https://github.com/ruwwww/ComfyUI-Spectrum-sdxl) | MIT License, copyright (c) 2026 A. Izzuddin Al Faruq | Reference for SDXL-oriented ComfyUI integration, manual parameters and final real-step guard behavior. |

SDMLX's default `fast` and `standard` policies are practical SDXL/MLX presets, not upstream defaults. They should be treated as SDMLX tuning choices layered on top of the Spectrum idea.

## SeaCache Acceleration

SDMLX includes an MLX adaptation of a SeaCache-style cache path for FLUX.1 sampling. SeaCache is described in the paper [SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](https://arxiv.org/abs/2602.18993) by Jiwoo Chung, Sangeek Hyun, MinKyu Lee, Byeongju Han, Geonho Cha, Dongyoon Wee, Youngjun Hong and Jae-Pil Heo.

The implementation in SDMLX is written for the SDMLX FLUX MLX runtime and uses SDMLX-specific defaults. The core cache idea and terminology come from the public SeaCache work.

| Component | Upstream source | License / terms | Notes |
| --- | --- | --- | --- |
| SeaCache paper | [arXiv:2602.18993](https://arxiv.org/abs/2602.18993) | Paper license as listed by arXiv | Original SeaCache method and terminology. |
| SeaCache code | [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) | Check upstream repository before reuse or redistribution | Practical reference for the FLUX-oriented cache path. |

SDMLX's SeaCache defaults are practical FLUX/MLX tuning choices, not upstream defaults.

## LUA Latent Upscale Adapter

SDMLX includes a FLUX-facing node for LUA-style latent upscaling. LUA is described in the paper [One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models](https://arxiv.org/abs/2511.10629) by Aleksandr Razin, Danil Kazantsev and Ilya Makarov.

The bundled adapter code is adapted from the official Apache-2.0 implementation. The default LUA weights are not bundled; they are downloaded on first use from the upstream Hugging Face repository unless the user supplies a local `lua_flux.pth` path.

| Component | Upstream source | License / terms | Notes |
| --- | --- | --- | --- |
| LUA paper | [arXiv:2511.10629](https://arxiv.org/abs/2511.10629) | Paper license as listed by arXiv | Original latent-upscale adapter method. |
| LUA code | [vaskers5/LUA](https://github.com/vaskers5/LUA) | Apache License 2.0 | Source for the small PyTorch LUA adapter module used by the SDMLX FLUX LUA Adapter. |
| LUA FLUX weights | [vaskers5/LUA-FLUX](https://huggingface.co/vaskers5/LUA-FLUX) | Check upstream model card before redistribution or commercial use | Default `lua_flux.pth` weights; supports FLUX/SD3-style 16-channel latents. |

## FLUX Runtime Reference

SDMLX's FLUX sampler uses an SDMLX MLX sampling loop with Euler-style flow updates and sigma/runtime scheduling adapted from `mflux`. SDMLX also includes a small local FLUX VAE implementation adapted from `mflux`, so the full `mflux` package is not required at runtime.

| Component | Upstream source | License | Notes |
| --- | --- | --- | --- |
| mflux | [filipstrand/mflux](https://github.com/filipstrand/mflux) | MIT License, copyright (c) 2026 Filip Strand | Reference for FLUX configuration, sigma scheduling behavior and the local FLUX VAE code adapted into SDMLX. The bundled license text is in `sdmlx_qwen_native/LICENSE.mflux`. |

## Qwen Runtime Reference

SDMLX's Qwen Image Edit 2511 path includes a local MLX runtime adapted from `mflux` 0.18.0 with SDMLX changes for the Qwen 2511 edit path, including reference handling, Qwen Edit Plus prompt/image prefix handling, guidance=1.0 single-pass sampling and the validated `linear` scheduler path.

| Component | Upstream source | License / terms | Notes |
| --- | --- | --- | --- |
| mflux | [filipstrand/mflux](https://github.com/filipstrand/mflux) | MIT License, copyright (c) 2026 Filip Strand | Reference and adapted source base for the local Qwen MLX runtime. The bundled license text is in `sdmlx_qwen_native/LICENSE.mflux`. |
| Qwen Image Edit 2511 | `Qwen/qwen-image-edit-2511` | Check upstream model card before use or redistribution | Base model family for the Qwen Image Edit path. |
| Qwen Image Edit 2511 8-bit MLX package | `mlx-community/qwen-image-edit-2511-8bit` | Check upstream package and base model terms | Default repo id for Qwen `.sdmlx` package manifests. |

### Qwen phr00t-style Conditioning Plus Reference

`SDMLX Qwen Image Edit Conditioning Plus` is an SDMLX implementation of the behavior pattern used by phr00t's fixed Qwen Image Edit Plus text-encode node: multi-image `Picture N` framing, a small Qwen-VL image path, and separately scaled VAE reference latents. SDMLX keeps this behavior inside its own MLX conditioning/runtime path and does not require users to replace ComfyUI files.

| Component | Upstream source | License / terms | Notes |
| --- | --- | --- | --- |
| phr00t fixed Qwen text-encode node | [Phr00t/Qwen-Image-Edit-Rapid-AIO fixed-textencode-node](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main/fixed-textencode-node) | Check upstream model card/repository terms before redistribution or commercial use | Behavioral reference for the SDMLX Plus conditioning contract. |
