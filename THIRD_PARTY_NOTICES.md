# Third-Party Notices

SDMLX source code is licensed separately under the GNU General Public License v3.0 or later. This notice file covers third-party model assets, model-derived acceleration patches, and optional companion models that SDMLX can use or download.

SDMLX does not claim ownership of third-party model weights, LoRAs, ControlNet weights, IP-Adapter weights, CLIP Vision weights, InsightFace models, or any derivatives of those assets. Those assets remain governed by their upstream licenses and terms.

This file is a practical attribution and license summary, not legal advice. Always check the upstream repository or model card before redistributing, modifying, or using these assets in a commercial context.

## SDMLX Acceleration Patches

SDMLX acceleration patches are transformed LoRA/model factors mapped for the SDMLX MLX runtime. They are not relicensed by SDMLX. They remain subject to the upstream licenses of the source LoRAs or models.

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
- Do not redistribute SDMLX acceleration patches unless the upstream source license permits redistribution of transformed weights.
- DMD2-derived patches should be treated as non-commercial.
- OpenRAIL and OpenRAIL++-M assets may allow broad use but include use-based restrictions that downstream users must follow.
- Keep attribution to upstream model authors when publishing workflows, derivative patches, or packaged releases.
