WEB_DIRECTORY = "./web"

_NODE_DISPLAY_NAMES = {
    "SDMLX_LoadImageScale": "🍏 SDMLX LoadImage Scale",
    "SDMLX_LoadImageAdvanced": "🍏 SDMLX Load Image Advanced",
    "SDMLX_GaussianBlurMask": "🍏 SDMLX Gaussian Blur Mask",
    "SDMLX_CheckpointLoader": "🍏 SDMLX Loader Universal",
    "SDMLX_Loader": "🍏 SDMLX Loader",
    "SDMLX_CLIPLoader": "🍏 SDMLX CLIP Loader",
    "SDMLX_DualCLIPLoader": "🍏 SDMLX Dual CLIP Loader",
    "SDMLX_VAELoader": "🍏 SDMLX VAE Loader",
    "SDMLX_CLIPTextEncode": "🍏 SDMLX CLIP Text Encode",
    "SDMLX_ConditioningZeroOut": "🍏 SDMLX Conditioning Zero Out",
    "SDMLX_FluxGuidance": "🍏 SDMLX FLUX Guidance",
    "SDMLX_LoraLoader": "🍏 SDMLX LoRA Loader",
    "SDMLX_MultiLoraLoader": "🍏 SDMLX Multi LoRA Loader",
    "SDMLX_SpeedPatchConverter": "🍏 SDMLX Speed Patch Converter",
    "SDMLX_SpectrumBoost": "🍏 SDMLX Spectrum Advanced",
    "SDMLX_LoraSchedule": "🍏 SDMLX Scheduler",
    "SDMLX_IPAdapterLoader": "🍏 SDMLX IP-Adapter Loader",
    "SDMLX_CLIPVisionLoader": "🍏 SDMLX CLIP Vision Loader",
    "SDMLX_IPAdapterMLXCLIPVisionEncode": "🍏 SDMLX CLIP Vision Encode",
    "SDMLX_InsightFaceLoader": "🍏 SDMLX InsightFace Loader",
    "SDMLX_InsightFaceAlignCrop": "🍏 SDMLX InsightFace Align Crop",
    "SDMLX_ApplyIPAdapter": "🍏 SDMLX Apply IP-Adapter",
    "SDMLX_ApplyIPAdapterFaceID": "🍏 SDMLX Apply IP-Adapter FaceID",
    "SDMLX_ApplyIPAdapterFaceIDAIO": "🍏 SDMLX FaceID AIO",
    "SDMLX_DifferentialDiffusion": "🍏 SDMLX Differential Diffusion",
    "SDMLX_KSampler": "🍏 SDMLX KSampler",
    "SDMLX_KSamplerAdvanced": "🍏 SDMLX KSampler Advanced",
    "SDMLX_VAEDecode": "🍏 SDMLX VAE Decode",
    "SDMLX_VAEEncode": "🍏 SDMLX VAE Encode",
    "SDMLX_PromptSlotSwitcher": "🍏 SDMLX Prompt Switcher",
    "SDMLX_ControlNetUnionLoader": "🍏 SDMLX ControlNet Union ProMax Loader",
    "SDMLX_ApplyControlNet": "🍏 SDMLX Apply ControlNet",
    "SDMLX_InpaintConditioning": "🍏 SDMLX Inpaint Conditioning",
    "SDMLX_InpaintDetailer": "🍏 SDMLX Inpaint Detailer",
    "SDMLX_HiresFix": "🍏 SDMLX Hires Fix",
    "SDMLX_TiledUpscale": "🍏 SDMLX Tiled Upscale",
    "SDMLX_NumberPicker": "🍏 SDMLX Number Picker",
    "SDMLXFluxNativeLoader": "🍏 SDMLX Load Diffusion Model",
    "SDMLX_CLIPTextEncodeFlux": "🍏 SDMLX CLIP Text Encode Flux",
    "SDMLXFluxNativeSampler": "🍏 SDMLX KSampler (FLUX.1)",
    "SDMLXFluxSeaCacheAdvanced": "🍏 SDMLX FLUX SeaCache Advanced",
    "SDMLXFluxLUAAdapter": "🍏 SDMLX FLUX LUA Adapter",
    "SDMLXFluxKontextImageScale": "🍏 SDMLX FLUX Kontext Scale",
    "SDMLXFluxEmptyLatentImage": "🍏 SDMLX Empty Latent Image FLUX.1",
    "SDMLXFlux2ScaleToMegapixels": "🍏 SDMLX Scale To Megapixels",
    "SDMLXFlux2EmptyLatentImage": "🍏 SDMLX Empty Latent Image FLUX.2",
    "SDMLXFlux2ReferenceLatent": "🍏 SDMLX Reference Latent",
    "SDMLXFlux2KleinKSampler": "🍏 SDMLX KSampler (FLUX.2-klein)",
    "SDMLXFlux2LabRuntimePlan": "🍏 SDMLX FLUX.2 Lab Runtime Plan",
    "SDMLXFlux2LabKSampler": "🍏 SDMLX KSampler (FLUX.2 Lab)",
    "SDMLXFlux2KleinEnhancerAdvanced": "🍏 SDMLX FLUX.2 Klein Enhancer Advanced",
    "SDMLXFlux2KleinEnhancedEditSampler": "🍏 SDMLX KSampler (FLUX.2-klein Enhanced Edit)",
    "SDMLXQwenImageEditConditioning": "🍏 SDMLX Qwen Image Edit Conditioning",
    "SDMLXQwenImageEditPlusConditioning": "🍏 SDMLX Qwen Image Edit Conditioning Plus",
    "SDMLXQwenTokenBudget": "🍏 SDMLX Qwen Token Budget",
    "SDMLXQwenModelSamplingAuraFlow": "🍏 SDMLX Qwen ModelSampling AuraFlow",
}

_NODE_CATEGORIES = {
    "SDMLX_LoadImageScale": "SDMLX/Image",
    "SDMLX_LoadImageAdvanced": "SDMLX/Image",
    "SDMLX_GaussianBlurMask": "SDMLX/Mask",
    "SDMLX_CheckpointLoader": "SDMLX/Loaders",
    "SDMLX_Loader": "SDMLX/Loaders",
    "SDMLX_CLIPLoader": "SDMLX/Loaders",
    "SDMLX_DualCLIPLoader": "SDMLX/Loaders",
    "SDMLX_VAELoader": "SDMLX/Loaders",
    "SDMLX_CLIPTextEncode": "SDMLX/Conditioning",
    "SDMLX_ConditioningZeroOut": "SDMLX/Conditioning",
    "SDMLX_FluxGuidance": "SDMLX/Conditioning",
    "SDMLX_LoraLoader": "SDMLX/LoRA",
    "SDMLX_MultiLoraLoader": "SDMLX/LoRA",
    "SDMLX_SpeedPatchConverter": "SDMLX/Utilities",
    "SDMLX_SpectrumBoost": "SDMLX/Advanced",
    "SDMLX_LoraSchedule": "SDMLX/Advanced",
    "SDMLX_IPAdapterLoader": "SDMLX/IPAdapter",
    "SDMLX_CLIPVisionLoader": "SDMLX/IPAdapter",
    "SDMLX_IPAdapterMLXCLIPVisionEncode": "SDMLX/IPAdapter",
    "SDMLX_InsightFaceLoader": "SDMLX/IPAdapter",
    "SDMLX_InsightFaceAlignCrop": "SDMLX/IPAdapter",
    "SDMLX_ApplyIPAdapter": "SDMLX/IPAdapter",
    "SDMLX_ApplyIPAdapterFaceID": "SDMLX/IPAdapter",
    "SDMLX_ApplyIPAdapterFaceIDAIO": "SDMLX/IPAdapter",
    "SDMLX_DifferentialDiffusion": "SDMLX/Inpaint",
    "SDMLX_KSampler": "SDMLX/Sampling",
    "SDMLX_KSamplerAdvanced": "SDMLX/Sampling",
    "SDMLX_VAEDecode": "SDMLX/Latent",
    "SDMLX_VAEEncode": "SDMLX/Latent",
    "SDMLX_PromptSlotSwitcher": "SDMLX/Utilities",
    "SDMLX_ControlNetUnionLoader": "SDMLX/ControlNet",
    "SDMLX_ApplyControlNet": "SDMLX/ControlNet",
    "SDMLX_InpaintConditioning": "SDMLX/Inpaint",
    "SDMLX_InpaintDetailer": "SDMLX/Inpaint",
    "SDMLX_HiresFix": "SDMLX/Upscale",
    "SDMLX_TiledUpscale": "SDMLX/Upscale",
    "SDMLX_NumberPicker": "SDMLX/Utilities",
    "SDMLXFluxNativeLoader": "SDMLX/Loaders",
    "SDMLX_CLIPTextEncodeFlux": "SDMLX/Conditioning",
    "SDMLXFluxNativeSampler": "SDMLX/Sampling",
    "SDMLXFluxSeaCacheAdvanced": "SDMLX/Advanced",
    "SDMLXFluxLUAAdapter": "SDMLX/Upscale",
    "SDMLXFluxKontextImageScale": "SDMLX/Image",
    "SDMLXFluxEmptyLatentImage": "SDMLX/Latent",
    "SDMLXFlux2ScaleToMegapixels": "SDMLX/Image",
    "SDMLXFlux2EmptyLatentImage": "SDMLX/Latent",
    "SDMLXFlux2ReferenceLatent": "SDMLX/Conditioning",
    "SDMLXFlux2KleinKSampler": "SDMLX/Sampling",
    "SDMLXFlux2LabRuntimePlan": "SDMLX/FLUX.2 Lab",
    "SDMLXFlux2LabKSampler": "SDMLX/FLUX.2 Lab",
    "SDMLXFlux2KleinEnhancerAdvanced": "SDMLX/Advanced",
    "SDMLXFlux2KleinEnhancedEditSampler": "SDMLX/Sampling",
    "SDMLXQwenImageEditConditioning": "SDMLX/Conditioning",
    "SDMLXQwenImageEditPlusConditioning": "SDMLX/Conditioning",
    "SDMLXQwenTokenBudget": "SDMLX/Conditioning",
    "SDMLXQwenModelSamplingAuraFlow": "SDMLX/Advanced",
}


def _make_unavailable_node(name):
    class UnavailableNode:
        CATEGORY = _NODE_CATEGORIES.get(name, "SDMLX/Utilities")
        RETURN_TYPES = ()
        FUNCTION = "unavailable"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def unavailable(self):
            raise RuntimeError(
                "SDMLX could not load its MLX runtime dependencies. "
                "Install SDMLX on Apple Silicon with the package requirements enabled."
            )

    UnavailableNode.__name__ = name
    return UnavailableNode


try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, SDMLX_VERSION
    from .flux_nodes import NODE_CLASS_MAPPINGS as FLUX_NODE_CLASS_MAPPINGS
    from .flux_nodes import NODE_DISPLAY_NAME_MAPPINGS as FLUX_NODE_DISPLAY_NAME_MAPPINGS
    from .flux2_nodes import NODE_CLASS_MAPPINGS as FLUX2_NODE_CLASS_MAPPINGS
    from .flux2_nodes import NODE_DISPLAY_NAME_MAPPINGS as FLUX2_NODE_DISPLAY_NAME_MAPPINGS
    from .qwen_nodes import NODE_CLASS_MAPPINGS as QWEN_NODE_CLASS_MAPPINGS
    from .qwen_nodes import NODE_DISPLAY_NAME_MAPPINGS as QWEN_NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS,
        **FLUX_NODE_CLASS_MAPPINGS,
        **FLUX2_NODE_CLASS_MAPPINGS,
        **QWEN_NODE_CLASS_MAPPINGS,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **FLUX_NODE_DISPLAY_NAME_MAPPINGS,
        **FLUX2_NODE_DISPLAY_NAME_MAPPINGS,
        **QWEN_NODE_DISPLAY_NAME_MAPPINGS,
    }
except ModuleNotFoundError as exc:
    # The Comfy Registry parser may import custom nodes on a non-macOS host where
    # mlx is unavailable. Keep the node list visible there; real runtime imports
    # still use .nodes on supported Apple Silicon installations.
    if exc.name and exc.name.startswith((
        "mlx",
        "numpy",
        "torch",
        "transformers",
        "PIL",
        "huggingface_hub",
        "safetensors",
        "gguf",
        "platformdirs",
        "piexif",
    )):
        SDMLX_VERSION = "0.1.22"
        NODE_CLASS_MAPPINGS = {
            node_name: _make_unavailable_node(node_name) for node_name in _NODE_DISPLAY_NAMES
        }
        NODE_DISPLAY_NAME_MAPPINGS = dict(_NODE_DISPLAY_NAMES)
    else:
        raise

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SDMLX_VERSION",
    "WEB_DIRECTORY",
]
