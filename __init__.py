WEB_DIRECTORY = "./web"

_NODE_DISPLAY_NAMES = {
    "SDMLX_GaussianBlurMask": "🍏 SDMLX Gaussian Blur Mask",
    "SDMLX_CheckpointLoader": "🍏 SDMLX Loader Universal",
    "SDMLX_Loader": "🍏 SDMLX Loader",
    "SDMLX_CLIPTextEncode": "🍏 SDMLX CLIP Text Encode",
    "SDMLX_LoraLoader": "🍏 SDMLX LoRA Loader",
    "SDMLX_MultiLoraLoader": "🍏 SDMLX Multi LoRA Loader",
    "SDMLX_SpeedPatchConverter": "🍏 SDMLX Speed Patch Converter",
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
    "SDMLX_VAEDecode": "🍏 SDMLX VAE Decode",
    "SDMLX_ControlNetUnionLoader": "🍏 SDMLX ControlNet Union ProMax Loader",
    "SDMLX_ApplyControlNet": "🍏 SDMLX Apply ControlNet",
    "SDMLX_InpaintConditioning": "🍏 SDMLX Inpaint Conditioning",
    "SDMLX_InpaintDetailer": "🍏 SDMLX Inpaint Detailer",
    "SDMLX_HiresFix": "🍏 SDMLX Hires Fix",
    "SDMLX_TiledUpscale": "🍏 SDMLX Tiled Upscale",
}


def _make_unavailable_node(name):
    class UnavailableNode:
        CATEGORY = "SDMLX"
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
    )):
        SDMLX_VERSION = "0.1.10"
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
