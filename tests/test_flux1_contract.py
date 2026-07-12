from __future__ import annotations

import os
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
import torch
from safetensors.numpy import save_file


SUITE_ROOT = Path(__file__).resolve().parents[1]
SUITE_PARENT = SUITE_ROOT.parent
COMFY_ROOT = os.environ.get("COMFYUI_ROOT")
if COMFY_ROOT:
    sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, str(SUITE_PARENT))

from .. import flux_nodes  # noqa: E402
from ..flux_native import native_flux_core  # noqa: E402


def flux1_keys(*, guidance: bool) -> set[str]:
    keys = {"img_in.weight", "txt_in.weight", "final_layer.linear.weight"}
    keys.update(f"double_blocks.{index}.img_attn.qkv.weight" for index in range(19))
    keys.update(f"single_blocks.{index}.linear1.weight" for index in range(38))
    if guidance:
        keys.add("guidance_in.in_layer.weight")
    return keys


def contract(
    *,
    source: str = "flux1-dev.safetensors",
    guidance: bool = True,
    fp8: int = 0,
    scales: int = 0,
    inputs: int = 0,
    marker: bool = False,
    gguf: bool = False,
):
    return flux_nodes._classify_flux1_checkpoint_contract(
        source_identity=source,
        normalized_keys=flux1_keys(guidance=guidance),
        fp8_2d_weights=fp8,
        scale_weights=scales,
        input_scales=inputs,
        scaled_marker=marker,
        is_gguf=gguf,
    )


class Flux1IdentityTests(unittest.TestCase):
    def test_dev_kontext_and_schnell_identity(self):
        self.assertEqual(contract(source="community.safetensors").variant, "dev")
        self.assertEqual(contract(source="flux1-kontext-dev.safetensors").variant, "kontext")
        schnell = contract(source="flux1-schnell.safetensors", guidance=False)
        self.assertEqual((schnell.variant, schnell.model_family), ("schnell", "schnell"))

    def test_structure_wins_over_non_authoritative_name(self):
        mislabeled = contract(source="custom-schnell-name.safetensors", guidance=True)
        self.assertEqual((mislabeled.variant, mislabeled.model_family), ("dev", "dev"))

    def test_other_families_and_wrong_depth_fail(self):
        with self.assertRaisesRegex(RuntimeError, "another model family"):
            contract(source="flux2-klein.safetensors")
        with self.assertRaisesRegex(RuntimeError, "19-double/38-single"):
            flux_nodes._classify_flux1_checkpoint_contract(
                source_identity="not-flux.safetensors",
                normalized_keys={"img_in.weight", "txt_in.weight", "final_layer.linear.weight"},
            )
        with self.assertRaisesRegex(RuntimeError, "nonstandard FLUX.1 img_in"):
            flux_nodes._classify_flux1_checkpoint_contract(
                source_identity="flux1-fill-dev.safetensors",
                normalized_keys=flux1_keys(guidance=True),
                img_in_features=384,
            )

    def test_header_inspector_and_detector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "qwen_flux2_archive"
            root.mkdir()
            weights = {key: np.zeros((1, 1), dtype=np.float16) for key in flux1_keys(guidance=True)}
            weights["img_in.weight"] = np.zeros((1, 64), dtype=np.float16)
            dev_path = root / "flux1-dev.safetensors"
            save_file(weights, str(dev_path), metadata={"modelspec.architecture": "Flux.1-dev"})
            inspected = flux_nodes.inspect_flux1_checkpoint_contract(dev_path)
            self.assertEqual((inspected.variant, inspected.quant_contract), ("dev", "dense"))
            self.assertTrue(flux_nodes.is_flux1_checkpoint_file(dev_path))
            as_kontext = flux_nodes.inspect_flux1_checkpoint_contract(
                dev_path,
                source_name="kontext/generic-edit-weights.safetensors",
            )
            self.assertEqual(as_kontext.variant, "kontext")

            other_path = root / "flux2-klein.safetensors"
            save_file(weights, str(other_path))
            self.assertFalse(flux_nodes.is_flux1_checkpoint_file(other_path))


class Flux1QuantContractTests(unittest.TestCase):
    def test_dense_raw_scaled_and_gguf_contracts(self):
        self.assertEqual(contract().quant_contract, flux_nodes.FLUX1_QUANT_DENSE)
        raw = contract(fp8=314)
        self.assertEqual((raw.quant_contract, raw.fp8_mode), (flux_nodes.FLUX1_QUANT_RAW_FP8, "native"))
        scaled = contract(fp8=314, scales=314, inputs=314)
        self.assertEqual(
            (scaled.quant_contract, scaled.fp8_mode, scaled.scale_weights, scaled.input_scales),
            (flux_nodes.FLUX1_QUANT_SCALED_FP8, "dequant", 314, 314),
        )
        self.assertEqual(contract(gguf=True).quant_contract, flux_nodes.FLUX1_QUANT_GGUF)

    def test_load_options_keep_base_contract_and_dense_patch_cache(self):
        source = Path("/tmp/flux1-dev.safetensors")
        patch_cache = Path("/tmp/flux1-dev-hyper.safetensors")
        scaled = contract(fp8=314, scales=314)
        raw = contract(fp8=314)

        self.assertEqual(flux_nodes._flux1_load_options(scaled, source, source_path=source)[0], "dequant")
        self.assertEqual(flux_nodes._flux1_load_options(raw, source, source_path=source)[0], "native")
        self.assertEqual(flux_nodes._flux1_load_options(raw, patch_cache, source_path=source)[0], "dequant")

    def test_scaled_rehydrate_uses_dequant(self):
        source = Path("/tmp/flux1-kontext-scaled.safetensors")
        old_transformer = native_flux_core.FluxNativeTransformer(weights={})
        replacement = native_flux_core.FluxNativeTransformer(weights={})
        model = flux_nodes.SDMLXFluxNativeModel(
            "flux1-kontext-scaled.safetensors",
            source,
            old_transformer,
            mx.float16,
            flux_nodes.ModelConfig.FLUX1_DEV,
            0,
            0,
            model_family="dev",
            checkpoint_contract=contract(source="flux1-kontext-dev.safetensors", fp8=314, scales=314),
            source_path=source,
        )
        with mock.patch.object(flux_nodes, "_transformer_has_model_weights", return_value=False), mock.patch.object(
            flux_nodes,
            "_load_prepared_flux_transformer",
            return_value=(replacement, 1, 2, 0.1),
        ) as loader:
            flux_nodes._ensure_flux_model_weights_loaded(model)

        self.assertEqual(loader.call_args.kwargs["fp8_mode"], "dequant")
        self.assertIs(model.transformer, replacement)


class Flux1SamplingContractTests(unittest.TestCase):
    def test_dev_and_schnell_sigma_contracts(self):
        config = flux_nodes.Config(num_inference_steps=4, width=1024, height=1024, guidance=3.5)
        dev = np.array(flux_nodes.RuntimeConfig(config, flux_nodes.ModelConfig.FLUX1_DEV).sigmas)
        schnell = np.array(flux_nodes.RuntimeConfig(config, flux_nodes.ModelConfig.FLUX1_SCHNELL).sigmas)
        linear = np.concatenate([np.linspace(1.0, 0.25, 4), np.zeros(1)]).astype(np.float32)
        shifted = np.exp(1.15) / (np.exp(1.15) + (1.0 / linear[:-1] - 1.0))
        expected_dev = np.concatenate([shifted, np.zeros(1)]).astype(np.float32)
        np.testing.assert_allclose(schnell, linear, rtol=0, atol=1e-7)
        np.testing.assert_allclose(dev, expected_dev, rtol=0, atol=1e-6)

    def test_kontext_ids_fill_step_and_reference_limits(self):
        latent_ids = native_flux_core.prepare_latent_image_ids(32, 32)
        kontext_ids = native_flux_core.prepare_kontext_image_ids(32, 32)
        mx.eval(latent_ids, kontext_ids)
        self.assertTrue(np.all(np.array(latent_ids)[..., 0] == 0))
        self.assertTrue(np.all(np.array(kontext_ids)[..., 0] == 1))
        self.assertEqual(flux_nodes._flux_kontext_cache_fill_step(20), 3)
        self.assertEqual(flux_nodes._flux_kontext_cache_fill_step(2), 2)

        context_contract = contract(source="flux1-kontext-dev.safetensors")
        self.assertTrue(flux_nodes._validate_flux1_reference_contract(context_contract, [object()]))
        with self.assertRaisesRegex(RuntimeError, "one reference latent"):
            flux_nodes._validate_flux1_reference_contract(context_contract, [object(), object()])
        with self.assertRaisesRegex(RuntimeError, "Kontext model checkpoint"):
            flux_nodes._validate_flux1_reference_contract(contract(), [object()])

    def test_runtime_reset_clears_cross_run_cache_state(self):
        transformer = native_flux_core.FluxNativeTransformer(weights={})
        transformer.forecast_single_linear2_plan = {3: ("linear", {0})}
        transformer.set_kontext_kv_cache(True, 4)
        flux_nodes._reset_transformer_runtime_state(transformer)
        self.assertEqual(transformer.forecast_single_linear2_history, {})
        self.assertFalse(transformer.kontext_kv_cache_enabled)
        self.assertEqual(transformer.teacache_mode, "off")


class Flux1ConditioningAndAdapterTests(unittest.TestCase):
    def setUp(self):
        flux_nodes.FLUX_TEXT_CONDITIONING_CACHE.clear()

    def tearDown(self):
        flux_nodes.FLUX_TEXT_CONDITIONING_CACHE.clear()

    def test_separate_clip_and_t5_prompts_and_cache_identity(self):
        class FakeClip:
            def __init__(self):
                self.calls = []

            def tokenize(self, text):
                return {"l": ("clip", text), "t5xxl": ("t5", text)}

            def encode_from_tokens_scheduled(self, tokens, add_dict):
                self.calls.append((dict(tokens), dict(add_dict)))
                return [["encoded", {"guidance": add_dict["guidance"]}]]

        clip = FakeClip()
        handle = {"type": "flux1", "clip": clip, "cache_key": "base"}
        node = flux_nodes.SDMLX_CLIPTextEncodeFlux()
        first = node.encode(handle, "short", "long prompt", 3.5)[0]
        second = node.encode(handle, "short", "long prompt", 3.5)[0]
        self.assertIs(first, second)
        self.assertEqual(len(clip.calls), 1)
        self.assertEqual(clip.calls[0][0]["l"], ("clip", "short"))
        self.assertEqual(clip.calls[0][0]["t5xxl"], ("t5", "long prompt"))
        self.assertEqual(clip.calls[0][1]["guidance"], 3.5)

        patched = dict(handle, cache_key="base:lora")
        node.encode(patched, "short", "long prompt", 3.5)
        self.assertEqual(len(clip.calls), 2)

    def test_lora_mapping_clone_and_duplicate_patch_guard(self):
        targets = flux_nodes._flux_lora_targets("lora_unet_double_blocks_0_img_attn_qkv")
        self.assertIn(("double_blocks.0.img_attn.qkv.weight", None, None), targets)
        self.assertTrue(flux_nodes._flux1_lora_text_key_candidate("lora_te1_text_model_encoder_layers_0.alpha"))

        source = Path("/tmp/flux1-dev.safetensors")
        transformer = native_flux_core.FluxNativeTransformer(weights={})
        model = flux_nodes.SDMLXFluxNativeModel(
            source.name,
            source,
            transformer,
            mx.float16,
            flux_nodes.ModelConfig.FLUX1_DEV,
            0,
            0,
            model_family="dev",
            checkpoint_contract=contract(),
            source_path=source,
        )
        cloned = flux_nodes._clone_flux_model_for_patch(model)
        self.assertIs(cloned.checkpoint_contract, model.checkpoint_contract)
        self.assertEqual(cloned.source_path, source)

        cloned.transformer.lora_sources = ["Hyper-Flux.1-Dev-4-step-Lora.safetensors"]
        with self.assertRaisesRegex(RuntimeError, "already baked"):
            flux_nodes._assert_no_duplicate_flux_acceleration_patch_lora(
                cloned,
                "Hyper FLUX Dev 4-step",
            )

    def test_forecast_linear_path_has_explicit_context(self):
        transformer = native_flux_core.FluxNativeTransformer(weights={})
        transformer.forecast_step_index = 3
        history = [(1, mx.zeros((1, 1, 2))), (2, mx.ones((1, 1, 2)))]
        forecast, gain = transformer.raw_single_forecast_value(
            history,
            "linear",
            index=0,
            scope="linear2",
        )
        self.assertIsNotNone(forecast)
        self.assertIsInstance(gain, float)
        mx.eval(forecast)

    def test_product_transformer_has_no_scout_branches(self):
        source = inspect.getsource(native_flux_core.FluxNativeTransformer)
        self.assertNotIn("token_scout", source)
        self.assertNotIn("attention_scout", source)
        self.assertNotIn("linear2_scout", source)

    def test_lua_disabled_passes_latent_through(self):
        class FakeVAE:
            def decode(self, latents):
                return mx.zeros((1, 3, 16, 16), dtype=mx.float16)

        samples = {"samples": torch.zeros((1, 16, 2, 2)), "sdmlx_lua_scale": "x4"}
        with mock.patch.object(flux_nodes, "_resolve_flux_vae", return_value=(FakeVAE(), "ae")), mock.patch.object(
            flux_nodes,
            "_apply_vae_cache_limit",
        ):
            image, output = flux_nodes.SDMLXFluxLUAAdapter().upscale(
                samples,
                {"type": "flux1"},
                "2x",
                "auto",
                "float32",
                enabled=False,
            )
        self.assertTrue(torch.equal(output["samples"], samples["samples"]))
        self.assertNotIn("sdmlx_lua_scale", output)
        self.assertEqual(tuple(image.shape), (1, 16, 16, 3))

    def test_lua_skips_decode_when_image_output_is_unconnected(self):
        samples = {
            "samples": torch.zeros((1, 16, 2, 2)),
            "sdmlx_original_size": (32, 32),
        }
        with mock.patch.object(flux_nodes, "_resolve_flux_vae") as resolve:
            image, output = flux_nodes.SDMLXFluxLUAAdapter().upscale(
                samples,
                {"type": "flux1"},
                "2x",
                "auto",
                "float32",
                enabled=False,
                unique_id="7",
                prompt={"9": {"inputs": {"latent": ["7", 1]}}},
            )
        self.assertIsNone(image)
        self.assertTrue(torch.equal(output["samples"], samples["samples"]))
        self.assertEqual(output["sdmlx_original_size"], (32, 32))
        resolve.assert_not_called()

    def test_lua_enabled_returns_upscaled_latent_without_decode(self):
        samples = {"samples": torch.zeros((1, 16, 2, 2)), "marker": "kept"}
        upscaled = torch.ones((1, 16, 4, 4))
        with mock.patch.object(flux_nodes, "_cached_flux_lua_model", return_value=object()), mock.patch.object(
            flux_nodes,
            "lua_upscale_latent",
            return_value=upscaled,
        ), mock.patch.object(flux_nodes, "_torch_sync"), mock.patch.object(
            flux_nodes,
            "_resolve_flux_vae",
        ) as resolve:
            image, output = flux_nodes.SDMLXFluxLUAAdapter().upscale(
                samples,
                {"type": "flux1"},
                "2x",
                "cpu",
                "float32",
                enabled=True,
                unique_id="7",
                prompt={"9": {"inputs": {"latent": ["7", 1]}}},
            )
        self.assertIsNone(image)
        self.assertTrue(torch.equal(output["samples"], upscaled))
        self.assertEqual(output["marker"], "kept")
        self.assertEqual(output["sdmlx_lua_scale"], "x2")
        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
