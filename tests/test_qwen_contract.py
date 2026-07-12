from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
import torch

SUITE_ROOT = Path(__file__).resolve().parents[1]
SUITE_PARENT = SUITE_ROOT.parent
COMFY_ROOT = os.environ.get("COMFYUI_ROOT")
if COMFY_ROOT:
    sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, str(SUITE_PARENT))

from .. import nodes as suite_nodes  # noqa: E402
from .. import qwen_nodes  # noqa: E402
from ..sdmlx_qwen_native.utils.image_util import ImageUtil  # noqa: E402
from ..sdmlx_qwen_native.models.common.weights.mapping.weight_mapper import WeightMapper  # noqa: E402
from ..sdmlx_qwen_native.models.common.weights.loading.weight_loader import WeightLoader  # noqa: E402
from ..sdmlx_qwen_native.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition  # noqa: E402
from ..sdmlx_qwen_native.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping  # noqa: E402
from ..sdmlx_qwen_native.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit  # noqa: E402
from ..sdmlx_qwen_native.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402


class QwenIdentityTests(unittest.TestCase):
    def test_2511_marker_without_prefix(self):
        self.assertEqual(
            qwen_nodes.resolve_qwen_model_identity(transformer_keys={"__index_timestep_zero__"}),
            ("qwen-image-edit", "2511"),
        )

    def test_2511_marker_with_prefix(self):
        self.assertEqual(
            qwen_nodes.resolve_qwen_model_identity(
                transformer_keys={"model.diffusion_model.__index_timestep_zero__"}
            ),
            ("qwen-image-edit", "2511"),
        )

    def test_markerless_2511_original_source(self):
        self.assertEqual(
            qwen_nodes.resolve_qwen_model_identity(
                transformer_keys={"img_in.weight"},
                source_identifiers=("qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors",),
            ),
            ("qwen-image-edit", "2511"),
        )

    def test_2512_original_source(self):
        self.assertEqual(
            qwen_nodes.resolve_qwen_model_identity(
                source_identifiers=("qwen_image_2512_fp8_e4m3fn.safetensors",)
            ),
            ("qwen-image", "2512"),
        )

    def test_full_and_version_only_manifests(self):
        self.assertEqual(
            qwen_nodes.resolve_qwen_model_identity(
                manifest={"qwen_variant": "qwen-image", "model_version": "2512"}
            ),
            ("qwen-image", "2512"),
        )
        self.assertEqual(
            qwen_nodes.resolve_qwen_model_identity(manifest={"model_version": "2511"}),
            ("qwen-image-edit", "2511"),
        )

    def test_contradictory_and_ambiguous_identity_fail(self):
        with self.assertRaisesRegex(RuntimeError, "contradictory manifest identity"):
            qwen_nodes.resolve_qwen_model_identity(
                manifest={"qwen_variant": "qwen-image", "model_version": "2511"}
            )
        with self.assertRaisesRegex(RuntimeError, "identity is ambiguous"):
            qwen_nodes.resolve_qwen_model_identity(root_hint="runtime-roots/model-cache")

    def test_runtime_root_name_is_not_an_identity_hint(self):
        runtime_root = Path("/tmp/sdmlx/qwen/runtime-roots/qwen_image_2512-cache")
        self.assertIsNone(qwen_nodes._qwen_identity_root_hint(runtime_root))

    def test_manifest_backfill_does_not_touch_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir)
            manifest_path = package / "manifest.json"
            weight_path = package / "transformer" / "weights.safetensors"
            weight_path.parent.mkdir()
            weight_path.write_bytes(b"weight-sentinel")
            manifest = {
                "package_format": qwen_nodes.QWEN_PACKAGE_FORMAT,
                "model_family": qwen_nodes.QWEN_MODEL_FAMILY,
                "model_version": "2511",
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            updated = qwen_nodes._backfill_qwen_manifest_identity(
                package,
                manifest,
                "qwen-image-edit",
                "2511",
            )

            self.assertEqual(updated["qwen_variant"], "qwen-image-edit")
            self.assertEqual(weight_path.read_bytes(), b"weight-sentinel")
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["qwen_variant"], "qwen-image-edit")

    def test_dummy_package_uses_local_2512_source_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir) / "package.sdmlx"
            qwen_nodes.create_qwen_dummy_package(
                package,
                model_path="/models/qwen_image_2512_fp8_e4m3fn",
            )
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["qwen_variant"], "qwen-image")
            self.assertEqual(manifest["model_version"], "2512")


class QwenConditioningTests(unittest.TestCase):
    def test_negative_text_survives_negative_images(self):
        positive_image = object()
        positive = qwen_nodes._qwen_conditioning_entry("positive", [positive_image])
        negative = qwen_nodes._qwen_conditioning_entry("negative text", [object()])

        _entry, prompt, negative_prompt, images = qwen_nodes._qwen_sampling_conditioning(
            positive,
            negative,
        )

        self.assertEqual(prompt, "positive")
        self.assertEqual(negative_prompt, "negative text")
        self.assertIs(images[0], positive_image)

    def test_negative_prompt_fallback_and_missing_negative(self):
        positive = qwen_nodes._qwen_conditioning_entry("positive", [object()])
        fallback = qwen_nodes._qwen_conditioning_entry("", [])
        fallback["negative_prompt"] = "fallback"
        self.assertEqual(
            qwen_nodes._qwen_sampling_conditioning(positive, fallback)[2],
            "fallback",
        )
        self.assertEqual(qwen_nodes._qwen_sampling_conditioning(positive, None)[2], "")


class QwenSamplerTests(unittest.TestCase):
    def setUp(self):
        self.input_latent = {
            "samples": mx.zeros((1, 16, 8, 8)),
            "sdmlx_decode_crop": (1, 2, 60, 61),
            "sdmlx_original_size": (60, 61),
            "sdmlx_padded_size": (64, 64),
        }
        self.model = {
            "model_family": qwen_nodes.QWEN_MODEL_FAMILY,
            "qwen_variant": "qwen-image-edit",
            "model_version": "2511",
        }
        self.vae = {"type": qwen_nodes.QWEN_MODEL_FAMILY}
        self.conditioning = {"model_family": qwen_nodes.QWEN_MODEL_FAMILY}

    def test_advanced_default_contract_and_rejections(self):
        suite_nodes._validate_qwen_advanced_sampling(True, 0, 10000, 4, "disable")
        with self.assertRaisesRegex(RuntimeError, "partial step ranges"):
            suite_nodes._validate_qwen_advanced_sampling(True, 1, 4, 4, "disable")
        with self.assertRaisesRegex(RuntimeError, "add_noise=true"):
            suite_nodes._validate_qwen_advanced_sampling(False, 0, 4, 4, "disable")
        with self.assertRaisesRegex(RuntimeError, "leftover-noise"):
            suite_nodes._validate_qwen_advanced_sampling(True, 0, 4, 4, "enable")

    def test_standard_and_advanced_return_generated_latent(self):
        generated_image = object()
        generated_latent = {"samples": mx.ones((1, 16, 8, 8)), "model_family": qwen_nodes.QWEN_MODEL_FAMILY}
        with mock.patch.object(
            qwen_nodes,
            "sample_qwen_image_edit",
            return_value=(generated_image, generated_latent),
        ) as sampler:
            standard = suite_nodes.SDMLX_KSampler().sample(
                self.model, self.vae, self.conditioning, self.conditioning, self.input_latent, 42, 4, 1.0,
                "euler", "simple", False, 1.0, "None", 1.0, "off", False,
            )
            advanced = suite_nodes.SDMLX_KSamplerAdvanced().sample(
                self.model, self.vae, self.conditioning, self.conditioning, self.input_latent, True, 42, 4, 1.0,
                "euler", "simple", False, 0, 10000, "None", 1.0, "off", "disable", False,
            )

        self.assertIs(standard[0], generated_image)
        self.assertIs(standard[1], generated_latent)
        self.assertIs(advanced[1], generated_latent)
        self.assertEqual(sampler.call_count, 2)
        self.assertIs(sampler.call_args.kwargs["latent_metadata"], self.input_latent)

    def test_generated_latent_metadata_and_vae_decode_route(self):
        latent = mx.ones((1, 16, 8, 8))
        generated = SimpleNamespace(latent=latent)
        output = qwen_nodes._qwen_generated_latent_output(generated, self.input_latent)
        self.assertIs(output["samples"], latent)
        for key in ("sdmlx_decode_crop", "sdmlx_original_size", "sdmlx_padded_size"):
            self.assertEqual(output[key], self.input_latent[key])

        decoded = object()
        with mock.patch.object(qwen_nodes, "decode_qwen_latent_with_vae", return_value=decoded):
            routed = suite_nodes.decode_mlx_latent_to_image(
                {"samples": latent},
                {"type": qwen_nodes.QWEN_MODEL_FAMILY},
            )
        self.assertIs(routed, decoded)


class QwenNativeResultTests(unittest.TestCase):
    def test_latent_metadata_does_not_change_direct_image(self):
        decoded = mx.zeros((1, 3, 8, 8))
        latent = mx.ones((1, 16, 1, 1))
        config = SimpleNamespace(
            model_config=SimpleNamespace(),
            num_inference_steps=4,
            guidance=1.0,
            precision=mx.bfloat16,
            height=8,
            width=8,
            controlnet_strength=None,
        )
        common = {
            "decoded_latents": decoded,
            "config": config,
            "seed": 42,
            "prompt": "test",
            "quantization": 8,
            "generation_time": 0.0,
        }
        without_latent = ImageUtil.to_image(**common)
        with_latent = ImageUtil.to_image(**common, latent=latent)
        self.assertTrue(np.array_equal(np.array(without_latent.image), np.array(with_latent.image)))
        self.assertIs(with_latent.latent, latent)

    def test_both_native_qwen_variants_attach_unpacked_latent(self):
        variant_paths = (
            SUITE_ROOT / "sdmlx_qwen_native/models/qwen/variants/edit/qwen_image_edit.py",
            SUITE_ROOT / "sdmlx_qwen_native/models/qwen/variants/txt2img/qwen_image.py",
        )
        for variant_path in variant_paths:
            tree = ast.parse(variant_path.read_text(encoding="utf-8"))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_image"
            ]
            self.assertTrue(
                any(any(keyword.arg == "latent" for keyword in call.keywords) for call in calls),
                variant_path.name,
            )


class QwenPerformanceContractTests(unittest.TestCase):
    def test_suite_uses_in_memory_images_without_temporary_png(self):
        captured = {}

        class FakeModel:
            def generate_image(self, *, images=None, **kwargs):
                captured["images"] = images
                captured["kwargs"] = kwargs
                return SimpleNamespace(
                    image=Image.new("RGB", (8, 8), "white"),
                    latent=mx.zeros((1, 16, 1, 1)),
                )

        model_handle = {
            "model_family": qwen_nodes.QWEN_MODEL_FAMILY,
            "qwen_variant": "qwen-image-edit",
            "model_version": "2511",
            "model_path": "unused",
        }
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        positive = qwen_nodes._qwen_conditioning_entry("edit", [image])
        negative = qwen_nodes._qwen_conditioning_entry("", [])
        with mock.patch.object(qwen_nodes, "_load_qwen_model", return_value=FakeModel()), mock.patch.object(
            qwen_nodes,
            "ensure_qwen_acceleration_patch",
            return_value=None,
        ), mock.patch.object(
            qwen_nodes.tempfile,
            "TemporaryDirectory",
            side_effect=AssertionError("temporary PNG path used"),
        ):
            output, latent = qwen_nodes.sample_qwen_image_edit(
                model_handle,
                positive,
                negative,
                8,
                8,
                42,
                1,
                1.0,
                "linear",
                None,
            )
        self.assertEqual(tuple(output.shape), (1, 8, 8, 3))
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 1, 1))
        self.assertEqual(len(captured["images"]), 1)
        self.assertIsInstance(captured["images"][0], Image.Image)
        self.assertNotIn("image_paths", captured["kwargs"])

    def test_vlm_cache_reuses_images_and_unchanged_negative_prompt(self):
        class FakeTokenizer:
            use_picture_prefix = True
            last_template_drop_idx = None

            def __init__(self):
                self.calls = 0

            def tokenize_with_image(self, *_args, **_kwargs):
                self.calls += 1
                return (
                    mx.zeros((1, 2), dtype=mx.int32),
                    mx.ones((1, 2), dtype=mx.int32),
                    mx.zeros((1, 3, 2, 2)),
                    mx.ones((1, 3), dtype=mx.int32),
                )

        class FakeEncoder:
            def __init__(self):
                self.calls = 0

            def __call__(self, **_kwargs):
                self.calls += 1
                return mx.full((1, 2, 4), self.calls), mx.ones((1, 2))

        tokenizer = FakeTokenizer()
        encoder = FakeEncoder()
        runtime = SimpleNamespace(
            tokenizers={"qwen_vl": tokenizer},
            qwen_vl_encoder=encoder,
            model_config=SimpleNamespace(model_name="Qwen-Image-Edit-2511"),
            _vlm_conditioning_cache={},
            _vlm_conditioning_cache_order=[],
            _vlm_conditioning_cache_max=6,
        )
        image = Image.new("RGB", (8, 8), "red")
        args = (runtime, "first", "negative", [image], None)
        QwenImageEdit._encode_prompts_with_images(*args)
        QwenImageEdit._encode_prompts_with_images(*args)
        self.assertEqual((tokenizer.calls, encoder.calls), (2, 2))
        QwenImageEdit._encode_prompts_with_images(runtime, "second", "negative", [image], None)
        self.assertEqual((tokenizer.calls, encoder.calls), (3, 3))

    def test_reference_vae_cache_keys_pil_image_content(self):
        QwenEditUtil.clear_image_conditioning_cache()
        vae = object()
        first = Image.new("RGB", (8, 8), "blue")
        second = first.copy()
        util_module = sys.modules[QwenEditUtil.__module__]
        with mock.patch.object(
            util_module.LatentCreator,
            "encode_image",
            return_value=mx.zeros((1, 16, 2, 2)),
        ) as encode, mock.patch.object(
            util_module.QwenLatentCreator,
            "pack_latents",
            return_value=mx.zeros((1, 4, 64)),
        ):
            one = QwenEditUtil._cached_image_conditioning(vae, first, 32, 32, None)
            two = QwenEditUtil._cached_image_conditioning(vae, second, 32, 32, None)
        self.assertIs(one[0], two[0])
        self.assertIs(one[1], two[1])
        self.assertEqual(one[2], two[2])
        self.assertEqual(encode.call_count, 1)
        QwenEditUtil.clear_image_conditioning_cache()


class QwenVAEContractTests(unittest.TestCase):
    def test_public_qwen_decode_uses_native_direct_image_rounding(self):
        decoded = mx.array(
            [[[[[-1.0, -0.75], [0.0, 0.75]]], [[[1.0, 0.25], [-0.25, -0.5]]], [[[0.1, -0.1], [0.5, -0.9]]]]]
        )
        direct = qwen_nodes._pil_to_image_tensor(ImageUtil._decoded_to_pil(decoded))
        public = qwen_nodes._qwen_decoded_to_image_tensor(decoded)

        self.assertTrue(np.array_equal(direct.numpy(), public.numpy()))

    def test_qwen_vae_component_preserves_bfloat16_source_bits(self):
        component = next(item for item in QwenWeightDefinition.get_components() if item.name == "vae")
        self.assertEqual(component.loading_mode, "single_mlx")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vae_dir = root / "vae"
            vae_dir.mkdir()
            mx.save_safetensors(
                str(vae_dir / "weights.safetensors"),
                {
                    "decoder.conv1.weight": mx.ones((2, 2, 1, 1, 1), dtype=mx.bfloat16),
                    "unrelated.weight": mx.ones((1,), dtype=mx.bfloat16),
                },
            )
            loaded = WeightLoader._load_safetensors(
                vae_dir,
                component.loading_mode,
                weight_prefix_filters=component.weight_prefix_filters,
            )

        self.assertEqual(set(loaded), {"decoder.conv1.weight"})
        self.assertEqual(loaded["decoder.conv1.weight"].dtype, mx.bfloat16)

    def test_comfy_wan21_mapping_covers_complete_native_qwen_vae(self):
        mapping = QwenWeightMapping.get_comfy_vae_mapping()
        flat = WeightMapper._build_flat_mapping(mapping)
        destinations = [target for targets in flat.values() for target, _transform in targets]

        self.assertEqual(len(flat), 192)
        self.assertEqual(len(destinations), 192)
        self.assertEqual(len(set(destinations)), 192)
        self.assertIn("conv1.weight", flat)
        self.assertIn("conv2.weight", flat)
        self.assertIn("decoder.upsamples.14.residual.6.weight", flat)
        self.assertIn("encoder.downsamples.8.time_conv.weight", flat)
        self.assertNotIn("encoder.downsamples.5.time_conv.weight", flat)

    def test_comfy_wan21_mapping_applies_mlx_layout_transforms(self):
        raw = {
            "conv2.weight": mx.arange(16).reshape(2, 2, 1, 2, 2),
            "decoder.middle.1.proj.weight": mx.arange(16).reshape(2, 2, 2, 2),
            "decoder.head.0.gamma": mx.ones((2, 1, 1, 1)),
        }
        mapped = WeightMapper.apply_mapping(raw, QwenWeightMapping.get_comfy_vae_mapping())

        self.assertEqual(tuple(mapped["post_quant_conv"]["conv3d"]["weight"].shape), (2, 1, 2, 2, 2))
        self.assertEqual(
            tuple(mapped["decoder"]["mid_block"]["attentions"][0]["proj"]["weight"].shape),
            (2, 2, 2, 2),
        )
        self.assertEqual(tuple(mapped["decoder"]["norm_out"]["weight"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
