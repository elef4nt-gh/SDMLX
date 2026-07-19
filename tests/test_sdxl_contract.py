import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from .. import nodes


def _pair(prefix, rank=2, in_dim=4, out_dim=4, layout="down_up"):
    down = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim) / 10
    up = torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank) / 10
    if layout == "ab":
        return {f"{prefix}.lora_A.weight": down, f"{prefix}.lora_B.weight": up}
    return {f"{prefix}.lora_down.weight": down, f"{prefix}.lora_up.weight": up}


class SDXLLoRAClassificationTests(unittest.TestCase):
    def setUp(self):
        nodes.LORA_MODULES_CACHE.clear()
        nodes.SDXL_CLIP_LORA_MODULES_CACHE.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name, state):
        path = self.root / name
        save_file(state, str(path))
        return str(path)

    def test_four_lora_classes(self):
        unet = _pair("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q")
        clip = _pair("lora_te1_text_model_encoder_layers_0_self_attn_q_proj")
        cases = {
            nodes.SDXL_LORA_UNET_ONLY: unet,
            nodes.SDXL_LORA_CLIP_ONLY: clip,
            nodes.SDXL_LORA_MIXED: {**unet, **clip},
            nodes.SDXL_LORA_INCOMPATIBLE: _pair("lora_transformer_blocks_0_attn_to_q"),
        }
        for expected, state in cases.items():
            with self.subTest(expected=expected):
                result = nodes.classify_sdxl_lora(self._write(f"{expected}.safetensors", state))
                self.assertEqual(result["kind"], expected)

    def test_diffusers_ab_layout_maps_unet_and_clip(self):
        state = {}
        state.update(_pair("unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q", layout="ab"))
        state.update(_pair("text_encoder.text_model.encoder.layers.0.self_attn.q_proj", layout="ab"))
        result = nodes.classify_sdxl_lora(self._write("diffusers_ab.safetensors", state))
        self.assertEqual(result["kind"], nodes.SDXL_LORA_MIXED)
        self.assertIn("diffusers_ab", result["layouts"])

    def test_clip_only_never_enters_model_stack(self):
        path = self._write(
            "clip_only.safetensors",
            _pair("lora_te1_text_model_encoder_layers_0_self_attn_q_proj"),
        )
        model = {"weights": {}, "cache_key": "base", "model_family": "sdxl"}
        clip = {"clip_l": {}, "clip_g": {}, "cache_key": "clip", "type": "sdxl"}
        with mock.patch.object(
            nodes,
            "apply_sdxl_clip_lora_to_mlx_clip",
            return_value=({**clip, "cache_key": "patched"}, {"applied": 1, "modules": 1}),
        ):
            patched_model, patched_clip, result = nodes.apply_sdxl_lora_contract(
                model, clip, "clip", path, 1.0
            )
        self.assertNotIn("loras", patched_model)
        self.assertEqual(patched_clip["cache_key"], "patched")
        self.assertEqual(result["kind"], nodes.SDXL_LORA_CLIP_ONLY)

    def test_clip_only_requires_clip_and_rejects_scheduler(self):
        path = self._write(
            "clip_only.safetensors",
            _pair("lora_te1_text_model_encoder_layers_0_self_attn_q_proj"),
        )
        model = {"weights": {}, "cache_key": "base", "model_family": "sdxl"}
        with self.assertRaisesRegex(ValueError, "Connect mlx_clip"):
            nodes.apply_sdxl_lora_contract(model, None, "clip", path, 1.0)
        with self.assertRaisesRegex(ValueError, "Scheduler controls only"):
            nodes.apply_sdxl_lora_contract(model, {}, "clip", path, 1.0, schedule={"mode": "linear"})

    def test_mixed_without_clip_keeps_unet_part(self):
        state = {}
        state.update(_pair("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"))
        state.update(_pair("lora_te1_text_model_encoder_layers_0_self_attn_q_proj"))
        path = self._write("mixed.safetensors", state)
        model = {"weights": {}, "cache_key": "base", "model_family": "sdxl"}
        patched_model, patched_clip, result = nodes.apply_sdxl_lora_contract(model, None, "mixed", path, 0.75)
        self.assertIsNone(patched_clip)
        self.assertEqual(len(patched_model["loras"]), 1)
        self.assertEqual(patched_model["loras"][0]["sdxl_lora_kind"], nodes.SDXL_LORA_MIXED)
        self.assertTrue(result["model_added"])


class SDXLPackageAndFamilyTests(unittest.TestCase):
    def test_runtime_family_resolution(self):
        self.assertEqual(nodes.sdmlx_runtime_family({"model_family": "sdxl"}), "sdxl")
        self.assertEqual(nodes.sdmlx_runtime_family({"model_family": "qwen-image-edit"}), "qwen")
        self.assertEqual(nodes.sdmlx_runtime_family({"model_family": "flux2-klein"}), "flux2-klein")
        self.assertEqual(nodes.sdmlx_runtime_family({"weights": {}, "cache_key": "legacy"}), "sdxl")
        with self.assertRaisesRegex(RuntimeError, "received flux2-klein"):
            nodes.require_sdmlx_family({"model_family": "flux2-klein"}, {"sdxl"}, "test")

    def test_manifest_backfill_without_weight_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "old.sdmlx"
            package.mkdir()
            manifest = {"package_format": "sdmlx-package-v3", "cache_version": nodes.SDMLX_CACHE_VERSION}
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(nodes, "mark_macos_package"):
                updated = nodes._backfill_sdxl_manifest_family(str(package), manifest)
            self.assertEqual(updated["base_model_family"], "sdxl")
            self.assertEqual(json.loads((package / "manifest.json").read_text())["base_model_family"], "sdxl")
            self.assertEqual(list(package.iterdir()), [package / "manifest.json"])

    def test_contradictory_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "bad.sdmlx"
            package.mkdir()
            manifest = {"package_format": "sdmlx-package-v3", "base_model_family": "qwen"}
            with self.assertRaisesRegex(RuntimeError, "incompatible model family"):
                nodes._backfill_sdxl_manifest_family(str(package), manifest)


class SDXLInpaintDetailerCompositeTests(unittest.TestCase):
    def test_unmasked_pixels_remain_pixel_identical(self):
        image = torch.arange(6 * 6 * 3, dtype=torch.float32).reshape(1, 6, 6, 3) / 108.0
        rendered = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        mask[:, :, 1:3, 1:3] = 1.0

        output, composite_mask = nodes.composite_inpaint_detailer_crop(
            image,
            rendered,
            mask,
            (1, 1, 5, 5),
            crop_blend=0,
        )

        expected = image.clone()
        expected[:, 2:4, 2:4, :] = 1.0
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(tuple(composite_mask.shape), (1, 4, 4, 1))
        self.assertTrue(torch.equal(output[:, 1, 1:5, :], image[:, 1, 1:5, :]))

    def test_crop_blend_feathers_the_input_mask_not_the_full_crop(self):
        image = torch.zeros((1, 9, 9, 3), dtype=torch.float32)
        rendered = torch.ones((1, 7, 7, 3), dtype=torch.float32)
        mask = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
        mask[:, :, 3, 3] = 1.0

        output, composite_mask = nodes.composite_inpaint_detailer_crop(
            image,
            rendered,
            mask,
            (1, 1, 8, 8),
            crop_blend=1,
        )

        self.assertGreater(float(composite_mask[0, 3, 3, 0]), 0.0)
        self.assertEqual(float(composite_mask[0, 0, 0, 0]), 0.0)
        self.assertTrue(torch.equal(output[:, 0, :, :], image[:, 0, :, :]))
        self.assertTrue(torch.equal(output[:, :, 0, :], image[:, :, 0, :]))


class SDXLInpaintConditioningCompositeTests(unittest.TestCase):
    def test_padding_continues_mask_that_touches_image_edge(self):
        pixels = torch.zeros((1, 1010, 64, 3), dtype=torch.float32)
        mask = torch.zeros((1, 1, 1010, 64), dtype=torch.float32)
        mask[:, :, 900:, :] = 1.0

        _, padded_mask, crop = nodes.pad_inpaint_pixels_and_mask(
            pixels,
            mask,
            target_width=64,
            target_height=1024,
        )
        grown = nodes.grow_mask_tensor(padded_mask, 4)
        latent_mask = torch.nn.functional.interpolate(grown, size=(128, 8), mode="area")

        self.assertEqual(crop, (0, 7, 64, 1010))
        self.assertTrue(torch.equal(padded_mask[:, :, -7:, :], torch.ones((1, 1, 7, 64))))
        self.assertEqual(float(latent_mask[0, 0, -1, 0]), 1.0)

    def test_padding_stays_unmasked_when_mask_ends_before_image_edge(self):
        pixels = torch.zeros((1, 1010, 64, 3), dtype=torch.float32)
        mask = torch.zeros((1, 1, 1010, 64), dtype=torch.float32)
        mask[:, :, 900:1000, :] = 1.0

        _, padded_mask, _ = nodes.pad_inpaint_pixels_and_mask(
            pixels,
            mask,
            target_width=64,
            target_height=1024,
        )

        self.assertEqual(float(torch.max(padded_mask[:, :, -17:, :])), 0.0)

    def test_conditioning_attaches_composite_metadata_only_with_noise_mask(self):
        pixels = torch.rand((1, 64, 64, 3), dtype=torch.float32)
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        mask[:, 20:44, 20:44] = 1.0
        latents = torch.zeros((1, 8, 8, 4), dtype=torch.float32)
        conditioning = {"model_family": "sdxl"}
        vae = {"model_family": "sdxl"}

        with mock.patch.object(nodes, "encode_pixels_to_latents", return_value=latents), mock.patch.object(
            nodes,
            "prepare_inpaint_mask",
            return_value=torch.ones((1, 8, 8, 1), dtype=torch.float32),
        ):
            _, _, masked = nodes.SDMLX_InpaintConditioning().encode(
                conditioning,
                conditioning,
                pixels,
                vae,
                mask,
                noise_mask=True,
            )
            _, _, unmasked = nodes.SDMLX_InpaintConditioning().encode(
                conditioning,
                conditioning,
                pixels,
                vae,
                mask,
                noise_mask=False,
            )

        self.assertTrue(masked["sdmlx_inpaint_pixel_composite"])
        self.assertTrue(torch.equal(masked["sdmlx_inpaint_source_image"], pixels))
        self.assertEqual(tuple(masked["sdmlx_inpaint_source_mask"].shape), (1, 64, 64, 1))
        self.assertNotIn("sdmlx_inpaint_pixel_composite", unmasked)
        self.assertNotIn("sdmlx_inpaint_source_image", unmasked)
        self.assertNotIn("sdmlx_inpaint_source_mask", unmasked)

    def test_decode_composites_only_the_input_mask(self):
        source = torch.arange(6 * 6 * 3, dtype=torch.float32).reshape(1, 6, 6, 3) / 108.0
        decoded = torch.ones((1, 6, 6, 3), dtype=torch.float32)
        mask = torch.zeros((1, 6, 6, 1), dtype=torch.float32)
        mask[:, 2:4, 2:4, :] = 1.0
        latent = {
            "sdmlx_inpaint_pixel_composite": True,
            "sdmlx_inpaint_source_image": source,
            "sdmlx_inpaint_source_mask": mask,
        }

        output = nodes.composite_decoded_inpaint_image(decoded, latent)

        expected = source.clone()
        expected[:, 2:4, 2:4, :] = 1.0
        self.assertTrue(torch.equal(output, expected))

    def test_sdxl_decode_uses_inpaint_composite_metadata(self):
        source = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        source[:, :, :, 1] = 0.25
        decoded = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        mask = torch.zeros((1, 8, 8, 1), dtype=torch.float32)
        mask[:, 3:5, 3:5, :] = 1.0
        latent = {
            "samples": torch.zeros((1, 1, 1, 4), dtype=torch.float32),
            "sdmlx_inpaint_pixel_composite": True,
            "sdmlx_inpaint_source_image": source,
            "sdmlx_inpaint_source_mask": mask,
        }

        with mock.patch.object(nodes, "decode_latents", return_value=decoded):
            output = nodes.decode_mlx_latent_to_image(latent, {"model_family": "sdxl"})

        expected = source.clone()
        expected[:, 3:5, 3:5, :] = 1.0
        self.assertTrue(torch.equal(output, expected))

    def test_noise_mask_false_keeps_full_decode(self):
        decoded = torch.rand((1, 4, 4, 3), dtype=torch.float32)
        self.assertIs(nodes.composite_decoded_inpaint_image(decoded, {}), decoded)

    def test_sampler_metadata_contract_preserves_inpaint_source(self):
        source_image = torch.rand((1, 4, 4, 3), dtype=torch.float32)
        source_mask = torch.rand((1, 4, 4, 1), dtype=torch.float32)
        latent = {
            "samples": object(),
            "sdmlx_decode_crop": (0, 0, 4, 4),
            "sdmlx_inpaint_pixel_composite": True,
            "sdmlx_inpaint_source_image": source_image,
            "sdmlx_inpaint_source_mask": source_mask,
            "unrelated": "drop",
        }

        output = nodes.copy_sdmlx_latent_metadata(latent, {"samples": object()})

        self.assertIs(output["sdmlx_inpaint_source_image"], source_image)
        self.assertIs(output["sdmlx_inpaint_source_mask"], source_mask)
        self.assertNotIn("unrelated", output)

    def test_incomplete_composite_metadata_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "metadata is incomplete"):
            nodes.composite_decoded_inpaint_image(
                torch.zeros((1, 4, 4, 3), dtype=torch.float32),
                {"sdmlx_inpaint_pixel_composite": True},
            )


if __name__ == "__main__":
    unittest.main()
