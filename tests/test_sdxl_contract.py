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


if __name__ == "__main__":
    unittest.main()
