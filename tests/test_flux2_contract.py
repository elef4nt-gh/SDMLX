from __future__ import annotations

import json
import os
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import mlx.core as mx
from huggingface_hub.errors import GatedRepoError

from .. import flux2_nodes


def _reference(value: float) -> dict:
    return {
        "latents": mx.full((1, 4, 2), value, dtype=mx.float32),
        "ids": mx.zeros((1, 4, 4), dtype=mx.float32),
        "width": 32,
        "height": 32,
    }


def _flux2_9b_config(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        text_encoder_overrides={
            "hidden_size": 4096,
            "intermediate_size": 12288,
        },
    )


def _write_legacy_text_encoder_cache(
    cache_root: Path,
    source_path: Path,
    *,
    entry_name: str,
    model_name: str,
    mtime_ns: int,
) -> Path:
    entry_root = cache_root / entry_name
    entry_root.mkdir(parents=True)
    cache_path = entry_root / "weights.safetensors"
    cache_path.write_bytes(b"prepared-cache")
    os.utime(cache_path, ns=(mtime_ns, mtime_ns))
    manifest = {
        "format": flux2_nodes._FLUX2_TEXT_ENCODER_CACHE_MANIFEST_FORMAT,
        "kind": "prepared_text_encoder",
        "model_family": flux2_nodes.FLUX2_MODEL_FAMILY,
        "component": "text_encoder",
        "cache_format": flux2_nodes._FLUX2_TEXT_ENCODER_CACHE_FORMAT,
        "source_identity": flux2_nodes._flux2_file_identity(source_path),
        "model_config": {
            "model_name": model_name,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 0,
            "num_attention_heads": 0,
            "num_key_value_heads": 0,
            "head_dim": 128,
        },
    }
    with (entry_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return cache_path


class Flux2EnhancedReferenceContractTests(unittest.TestCase):
    sampler = flux2_nodes.SDMLXFlux2KleinEnhancedEditSampler

    def test_positive_references_win_without_adding_negative(self):
        positive = {"reference_latents": [_reference(1.0), _reference(2.0)]}
        negative = {"reference_latents": [_reference(3.0)]}
        refs = self.sampler._conditioning_refs_ordered(positive, negative)
        self.assertEqual(len(refs), 2)
        self.assertEqual(
            [ref["_sdmlx_reference_source"] for ref in refs],
            ["conditioning_positive", "conditioning_positive"],
        )
        self.assertEqual([ref["_sdmlx_reference_slot"] for ref in refs], [1, 2])

    def test_negative_references_are_fallback_only(self):
        refs = self.sampler._conditioning_refs_ordered(
            {"reference_latents": []},
            {"reference_latents": [_reference(3.0)]},
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["_sdmlx_reference_source"], "conditioning_negative")

    def test_direct_slot_gaps_keep_original_slot_and_final_sequence(self):
        optional = {"reference_image_1": object(), "reference_image_3": object()}

        def encoded(_image, _vae, *, index):
            ref = _reference(float(index))
            ref["encoded_index"] = index
            return ref

        with mock.patch.object(flux2_nodes, "_flux2_enhanced_direct_reference", side_effect=encoded):
            direct = self.sampler._direct_references({}, 2, optional)
        refs = self.sampler._conditioning_refs_ordered(
            {"reference_latents": [_reference(1.0), _reference(2.0)]},
            {},
        )
        refs.extend(direct)
        self.sampler._finalize_reference_sequence(refs)
        self.assertEqual([ref["_sdmlx_reference_slot"] for ref in direct], [1, 3])
        self.assertEqual([ref["encoded_index"] for ref in direct], [2, 3])
        self.assertEqual([ref["_sdmlx_reference_sequence"] for ref in refs], [1, 2, 3, 4])

    def test_masks_apply_only_to_matching_direct_slots(self):
        conditioning = _reference(0.0)
        conditioning.update(
            {"_sdmlx_reference_source": "conditioning_positive", "_sdmlx_reference_slot": 1}
        )
        direct_1 = _reference(1.0)
        direct_1.update({"_sdmlx_reference_source": "direct", "_sdmlx_reference_slot": 1})
        direct_3 = _reference(3.0)
        direct_3.update({"_sdmlx_reference_source": "direct", "_sdmlx_reference_slot": 3})
        mask_1 = object()
        mask_3 = object()
        seen = []

        def mask_indices(mask, _ref, _count, _threshold):
            seen.append(mask)
            return None

        with mock.patch.object(flux2_nodes, "_flux2_mask_keep_indices", side_effect=mask_indices):
            self.sampler._mask_keep_list(
                [conditioning, direct_1, direct_3],
                [4, 4, 4],
                {"subject_mask_1": mask_1, "subject_mask_3": mask_3},
                0.5,
            )
        self.assertEqual(seen, [None, mask_1, mask_3])

    def test_native_only_is_a_feature_transfer_no_op(self):
        state = self.sampler._build_state(
            references=[_reference(1.0)],
            text_token_count=8,
            enhance_preset="native_only",
            identity_strength=1.0,
            reference_balance=0.2,
            color_anchor_strength=0.0,
            mask_behavior="focus_only",
            similarity_floor=0.2,
            softmax_temperature=0.07,
            mask_threshold=0.5,
            double_blocks="0-7:mid_img=0.5",
            single_blocks="0:mid_img=0.5",
            reference_indices="all",
            active_scale=1.0,
            per_token_whiten=0.0,
            early_layer_scale=1.0,
            mid_layer_scale=1.0,
            late_layer_scale=1.0,
            optional={},
            debug=False,
        )
        self.assertEqual(state.identity_strength, 0.0)
        self.assertEqual(state.double_map, {})
        self.assertEqual(state.single_map, {})
        self.assertEqual((state.text_scale, state.reference_scale), (1.0, 1.0))


class Flux2PreparedTextEncoderCacheContractTests(unittest.TestCase):
    def test_official_9b_variants_share_canonical_cache_key(self):
        configs = [
            _flux2_9b_config("black-forest-labs/FLUX.2-klein-base-9B"),
            _flux2_9b_config("black-forest-labs/FLUX.2-klein-9B"),
            _flux2_9b_config("black-forest-labs/FLUX.2-klein-9b-kv"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "qwen_3_8b_fp8mixed.safetensors"
            source.write_bytes(b"same-text-encoder")
            cache_root = root / "cache"
            with mock.patch.object(
                flux2_nodes,
                "_flux2_text_encoder_prepared_cache_dir",
                return_value=cache_root,
            ):
                paths = {
                    flux2_nodes._flux2_text_encoder_cache_path(source, config)
                    for config in configs
                }
        self.assertEqual(len(paths), 1)

    def test_legacy_resolution_selects_newest_matching_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "qwen_3_8b_fp8mixed.safetensors"
            source.write_bytes(b"same-text-encoder")
            cache_root = root / "cache"
            older = _write_legacy_text_encoder_cache(
                cache_root,
                source,
                entry_name="older",
                model_name="black-forest-labs/FLUX.2-klein-base-9B",
                mtime_ns=1_000_000_000,
            )
            newest = _write_legacy_text_encoder_cache(
                cache_root,
                source,
                entry_name="newest",
                model_name="black-forest-labs/FLUX.2-klein-9b-kv",
                mtime_ns=3_000_000_000,
            )
            _write_legacy_text_encoder_cache(
                cache_root,
                source,
                entry_name="middle",
                model_name="black-forest-labs/FLUX.2-klein-9B",
                mtime_ns=2_000_000_000,
            )
            with mock.patch.object(
                flux2_nodes,
                "_flux2_text_encoder_prepared_cache_dir",
                return_value=cache_root,
            ):
                resolved = flux2_nodes._flux2_resolve_text_encoder_cache_path(
                    source,
                    _flux2_9b_config("black-forest-labs/FLUX.2-klein-9B"),
                )
        self.assertNotEqual(resolved, older)
        self.assertEqual(resolved, newest)

    def test_legacy_cache_metadata_is_valid_for_another_9b_variant(self):
        source_identity = {
            "source_size": 123,
            "source_mtime_ns": 456,
            "source_content_digest": "same-digest",
        }
        legacy_metadata = {
            "cache_format": flux2_nodes._FLUX2_TEXT_ENCODER_CACHE_FORMAT,
            "source_size": "123",
            "source_mtime_ns": "456",
            "source_content_digest": "same-digest",
            "model_config": json.dumps(
                {
                    "model_name": "black-forest-labs/FLUX.2-klein-9b-kv",
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "num_hidden_layers": 0,
                    "num_attention_heads": 0,
                    "num_key_value_heads": 0,
                    "head_dim": 128,
                }
            ),
        }
        self.assertTrue(
            flux2_nodes._flux2_text_encoder_cache_metadata_matches(
                legacy_metadata,
                source_identity,
                _flux2_9b_config("black-forest-labs/FLUX.2-klein-base-9B"),
            )
        )

    def test_changed_source_mtime_invalidates_legacy_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "encoder.safetensors"
            source.write_bytes(b"same-source")
            cache_root = root / "cache"
            _write_legacy_text_encoder_cache(
                cache_root,
                source,
                entry_name="legacy",
                model_name="black-forest-labs/FLUX.2-klein-9B",
                mtime_ns=1_000_000_000,
            )
            source_stat = source.stat()
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1),
            )
            with mock.patch.object(
                flux2_nodes,
                "_flux2_text_encoder_prepared_cache_dir",
                return_value=cache_root,
            ):
                candidates = flux2_nodes._flux2_text_encoder_cache_candidates(
                    source,
                    _flux2_9b_config("black-forest-labs/FLUX.2-klein-9B"),
                )
        self.assertEqual(candidates, [])

    def test_different_encoder_geometry_does_not_reuse_9b_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "encoder.safetensors"
            source.write_bytes(b"same-source")
            cache_root = root / "cache"
            _write_legacy_text_encoder_cache(
                cache_root,
                source,
                entry_name="nine-b",
                model_name="black-forest-labs/FLUX.2-klein-9B",
                mtime_ns=1_000_000_000,
            )
            config_4b = SimpleNamespace(
                model_name="black-forest-labs/FLUX.2-klein-4B",
                text_encoder_overrides={
                    "hidden_size": 2560,
                    "intermediate_size": 9728,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                },
            )
            with mock.patch.object(
                flux2_nodes,
                "_flux2_text_encoder_prepared_cache_dir",
                return_value=cache_root,
            ):
                candidates = flux2_nodes._flux2_text_encoder_cache_candidates(
                    source,
                    config_4b,
                )
        self.assertEqual(candidates, [])


class Flux2PackageAccessContractTests(unittest.TestCase):
    def test_gated_tokenizer_download_has_concise_actionable_error(self):
        response = httpx.Response(
            401,
            request=httpx.Request(
                "HEAD",
                "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
            ),
        )
        gated = GatedRepoError("restricted", response=response)

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("huggingface_hub.snapshot_download", side_effect=gated):
                try:
                    flux2_nodes._flux2_download_tokenizer_assets(
                        Path(temp_dir),
                        "flux2-klein-9b",
                    )
                except RuntimeError as exc:
                    rendered = "".join(traceback.format_exception(exc))
                else:
                    self.fail("gated download did not fail")

        self.assertIn("Hugging Face access is required", rendered)
        self.assertIn("black-forest-labs/FLUX.2-klein-9B", rendered)
        self.assertIn("Git credentials and Xcode are not required", rendered)
        self.assertNotIn("GatedRepoError", rendered)


if __name__ == "__main__":
    unittest.main()
