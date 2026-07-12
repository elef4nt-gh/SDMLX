from __future__ import annotations

import unittest
from unittest import mock

import mlx.core as mx

from .. import flux2_nodes


def _reference(value: float) -> dict:
    return {
        "latents": mx.full((1, 4, 2), value, dtype=mx.float32),
        "ids": mx.zeros((1, 4, 4), dtype=mx.float32),
        "width": 32,
        "height": 32,
    }


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


if __name__ == "__main__":
    unittest.main()
