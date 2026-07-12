from __future__ import annotations

import unittest

from .. import nodes


def _handle(family: str) -> dict:
    if family == "sdxl":
        return {"model_family": "sdxl", "type": "sdxl", "weights": {}, "cache_key": "sdxl"}
    if family == "qwen":
        return {"model_family": "qwen-image-edit", "type": "qwen-image-edit"}
    if family == "flux1":
        return {"model_family": "flux1", "type": "flux1"}
    return {"model_family": "flux2-klein", "type": "flux2-klein"}


def _conditioning(family: str) -> dict:
    if family == "sdxl":
        return {"model_family": "sdxl", "cond": object(), "pooled": object()}
    return {"model_family": "qwen-image-edit" if family == "qwen" else family}


class RuntimeFamilyResolverTests(unittest.TestCase):
    def test_all_supported_families_resolve(self):
        self.assertEqual(nodes.sdmlx_runtime_family(_handle("sdxl")), "sdxl")
        self.assertEqual(nodes.sdmlx_runtime_family(_handle("qwen")), "qwen")
        self.assertEqual(nodes.sdmlx_runtime_family(_handle("flux1")), "flux1")
        self.assertEqual(nodes.sdmlx_runtime_family(_handle("flux2-klein")), "flux2-klein")

    def test_contradictory_markers_fail(self):
        with self.assertRaisesRegex(RuntimeError, "contradictory runtime family"):
            nodes.sdmlx_runtime_family({"model_family": "qwen-image-edit", "type": "sdxl"})

    def test_standard_sampler_pairings_accept_sdxl_and_qwen(self):
        for family in ("sdxl", "qwen"):
            with self.subTest(family=family):
                result = nodes.require_sdmlx_runtime_pairing(
                    "test",
                    _handle(family),
                    allowed={"sdxl", "qwen"},
                    vae=_handle(family),
                    positive=_conditioning(family),
                    negative=_conditioning(family),
                )
                self.assertEqual(result, family)

    def test_cross_family_vae_and_conditioning_fail(self):
        with self.assertRaisesRegex(RuntimeError, "expected VAE family sdxl"):
            nodes.require_sdmlx_runtime_pairing(
                "test",
                _handle("sdxl"),
                allowed={"sdxl", "qwen"},
                vae=_handle("qwen"),
            )
        with self.assertRaisesRegex(RuntimeError, "positive conditioning family sdxl"):
            nodes.require_sdmlx_runtime_pairing(
                "test",
                _handle("sdxl"),
                allowed={"sdxl", "qwen"},
                positive=_conditioning("qwen"),
            )

    def test_standard_sampler_rejects_flux_families_before_sampling(self):
        sampler = nodes.SDMLX_KSampler()
        for family in ("flux1", "flux2-klein"):
            with self.subTest(family=family), self.assertRaisesRegex(RuntimeError, "expected model family"):
                sampler.sample(
                    _handle(family),
                    _handle(family),
                    _conditioning(family),
                    _conditioning(family),
                    {"samples": object()},
                    0,
                    4,
                    1.0,
                    "euler",
                    "simple",
                    False,
                    1.0,
                    "none",
                    1.0,
                    "off",
                    False,
                )

    def test_inpaint_conditioning_rejects_qwen(self):
        with self.assertRaisesRegex(RuntimeError, "expected VAE family sdxl"):
            nodes.SDMLX_InpaintConditioning().encode(
                _conditioning("qwen"),
                _conditioning("qwen"),
                object(),
                _handle("qwen"),
                object(),
            )


if __name__ == "__main__":
    unittest.main()
