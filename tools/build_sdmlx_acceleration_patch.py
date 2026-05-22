#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import numpy as np
from safetensors.numpy import save_file


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_import_paths():
    root = repo_root()
    parent = os.path.dirname(root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        import folder_paths  # noqa: F401
    except Exception:
        comfy_root = os.path.expanduser(
            "~/ComfyUI-Easy-Install/ComfyUI-Easy-Install/ComfyUI"
        )
        if os.path.isdir(comfy_root) and comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)


def np_array(value):
    return np.ascontiguousarray(np.asarray(value))


def build_patch(source, output_dir, package_name, source_repo=None, source_file=None, license_name=None):
    add_import_paths()
    from SDMLX import nodes

    modules_info = nodes.load_lora_modules(source)
    modules = modules_info["modules"]
    if not modules:
        raise RuntimeError(f"No UNet LoRA modules found in {source}")

    package_dir = os.path.join(output_dir, package_name)
    os.makedirs(package_dir, exist_ok=True)

    factors = {}
    manifest_modules = []
    for module in modules:
        target_base = module["target_base"]
        factors[f"{target_base}.lora_up"] = np_array(module["up"])
        factors[f"{target_base}.lora_down"] = np_array(module["down"])
        factors[f"{target_base}.alpha"] = np.array(float(module["alpha"]), dtype=np.float32)
        manifest_modules.append(
            {
                "target": f"{target_base}.weight",
                "source_prefix": module.get("source_prefix"),
                "split_index": module.get("split_index"),
                "rank": int(module["rank"]),
                "alpha": float(module["alpha"]),
                "up_shape": list(np_array(module["up"]).shape),
                "down_shape": list(np_array(module["down"]).shape),
            }
        )

    patch_path = os.path.join(package_dir, "patch.safetensors")
    save_file(factors, patch_path)

    manifest = {
        "format": "sdmlx-acceleration-patch-v1",
        "base_model_family": "sdxl",
        "created_at_unix": int(time.time()),
        "module_count": len(manifest_modules),
        "components": {"factors": "patch.safetensors"},
        "factor_layout": {
            "linear": {
                "lora_down": "[rank, in]",
                "lora_up": "[out, rank]",
            },
            "conv2d": {
                "lora_down": "[rank, kernel_h, kernel_w, in]",
                "lora_up": "[out, rank]",
            },
            "scale": "runtime_strength * alpha / rank",
        },
        "modules": manifest_modules,
    }
    with open(os.path.join(package_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    metadata = {
        "source_path": os.path.abspath(source),
        "source_repo": source_repo,
        "source_file": source_file,
        "license": license_name,
        "text_prefixes": modules_info.get("text_prefixes", 0),
        "skipped_prefixes": modules_info.get("skipped_prefixes", []),
    }
    with open(os.path.join(package_dir, "source_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"built {package_name}")
    print(f"  modules: {len(manifest_modules)}")
    print(f"  patch:   {patch_path}")


def main():
    parser = argparse.ArgumentParser(description="Build an SDMLX acceleration patch from an SDXL UNet LoRA.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--source-repo")
    parser.add_argument("--source-file")
    parser.add_argument("--license")
    args = parser.parse_args()
    build_patch(
        args.source,
        args.output_dir,
        args.package_name,
        args.source_repo,
        args.source_file,
        args.license,
    )


if __name__ == "__main__":
    main()
