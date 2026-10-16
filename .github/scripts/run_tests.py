"""Run the SDMLX test suite on a machine without Apple Silicon (used by CI).

Expects the working directory to be ComfyUI/custom_nodes, with this
repository checked out there as SDMLX and ComfyUI on PYTHONPATH.
"""

import sys
import unittest

# ComfyUI picks its torch device on import. Without a GPU it has to be told
# to use the CPU, the same as starting it with `python main.py --cpu`.
import comfy.options

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import comfy.cli_args  # noqa: E402,F401

suite = unittest.defaultTestLoader.discover("SDMLX/tests", top_level_dir=".")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
