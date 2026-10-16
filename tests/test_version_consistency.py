from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class VersionConsistencyTests(unittest.TestCase):
    def test_all_version_markers_match_pyproject(self):
        pyproject = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M).group(1)
        nodes_version = re.search(r'^SDMLX_VERSION = "([^"]+)"', _read("nodes.py"), re.M).group(1)
        fallback_versions = re.findall(r'SDMLX_VERSION = "([^"]+)"', _read("__init__.py"))
        self.assertEqual(nodes_version, pyproject)
        self.assertTrue(fallback_versions)
        for version in fallback_versions:
            self.assertEqual(version, pyproject)


if __name__ == "__main__":
    unittest.main()
