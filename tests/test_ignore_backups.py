from __future__ import annotations

import fnmatch
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _ignore_patterns(name: str) -> list[str]:
    lines = (line.strip() for line in _read(name).splitlines())
    return [line for line in lines if line and not line.startswith("#")]


class BackupFilesIgnoredTests(unittest.TestCase):
    BACKUP_NAMES = (
        "nodes.py.bak-20260809-transient-dmd",
        "nodes.py.before-inpaint-neutralization-20260731",
        "nodes.py.pre-ltx-auto-20260809",
        "THIRD_PARTY_NOTICES.md.pre-ltx-auto-20260809",
        ".tracking",
    )

    def test_backup_files_are_excluded_from_git_and_registry(self):
        for ignore_file in (".gitignore", ".comfyignore"):
            patterns = _ignore_patterns(ignore_file)
            for name in self.BACKUP_NAMES:
                with self.subTest(ignore_file=ignore_file, name=name):
                    self.assertTrue(any(fnmatch.fnmatch(name, p) for p in patterns))

    def test_source_files_are_not_excluded(self):
        for ignore_file in (".gitignore", ".comfyignore"):
            patterns = [p for p in _ignore_patterns(ignore_file) if "*" in p and "/" not in p]
            for name in (
                "nodes.py",
                "pyproject.toml",
                "README.md",
                "web/js/sdmlx_ksampler.js",
                ".pre-commit-config.yaml",
            ):
                with self.subTest(ignore_file=ignore_file, name=name):
                    base = name.rsplit("/", 1)[-1]
                    self.assertFalse(any(fnmatch.fnmatch(base, p) for p in patterns))


if __name__ == "__main__":
    unittest.main()
