from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "training_ncp_mse.py"


def write_tensorflow_stub(root: Path) -> None:
    (root / "tensorflow" / "keras").mkdir(parents=True, exist_ok=True)
    (root / "tensorflow" / "__init__.py").write_text(
        "__version__ = '0.0-test'\n"
        "from . import keras\n"
        "class _Config:\n"
        "    @staticmethod\n"
        "    def list_physical_devices(kind):\n"
        "        return []\n"
        "config = _Config()\n"
        "class random:\n"
        "    @staticmethod\n"
        "    def set_seed(seed):\n"
        "        return None\n",
        encoding="utf-8",
    )
    (root / "tensorflow" / "keras" / "__init__.py").write_text(
        "from . import backend, layers, optimizers, utils\n",
        encoding="utf-8",
    )
    (root / "tensorflow" / "keras" / "backend.py").write_text("", encoding="utf-8")
    (root / "tensorflow" / "keras" / "layers.py").write_text("", encoding="utf-8")
    (root / "tensorflow" / "keras" / "utils.py").write_text(
        "def plot_model(*args, **kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (root / "tensorflow" / "keras" / "optimizers.py").write_text(
        "class legacy:\n"
        "    pass\n",
        encoding="utf-8",
    )


class TrainingScriptSmokeTests(unittest.TestCase):
    def run_smoke_test(self, cwd: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            stub_root = Path(tmpdir)
            write_tensorflow_stub(stub_root)
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(stub_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
            return subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_smoke_imports_from_repository_root(self) -> None:
        result = self.run_smoke_test(
            cwd=REPO_ROOT,
            command=[sys.executable, str(SCRIPT_PATH), "--smoke-test-imports"],
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("Smoke test imports completed successfully.", result.stdout + result.stderr)

    def test_smoke_imports_from_scripts_directory(self) -> None:
        result = self.run_smoke_test(
            cwd=REPO_ROOT / "scripts",
            command=[sys.executable, "training_ncp_mse.py", "--smoke-test-imports"],
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("Smoke test imports completed successfully.", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
