import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_engine import AvatarEngineService


ROOT = Path(__file__).resolve().parents[1]


class ColabBackendRegressionTests(unittest.TestCase):
    def test_avatar_processes_force_headless_matplotlib(self):
        with patch.dict(os.environ, {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}):
            env = AvatarEngineService._headless_process_env()
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertTrue(env["MPLCONFIGDIR"])

    def test_ditto_installer_overrides_colab_backend(self):
        installer = (ROOT / "scripts" / "install_ditto_a100.sh").read_text()
        self.assertIn("export MPLBACKEND=Agg", installer)
        self.assertIn(".softmeta_ditto_pytorch_v098", installer)
        self.assertIn('if [[ "$FORCE" != "1" ]] && runtime_ok; then', installer)

    def test_colab_sets_headless_backend_in_all_runtime_paths(self):
        notebook_path = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v0.9.8.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertGreaterEqual(source.count("MPLBACKEND"), 4)
        self.assertIn('os.environ["MPLBACKEND"] = "Agg"', source)
        self.assertIn('"MPLBACKEND": "Agg"', source)


if __name__ == "__main__":
    unittest.main()
