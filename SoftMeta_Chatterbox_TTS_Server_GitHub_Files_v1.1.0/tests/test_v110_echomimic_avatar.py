import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_engine import AvatarEngineService
from models import VideoJobCreate
from scripts.patch_echomimic_v3_flash import patch_runtime


ROOT = Path(__file__).resolve().parents[1]


class EchoMimicV3RegressionTests(unittest.TestCase):
    def test_headless_gpu_environment(self):
        with patch.dict(os.environ, {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}):
            env = AvatarEngineService._headless_process_env()
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")

    def test_request_defaults_to_echomimic_auto(self):
        request = VideoJobCreate(avatar_filename="portrait.png", audio_job_id="audio-id", consent=True)
        self.assertEqual(request.engine, "auto")
        self.assertEqual(request.motion_style, "natural")

    def test_installer_is_pinned_selective_and_verified(self):
        installer = (ROOT / "scripts/install_echomimic_v3_flash_a100.sh").read_text()
        for revision in (
            "7e89489ca51c0d008fc1963ec6c03fc5bd0b9397",
            "fc913c34361f4ec879e2f9c78b4f11ae50a937d1",
            "311e176905a8c4c24b240b530488fe636ce4d249",
            "3991242c806928916fff4a8c0e4f76acf661b743",
        ):
            self.assertIn(revision, installer)
        self.assertIn("EchoMimicV3 Flash is ready", installer)
        self.assertIn(".softmeta_echomimic_v3_flash_v110", installer)

    def test_guarded_batch_patch_applies_and_compiles(self):
        official = Path("/tmp/echomimic_v3_official/infer_flash.py")
        if not official.is_file():
            self.skipTest("official source fixture is unavailable")
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            (repo / "infer_flash.py").write_text(official.read_text())
            patch_runtime(repo)
            source = (repo / "infer_flash.py").read_text()
            ast.parse(source)
            self.assertIn("--batch_manifest", source)
            self.assertIn("SOFTMETA_PROGRESS", source)

    def test_colab_uses_echo_runtime_only_when_enabled(self):
        notebook_path = ROOT / "colab/SoftMeta_Chatterbox_TTS_Colab_v1.1.0.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("install_echomimic_v3_flash_a100.sh", source)
        self.assertIn('env_python("echo310")', source)
        self.assertIn('"SOFTMETA_ECHOMIMIC_MODELS": "/content/echomimic_v3_models"', source)
        self.assertNotIn("LongCat", source)

    def test_ui_names_echomimic_backend(self):
        html = (ROOT / "ui/index.html").read_text()
        js = (ROOT / "ui/app.js").read_text()
        self.assertIn("EchoMimicV3 Flash", html)
        self.assertIn("echomimic_v3_flash", html)
        self.assertIn("8-step TeaCache", js)


if __name__ == "__main__":
    unittest.main()
