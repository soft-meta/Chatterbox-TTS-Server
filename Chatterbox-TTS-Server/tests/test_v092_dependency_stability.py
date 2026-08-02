from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_voice_requirements_do_not_override_moss_runtime():
    requirements = (ROOT / "requirements-voice.txt").read_text()
    assert "speechbrain==1.1.0" in requirements
    assert "scipy==" not in requirements
    assert "transformers==" not in requirements
    assert "torch==" not in requirements
    assert "torchaudio==" not in requirements


def test_moss_installer_uses_official_runtime_and_pip_check():
    installer = (ROOT / "scripts/install_moss_a100.sh").read_text()
    assert 'MOSS_DIR[torch-runtime]' in installer
    assert "python -m pip check" in installer
    assert "speechbrain.inference.speaker" in installer


def test_ditto_colab_defaults_to_pytorch_without_legacy_trt_build():
    installer = (ROOT / "scripts/install_ditto_a100.sh").read_text()
    assert 'TRY_TENSORRT="${SOFTMETA_TRY_TENSORRT:-0}"' in installer
    assert 'if [[ "$TRY_TENSORRT" == "1" ]]' in installer
    assert "Ditto PyTorch is the stable A100 40GB backend" in installer


def test_avatar_auto_backend_does_not_select_unimportable_tensorrt():
    source = (ROOT / "avatar_engine.py").read_text()
    assert 'self._python_can_import("tensorrt")' in source
    assert 'os.getenv("SOFTMETA_ENABLE_TENSORRT", "0") == "1"' in source
    assert 'order = ("ditto_trt", "ditto_pytorch") if trt_enabled else ("ditto_pytorch",)' in source


def test_voice_worker_has_cached_speechbrain_compatibility_guard():
    source = (ROOT / "voice_worker.py").read_text()
    assert "def _prepare_speechbrain_audio_compatibility" in source
    assert 'torchaudio.list_audio_backends = lambda: ["soundfile"]' in source


def test_v092_notebook_is_self_contained_and_uses_stable_backends():
    notebook_path = ROOT / "colab/SoftMeta_Chatterbox_TTS_Colab_v0.9.2.ipynb"
    data = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
    )
    assert "Applied embedded SoftMeta v0.9.2 patch" in source
    assert "install_moss_a100.sh" in source
    assert "install_ditto_a100.sh" in source
    assert "'SOFTMETA_ENABLE_TENSORRT': '0'" in source
    assert "softmeta_chatterbox_v092.log" in source
