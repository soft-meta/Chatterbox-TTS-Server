from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_is_small_and_has_no_embedded_patch() -> None:
    notebook_path = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.1.0.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = notebook_path.read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert notebook_path.stat().st_size < 60_000
    assert "SOFTMETA_PATCH" not in text
    assert "base64" not in text
    assert "--branch main --depth 1 https://github.com/soft-meta/Chatterbox-TTS-Server.git" in text
    assert "INSTALL_MOSS_RUNTIME = False" in text
    assert "INSTALL_AVATAR_RUNTIME" not in text
    assert "Generate Video" not in text
    assert "EchoMimic" not in text


def test_optional_moss_installer_is_idempotent() -> None:
    moss = (ROOT / "scripts" / "install_moss_a100.sh").read_text(encoding="utf-8")
    assert ".softmeta_moss_runtime_v100" in moss
