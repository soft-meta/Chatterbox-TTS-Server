from __future__ import annotations

import json
from pathlib import Path

from config import DEFAULT_CONFIG
from models import AudioJobCreate

ROOT = Path(__file__).resolve().parents[1]


def test_turbo_is_default_and_motivational_has_turbo_overrides() -> None:
    assert DEFAULT_CONFIG["tts_engine"]["default_model"] == "chatterbox"
    assert DEFAULT_CONFIG["generation_defaults"]["preset"] == "Motivational Speech"
    assert DEFAULT_CONFIG["generation_defaults"]["chunk_words"] == 85
    source = (ROOT / "server.py").read_text(encoding="utf-8")
    assert '"name": "Motivational Speech"' in source
    assert '"turbo_overrides"' in source


def test_generate_voice_feature_is_removed() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    assert 'data-voice-mode="generated"' not in html
    assert '/api/voice-designer/' not in server
    assert 'generateDesignedVoice' not in js
    assert not (ROOT / "voice_designer.py").exists()
    assert not (ROOT / "voice_worker.py").exists()
    assert not (ROOT / "requirements-voice.txt").exists()


def test_boxed_desktop_layout() -> None:
    css = (ROOT / "ui" / "styles.css").read_text(encoding="utf-8")
    assert 'width: min(1280px, calc(100% - 64px));' in css


def test_colab_is_tts_only_and_turbo_default() -> None:
    notebook_path = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.5.4.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = notebook_path.read_text(encoding="utf-8")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "SOFTMETA_PATCH" not in text
    assert "INSTALL_MOSS_RUNTIME" not in text
    assert "moss312" not in text
    assert '"SOFTMETA_MODEL": "chatterbox"' in source


def test_generated_voice_mode_is_not_accepted() -> None:
    schema = AudioJobCreate.model_json_schema()
    assert "generated" not in str(schema)
