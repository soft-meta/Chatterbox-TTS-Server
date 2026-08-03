from __future__ import annotations

import json
import subprocess
from pathlib import Path

from avatar_engine import AvatarEngineService


ROOT = Path(__file__).resolve().parents[1]


def test_avatar_concat_is_python311_safe() -> None:
    source = (ROOT / "avatar_engine.py").read_text(encoding="utf-8")
    assert 'f"file \'{clip.as_posix().replace(' not in source
    assert 'lines.append("file \'" + escaped_path + "\'")' in source


def test_avatar_concat_escapes_single_quotes(tmp_path: Path) -> None:
    service = object.__new__(AvatarEngineService)
    service.ffmpeg = "ffmpeg"
    service.media_duration = lambda _: 1.0  # type: ignore[method-assign]

    destination = tmp_path / "joined.mp4"
    clip_a = tmp_path / "normal.mp4"
    clip_b = tmp_path / "speaker's clip.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")

    def fake_run(command: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
        destination.write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, "", "")

    service._run_capture = fake_run  # type: ignore[method-assign]
    service._concat_segments([clip_a, clip_b], destination, tmp_path)

    concat_text = (tmp_path / "concat.txt").read_text(encoding="utf-8")
    assert "file '" in concat_text
    assert "speaker'\\''s clip.mp4" in concat_text


def test_colab_is_small_and_has_no_embedded_patch() -> None:
    notebook_path = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.0.0.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = notebook_path.read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert notebook_path.stat().st_size < 60_000
    assert "SOFTMETA_PATCH" not in text
    assert "base64" not in text
    assert "--branch main --depth 1 https://github.com/soft-meta/Chatterbox-TTS-Server.git" in text
    assert "INSTALL_MOSS_RUNTIME = False" in text
    assert "INSTALL_AVATAR_RUNTIME = False" in text


def test_optional_installers_are_idempotent() -> None:
    moss = (ROOT / "scripts" / "install_moss_a100.sh").read_text(encoding="utf-8")
    avatar = (ROOT / "scripts" / "install_longcat_avatar_a100.sh").read_text(encoding="utf-8")
    assert ".softmeta_moss_runtime_v100" in moss
    assert ".softmeta_longcat_avatar_v15_v100" in avatar
    assert "runtime_ok" in avatar
    assert "allow_patterns" in avatar
    assert '"base_model_int8/**"' in avatar
    assert '"whisper-large-v3/model.safetensors"' in avatar
    assert "AVATAR_MODEL_REVISION" in avatar
    assert "free_gb < 55" in avatar
    requirements = (ROOT / "requirements-avatar.txt").read_text(encoding="utf-8")
    assert "huggingface_hub" in requirements
