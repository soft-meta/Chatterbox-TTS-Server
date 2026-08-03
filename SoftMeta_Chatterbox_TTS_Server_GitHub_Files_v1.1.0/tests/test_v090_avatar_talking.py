from pathlib import Path

import pytest
from pydantic import ValidationError

from avatar_engine import AvatarEngineService
from models import VideoJobCreate
from storage import Storage

ROOT = Path(__file__).resolve().parents[1]


def valid_request(**overrides):
    values = {
        "title": "Long Avatar Story",
        "avatar_filename": "avatar.png",
        "audio_source": "audio_job",
        "audio_job_id": "job-123",
        "engine": "auto",
        "render_mode": "continuous",
        "segment_seconds": 180,
        "aspect_ratio": "9:16",
        "resolution": "1080p",
        "fps": 25,
        "framing": "upper",
        "image_fit": "cover",
        "quality": "high",
        "consent": True,
    }
    values.update(overrides)
    return VideoJobCreate(**values)


def test_v090_video_request_requires_permission_and_audio() -> None:
    request = valid_request()
    assert request.consent is True
    assert request.audio_job_id == "job-123"
    with pytest.raises(ValidationError):
        valid_request(consent=False)
    with pytest.raises(ValidationError):
        valid_request(audio_job_id=None)
    upload = valid_request(audio_source="upload", audio_job_id=None, audio_filename="voice.wav")
    assert upload.audio_filename == "voice.wav"


def test_v090_generate_video_tab_follows_audio_tabs() -> None:
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    build_start = script.index("function buildTabs()")
    build_end = script.index("function addTab()", build_start)
    build = script[build_start:build_end]
    assert "state.tabs.forEach" in build
    assert "Generate Video" in build
    assert build.index("state.tabs.forEach") < build.index("Generate Video") < build.index("add-tab")
    assert "VIDEO_VIEW_ID" in script


def test_v090_avatar_ui_has_complete_long_video_controls() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    for text in (
        "Avatar Talking",
        "Upload Avatar Image",
        "Select Completed Audio",
        "Continuous · Best motion continuity",
        "Checkpointed · Safer for 10–30 minute jobs",
        "EchoMimicV3 Flash · Realistic motion",
        "Natural · Realistic eyes and body",
        "Generate Video",
        "Download MP4",
    ):
        assert text in html


def test_v110_avatar_engine_uses_official_echomimic_cli() -> None:
    source = (ROOT / "avatar_engine.py").read_text(encoding="utf-8")
    assert "infer_flash.py" in source
    assert '"--batch_manifest"' in source
    assert '"--num_inference_steps", "8"' in source
    assert '"--teacache_threshold", "0.1"' in source
    assert "SOFTMETA_PROGRESS" in source
    assert "freezedetect" in source


def test_v090_storage_has_separate_avatar_directories() -> None:
    storage = Storage()
    assert storage.avatar_images.is_dir()
    assert storage.video_audio.is_dir()
    assert storage.video_outputs.is_dir()
    assert storage.video_work.is_dir()
