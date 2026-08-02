from pathlib import Path


def test_avatar_installer_is_packaged():
    installer = Path(__file__).resolve().parents[1] / "scripts" / "install_ditto_a100.sh"
    assert installer.is_file()
    text = installer.read_text(encoding="utf-8")
    assert "digital-avatar/ditto-talkinghead" in text
    assert "ditto_pytorch/**" in text
    assert "tensorrt==8.6.1" not in text


def test_long_video_defaults_are_checkpointed():
    import config
    from models import VideoJobCreate

    assert config.DEFAULT_CONFIG["avatar"]["default_render_mode"] == "checkpointed"
    assert config.DEFAULT_CONFIG["avatar"]["default_segment_seconds"] == 120
    fields = VideoJobCreate.model_fields
    assert fields["render_mode"].default == "checkpointed"
    assert fields["segment_seconds"].default == 120


def test_ui_marks_checkpointed_as_default():
    root = Path(__file__).resolve().parents[1]
    html = (root / "ui" / "index.html").read_text(encoding="utf-8")
    app = (root / "ui" / "app.js").read_text(encoding="utf-8")
    assert 'value="checkpointed" selected' in html
    assert "render_mode: 'checkpointed'" in app
