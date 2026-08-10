from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from config import DEFAULT_CONFIG
from emotion_director import analyze_serious_senior_advisor, analyze_turbo_avatar_performance
from professional_audio import analyze_prosody_quality
from speech_pipeline import build_long_form_segments

ROOT = Path(__file__).resolve().parents[1]


BASE = """Many older adults think feeling tired is simply part of aging. But that is not always the whole story. Most people do not realize how much a few small habits can change the way they feel. The good news is that simple changes can help you feel steadier and safer. Remember this, because the goal is not to push yourself harder. The goal is to understand what your body is telling you and respond calmly.

Number One: Morning Routine
A steady morning can make the rest of the day easier. Keep the first hour simple and give yourself enough time to move without rushing. I learned too late that ignoring small warning signs can become a bigger problem. Pay attention to changes in your balance, breathing, and energy. Fortunately, many people improve when they make a few practical adjustments.

Number Two: Hydration
You may think thirst always tells you when to drink, but older adults may not feel thirst as strongly. What surprised me is how often people wait until they already feel weak. A small routine can help. Keep fluids nearby if your doctor has not limited them. This is important because dehydration can make dizziness and fatigue worse.

Number Three: Movement
Do not ignore sudden chest pressure or severe shortness of breath. Call emergency services if symptoms are severe. For ordinary stiffness, gentle movement can help you stay stronger. The encouraging part is that progress does not have to be dramatic. A few minutes at a time can still make a difference. Remember this when you feel discouraged.

Number Four: Sleep
Most people do not realize how much poor sleep can affect mood, balance, and memory. You might think more time in bed always helps, but that is not always the whole story. A regular schedule may help you feel better. Keep this in mind and talk with your doctor if sleep problems continue.

Number Five: Support
I was lonely for a long time before I learned how much a simple phone call could change the day. The good news is that support does not have to be complicated. You can still protect your independence while accepting help when you need it. This matters because small connections can reduce the feeling of carrying everything alone.
"""


def test_turbo_plans_semantic_audible_events_without_changing_original_planner() -> None:
    text = "\n".join([BASE] * 4)
    original = analyze_serious_senior_advisor(text)
    turbo = analyze_turbo_avatar_performance(text)

    assert original.mode == "Serious Senior Advisor"
    assert original.avatar_window_words == 0
    assert original.avatar_applied_count == 0

    assert turbo.mode == "Turbo Human Performance Events"
    assert 2 <= turbo.avatar_target_count <= 10
    assert turbo.avatar_applied_count >= 4
    assert turbo.applied_count - turbo.avatar_applied_count <= 2
    assert set(turbo.by_tag).issubset({
        "laugh", "chuckle", "sigh", "gasp", "clear throat",
        "groan", "sniff", "cough", "shush",
    })
    # This serious advice sample genuinely contains regret/surprise language but no
    # humour, so the planner should not manufacture a laugh just to hit a quota.
    assert "laugh" not in turbo.by_tag
    assert "chuckle" not in turbo.by_tag
    assert "sigh" in turbo.by_tag
    assert "gasp" in turbo.by_tag


def test_turbo_avatar_segments_get_wider_local_pace_band_only_in_turbo_mode() -> None:
    neutral = " ".join(f"steady{i}" for i in range(78)) + "."
    tagged = neutral + " [surprised] Most people do not realize this can happen. The next sentence stays calm and explains the idea clearly."
    original = build_long_form_segments(tagged, max_words=85, age_profile="70s")
    turbo = build_long_form_segments(
        tagged,
        max_words=85,
        age_profile="70s",
        turbo_avatar_mode=True,
        avatar_window_words=740,
    )
    original_emotion = next(item for item in original if item.emotion_tag == "surprised")
    turbo_emotion = next(item for item in turbo if item.emotion_tag == "surprised")
    assert original_emotion.min_wpm == original_emotion.target_wpm - 13
    assert turbo_emotion.min_wpm == turbo_emotion.target_wpm - 16
    assert turbo_emotion.target_wpm != original_emotion.target_wpm


def _write_signal(path: Path, *, dynamic: bool, seconds: int = 60, sr: int = 24000) -> None:
    t = np.arange(sr * seconds, dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * 180 * t)
    if dynamic:
        block = np.arange(sr * seconds) // (sr * 5)
        amp = np.where(block % 2 == 0, 0.035, 0.10).astype(np.float32)
    else:
        amp = np.full(sr * seconds, 0.06, dtype=np.float32)
    sf.write(path, (carrier * amp).astype(np.float32), sr, subtype="PCM_16")


def test_prosody_advisor_rewards_real_dynamic_movement(tmp_path: Path) -> None:
    flat = tmp_path / "flat.wav"
    dynamic = tmp_path / "dynamic.wav"
    _write_signal(flat, dynamic=False)
    _write_signal(dynamic, dynamic=True)
    summary = {
        "total_words": 150,
        "avatar_target_count": 8,
        "placements": [{"word_position": p} for p in (12, 30, 48, 66, 84, 102, 120, 138)],
    }
    flat_result = analyze_prosody_quality(flat, summary, avatar_minutes=5.0)
    dynamic_result = analyze_prosody_quality(dynamic, summary, avatar_minutes=5.0)
    assert dynamic_result["score"] > flat_result["score"] + 8
    assert dynamic_result["first5_rms_range_db"] > flat_result["first5_rms_range_db"] + 3
    assert dynamic_result["lra_lu"] >= flat_result["lra_lu"]


def test_v155_is_turbo_first_in_server_ui_and_colab() -> None:
    assert DEFAULT_CONFIG["tts_engine"]["default_model"] == "chatterbox-turbo"
    app = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    notebook = (ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.5.7.ipynb").read_text(encoding="utf-8")
    assert "model === 'chatterbox-turbo'" in app
    assert "analyze_turbo_avatar_performance" in server
    assert 'APP_VERSION = "1.5.7"' in server
    assert '\\"SOFTMETA_MODEL\\": \\"chatterbox-turbo\\"' in notebook
