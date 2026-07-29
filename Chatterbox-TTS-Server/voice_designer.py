from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Literal

from utils import safe_filename

Gender = Literal["male", "female"]
Emotion = Literal["warm", "calm", "reflective", "concerned", "serious", "hopeful"]


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    age: int
    gender: Gender
    language: str
    emotion: Emotion
    speaker_name: str
    age_label: str
    pace_label: str
    effective_description: str
    sample_tempo: float
    recommended_speed_factor: float


@dataclass(frozen=True, slots=True)
class DesignedVoiceResult:
    path: Path
    profile: VoiceProfile


class VoiceDesigner:
    """Generate a short age-aware reference voice in an isolated process.

    Chatterbox and Parler-TTS require incompatible Transformers versions, so
    Parler-TTS runs in a separate Python environment. The description builder
    converts simple UI fields into a concise prompt that Parler-TTS can follow.
    """

    MODEL_ID = "parler-tts/parler-tts-mini-v1.1"

    # These are official Parler-TTS Mini training speakers with strong published
    # consistency scores. The seed selects one stable identity per generation.
    MALE_SPEAKERS = ("Jon", "Gary", "Mike", "Will", "Patrick", "Eric", "Rick", "Bill", "James", "Jerry")
    FEMALE_SPEAKERS = ("Lea", "Jenna", "Laura", "Lauren", "Eileen", "Alisa", "Karen", "Barbara", "Carol", "Emily", "Rose", "Anna", "Tina")

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda",
        *,
        python_executable: str | None = None,
        worker_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.python_executable = (
            python_executable
            or os.getenv("SOFTMETA_VOICE_PYTHON")
            or sys.executable
        )
        self.worker_path = worker_path or Path(__file__).with_name("voice_worker.py")
        self.timeout_seconds = timeout_seconds
        self._lock = RLock()

    @classmethod
    def _speaker_for(cls, gender: Gender, seed: int) -> str:
        speakers = cls.MALE_SPEAKERS if gender == "male" else cls.FEMALE_SPEAKERS
        return speakers[int(seed) % len(speakers)]

    @staticmethod
    def _age_settings(age: int) -> tuple[str, str, str, float, float]:
        if age < 40:
            return (
                "adult",
                "natural conversational pace",
                "steady energy, natural confidence, flexible pitch, and relaxed breathing",
                1.0,
                1.0,
            )
        if age < 50:
            return (
                "mature adult",
                "normal, unhurried conversational pace",
                "mature confidence and steady energy without sounding old or weak",
                1.0,
                1.0,
            )
        if age < 60:
            return (
                "mature and experienced",
                "slightly slower, thoughtful pace",
                "grounded energy, small pauses before meaningful thoughts, and controlled emotion",
                0.99,
                0.96,
            )
        if age < 70:
            return (
                "older and experienced",
                "slow, clear pace",
                "warm vocal weight, gentle pauses between ideas, occasional soft breaths, and calm clarity",
                0.97,
                0.90,
            )
        if age < 80:
            return (
                "elderly but strong and mentally clear",
                "noticeably slow, considered pace",
                "softer projection, longer meaningful pauses, mild aged texture, careful pronunciation, and emotional depth",
                0.94,
                0.84,
            )
        if age < 90:
            return (
                "very elderly, thoughtful, and mentally present",
                "very slow, careful pace",
                "soft vocal power, clear pauses inside longer sentences, gentle roughness, occasional quiet breaths, and no exaggerated weakness",
                0.91,
                0.76,
            )
        return (
            "very elderly, fragile but understandable and mentally present",
            "very slow and careful pace",
            "low physical energy, long natural pauses for breath and thought, mild aged texture, gentle breathiness, and only occasional subtle trembling",
            0.88,
            0.68,
        )

    @staticmethod
    def _emotion_text(emotion: Emotion) -> str:
        return {
            "warm": "warm, sincere, emotionally present, and gently reassuring",
            "calm": "calm, composed, quiet, and emotionally steady",
            "reflective": "thoughtful, reflective, slightly intimate, and connected to personal experience",
            "concerned": "careful, concerned, quietly firm, and never alarmist",
            "serious": "serious, grounded, restrained, and never theatrical",
            "hopeful": "gentle and hopeful, with slightly more light in meaningful final words",
        }[emotion]

    @classmethod
    def build_profile(
        cls,
        *,
        age: int,
        gender: Gender,
        language: str,
        emotion: Emotion,
        description: str,
        seed: int,
    ) -> VoiceProfile:
        age = max(18, min(110, int(age)))
        speaker_name = cls._speaker_for(gender, seed)
        age_label, pace_label, age_behaviour, sample_tempo, recommended_speed = cls._age_settings(age)
        gender_noun = "man" if gender == "male" else "woman"
        gender_pitch = "medium-low pitch" if gender == "male" else "natural medium pitch with mature warmth"
        notes = " ".join(description.strip().split())
        if len(notes) > 430:
            notes = notes[:427].rstrip() + "..."

        effective = (
            f"{speaker_name} speaks as a {age}-year-old American {gender_noun}. "
            f"The voice is {age_label}, with {gender_pitch}. The delivery uses a {pace_label}, "
            f"with {age_behaviour}. The emotional tone is {cls._emotion_text(emotion)}. "
            "Use neutral General American English pronunciation. Speak as if privately sharing a real memory, warning, or piece of advice with one trusted listener in a quiet room. "
            "Use subtle human variation in pitch, timing, energy, breath, emphasis, and sentence rhythm. Allow tiny natural imperfections and soft connections between unimportant words. "
            "Do not sound like an AI assistant, audiobook narrator, advertisement, newsreader, radio host, customer-service agent, documentary narrator, or studio announcer. "
            "Avoid identical sentence rhythms, perfect timing, excessive clarity, constant dramatic emphasis, and repeated falling tones. "
            "The recording is very clear, close-microphone, dry, intimate, and has almost no background noise or reverberation."
        )
        if notes:
            effective += f" Additional speaker notes: {notes}"

        return VoiceProfile(
            age=age,
            gender=gender,
            language=language,
            emotion=emotion,
            speaker_name=speaker_name,
            age_label=age_label,
            pace_label=pace_label,
            effective_description=effective,
            sample_tempo=sample_tempo,
            recommended_speed_factor=recommended_speed,
        )

    def _next_output_path(self, name: str, seed: int) -> Path:
        stem = safe_filename(name, "Generated_Voice")
        candidate = self.output_dir / f"{stem}_{seed}.wav"
        counter = 2
        while candidate.exists():
            candidate = self.output_dir / f"{stem}_{seed}_{counter}.wav"
            counter += 1
        return candidate

    def _validate_runner(self) -> None:
        python_path = Path(self.python_executable)
        if not python_path.exists():
            raise RuntimeError(
                "The isolated Generate Voice environment is missing. "
                "Run the SoftMeta Colab installation cell again from a fresh runtime. "
                f"Expected Python executable: {python_path}"
            )
        if not self.worker_path.exists():
            raise RuntimeError(f"Generate Voice worker was not found: {self.worker_path}")

    def generate(
        self,
        *,
        name: str,
        description: str,
        sample_text: str,
        seed: int,
        age: int = 50,
        gender: Gender = "male",
        language: str = "en-US",
        emotion: Emotion = "warm",
    ) -> DesignedVoiceResult:
        with self._lock:
            self._validate_runner()
            profile = self.build_profile(
                age=age,
                gender=gender,
                language=language,
                emotion=emotion,
                description=description,
                seed=seed,
            )
            output_path = self._next_output_path(name, seed)
            request = {
                "model_id": self.MODEL_ID,
                "description": profile.effective_description,
                "sample_text": sample_text.strip(),
                "seed": int(seed),
                "device": self.device,
                "output_path": str(output_path),
                "sample_tempo": profile.sample_tempo,
            }

            request_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix="softmeta_voice_",
                    delete=False,
                    encoding="utf-8",
                ) as handle:
                    json.dump(request, handle, ensure_ascii=False)
                    request_path = Path(handle.name)

                env = {
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
                result = subprocess.run(
                    [
                        self.python_executable,
                        "-u",
                        str(self.worker_path),
                        "--request",
                        str(request_path),
                    ],
                    cwd=str(self.worker_path.parent),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_seconds,
                    check=False,
                )

                if result.returncode != 0:
                    output_path.unlink(missing_ok=True)
                    details = (result.stdout or "No worker output.")[-8000:]
                    raise RuntimeError(
                        "The isolated Generate Voice worker failed.\n"
                        f"Python: {self.python_executable}\n"
                        f"Exit code: {result.returncode}\n"
                        f"Worker output:\n{details}"
                    )
                if not output_path.exists() or output_path.stat().st_size < 1000:
                    output_path.unlink(missing_ok=True)
                    details = (result.stdout or "No worker output.")[-4000:]
                    raise RuntimeError(
                        "Generate Voice finished without creating a valid WAV file.\n"
                        f"Worker output:\n{details}"
                    )

                metadata = {
                    "name": name,
                    "filename": output_path.name,
                    "seed": int(seed),
                    "sample_text": sample_text.strip(),
                    **asdict(profile),
                }
                output_path.with_suffix(".json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return DesignedVoiceResult(path=output_path, profile=profile)
            except subprocess.TimeoutExpired as error:
                output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Generate Voice exceeded the {self.timeout_seconds // 60}-minute timeout."
                ) from error
            finally:
                if request_path is not None:
                    request_path.unlink(missing_ok=True)
