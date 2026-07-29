from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from utils import resolve_inside, safe_filename

Gender = Literal["male", "female"]
Emotion = Literal["warm", "calm", "reflective", "concerned", "serious", "hopeful"]


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    age: int
    gender: Gender
    language: str
    emotion: Emotion
    age_label: str
    pace_label: str
    phrase_guidance: str
    identity_traits: dict[str, str]
    effective_description: str
    recommended_speed_factor: float


class VoiceDesigner:
    """Create new, reusable human voice references with Qwen3-TTS VoiceDesign.

    Qwen3-TTS runs in an isolated Python environment because its Transformers
    dependency is intentionally separate from Chatterbox. The worker loads the
    official VoiceDesign checkpoint once per request, produces several distinct
    candidates, and optionally compares speaker embeddings with saved voices.
    """

    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    MODEL_REVISION = "fa0251e3279a10b4936dc49d69a59c41b07cbfc0"
    RESULT_PREFIX = "SOFTMETA_RESULT="

    _MALE_PITCH = (
        "deep bass-baritone range",
        "low rounded baritone range",
        "mid-low conversational baritone",
        "lean tenor-baritone range",
        "clear mature tenor range",
        "compact midrange with a low centre",
        "soft low-mid range",
        "broad chest-led baritone range",
    )
    _FEMALE_PITCH = (
        "deep contralto range",
        "warm low alto range",
        "balanced alto range",
        "soft mezzo range",
        "clear mature midrange",
        "light but grounded midrange",
        "rounded low-mid range",
        "airy alto-mezzo range",
    )
    _VOICE_FAMILIES = (
        "mellow neighbour",
        "retired professional",
        "plain-spoken storyteller",
        "quiet family adviser",
        "dry-humoured observer",
        "gentle community elder",
        "reserved former teacher",
        "practical lifelong worker",
        "soft-spoken caregiver",
        "thoughtful private conversationalist",
        "confident small-town speaker",
        "warm reflective grandparent",
    )
    _RESONANCE = (
        "chest-forward resonance",
        "balanced oral and chest resonance",
        "compact dry resonance",
        "rounded mellow resonance",
        "lightly airy resonance",
        "close and intimate resonance",
        "narrow focused resonance",
        "broad open resonance",
        "slightly nasal everyday resonance",
        "soft back-of-mouth resonance",
    )
    _TEXTURE = (
        "clean texture with tiny natural grain",
        "slightly husky texture",
        "soft dry texture",
        "velvety texture with subtle breath",
        "lightly weathered texture",
        "smooth texture with occasional rough edges",
        "papery but healthy texture",
        "rounded texture with faint rasp",
        "clear texture with a dry edge",
        "soft smoky texture without hoarseness",
        "light reedy texture",
        "dense smooth texture with restrained breathiness",
    )
    _VOCAL_WEIGHT = (
        "light vocal weight",
        "medium-light vocal weight",
        "balanced medium vocal weight",
        "full but relaxed vocal weight",
        "compact firm vocal weight",
        "softened low-energy vocal weight",
        "broad grounded vocal weight",
        "slender intimate vocal weight",
    )
    _ARTICULATION = (
        "soft consonants and naturally connected everyday words",
        "clear but never over-pronounced articulation",
        "rounded vowels and relaxed consonants",
        "careful key words with softer unimportant words",
        "slightly compact articulation with conversational contractions",
        "gentle articulation with small timing imperfections",
        "crisp key consonants with relaxed word endings",
        "unpolished everyday articulation with occasional softened syllables",
        "deliberate vowels with light consonant reduction",
        "casual connected speech with clear advice words",
    )
    _PERSONALITY = (
        "warm, practical, and quietly confident",
        "reserved, sincere, and thoughtful",
        "gentle, emotionally observant, and reassuring",
        "grounded, direct, and quietly humorous",
        "reflective, humble, and personally experienced",
        "calm, private, and trustworthy",
        "matter-of-fact, patient, and dependable",
        "soft-spoken, curious, and emotionally careful",
        "plain-spoken, independent, and kind",
        "confident, sociable, and naturally informal",
        "serious, observant, and quietly protective",
        "easygoing, candid, and gently expressive",
    )
    _CADENCE = (
        "an unforced conversational cadence with varied phrase lengths",
        "a measured cadence with occasional quick connecting words",
        "a reflective cadence with uneven but meaningful pauses",
        "a calm cadence with subtle changes in energy and emphasis",
        "a private one-to-one cadence that avoids performance rhythm",
        "a lived-in cadence with gentle starts and natural sentence release",
        "a practical cadence with short setup phrases and firmer advice phrases",
        "a slightly wandering memory-based cadence that always remains clear",
        "a compact cadence with brief pauses and occasional longer reflection",
        "a warm storytelling cadence with irregular breath points",
        "a low-energy cadence with careful starts and softly released endings",
        "a direct everyday cadence with natural contractions and small hesitations",
    )
    _MELODY = (
        "mostly level sentence melody with selective rises on genuine questions",
        "gentle pitch arcs that avoid repeating the same ending",
        "restrained melody with quiet emphasis on personally meaningful words",
        "subtle upward movement in connecting phrases and relaxed final release",
        "low melodic movement with occasional expressive lift",
        "more varied conversational melody without sounding performative",
        "soft downward turns mixed with unfinished-sounding thought transitions",
        "narrow pitch movement with a few warm expressive peaks",
    )
    _BREATH_STYLE = (
        "small quiet breaths between selected thought groups",
        "mostly silent breathing with an occasional audible intake before important advice",
        "soft breath release at a few sentence endings",
        "brief natural breaths that do not occur at fixed intervals",
        "lightly breathy starts on reflective sentences",
        "controlled breathing with one or two imperfect pauses",
        "short recovery breaths after longer phrases",
        "gentle breath support with no gasping or theatrical fragility",
    )

    def __init__(
        self,
        output_dir: Path,
        candidate_root: Path,
        device: str = "cuda",
        *,
        python_executable: str | None = None,
        worker_path: Path | None = None,
        timeout_seconds: int = 2700,
        model_cache_dir: Path | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_root = candidate_root
        self.candidate_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.python_executable = (
            python_executable
            or os.getenv("SOFTMETA_VOICE_PYTHON")
            or sys.executable
        )
        self.worker_path = worker_path or Path(__file__).with_name("voice_worker.py")
        self.timeout_seconds = timeout_seconds
        self.model_cache_dir = model_cache_dir or Path(
            os.getenv("SOFTMETA_QWEN_MODEL_DIR", "/content/softmeta_models/qwen3_voice_design")
        )
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @staticmethod
    def _age_settings(age: int) -> tuple[str, str, str]:
        if age < 40:
            return (
                "adult",
                "natural conversational pace with flexible energy",
                "Use varied thought groups, usually around 8 to 14 words, with short natural pauses.",
            )
        if age < 50:
            return (
                "mature adult",
                "normal, unhurried conversational pace",
                "Use mature phrasing, usually around 7 to 12 words per thought group, with natural breathing.",
            )
        if age < 60:
            return (
                "mature and experienced",
                "slightly slower and more thoughtful than a young adult",
                "Use thought groups of roughly 6 to 10 words and small pauses before meaningful ideas.",
            )
        if age < 70:
            return (
                "older and experienced",
                "slow, clear, and grounded without stretched words",
                "Use thought groups of roughly 5 to 9 words, gentle pauses between ideas, and occasional quiet breaths.",
            )
        if age < 80:
            return (
                "elderly, strong, and mentally clear",
                "noticeably measured and considered, not digitally slowed",
                "Use short thought groups of roughly 4 to 7 words, variable pauses, and a breath before selected important ideas.",
            )
        if age < 90:
            return (
                "very elderly, thoughtful, and mentally present",
                "very careful and spacious while keeping each word natural",
                "Use short thought groups of roughly 3 to 6 words, clear variable pauses, softer projection, and occasional quiet breaths.",
            )
        return (
            "very elderly, physically gentle, understandable, and mentally present",
            "very careful and spacious without theatrical fragility",
            "Use thought groups of roughly 3 to 5 words, longer varied pauses for breath and thought, and low physical energy with clear meaning.",
        )

    @staticmethod
    def _age_vocal_character(age: int) -> str:
        if age < 40:
            return "full adult vocal energy, easy breath support, and flexible conversational timing"
        if age < 50:
            return "mature vocal weight, stable breath support, and less youthful brightness"
        if age < 60:
            return "grounded mature resonance, controlled energy, and a slightly more considered sentence release"
        if age < 70:
            return "subtly softened projection, mild lived-in grain, relaxed breath support, and thoughtful phrase planning"
        if age < 80:
            return "reduced youthful brightness, gentle age texture, careful breath placement, and lower but healthy physical energy"
        if age < 90:
            return "soft projection, clear aged texture, lighter breath support, occasional roughness, and spacious thought-led delivery"
        return "fragile but intelligible projection, mild breathiness, occasional subtle tremor, low physical energy, and strong emotional meaning"

    @staticmethod
    def _emotion_text(emotion: Emotion) -> str:
        return {
            "warm": "warm, sincere, emotionally present, and gently reassuring",
            "calm": "calm, composed, quiet, and emotionally steady",
            "reflective": "thoughtful, reflective, intimate, and connected to personal experience",
            "concerned": "careful, concerned, quietly firm, and never alarmist",
            "serious": "serious, grounded, restrained, and never theatrical",
            "hopeful": "gentle and hopeful, with a little more light in meaningful final words",
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
        candidate_index: int,
    ) -> VoiceProfile:
        age = max(18, min(110, int(age)))
        randomiser = random.Random(int(seed))
        gender_noun = "man" if gender == "male" else "woman"

        # Use rotated, coprime steps instead of independent random choices so
        # candidates in one batch deliberately occupy different vocal regions.
        def distinct(sequence: tuple[str, ...], step: int, salt: int) -> str:
            start = randomiser.randrange(len(sequence))
            return sequence[(start + candidate_index * step + salt) % len(sequence)]

        pitch_pool = cls._MALE_PITCH if gender == "male" else cls._FEMALE_PITCH
        age_label, pace_label, phrase_guidance = cls._age_settings(age)
        age_texture = cls._age_vocal_character(age)
        traits = {
            "voice_family": distinct(cls._VOICE_FAMILIES, 5, 1),
            "pitch": distinct(pitch_pool, 3, 0),
            "resonance": distinct(cls._RESONANCE, 3, 2),
            "texture": distinct(cls._TEXTURE, 5, 3),
            "vocal_weight": distinct(cls._VOCAL_WEIGHT, 3, 4),
            "articulation": distinct(cls._ARTICULATION, 7, 5),
            "personality": distinct(cls._PERSONALITY, 5, 6),
            "cadence": distinct(cls._CADENCE, 7, 7),
            "melody": distinct(cls._MELODY, 3, 1),
            "breath_style": distinct(cls._BREATH_STYLE, 5, 2),
            "age_character": age_texture,
        }
        notes = " ".join(description.strip().split())
        if len(notes) > 700:
            notes = notes[:697].rstrip() + "..."

        effective = (
            f"Create one original, believable {age}-year-old American {gender_noun}. "
            "This must be a new fictional speaker identity and must not imitate any known real person. "
            "Use natural General American English pronunciation, but do not make the voice generic or identical to other American speakers. "
            f"The speaker's social voice family is a {traits['voice_family']}. "
            f"The vocal identity has {traits['pitch']}, {traits['resonance']}, {traits['vocal_weight']}, and {traits['texture']}. "
            f"Use {traits['melody']}. The person is {traits['personality']}. "
            f"Use {traits['articulation']}, {traits['cadence']}, and {traits['breath_style']}. "
            f"The age impression is {age_label}: {age_texture}. The speaking pace is {pace_label}. {phrase_guidance} "
            "The requested age must change the natural vocal texture, physical energy, phrase planning, breath placement, and sentence release. "
            "Do not create a young voice and merely slow the playback. Create the age naturally, not by stretching vowels or globally time-stretching the recording. "
            f"The emotional baseline is {cls._emotion_text(emotion)}. "
            "Speak privately to one trusted listener in a quiet room, as if sharing a real memory, warning, or piece of advice from personal experience. "
            "Use tiny human variations in pitch, timing, energy, breath, emphasis, and sentence rhythm. Connect ordinary words naturally and give meaningful words quiet attention. "
            "Do not sound like an AI assistant, audiobook narrator, advertisement, newsreader, radio host, documentary narrator, customer-service agent, or motivational stage speaker. "
            "Avoid perfect timing, repeated sentence melody, identical pause lengths, excessive clarity, constant dramatic emphasis, and a repeated falling tone. "
            "Keep the speaker mentally clear and emotionally connected. Do not exaggerate weakness, shaking, gasping, sickness, confusion, or frailty. "
            "The recording is close-microphone, dry, intimate, clear, and free of music, echo, and background noise."
        )
        if notes:
            effective += f" Additional direction: {notes}"

        return VoiceProfile(
            age=age,
            gender=gender,
            language=language,
            emotion=emotion,
            age_label=age_label,
            pace_label=pace_label,
            phrase_guidance=phrase_guidance,
            identity_traits=traits,
            effective_description=effective,
            recommended_speed_factor=1.0,
        )

    def _validate_runner(self) -> None:
        python_path = Path(self.python_executable)
        if not python_path.exists():
            raise RuntimeError(
                "The isolated Qwen3-TTS Generate Voice environment is missing. "
                "Restart the Colab runtime and run every installation cell again. "
                f"Expected Python executable: {python_path}"
            )
        if not self.worker_path.exists():
            raise RuntimeError(f"Generate Voice worker was not found: {self.worker_path}")

    @staticmethod
    def _extract_result(stdout: str) -> dict[str, Any]:
        for line in reversed(stdout.splitlines()):
            if line.startswith(VoiceDesigner.RESULT_PREFIX):
                payload = line[len(VoiceDesigner.RESULT_PREFIX):]
                result = json.loads(payload)
                if not isinstance(result, dict):
                    break
                return result
        raise RuntimeError(f"Qwen3-TTS worker did not return a result payload.\n{stdout[-8000:]}")

    def generate_candidates(
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
        candidate_count: int = 3,
        uniqueness_threshold: float = 0.78,
    ) -> dict[str, Any]:
        with self._lock:
            self._validate_runner()
            candidate_count = max(1, min(4, int(candidate_count)))
            session_id = uuid.uuid4().hex
            session_dir = self.candidate_root / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            profiles = [
                self.build_profile(
                    age=age,
                    gender=gender,
                    language=language,
                    emotion=emotion,
                    description=description,
                    seed=seed,
                    candidate_index=index,
                )
                for index in range(candidate_count)
            ]
            request = {
                "model_id": self.MODEL_ID,
                "model_revision": self.MODEL_REVISION,
                "model_cache_dir": str(self.model_cache_dir),
                "name": name.strip(),
                "sample_text": sample_text.strip(),
                "base_seed": int(seed),
                "device": self.device,
                "session_id": session_id,
                "session_dir": str(session_dir),
                "saved_voice_dir": str(self.output_dir),
                "candidate_count": candidate_count,
                "uniqueness_threshold": float(uniqueness_threshold),
                "profiles": [asdict(profile) for profile in profiles],
            }

            request_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix="softmeta_qwen_voice_",
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
                    shutil.rmtree(session_dir, ignore_errors=True)
                    details = (result.stdout or "No worker output.")[-12000:]
                    raise RuntimeError(
                        "The isolated Qwen3-TTS Generate Voice worker failed.\n"
                        f"Python: {self.python_executable}\n"
                        f"Exit code: {result.returncode}\n"
                        f"Worker output:\n{details}"
                    )
                payload = self._extract_result(result.stdout or "")
                candidates = payload.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    raise RuntimeError("Qwen3-TTS finished without creating voice candidates.")
                return payload
            except subprocess.TimeoutExpired as error:
                shutil.rmtree(session_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Generate Voice exceeded the {self.timeout_seconds // 60}-minute timeout."
                ) from error
            finally:
                if request_path is not None:
                    request_path.unlink(missing_ok=True)

    def save_candidate(self, *, session_id: str, filename: str, voice_name: str) -> Path:
        with self._lock:
            session_dir = resolve_inside(self.candidate_root, session_id)
            if not session_dir.is_dir():
                raise FileNotFoundError(session_id)
            source = resolve_inside(session_dir, filename)
            if not source.is_file() or source.suffix.lower() != ".wav":
                raise FileNotFoundError(filename)

            metadata_source = source.with_suffix(".json")
            metadata: dict[str, Any] = {}
            if metadata_source.exists():
                try:
                    loaded = json.loads(metadata_source.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metadata = loaded
                except (OSError, json.JSONDecodeError):
                    metadata = {}

            stem = safe_filename(voice_name, "Generated_Voice")
            destination = self.output_dir / f"{stem}.wav"
            counter = 2
            while destination.exists():
                destination = self.output_dir / f"{stem}_{counter}.wav"
                counter += 1
            shutil.copy2(source, destination)

            embedding_source = source.with_suffix(".ecapa.npy")
            if embedding_source.exists():
                shutil.copy2(embedding_source, destination.with_suffix(".ecapa.npy"))

            metadata.update(
                {
                    "name": voice_name.strip(),
                    "filename": destination.name,
                    "saved_at": time.time(),
                    "source": "Qwen3-TTS VoiceDesign",
                    "model_id": self.MODEL_ID,
                    "model_revision": self.MODEL_REVISION,
                }
            )
            destination.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return destination
