from __future__ import annotations

import hashlib
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
    identity_code: str
    identity_summary: str


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
    _VOCAL_ANATOMY = (
        "a long vocal tract impression with a broad lower resonance",
        "a compact vocal tract impression with focused midrange energy",
        "a narrow throat setting with lean, dry resonance",
        "an open pharyngeal setting with rounded low harmonics",
        "a smaller oral cavity impression with a brighter intimate tone",
        "a broad oral cavity impression with dense chest support",
        "a relaxed laryngeal setting with mellow lower overtones",
        "a slightly raised laryngeal setting with a clear forward tone",
        "a deep-set speaking mechanism with restrained upper brightness",
        "a light, agile speaking mechanism with compact resonance",
        "a heavy-set speaking mechanism with slow harmonic release",
        "a balanced everyday speaking mechanism with an unusual dry edge",
    )
    _NASAL_BALANCE = (
        "almost no nasal colour",
        "a faint natural nasal edge",
        "a moderately forward nasal balance",
        "a warm oral tone with only slight nasal leakage",
        "a dry, focused nasal-orality mix",
        "a broad oral tone with occasional nasal consonant colour",
    )
    _VOWEL_SHAPE = (
        "wide relaxed vowels",
        "compact centred vowels",
        "rounded back vowels",
        "slightly flattened everyday vowels",
        "clear front vowels with softened endings",
        "narrow vowels with warm lower resonance",
        "open conversational vowels with natural reduction",
        "small precise vowels without announcer polish",
    )
    _CONSONANT_ATTACK = (
        "soft consonant attacks",
        "firm but unforced consonant attacks",
        "slightly breath-led consonant starts",
        "dry compact consonants",
        "clear advice words with relaxed ordinary consonants",
        "casual consonant reduction in unimportant words",
        "light tongue-forward consonants",
        "rounded consonants with softened final sounds",
    )
    _SPECTRAL_COLOUR = (
        "dark spectral colour with restrained brightness",
        "warm balanced spectral colour",
        "dry mid-forward spectral colour",
        "soft smoky spectral colour",
        "clear upper-mid colour without youthful sparkle",
        "dense low-mid colour with a thin airy edge",
        "light reedy colour with grounded resonance",
        "broad mellow colour with gentle high-frequency roll-off",
    )
    _SPEAKING_HABITS = (
        "often begins thoughts softly before settling into them",
        "uses brief direct openings and more reflective endings",
        "occasionally holds a meaningful word for a fraction longer",
        "connects everyday words but separates advice clearly",
        "uses small hesitations before personal memories",
        "lets some sentences trail gently instead of always closing firmly",
        "uses quiet emphasis instead of extra loudness",
        "sometimes starts the next thought before the previous energy fully disappears",
        "alternates concise statements with slower reflective phrases",
        "uses restrained questions that sound genuinely curious",
    )

    def __init__(
        self,
        output_dir: Path,
        candidate_root: Path,
        device: str = "cuda",
        *,
        python_executable: str | None = None,
        worker_path: Path | None = None,
        timeout_seconds: int = 5400,
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
        identity_seed = (int(seed) + (candidate_index + 1) * 104_729 + 7_919) % 2_147_483_647
        randomiser = random.Random(identity_seed)
        gender_noun = "man" if gender == "male" else "woman"

        # Each attempt is intentionally pushed into different vocal regions.
        def choose(sequence: tuple[str, ...], salt: int) -> str:
            local = random.Random(identity_seed ^ (salt * 1_000_003))
            return sequence[local.randrange(len(sequence))]

        pitch_pool = cls._MALE_PITCH if gender == "male" else cls._FEMALE_PITCH
        age_label, pace_label, phrase_guidance = cls._age_settings(age)
        age_texture = cls._age_vocal_character(age)
        traits = {
            "voice_family": choose(cls._VOICE_FAMILIES, 1),
            "pitch": choose(pitch_pool, 2),
            "vocal_anatomy": choose(cls._VOCAL_ANATOMY, 3),
            "resonance": choose(cls._RESONANCE, 4),
            "spectral_colour": choose(cls._SPECTRAL_COLOUR, 5),
            "nasal_balance": choose(cls._NASAL_BALANCE, 6),
            "texture": choose(cls._TEXTURE, 7),
            "vocal_weight": choose(cls._VOCAL_WEIGHT, 8),
            "vowel_shape": choose(cls._VOWEL_SHAPE, 9),
            "consonant_attack": choose(cls._CONSONANT_ATTACK, 10),
            "articulation": choose(cls._ARTICULATION, 11),
            "personality": choose(cls._PERSONALITY, 12),
            "speaking_habit": choose(cls._SPEAKING_HABITS, 13),
            "cadence": choose(cls._CADENCE, 14),
            "melody": choose(cls._MELODY, 15),
            "breath_style": choose(cls._BREATH_STYLE, 16),
            "age_character": age_texture,
        }
        fingerprint_source = json.dumps(
            {"age": age, "gender": gender, "identity_seed": identity_seed, "traits": traits},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
        identity_code = "SM-" + hashlib.sha256(fingerprint_source).hexdigest()[:10].upper()
        identity_summary = (
            f"{traits['pitch']}; {traits['vocal_anatomy']}; {traits['spectral_colour']}; "
            f"{traits['nasal_balance']}; {traits['texture']}"
        )

        notes = " ".join(description.strip().split())
        if len(notes) > 700:
            notes = notes[:697].rstrip() + "..."

        effective = (
            "Create one completely original fictional speaker. Speaker identity is the primary requirement; acting style is secondary. "
            f"Identity code: {identity_code}. The speaker is a {age}-year-old American {gender_noun}. "
            "Do not reuse a familiar house narrator, default male voice, or recurring synthetic speaker. "
            "Do not create the same person with only a different pitch, tempo, age performance, or emotional tune. "
            "Construct a genuinely different perceived vocal anatomy and stable human identity. "
            f"Use {traits['vocal_anatomy']}, {traits['pitch']}, {traits['resonance']}, {traits['spectral_colour']}, "
            f"{traits['nasal_balance']}, {traits['vocal_weight']}, and {traits['texture']}. "
            f"Shape speech with {traits['vowel_shape']} and {traits['consonant_attack']}. "
            f"The social speaking character is a {traits['voice_family']}; the person is {traits['personality']} and {traits['speaking_habit']}. "
            "Use General American English pronunciation, but preserve an individual human voice rather than a generic American announcer. "
            f"Use {traits['articulation']}, {traits['cadence']}, {traits['melody']}, and {traits['breath_style']}. "
            f"Only after establishing the unique identity, apply the age impression: {age_label}; {age_texture}. "
            f"The natural speaking pace is {pace_label}. {phrase_guidance} "
            "Age must affect vocal texture, projection, breath support, thought grouping, phrase planning, and sentence release. "
            "Do not imitate age by globally slowing or time-stretching the recording. Create age naturally, not by stretching vowels or unnaturally lengthening individual words. "
            f"The emotional baseline is {cls._emotion_text(emotion)}. "
            "Speak privately to one trusted listener in a quiet room, as if sharing a real memory, warning, or piece of advice from personal experience. "
            "Use naturally imperfect timing and small changes in pitch, energy, breath, emphasis, and rhythm. "
            "Do not pronounce every ordinary word perfectly. Connect unimportant words naturally and give meaningful words quiet attention. "
            "Do not sound like an AI assistant, audiobook narrator, advertisement, newsreader, radio host, documentary narrator, customer-service agent, or motivational stage speaker. "
            "Avoid repeated sentence melody, identical pause lengths, exaggerated frailty, theatrical trembling, and polished synthetic smoothness. "
            "Keep the speaker mentally clear, emotionally connected, and physically believable. "
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
            identity_code=identity_code,
            identity_summary=identity_summary,
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
        uniqueness_threshold: float = 0.72,
    ) -> dict[str, Any]:
        with self._lock:
            self._validate_runner()
            candidate_count = max(1, min(4, int(candidate_count)))
            max_attempts = min(12, max(candidate_count * 3, candidate_count + 4))
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
                for index in range(max_attempts)
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
                "max_attempts": max_attempts,
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
