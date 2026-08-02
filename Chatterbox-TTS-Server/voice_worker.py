from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

RESULT_PREFIX = "SOFTMETA_RESULT="
DEFAULT_MODEL_REVISION = "97521ec"
REQUIRED_MODEL_FILES = (
    "config.json",
    "processor_config.json",
    "model.safetensors",
    "configuration_moss_tts.py",
    "modeling_moss_tts.py",
    "processing_moss_tts.py",
    "inference_utils.py",
    "tokenizer_config.json",
)


def load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_id",
        "sample_text",
        "base_seed",
        "device",
        "session_id",
        "session_dir",
        "saved_voice_dir",
        "candidate_count",
        "profiles",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Voice request is missing: {', '.join(missing)}")
    return data


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < 1000 or not np.all(np.isfinite(audio)):
        raise RuntimeError("MOSS VoiceGenerator returned an invalid audio sample.")
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio * (0.96 / peak)
    return audio


def resolve_verified_model_snapshot(data: dict[str, Any]) -> Path:
    """Download and validate a pinned official MOSS VoiceGenerator snapshot."""

    try:
        from huggingface_hub import snapshot_download
    except Exception as error:
        raise RuntimeError(
            "huggingface-hub is required to prepare MOSS VoiceGenerator. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    model_id = str(data["model_id"])
    revision = str(data.get("model_revision") or DEFAULT_MODEL_REVISION)
    cache_root = Path(
        str(
            data.get("model_cache_dir")
            or os.getenv("SOFTMETA_MOSS_MODEL_DIR")
            or "/content/softmeta_models/moss_voice_generator"
        )
    ).resolve()
    local_dir = cache_root / revision[:12]
    local_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_MODEL_FILES if not (local_dir / name).is_file()]
    if missing:
        print(
            "Preparing official MOSS VoiceGenerator snapshot "
            f"{revision[:12]}... The first download is several gigabytes."
        )
        try:
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(local_dir),
                max_workers=4,
            )
        except Exception as error:
            raise RuntimeError(
                "MOSS VoiceGenerator could not be downloaded. "
                f"Model: {model_id} Revision: {revision}. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error

    missing = [name for name in REQUIRED_MODEL_FILES if not (local_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "The downloaded MOSS VoiceGenerator snapshot is incomplete. Missing: "
            + ", ".join(missing)
            + ". Delete the local MOSS model directory and try again."
        )

    print(f"Verified MOSS VoiceGenerator model snapshot: {local_dir}")
    return local_dir


def load_embedding_model():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as error:
        print(f"Warning: speaker uniqueness checker is unavailable: {type(error).__name__}: {error}")
        return None

    try:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/content/hf_home/speechbrain-spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    except Exception as error:
        print(f"Warning: speaker embedding model could not load: {type(error).__name__}: {error}")
        return None


def audio_for_embedding(path: Path, target_sr: int = 16000) -> torch.Tensor:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sample_rate), target_sr)
        audio = resample_poly(audio, target_sr // gcd, int(sample_rate) // gcd).astype(np.float32)
    if audio.size < target_sr:
        audio = np.pad(audio, (0, target_sr - audio.size))
    return torch.from_numpy(audio).unsqueeze(0)


def embedding_for(path: Path, classifier) -> np.ndarray:
    cache = path.with_suffix(".ecapa.npy")
    if cache.exists():
        try:
            embedding = np.load(cache)
            if embedding.ndim == 1 and embedding.size > 0:
                return embedding.astype(np.float32, copy=False)
        except Exception:
            pass
    signal = audio_for_embedding(path)
    with torch.inference_mode():
        embedding = classifier.encode_batch(signal).squeeze().detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding /= norm
    np.save(cache, embedding)
    return embedding


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denominator)


def existing_embeddings(directory: Path, classifier) -> list[tuple[str, np.ndarray]]:
    if classifier is None or not directory.exists():
        return []
    result: list[tuple[str, np.ndarray]] = []
    for path in sorted(directory.glob("*.wav")):
        try:
            result.append((path.name, embedding_for(path, classifier)))
        except Exception as error:
            print(f"Warning: could not analyse {path.name}: {type(error).__name__}: {error}")
    return result


def evaluate_uniqueness(
    embedding: np.ndarray | None,
    references: list[tuple[str, np.ndarray]],
    threshold: float,
) -> dict[str, Any]:
    if embedding is None:
        return {
            "checked": False,
            "max_similarity": None,
            "similarity_percent": None,
            "difference_score": None,
            "closest_voice": None,
            "reference_count": len(references),
            "status": "not_checked",
        }
    if not references:
        return {
            "checked": True,
            "max_similarity": None,
            "similarity_percent": None,
            "difference_score": None,
            "closest_voice": None,
            "reference_count": 0,
            "status": "baseline",
        }
    similarities = [(name, cosine_similarity(embedding, reference)) for name, reference in references]
    closest_name, maximum = max(similarities, key=lambda item: item[1])
    maximum = max(-1.0, min(1.0, maximum))
    similarity_percent = max(0.0, min(100.0, maximum * 100.0))
    difference = max(0.0, min(100.0, (1.0 - maximum) * 100.0))
    review_floor = max(0.50, threshold - 0.10)
    if maximum >= threshold:
        status = "too_similar"
    elif maximum >= review_floor:
        status = "review"
    else:
        status = "unique"
    return {
        "checked": True,
        "max_similarity": round(maximum, 4),
        "similarity_percent": round(similarity_percent, 1),
        "difference_score": round(difference, 1),
        "closest_voice": closest_name,
        "reference_count": len(references),
        "status": status,
    }


def _frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    frame_count = 1 + max(0, (audio.size - frame_length) // hop_length)
    values = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_length
        frame = audio[start : start + frame_length]
        values[index] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    return values


def evaluate_acoustic_quality(audio: np.ndarray, sample_rate: int, age: int) -> dict[str, Any]:
    """Reject broken audio without forcing every older voice into one pitch range.

    The score intentionally avoids a hard pitch rule. Real older American men and
    women can have low, medium, or high voices. We screen only obvious technical
    failures, excessive dead air, clipping, and mechanically weak dynamics.
    """

    duration = float(audio.size / max(1, sample_rate))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-9))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if audio.size else 1.0

    frame_length = max(256, int(sample_rate * 0.025))
    hop_length = max(128, int(sample_rate * 0.010))
    frame_rms = _frame_rms(audio, frame_length, hop_length)
    active_reference = max(float(np.percentile(frame_rms, 90)), 1e-5)
    silence_threshold = max(active_reference * 0.055, 10 ** (-48 / 20))
    silence_ratio = float(np.mean(frame_rms < silence_threshold)) if frame_rms.size else 1.0
    active_duration = duration * (1.0 - silence_ratio)

    p10 = max(float(np.percentile(frame_rms, 10)), 1e-6)
    p90 = max(float(np.percentile(frame_rms, 90)), p10)
    dynamic_range_db = 20.0 * math.log10(p90 / p10)

    if age < 60:
        preferred_pause = (0.04, 0.28)
    elif age < 70:
        preferred_pause = (0.06, 0.31)
    elif age < 80:
        preferred_pause = (0.08, 0.35)
    elif age < 90:
        preferred_pause = (0.10, 0.40)
    else:
        preferred_pause = (0.12, 0.45)

    reasons: list[str] = []
    hard_reject = False
    if duration < 1.5:
        reasons.append("too short")
        hard_reject = True
    if active_duration < 1.2:
        reasons.append("too little active speech")
        hard_reject = True
    if silence_ratio > 0.62:
        reasons.append("excessive dead air")
        hard_reject = True
    if clipping_ratio > 0.01:
        reasons.append("clipping")
        hard_reject = True
    if rms_db < -38.0:
        reasons.append("audio level too low")
        hard_reject = True

    score = 100.0
    if silence_ratio < preferred_pause[0]:
        score -= min(14.0, (preferred_pause[0] - silence_ratio) * 90.0)
    elif silence_ratio > preferred_pause[1]:
        score -= min(24.0, (silence_ratio - preferred_pause[1]) * 90.0)
    if dynamic_range_db < 7.0:
        score -= min(18.0, (7.0 - dynamic_range_db) * 2.5)
    if rms_db < -28.0:
        score -= min(15.0, (-28.0 - rms_db) * 1.5)
    if clipping_ratio > 0.001:
        score -= min(20.0, clipping_ratio * 1000.0)
    score = max(0.0, min(100.0, score))

    if hard_reject:
        status = "reject"
    elif score < 72.0:
        status = "review"
        if not reasons:
            reasons.append("cadence or dynamics should be reviewed")
    else:
        status = "pass"

    return {
        "checked": True,
        "status": status,
        "score": round(score, 1),
        "duration": round(duration, 3),
        "active_duration": round(active_duration, 3),
        "silence_ratio": round(silence_ratio, 4),
        "rms_db": round(rms_db, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "clipping_ratio": round(clipping_ratio, 6),
        "preferred_pause_min": preferred_pause[0],
        "preferred_pause_max": preferred_pause[1],
        "reasons": reasons,
    }


def resolve_attn_implementation(device: torch.device, dtype: torch.dtype) -> str:
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"
    if device.type == "cuda":
        return "sdpa"
    return "eager"


def generate_candidates(data: dict[str, Any]) -> dict[str, Any]:
    try:
        from transformers import AutoModel, AutoProcessor
    except Exception as error:
        raise RuntimeError(
            "MOSS VoiceGenerator could not be imported in the isolated Generate Voice environment. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    requested_device = str(data.get("device", "cuda"))
    device = torch.device("cuda:0" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and device.type == "cpu":
        raise RuntimeError("CUDA was requested for Generate Voice but is unavailable.")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    session_dir = Path(str(data["session_dir"])).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    profiles = list(data["profiles"])
    requested_count = min(int(data["candidate_count"]), len(profiles))
    max_attempts = min(int(data.get("max_attempts", len(profiles))), len(profiles))
    base_seed = int(data["base_seed"])
    sample_text = str(data["sample_text"]).strip()
    threshold = float(data.get("uniqueness_threshold", 0.72))

    classifier = load_embedding_model()
    saved_references = existing_embeddings(Path(str(data["saved_voice_dir"])), classifier)
    batch_references: list[tuple[str, np.ndarray]] = []
    accepted: list[dict[str, Any]] = []
    review_pool: list[dict[str, Any]] = []
    rejected_count = 0
    duplicate_rejected_count = 0
    quality_rejected_count = 0
    attempted_count = 0

    model_path = resolve_verified_model_snapshot(data)
    model = None
    processor = None
    try:
        attn_implementation = resolve_attn_implementation(device, dtype)
        print(f"MOSS attention backend: {attn_implementation}")
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            normalize_inputs=True,
        )
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))

        temperatures = (1.35, 1.50, 1.65, 1.42, 1.58, 1.72, 1.46, 1.62)
        top_ps = (0.54, 0.60, 0.66, 0.57, 0.63, 0.69, 0.59, 0.65)
        top_ks = (40, 50, 65, 45, 60, 75, 55, 70)
        repetition_penalties = (1.08, 1.10, 1.12, 1.09, 1.11, 1.13)

        for attempt_index in range(max_attempts):
            if len(accepted) >= requested_count:
                break
            attempted_count += 1
            profile = profiles[attempt_index]
            candidate_seed = (base_seed + attempt_index * 104_729 + 7_919) % 2_147_483_647
            set_seed(candidate_seed)

            conversations = [[processor.build_user_message(
                text=sample_text,
                instruction=str(profile["effective_description"]),
            )]]
            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=4096,
                    audio_temperature=temperatures[attempt_index % len(temperatures)],
                    audio_top_p=top_ps[attempt_index % len(top_ps)],
                    audio_top_k=top_ks[attempt_index % len(top_ks)],
                    audio_repetition_penalty=repetition_penalties[
                        attempt_index % len(repetition_penalties)
                    ],
                )
            messages = processor.decode(outputs)
            if not messages or messages[0] is None or not messages[0].audio_codes_list:
                raise RuntimeError(f"MOSS VoiceGenerator did not return attempt {attempt_index + 1}.")
            raw_audio = messages[0].audio_codes_list[0]
            if isinstance(raw_audio, torch.Tensor):
                raw_audio = raw_audio.detach().float().cpu().numpy()
            audio = normalise_audio(np.asarray(raw_audio, dtype=np.float32))

            attempt_path = session_dir / f"attempt_{attempt_index + 1}.wav"
            sf.write(attempt_path, audio, sample_rate, subtype="PCM_16")

            quality = evaluate_acoustic_quality(audio, sample_rate, int(profile["age"]))
            embedding: np.ndarray | None = None
            if classifier is not None:
                try:
                    embedding = embedding_for(attempt_path, classifier)
                except Exception as error:
                    print(
                        f"Warning: uniqueness check failed for {attempt_path.name}: "
                        f"{type(error).__name__}: {error}"
                    )
            comparison_references = saved_references + batch_references
            uniqueness = evaluate_uniqueness(embedding, comparison_references, threshold)
            if embedding is not None:
                batch_references.append((f"Attempt {attempt_index + 1}", embedding))

            metadata = {
                "name": str(data.get("name") or "Generated Voice"),
                "filename": attempt_path.name,
                "candidate_number": 0,
                "attempt_number": attempt_index + 1,
                "seed": candidate_seed,
                "sample_text": sample_text,
                "model_id": str(data["model_id"]),
                "model_revision": str(data.get("model_revision") or DEFAULT_MODEL_REVISION),
                "source": "MOSS VoiceGenerator",
                "uniqueness": uniqueness,
                "quality": quality,
                "sampling": {
                    "audio_temperature": temperatures[attempt_index % len(temperatures)],
                    "audio_top_p": top_ps[attempt_index % len(top_ps)],
                    "audio_top_k": top_ks[attempt_index % len(top_ks)],
                    "audio_repetition_penalty": repetition_penalties[
                        attempt_index % len(repetition_penalties)
                    ],
                },
                **profile,
            }
            info = sf.info(attempt_path)
            artifact = {
                **metadata,
                "duration": round(float(info.duration), 3),
                "size": attempt_path.stat().st_size,
                "_path": attempt_path,
                "_embedding": embedding,
            }

            uniqueness_status = uniqueness["status"]
            quality_status = quality["status"]
            if uniqueness_status == "too_similar":
                duplicate_rejected_count += 1
                rejected_count += 1
                attempt_path.unlink(missing_ok=True)
                attempt_path.with_suffix(".ecapa.npy").unlink(missing_ok=True)
            elif quality_status == "reject":
                quality_rejected_count += 1
                rejected_count += 1
                attempt_path.unlink(missing_ok=True)
                attempt_path.with_suffix(".ecapa.npy").unlink(missing_ok=True)
            elif uniqueness_status in {"baseline", "unique", "not_checked"} and quality_status == "pass":
                accepted.append(artifact)
            else:
                review_pool.append(artifact)

        if len(accepted) < requested_count and review_pool:
            review_pool.sort(
                key=lambda item: (
                    0 if item["uniqueness"].get("status") in {"baseline", "unique", "not_checked"} else 1,
                    -float(item["quality"].get("score") or 0.0),
                    float(item["uniqueness"].get("max_similarity") or -1.0),
                )
            )
            needed = requested_count - len(accepted)
            accepted.extend(review_pool[:needed])
            for extra in review_pool[needed:]:
                path = extra["_path"]
                path.unlink(missing_ok=True)
                path.with_suffix(".ecapa.npy").unlink(missing_ok=True)
        else:
            for extra in review_pool:
                path = extra["_path"]
                path.unlink(missing_ok=True)
                path.with_suffix(".ecapa.npy").unlink(missing_ok=True)

        candidates: list[dict[str, Any]] = []
        for candidate_number, artifact in enumerate(accepted[:requested_count], start=1):
            old_path: Path = artifact.pop("_path")
            artifact.pop("_embedding", None)
            output_path = session_dir / f"candidate_{candidate_number}.wav"
            old_path.replace(output_path)
            old_embedding = old_path.with_suffix(".ecapa.npy")
            if old_embedding.exists():
                old_embedding.replace(output_path.with_suffix(".ecapa.npy"))
            artifact["filename"] = output_path.name
            artifact["candidate_number"] = candidate_number
            output_path.with_suffix(".json").write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            candidates.append(artifact)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "ok": True,
        "session_id": str(data["session_id"]),
        "model_id": str(data["model_id"]),
        "model_revision": str(data.get("model_revision") or DEFAULT_MODEL_REVISION),
        "requested_count": requested_count,
        "candidate_count": len(candidates),
        "attempted_count": attempted_count,
        "rejected_count": rejected_count,
        "duplicate_rejected_count": duplicate_rejected_count,
        "quality_rejected_count": quality_rejected_count,
        "search_exhausted": len(candidates) < requested_count,
        "candidates": candidates,
        "uniqueness_threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    result = generate_candidates(load_request(args.request))
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
