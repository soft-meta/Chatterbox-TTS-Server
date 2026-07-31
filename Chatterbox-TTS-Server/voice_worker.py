from __future__ import annotations

import argparse
import gc
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
DEFAULT_MODEL_REVISION = "fa0251e3279a10b4936dc49d69a59c41b07cbfc0"
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json",
    "speech_tokenizer/model.safetensors",
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
        raise RuntimeError("Qwen3-TTS returned an invalid audio sample.")
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio * (0.96 / peak)
    return audio


def resolve_verified_model_snapshot(data: dict[str, Any]) -> Path:
    """Download and validate the exact official VoiceDesign snapshot.

    Qwen3TTSModel loads its processor from the same local directory as the
    model. Using a verified local snapshot prevents an older partial Hub cache
    from omitting speech_tokenizer/preprocessor_config.json.
    """

    try:
        from huggingface_hub import snapshot_download
    except Exception as error:
        raise RuntimeError(
            "huggingface-hub is required to prepare the verified Qwen3-TTS model snapshot. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    model_id = str(data["model_id"])
    revision = str(data.get("model_revision") or DEFAULT_MODEL_REVISION)
    cache_root = Path(
        str(
            data.get("model_cache_dir")
            or os.getenv("SOFTMETA_QWEN_MODEL_DIR")
            or "/content/softmeta_models/qwen3_voice_design"
        )
    ).resolve()
    local_dir = cache_root / revision[:12]
    local_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_MODEL_FILES if not (local_dir / name).is_file()]
    if missing:
        print(
            "Preparing verified Qwen3-TTS VoiceDesign snapshot "
            f"{revision[:12]}... This first download is several gigabytes."
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
                "The verified Qwen3-TTS VoiceDesign snapshot could not be downloaded. "
                f"Model: {model_id} Revision: {revision}. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error

    missing = [name for name in REQUIRED_MODEL_FILES if not (local_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "The downloaded Qwen3-TTS snapshot is incomplete. Missing: "
            + ", ".join(missing)
            + ". Delete the local Qwen model directory and try again."
        )

    print(f"Verified Qwen3-TTS model snapshot: {local_dir}")
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


def generate_candidates(data: dict[str, Any]) -> dict[str, Any]:
    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as error:
        raise RuntimeError(
            "Qwen3-TTS could not be imported in the isolated Generate Voice environment. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    requested_device = str(data.get("device", "cuda"))
    device = "cuda:0" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and device == "cpu":
        raise RuntimeError("CUDA was requested for Generate Voice but is unavailable.")
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.startswith("cuda") else torch.float32
    )

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
    attempted_count = 0

    model_path = resolve_verified_model_snapshot(data)
    model = None
    try:
        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        temperatures = (0.76, 0.88, 0.98, 1.06, 0.82, 0.94)
        top_ps = (0.86, 0.91, 0.97, 0.99, 0.89, 0.95)
        top_ks = (35, 50, 70, 90, 45, 80)

        for attempt_index in range(max_attempts):
            if len(accepted) >= requested_count:
                break
            attempted_count += 1
            profile = profiles[attempt_index]
            candidate_seed = (base_seed + attempt_index * 104_729 + 7_919) % 2_147_483_647
            set_seed(candidate_seed)
            wavs, sample_rate = model.generate_voice_design(
                text=sample_text,
                language="English",
                instruct=str(profile["effective_description"]),
                do_sample=True,
                temperature=temperatures[attempt_index % len(temperatures)],
                top_p=top_ps[attempt_index % len(top_ps)],
                top_k=top_ks[attempt_index % len(top_ks)],
                repetition_penalty=1.07,
                max_new_tokens=2048,
            )
            if not wavs:
                raise RuntimeError(f"Qwen3-TTS did not return attempt {attempt_index + 1}.")

            audio = normalise_audio(wavs[0])
            attempt_path = session_dir / f"attempt_{attempt_index + 1}.wav"
            sf.write(attempt_path, audio, int(sample_rate), subtype="PCM_16")

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
                "source": "Qwen3-TTS VoiceDesign",
                "uniqueness": uniqueness,
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

            status = uniqueness["status"]
            if status in {"baseline", "unique", "not_checked"}:
                accepted.append(artifact)
            elif status == "review":
                review_pool.append(artifact)
            else:
                rejected_count += 1
                attempt_path.unlink(missing_ok=True)
                attempt_path.with_suffix(".ecapa.npy").unlink(missing_ok=True)

        if len(accepted) < requested_count and review_pool:
            review_pool.sort(
                key=lambda item: float(item["uniqueness"].get("max_similarity") or -1.0)
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
