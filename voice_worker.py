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
            "difference_score": None,
            "closest_voice": None,
            "status": "not_checked",
        }
    if not references:
        return {
            "checked": True,
            "max_similarity": 0.0,
            "difference_score": 100.0,
            "closest_voice": None,
            "status": "unique",
        }
    similarities = [(name, cosine_similarity(embedding, reference)) for name, reference in references]
    closest_name, maximum = max(similarities, key=lambda item: item[1])
    maximum = max(-1.0, min(1.0, maximum))
    difference = max(0.0, min(100.0, (1.0 - maximum) * 100.0))
    if maximum >= threshold:
        status = "too_similar"
    elif maximum >= threshold - 0.08:
        status = "review"
    else:
        status = "unique"
    return {
        "checked": True,
        "max_similarity": round(maximum, 4),
        "difference_score": round(difference, 1),
        "closest_voice": closest_name,
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
    candidate_count = min(int(data["candidate_count"]), len(profiles))
    base_seed = int(data["base_seed"])
    sample_text = str(data["sample_text"]).strip()
    threshold = float(data.get("uniqueness_threshold", 0.78))

    model_path = resolve_verified_model_snapshot(data)
    model = None
    candidates: list[dict[str, Any]] = []
    try:
        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        for index in range(candidate_count):
            candidate_seed = (base_seed + index * 104_729) % 2_147_483_647
            set_seed(candidate_seed)
            profile = profiles[index]
            wavs, sample_rate = model.generate_voice_design(
                text=sample_text,
                language="English",
                instruct=str(profile["effective_description"]),
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=2048,
            )
            if not wavs:
                raise RuntimeError(f"Qwen3-TTS did not return candidate {index + 1}.")
            audio = normalise_audio(wavs[0])
            output_path = session_dir / f"candidate_{index + 1}.wav"
            sf.write(output_path, audio, int(sample_rate), subtype="PCM_16")
            metadata = {
                "name": str(data.get("name") or "Generated Voice"),
                "filename": output_path.name,
                "candidate_number": index + 1,
                "seed": candidate_seed,
                "sample_text": sample_text,
                "model_id": str(data["model_id"]),
                "model_revision": str(data.get("model_revision") or DEFAULT_MODEL_REVISION),
                "source": "Qwen3-TTS VoiceDesign",
                **profile,
            }
            output_path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            info = sf.info(output_path)
            candidates.append(
                {
                    **metadata,
                    "duration": round(float(info.duration), 3),
                    "size": output_path.stat().st_size,
                }
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    classifier = load_embedding_model()
    references = existing_embeddings(Path(str(data["saved_voice_dir"])), classifier)
    candidate_references: list[tuple[str, np.ndarray]] = []
    for candidate in candidates:
        path = session_dir / candidate["filename"]
        embedding: np.ndarray | None = None
        if classifier is not None:
            try:
                embedding = embedding_for(path, classifier)
            except Exception as error:
                print(f"Warning: uniqueness check failed for {path.name}: {type(error).__name__}: {error}")
        uniqueness = evaluate_uniqueness(embedding, references + candidate_references, threshold)
        candidate["uniqueness"] = uniqueness
        if embedding is not None:
            candidate_references.append((f"Candidate {candidate['candidate_number']}", embedding))
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["uniqueness"] = uniqueness
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "session_id": str(data["session_id"]),
        "model_id": str(data["model_id"]),
        "model_revision": str(data.get("model_revision") or DEFAULT_MODEL_REVISION),
        "candidate_count": len(candidates),
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
