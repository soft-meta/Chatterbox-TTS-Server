from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_request(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_id", "description", "sample_text", "seed", "device", "output_path"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Voice request is missing: {', '.join(missing)}")
    return data


def apply_tempo(path: Path, tempo: float, sample_rate: int) -> None:
    """Apply gentle age pacing without changing pitch.

    Parler-TTS is prompted to speak at the requested pace first. This small DSP
    correction makes age bands more dependable while avoiding extreme stretching.
    """
    tempo = max(0.5, min(2.0, float(tempo)))
    if abs(tempo - 1.0) < 0.005 or not shutil.which("ffmpeg"):
        return
    temporary = path.with_name(path.stem + ".tempo.wav")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-filter:a",
        f"atempo={tempo:.4f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0 and temporary.exists() and temporary.stat().st_size > 1000:
        temporary.replace(path)
    else:
        temporary.unlink(missing_ok=True)
        print(f"Warning: age tempo correction was skipped: {result.stderr[-1000:]}")


def generate_voice(data: dict) -> Path:
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except Exception as error:
        raise RuntimeError(
            "Parler-TTS could not be imported in the isolated voice environment. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    seed = int(data["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = str(data.get("device", "cuda"))
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA was requested for Generate Voice but is unavailable.")

    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = None
    try:
        model_id = str(data["model_id"])
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()

        prompt_tokenizer = AutoTokenizer.from_pretrained(model_id)
        description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path
        )

        description_inputs = description_tokenizer(
            str(data["description"]),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        prompt_inputs = prompt_tokenizer(
            str(data["sample_text"]),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            generation = model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=True,
                temperature=1.0,
                min_new_tokens=10,
            )

        audio = generation.detach().float().cpu().numpy().squeeze()
        if audio.ndim != 1 or audio.size < 1000:
            raise RuntimeError("Parler-TTS returned an invalid audio sample.")

        output_path = Path(str(data["output_path"])).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(model.config.sampling_rate)
        sf.write(
            output_path,
            audio,
            sample_rate,
            subtype="PCM_16",
        )
        apply_tempo(output_path, float(data.get("sample_tempo", 1.0)), sample_rate)
        print(json.dumps({"ok": True, "output_path": str(output_path)}))
        return output_path
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    generate_voice(load_request(args.request))


if __name__ == "__main__":
    main()
