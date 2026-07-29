from __future__ import annotations

import argparse
import gc
import json
import random
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

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = None
    try:
        model_id = str(data["model_id"])
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        prompt_tokenizer = AutoTokenizer.from_pretrained(model_id)
        description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path
        )

        description_ids = description_tokenizer(
            str(data["description"]),
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_ids = prompt_tokenizer(
            str(data["sample_text"]),
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.inference_mode():
            generation = model.generate(
                input_ids=description_ids,
                prompt_input_ids=prompt_ids,
                do_sample=True,
            )

        audio = generation.detach().float().cpu().numpy().squeeze()
        if audio.ndim != 1 or audio.size < 1000:
            raise RuntimeError("Parler-TTS returned an invalid audio sample.")

        output_path = Path(str(data["output_path"])).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            output_path,
            audio,
            int(model.config.sampling_rate),
            subtype="PCM_16",
        )
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
