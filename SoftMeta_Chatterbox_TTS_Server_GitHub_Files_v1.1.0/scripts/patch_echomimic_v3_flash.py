from __future__ import annotations

import sys
from pathlib import Path


MARKER = ".softmeta_echomimic_v3_flash_v110"


def patch_runtime(repo: Path) -> None:
    target = repo / "infer_flash.py"
    if not target.is_file():
        raise SystemExit(f"EchoMimicV3 Flash entrypoint was not found: {target}")
    marker = repo / MARKER
    if marker.is_file():
        return

    source = target.read_text(encoding="utf-8")
    replacements = {
        '    args = parse_args()':
            '    args = parse_args()\n    print("SOFTMETA_STAGE loading_models", flush=True)',
        'parser.add_argument("--save_path", type=str, default="outputs", help="Save path")':
            'parser.add_argument("--save_path", type=str, default="outputs", help="Save path")\n'
            '    parser.add_argument("--batch_manifest", type=str, default=None, help="SoftMeta JSON batch manifest")',
        'parser.add_argument("--image_path", type=str, required=True, help="Input image path")':
            'parser.add_argument("--image_path", type=str, default=None, help="Input image path")',
        'parser.add_argument("--audio_path", type=str, required=True, help="Input audio path")':
            'parser.add_argument("--audio_path", type=str, default=None, help="Input audio path")',
        'parser.add_argument("--prompt", type=str, required=True, help="Text prompt")':
            'parser.add_argument("--prompt", type=str, default="A person is speaking.", help="Text prompt")',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise SystemExit(f"EchoMimicV3 argument block changed; refusing unsafe patch: {old}")
        source = source.replace(old, new, 1)

    start_token = "    # Create output directory\n"
    end_token = "\nif __name__ == \"__main__\":\n"
    if source.count(start_token) != 1 or source.count(end_token) != 1:
        raise SystemExit("EchoMimicV3 inference body changed; refusing unsafe batch patch.")
    start = source.index(start_token)
    end = source.index(end_token)
    body = source[start:end]
    indented_body = "\n".join("    " + line if line else line for line in body.splitlines())
    replacement = '''    if args.batch_manifest:
        with open(args.batch_manifest, "r", encoding="utf-8") as stream:
            batch_items = json.load(stream)
        if not isinstance(batch_items, list) or not batch_items:
            raise ValueError("batch_manifest must contain a non-empty JSON list")
    else:
        if not image_path or not audio_path:
            raise ValueError("image_path and audio_path are required without batch_manifest")
        batch_items = [{
            "image_path": image_path,
            "audio_path": audio_path,
            "save_path": save_path,
            "prompt": prompt,
            "video_length": video_length,
            "seed": seed,
        }]

    batch_total = len(batch_items)
    print(f"SOFTMETA_STAGE models_ready batch_total={batch_total}", flush=True)
    for batch_index, batch_item in enumerate(batch_items, start=1):
        image_path = str(batch_item["image_path"])
        audio_path = str(batch_item["audio_path"])
        save_path = str(batch_item["save_path"])
        prompt = str(batch_item.get("prompt") or prompt)
        video_length = int(batch_item.get("video_length") or video_length)
        seed = int(batch_item.get("seed") or seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"SOFTMETA_PROGRESS {batch_index - 1} {batch_total} preparing", flush=True)
''' + indented_body + '''
        print(f"SOFTMETA_PROGRESS {batch_index} {batch_total} completed", flush=True)
'''
    source = source[:start] + replacement + source[end:]
    target.write_text(source, encoding="utf-8")
    marker.write_text("SoftMeta EchoMimicV3 Flash batch patch v1.1.0\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_echomimic_v3_flash.py /path/to/echomimic_v3")
    patch_runtime(Path(sys.argv[1]).resolve())
