from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = ".softmeta_longcat_runtime_patch_v105"
PINNED_SOURCE_SHA = "5fc822ab6c27cf53fef0d9acb5821fec32846d76"
PINNED_PIPELINE_SHA = "5caf1c5a092c7f6a890310b0224d8d506ed2d2e8"


def patch_runtime(repo: Path) -> None:
    target = repo / "run_demo_avatar_single_audio_to_video.py"
    if not target.is_file():
        raise SystemExit(f"LongCat avatar entrypoint was not found: {target}")

    source = target.read_text(encoding="utf-8")
    seed_old = "    global_seed = 42\n"
    seed_new = '    global_seed = int(os.getenv("SOFTMETA_LONGCAT_SEED", "42"))\n'
    save_old = (
        "        if cp_rank == 0:\n"
        "            output_tensor = torch.from_numpy(np.array(all_generated_frames))\n"
        "            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f\"video_continue_{segment_idx+1}\"), raw_speech_path, fps=save_fps, quality=5)\n"
        "            del output_tensor\n"
    )
    save_new = (
        "        if cp_rank == 0:\n"
        "            output_tensor = torch.from_numpy(np.array(new_video[num_cond_frames:]))\n"
        "            softmeta_chunk_base = os.path.join(output_dir, f\"softmeta_chunk_{segment_idx+1:05d}\")\n"
        "            save_video_ffmpeg(output_tensor, softmeta_chunk_base, raw_speech_path, fps=save_fps, quality=5)\n"
        "            del output_tensor\n"
        "            softmeta_chunk_paths.append(softmeta_chunk_base + \".mp4\")\n"
        "            if segment_idx == num_segments - 1:\n"
        "                softmeta_concat_video_chunks(softmeta_chunk_paths, os.path.join(output_dir, f\"video_continue_{segment_idx+1}.mp4\"))\n"
    )
    import_old = "import argparse\n"
    import_new = "import argparse\nimport subprocess\n"
    helper_anchor = (
        "def torch_gc():\n"
        "    torch.cuda.empty_cache()\n"
        "    torch.cuda.ipc_collect()\n"
    )
    helper_new = helper_anchor + (
        "\n"
        "def softmeta_concat_video_chunks(chunks, output_path):\n"
        "    list_path = output_path + \".txt\"\n"
        "    with open(list_path, \"w\", encoding=\"utf-8\") as handle:\n"
        "        for chunk in chunks:\n"
        "            escaped = Path(chunk).resolve().as_posix().replace(\"'\", \"'\\\\''\")\n"
        "            handle.write(\"file '\" + escaped + \"'\\n\")\n"
        "    subprocess.run([\"ffmpeg\", \"-y\", \"-f\", \"concat\", \"-safe\", \"0\", \"-i\", list_path, \"-map\", \"0:v:0\", \"-an\", \"-c:v\", \"copy\", output_path], check=True)\n"
        "    os.remove(list_path)\n"
    )
    frames_old = "    all_generated_frames = video\n"
    frames_new = (
        "    first_clip_name = \"ai2v_demo_1.mp4\" if stage_1 == \"ai2v\" else \"at2v_demo_1.mp4\"\n"
        "    softmeta_chunk_paths = [os.path.join(output_dir, first_clip_name)]\n"
    )
    extend_old = "        all_generated_frames.extend(new_video[num_cond_frames:])\n"
    t5_old = (
        "    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', "
        "'LongCat-Video'), subfolder=\"text_encoder\", torch_dtype=torch.bfloat16)\n"
    )
    t5_new = (
        "    # SoftMeta: load the large T5 encoder after the quantized DiT to reduce host-RAM peak.\n"
        + t5_old
    )
    audio_anchor = "    # initialize audio models\n"

    if seed_new not in source:
        if source.count(seed_old) != 1:
            raise SystemExit("LongCat seed location changed; refusing an unsafe patch.")
        source = source.replace(seed_old, seed_new, 1)
    if import_new not in source:
        if source.count(import_old) != 1:
            raise SystemExit("LongCat import block changed; refusing an unsafe patch.")
        source = source.replace(import_old, import_new, 1)
    if "def softmeta_concat_video_chunks" not in source:
        if source.count(helper_anchor) != 1:
            raise SystemExit("LongCat utility block changed; refusing an unsafe patch.")
        source = source.replace(helper_anchor, helper_new, 1)
    if frames_new not in source:
        if source.count(frames_old) != 1 or source.count(extend_old) != 1:
            raise SystemExit("LongCat frame accumulator changed; refusing an unsafe patch.")
        source = source.replace(frames_old, frames_new, 1)
        source = source.replace(extend_old, "", 1)
    if save_new not in source:
        if source.count(save_old) != 1:
            raise SystemExit("LongCat continuation-save block changed; refusing an unsafe patch.")
        source = source.replace(save_old, save_new, 1)
    if t5_new not in source:
        if source.count(t5_old) != 1 or source.count(audio_anchor) != 1:
            raise SystemExit("LongCat text-encoder load block changed; refusing an unsafe patch.")
        source = source.replace(t5_old, "", 1)
        source = source.replace(audio_anchor, t5_new + "\n" + audio_anchor, 1)

    target.write_text(source, encoding="utf-8")

    pipeline_target = repo / "longcat_video" / "pipeline_longcat_video_avatar.py"
    if not pipeline_target.is_file():
        raise SystemExit(f"LongCat avatar pipeline was not found: {pipeline_target}")
    pipeline = pipeline_target.read_text(encoding="utf-8")
    encode_old = (
        "        prompt_embeds = self.text_encoder(text_input_ids.to(device), "
        "mask.to(device)).last_hidden_state\n"
    )
    encode_new = (
        "        encoder_device = next(self.text_encoder.parameters()).device\n"
        "        prompt_embeds = self.text_encoder(text_input_ids.to(encoder_device), "
        "mask.to(encoder_device)).last_hidden_state\n"
    )
    to_old = (
        "        if self.text_encoder is not None:\n"
        "            self.text_encoder = self.text_encoder.to(device, non_blocking=True)\n"
    )
    to_new = (
        "        if self.text_encoder is not None and os.getenv("
        "\"SOFTMETA_LONGCAT_TEXT_ENCODER_CPU\", \"0\") != \"1\":\n"
        "            self.text_encoder = self.text_encoder.to(device, non_blocking=True)\n"
    )
    if encode_new not in pipeline:
        if pipeline.count(encode_old) != 1:
            raise SystemExit("LongCat prompt-encoding block changed; refusing an unsafe patch.")
        pipeline = pipeline.replace(encode_old, encode_new, 1)
    if to_new not in pipeline:
        if pipeline.count(to_old) != 1:
            raise SystemExit("LongCat device-transfer block changed; refusing an unsafe patch.")
        pipeline = pipeline.replace(to_old, to_new, 1)
    pipeline_target.write_text(pipeline, encoding="utf-8")

    marker = repo / PATCH_MARKER
    marker.write_text(
        "SoftMeta LongCat v1.5 single-GPU patch\n"
        "seed is configurable per checkpoint\n"
        "long continuation streams video chunks instead of retaining every RGB frame\n"
        "T5 text encoder can stay on CPU for a single A100 40 GB\n"
        "quantized DiT is loaded before T5 to lower host-RAM peak\n"
        f"expected upstream source sha: {PINNED_SOURCE_SHA}\n"
        f"expected upstream pipeline sha: {PINNED_PIPELINE_SHA}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    patch_runtime(args.repo.resolve())


if __name__ == "__main__":
    main()
