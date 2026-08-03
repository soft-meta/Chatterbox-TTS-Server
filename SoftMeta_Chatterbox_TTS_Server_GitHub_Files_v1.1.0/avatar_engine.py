from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageFilter, ImageOps

from storage import Storage
from utils import safe_filename

logger = logging.getLogger("softmeta.avatar")

ProgressCallback = Callable[[float, str, float | None], None]
CancelCallback = Callable[[], bool]


class AvatarEngineError(RuntimeError):
    pass


class AvatarCancelled(AvatarEngineError):
    pass


@dataclass(frozen=True)
class AvatarBackend:
    id: str
    label: str


@dataclass(frozen=True)
class RenderProfile:
    generation_size: tuple[int, int]
    final_size: tuple[int, int]
    crf: int
    preset: str


class AvatarEngineService:
    """External-worker adapter for realistic long-form avatar rendering.

    EchoMimicV3 Flash is installed in an isolated Python environment and invoked
    through its pinned official image-and-audio inference entrypoint. SoftMeta's
    guarded batch patch loads the model once for every long-form job.
    """

    def __init__(self, storage: Storage, config: dict[str, Any]) -> None:
        avatar_cfg = config.get("avatar", {})
        self.storage = storage
        default_avatar_python = (
            Path(os.getenv("MAMBA_ROOT_PREFIX", "/root/.local/share/mamba"))
            / "envs"
            / "echo310"
            / "bin"
            / "python"
        )
        self.python = Path(
            os.getenv("SOFTMETA_AVATAR_PYTHON")
            or avatar_cfg.get("python")
            or default_avatar_python
        ).expanduser()
        self.echo_dir = Path(
            os.getenv("SOFTMETA_ECHOMIMIC_DIR")
            or avatar_cfg.get("echomimic_dir")
            or "/content/EchoMimicV3"
        ).expanduser()
        self.model_root = Path(
            os.getenv("SOFTMETA_ECHOMIMIC_MODELS")
            or avatar_cfg.get("echomimic_models")
            or "/content/echomimic_v3_models"
        ).expanduser()
        self.flash_root = self.model_root / "flash"
        self.base_model = self.flash_root / "Wan2.1-Fun-V1.1-1.3B-InP"
        self.audio_model = self.flash_root / "chinese-wav2vec2-base"
        self.transformer_model = self.flash_root / "transformer" / "diffusion_pytorch_model.safetensors"
        self.ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe = shutil.which("ffprobe") or "ffprobe"
        self._active_process: subprocess.Popen[str] | None = None
        self._backend_status_cache: tuple[float, list[dict[str, Any]]] | None = None

    @staticmethod
    def _run_capture(command: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=AvatarEngineService._headless_process_env(),
        )

    @staticmethod
    def _indexed_model_complete(index: Path) -> bool:
        """Return true only when every safetensors shard named by an index exists."""
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            shards = set(weight_map.values())
            return bool(shards) and all((index.parent / shard).is_file() for shard in shards)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def _headless_process_env() -> dict[str, str]:
        """Return a stable environment for the isolated avatar process.

        Colab exports notebook-only UI variables. EchoMimic runs as a headless
        distributed worker and uses one GPU process, deterministic tokenization,
        the shared Hugging Face cache, and conservative CUDA allocation.
        """

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env.setdefault("MPLCONFIGDIR", "/tmp/softmeta-matplotlib")
        env.setdefault("HF_HOME", "/content/hf_home")
        env.setdefault("HF_HUB_CACHE", "/content/hf_home/hub")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_IB_DISABLE", "1")
        return env

    def _python_can_import(self, *modules: str) -> bool:
        """Check the isolated runtime in one process instead of one per module."""

        executable = str(self.python)
        if not (self.python.is_file() or shutil.which(executable)):
            return False
        try:
            result = self._run_capture(
                [executable, "-c", "; ".join(f"import {module}" for module in modules)],
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def cancel_active(self) -> None:
        process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=12)
        except (ProcessLookupError, PermissionError):
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()

    def gpu_info(self) -> dict[str, Any]:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return {"available": False, "name": None, "memory_mb": None}
        result = self._run_capture(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "name": None, "memory_mb": None}
        first = result.stdout.strip().splitlines()[0]
        name, _, memory = first.rpartition(",")
        try:
            memory_mb = int(memory.strip())
        except ValueError:
            memory_mb = None
        return {
            "available": True,
            "name": name.strip() or first.strip(),
            "memory_mb": memory_mb,
        }

    def _backends(self) -> list[AvatarBackend]:
        return [AvatarBackend(id="echomimic_v3_flash", label="EchoMimicV3 Flash")]

    def backend_status(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self._backend_status_cache and now - self._backend_status_cache[0] < 30:
            return [dict(item) for item in self._backend_status_cache[1]]
        inference = self.echo_dir / "infer_flash.py"
        python_ok = self.python.is_file() or shutil.which(str(self.python)) is not None
        runtime_ok = python_ok and self._python_can_import(
            "torch", "transformers", "diffusers", "librosa", "moviepy", "pyloudnorm"
        )
        files_ok = all(
            path.exists()
            for path in (
                self.echo_dir / ".softmeta_echomimic_v3_flash_v110",
                self.base_model / "diffusion_pytorch_model.safetensors",
                self.base_model / "Wan2.1_VAE.pth",
                self.base_model / "models_t5_umt5-xxl-enc-bf16.pth",
                self.base_model / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                self.audio_model / "config.json",
                self.transformer_model,
            )
        )
        result: list[dict[str, Any]] = []
        for item in self._backends():
            ready = bool(python_ok and inference.is_file() and runtime_ok and files_ok)
            result.append(
                {
                    "id": item.id,
                    "label": item.label,
                    "ready": ready,
                    "runtime_available": runtime_ok,
                    "model_files_available": files_ok,
                    "int8": False,
                    "distilled_steps": 8,
                }
            )
        self._backend_status_cache = (now, [dict(item) for item in result])
        return result

    def status(self) -> dict[str, Any]:
        backends = self.backend_status()
        ready = any(item["ready"] for item in backends)
        recommended = "echomimic_v3_flash"
        return {
            "ready": ready,
            "gpu": self.gpu_info(),
            "python": str(self.python),
            "echomimic_dir": str(self.echo_dir),
            "model_root": str(self.model_root),
            "backends": backends,
            "recommended": recommended,
            "message": (
                "EchoMimicV3 Flash is ready for realistic lip, eye, head and upper-body motion."
                if ready
                else "In Colab, enable INSTALL_AVATAR_RUNTIME and run the optional Avatar installer cell."
            ),
        }

    def _resolve_backend(self, requested: str) -> AvatarBackend:
        available = {item.id: item for item in self._backends()}
        backend_id = "echomimic_v3_flash" if requested == "auto" else requested
        item = available.get(backend_id)
        ready_ids = {entry["id"] for entry in self.backend_status() if entry["ready"]}
        if item and backend_id in ready_ids:
            return item
        status = self.status()
        raise AvatarEngineError(
            "EchoMimicV3 Flash is not ready. Run the optional Avatar installer cell first. "
            f"Detected status: {json.dumps(status, ensure_ascii=False)}"
        )

    def media_duration(self, path: Path) -> float:
        result = self._run_capture(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            timeout=60,
        )
        if result.returncode != 0:
            raise AvatarEngineError(result.stderr.strip() or f"Could not inspect {path.name}.")
        try:
            return max(0.0, float(result.stdout.strip()))
        except ValueError as error:
            raise AvatarEngineError(f"Invalid duration reported for {path.name}.") from error

    def _profile(self, aspect_ratio: str, resolution: str, quality: str) -> RenderProfile:
        final_sizes = {
            ("9:16", "720p"): (720, 1280),
            ("9:16", "1080p"): (1080, 1920),
            ("16:9", "720p"): (1280, 720),
            ("16:9", "1080p"): (1920, 1080),
            ("1:1", "720p"): (720, 720),
            ("1:1", "1080p"): (1080, 1080),
        }
        final_size = final_sizes[(aspect_ratio, resolution)]
        # EchoMimic preserves the source aspect ratio within its 512/768 budget. This image is
        # only the identity/reference canvas, so preserve enough source detail
        # for wrinkles, eyes, hair, and beard boundaries before model bucketing.
        generation_size = {
            "9:16": (720, 1280),
            "16:9": (1280, 720),
            "1:1": (768, 768),
        }[aspect_ratio]
        return RenderProfile(
            generation_size=generation_size,
            final_size=final_size,
            crf=17 if quality == "high" else 20,
            preset="slow" if quality == "high" else "medium",
        )

    @staticmethod
    def _prepare_image(
        source: Path,
        destination: Path,
        size: tuple[int, int],
        fit: str,
        framing: str,
    ) -> None:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < 384:
                raise AvatarEngineError("Avatar image is too small. Use at least 768 px for best results.")
            if framing == "head":
                crop_ratio = 0.86
                crop_w = max(1, round(image.width * crop_ratio))
                crop_h = max(1, round(image.height * crop_ratio))
                left = max(0, (image.width - crop_w) // 2)
                top = max(0, round((image.height - crop_h) * 0.26))
                image = image.crop((left, top, left + crop_w, top + crop_h))
            elif framing == "mid":
                # Keep more of the original body when the user selected a medium portrait.
                fit = "contain" if fit == "cover" else fit

            if fit == "cover":
                centering_y = 0.38 if framing == "head" else 0.42
                prepared = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, centering_y))
            else:
                background = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS).filter(
                    ImageFilter.GaussianBlur(radius=28)
                )
                contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
                x = (size[0] - contained.width) // 2
                y = (size[1] - contained.height) // 2
                background.paste(contained, (x, y))
                prepared = background
            prepared.save(destination, format="PNG", optimize=True)

    def _prepare_audio(self, source: Path, destination: Path) -> None:
        result = self._run_capture(
            [
                self.ffmpeg,
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            timeout=max(300, int(self.media_duration(source) * 2)),
        )
        if result.returncode != 0:
            raise AvatarEngineError(result.stderr[-4000:] or "Audio preparation failed.")

    def _silence_points(self, audio: Path) -> list[float]:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-i",
            str(audio),
            "-af",
            "silencedetect=noise=-36dB:d=0.45",
            "-f",
            "null",
            "-",
        ]
        result = self._run_capture(command, timeout=max(300, int(self.media_duration(audio) * 1.5)))
        text = f"{result.stdout}\n{result.stderr}"
        return [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", text)]

    def _segment_boundaries(self, audio: Path, target_seconds: int) -> list[tuple[float, float]]:
        duration = self.media_duration(audio)
        if duration <= target_seconds * 1.35:
            return [(0.0, duration)]
        silence = self._silence_points(audio)
        boundaries = [0.0]
        cursor = float(target_seconds)
        while cursor < duration - 20:
            candidates = [point for point in silence if cursor - 28 <= point <= cursor + 28]
            chosen = min(candidates, key=lambda point: abs(point - cursor)) if candidates else cursor
            if chosen - boundaries[-1] < 45:
                chosen = min(duration, boundaries[-1] + target_seconds)
            boundaries.append(chosen)
            cursor = chosen + target_seconds
        boundaries.append(duration)
        return [
            (round(boundaries[index], 3), round(boundaries[index + 1], 3))
            for index in range(len(boundaries) - 1)
            if boundaries[index + 1] - boundaries[index] > 1.0
        ]

    def _extract_segment(self, audio: Path, destination: Path, start: float, end: float) -> None:
        result = self._run_capture(
            [
                self.ffmpeg,
                "-y",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                str(audio),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            timeout=max(120, int(end - start) * 2),
        )
        if result.returncode != 0:
            raise AvatarEngineError(result.stderr[-4000:] or "Could not create an audio segment.")

    @staticmethod
    def _expected_render_seconds(audio_seconds: float, native_resolution: str) -> float:
        factor = 5.0 if native_resolution == "720p" else 3.0
        return max(90.0, audio_seconds * factor + 60.0)

    def _run_process(
        self,
        command: list[str],
        log_path: Path,
        expected_seconds: float,
        progress_start: float,
        progress_end: float,
        stage: str,
        progress: ProgressCallback,
        cancelled: CancelCallback,
        cwd: Path | None = None,
        process_env: dict[str, str] | None = None,
    ) -> None:
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("\n$ " + shlex.join(command) + "\n")
            log.flush()
            self._active_process = subprocess.Popen(
                command,
                cwd=cwd or self.echo_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=process_env or self._headless_process_env(),
            )
            try:
                while self._active_process.poll() is None:
                    if cancelled():
                        self.cancel_active()
                        raise AvatarCancelled("Video generation was cancelled.")
                    elapsed = time.monotonic() - started
                    fraction = min(0.90, elapsed / max(expected_seconds, 1.0))
                    recent = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                    markers = re.findall(r"SOFTMETA_PROGRESS\s+(\d+)\s+(\d+)\s+(\w+)", recent)
                    live_stage = stage
                    stages = re.findall(r"SOFTMETA_STAGE\s+([^\n\r]+)", recent)
                    if stages:
                        live_stage = f"{stage} · {stages[-1].replace('_', ' ')}"
                    if markers:
                        completed, total, phase = markers[-1]
                        total_value = max(1, int(total))
                        fraction = max(fraction, min(0.98, int(completed) / total_value))
                        live_stage = f"{stage} · chunk {min(int(completed) + 1, total_value)} of {total_value} · {phase}"
                    percent = progress_start + (progress_end - progress_start) * fraction
                    eta = max(1.0, expected_seconds - elapsed)
                    progress(percent, live_stage, eta)
                    time.sleep(1.5)
                code = self._active_process.returncode
            finally:
                self._active_process = None
        if code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise AvatarEngineError(f"EchoMimicV3 Flash exited with code {code}. Recent log:\n{tail}")

    @staticmethod
    def _motion_prompt(job: dict[str, Any]) -> str:
        custom = str(job.get("motion_prompt") or "").strip()
        if custom:
            return custom[:900]
        style = str(job.get("motion_style", "natural"))
        prompts = {
            "calm": (
                "The person speaks calmly and directly to the camera with precise relaxed lip movements, "
                "natural jaw and cheek motion, soft spontaneous blinking, steady eye contact, tiny realistic "
                "head turns, and occasional gentle shoulder shifts. Preserve the exact identity, age, skin "
                "texture, clothing, lighting, and background. The camera remains fixed."
            ),
            "expressive": (
                "The person speaks warmly and confidently to the camera with accurate lip synchronization, "
                "responsive cheeks and jaw, expressive eyebrows, natural blinking and gaze, moderate realistic "
                "head movement, and conversational upper-body gestures. Preserve identity, age, skin texture, "
                "clothing, lighting, and background. The camera remains fixed."
            ),
            "natural": (
                "The person speaks naturally to the camera with accurate subtle lip shapes, coordinated jaw and "
                "cheek motion, spontaneous blinks, living eye motion, small conversational head turns, and gentle "
                "neck and shoulder movement. Preserve the exact identity, age, wrinkles, hair, clothing, lighting, "
                "and background. Keep the camera stable and the movement realistic, not exaggerated."
            ),
        }
        return prompts.get(style, prompts["natural"])

    def _render_segment(
        self,
        backend: AvatarBackend,
        avatar: Path,
        audio: Path,
        output: Path,
        log_path: Path,
        progress_start: float,
        progress_end: float,
        label: str,
        progress: ProgressCallback,
        cancelled: CancelCallback,
        native_resolution: str,
        motion_prompt: str,
        seed: int,
    ) -> str:
        audio_seconds = self.media_duration(audio)
        output_dir = output.parent / f"{output.stem}_echomimic"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Flash is stable up to 137 frames per call. All chunks are submitted in
        # one patched batch so the 1.3B model is loaded exactly once per job.
        chunk_seconds = 137 / 25
        boundaries: list[tuple[float, float]] = []
        cursor = 0.0
        while cursor < audio_seconds - 0.01:
            end = min(audio_seconds, cursor + chunk_seconds)
            if audio_seconds - end < 0.8:
                end = audio_seconds
            boundaries.append((cursor, end))
            cursor = end
        manifest: list[dict[str, Any]] = []
        generated_clips: list[Path] = []
        for chunk_index, (start, end) in enumerate(boundaries, start=1):
            chunk_audio = output_dir / f"audio_{chunk_index:03d}.wav"
            chunk_output = output_dir / f"chunk_{chunk_index:03d}"
            chunk_output.mkdir(parents=True, exist_ok=True)
            self._extract_segment(audio, chunk_audio, start, end)
            frame_count = max(5, int(((end - start) * 25 - 1) // 4 * 4 + 1))
            manifest.append({
                "image_path": str(avatar.resolve()),
                "audio_path": str(chunk_audio.resolve()),
                "save_path": str(chunk_output.resolve()),
                "prompt": motion_prompt,
                "video_length": frame_count,
                "seed": seed + chunk_index,
            })
            generated_clips.append(chunk_output / f"{avatar.stem}_output.mp4")
        manifest_path = output_dir / "batch.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [
            str(self.python),
            "-u",
            str(self.echo_dir / "infer_flash.py"),
            "--batch_manifest", str(manifest_path),
            "--config_path", str(self.echo_dir / "config/config.yaml"),
            "--model_name", str(self.base_model),
            "--transformer_path", str(self.transformer_model),
            "--wav2vec_model_dir", str(self.audio_model),
            "--num_inference_steps", "8",
            "--sampler_name", "Flow_Unipc",
            "--guidance_scale", "5.0",
            "--audio_guidance_scale", "2.2",
            "--audio_scale", "1.0",
            "--teacache_threshold", "0.1",
            "--num_skip_start_steps", "5",
            "--weight_dtype", "bfloat16",
            "--sample_size", *(("768", "768") if native_resolution == "720p" else ("512", "512")),
            "--fps", "25",
        ]
        process_env = self._headless_process_env()
        process_env["SOFTMETA_ECHOMIMIC_SEED"] = str(seed)
        try:
            self._run_process(
                command=command,
                log_path=log_path,
                expected_seconds=self._expected_render_seconds(audio_seconds, native_resolution),
                progress_start=progress_start,
                progress_end=progress_end,
                stage=f"{label} · EchoMimicV3 Flash {native_resolution}",
                progress=progress,
                cancelled=cancelled,
                cwd=self.echo_dir,
                process_env=process_env,
            )
        except AvatarEngineError:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:].lower()
            oom = any(token in tail for token in ("out of memory", "cuda oom", "allocation on device"))
            if native_resolution != "720p" or not oom or cancelled():
                raise
            return self._render_segment(
                backend=backend,
                avatar=avatar,
                audio=audio,
                output=output,
                log_path=log_path,
                progress_start=progress_start,
                progress_end=progress_end,
                label=f"{label} · automatic memory-safe retry",
                progress=progress,
                cancelled=cancelled,
                native_resolution="480p",
                motion_prompt=motion_prompt,
                seed=seed,
            )

        missing = [str(path) for path in generated_clips if not path.is_file() or path.stat().st_size < 20_000]
        if missing:
            raise AvatarEngineError("EchoMimicV3 did not produce valid chunk MP4 files: " + ", ".join(missing))
        self._concat_segments(generated_clips, output, output_dir)
        if not output.is_file() or output.stat().st_size < 20_000:
            raise AvatarEngineError("EchoMimicV3 did not produce a valid MP4 file.")
        shutil.rmtree(output_dir, ignore_errors=True)
        return native_resolution

    def _concat_segments(self, clips: list[Path], destination: Path, work: Path) -> None:
        if len(clips) == 1:
            shutil.copy2(clips[0], destination)
            return
        list_path = work / "concat.txt"
        lines: list[str] = []
        for clip in clips:
            escaped_path = clip.as_posix().replace("'", "'\\''")
            lines.append("file '" + escaped_path + "'")
        list_path.write_text("\n".join(lines), encoding="utf-8")
        copy_result = self._run_capture(
            [
                self.ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(destination),
            ],
            timeout=max(600, int(sum(self.media_duration(clip) for clip in clips) * 2)),
        )
        if copy_result.returncode == 0 and destination.is_file():
            return
        # Codec parameters can occasionally differ; re-encode as a reliable fallback.
        result = self._run_capture(
            [
                self.ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ],
            timeout=max(900, int(sum(self.media_duration(clip) for clip in clips) * 3)),
        )
        if result.returncode != 0:
            raise AvatarEngineError(result.stderr[-8000:] or "Could not join video segments.")

    def _finalize(
        self,
        raw_video: Path,
        audio: Path,
        destination: Path,
        profile: RenderProfile,
        fps: int,
    ) -> None:
        width, height = profile.final_size
        filter_graph = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps}"
        )
        result = self._run_capture(
            [
                self.ffmpeg,
                "-y",
                "-i",
                str(raw_video),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                filter_graph,
                "-c:v",
                "libx264",
                "-preset",
                profile.preset,
                "-crf",
                str(profile.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(destination),
            ],
            timeout=max(1200, int(self.media_duration(audio) * 4)),
        )
        if result.returncode != 0:
            raise AvatarEngineError(result.stderr[-10000:] or "Final video encoding failed.")

    def _freeze_seconds(self, video: Path) -> float:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            "freezedetect=n=0.0015:d=3",
            "-an",
            "-f",
            "null",
            "-",
        ]
        result = self._run_capture(command, timeout=max(300, int(self.media_duration(video) * 1.5)))
        text = f"{result.stdout}\n{result.stderr}"
        return round(sum(float(value) for value in re.findall(r"freeze_duration:\s*([0-9.]+)", text)), 3)

    def _quality_report(self, video: Path, audio_duration: float) -> dict[str, Any]:
        video_duration = self.media_duration(video)
        drift = abs(video_duration - audio_duration)
        try:
            freeze_seconds = self._freeze_seconds(video)
        except Exception:
            freeze_seconds = 0.0
        freeze_ratio = freeze_seconds / max(video_duration, 0.001)
        passed = drift <= max(1.0, audio_duration * 0.006) and freeze_ratio < 0.12
        return {
            "passed": passed,
            "video_duration": round(video_duration, 3),
            "audio_duration": round(audio_duration, 3),
            "duration_drift": round(drift, 3),
            "freeze_seconds": freeze_seconds,
            "freeze_ratio": round(freeze_ratio, 4),
            "note": (
                "Technical checks passed. Watch the full result before publishing."
                if passed
                else "Technical review recommended. Check sync and any long frozen sections."
            ),
        }

    def render(
        self,
        job: dict[str, Any],
        avatar_source: Path,
        audio_source: Path,
        progress: ProgressCallback,
        cancelled: CancelCallback,
    ) -> dict[str, Any]:
        backend = self._resolve_backend(str(job.get("engine", "auto")))
        profile = self._profile(
            str(job.get("aspect_ratio", "9:16")),
            str(job.get("resolution", "1080p")),
            str(job.get("quality", "high")),
        )
        work = self.storage.video_work / job["id"]
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        log_path = self.storage.logs / f"avatar_{job['id']}.log"
        prepared_avatar = work / "avatar.png"
        prepared_audio = work / "audio.wav"

        progress(2.0, "Validating avatar and audio", None)
        if cancelled():
            raise AvatarCancelled("Video generation was cancelled.")
        self._prepare_image(
            avatar_source,
            prepared_avatar,
            profile.generation_size,
            str(job.get("image_fit", "cover")),
            str(job.get("framing", "upper")),
        )
        progress(5.0, "Preparing clean 16 kHz speech audio", None)
        self._prepare_audio(audio_source, prepared_audio)
        audio_duration = self.media_duration(prepared_audio)
        if audio_duration < 1.0:
            raise AvatarEngineError("Audio is too short to create a talking video.")

        render_mode = str(job.get("render_mode", "continuous"))
        requested_segment_seconds = int(job.get("segment_seconds", 300))
        native_resolution = str(job.get("native_resolution", "720p"))
        motion_prompt = self._motion_prompt(job)
        gpu = self.gpu_info()
        memory_mb = gpu.get("memory_mb") or 0
        auto_checkpointed = False
        # Long jobs are divided near silence at the outer level. Each section is
        # internally batched into 137-frame Flash clips with one model load.
        if render_mode == "continuous" and audio_duration > 600:
            render_mode = "checkpointed"
            requested_segment_seconds = min(max(requested_segment_seconds, 180), 300)
            auto_checkpointed = True
        if render_mode == "checkpointed":
            boundaries = self._segment_boundaries(
                prepared_audio,
                requested_segment_seconds,
            )
        else:
            boundaries = [(0.0, audio_duration)]

        clips: list[Path] = []
        actual_native_resolutions: list[str] = []
        total = len(boundaries)
        for index, (start, end) in enumerate(boundaries, start=1):
            if cancelled():
                raise AvatarCancelled("Video generation was cancelled.")
            segment_audio = prepared_audio
            if total > 1:
                segment_audio = work / f"audio_{index:03d}.wav"
                self._extract_segment(prepared_audio, segment_audio, start, end)
            clip = work / f"clip_{index:03d}.mp4"
            start_percent = 8.0 + ((index - 1) / total) * 76.0
            end_percent = 8.0 + (index / total) * 76.0
            actual_native = self._render_segment(
                backend=backend,
                avatar=prepared_avatar,
                audio=segment_audio,
                output=clip,
                log_path=log_path,
                progress_start=start_percent,
                progress_end=end_percent,
                label=f"Rendering avatar segment {index} of {total}",
                progress=progress,
                cancelled=cancelled,
                native_resolution=native_resolution,
                motion_prompt=motion_prompt,
                seed=42 + index * 1009,
            )
            clips.append(clip)
            actual_native_resolutions.append(actual_native)

        progress(86.0, "Joining rendered sections", None)
        raw_joined = work / "joined.mp4"
        self._concat_segments(clips, raw_joined, work)
        if cancelled():
            raise AvatarCancelled("Video generation was cancelled.")

        title = safe_filename(str(job.get("title") or "avatar_video"), "avatar_video")
        destination = self.storage.video_outputs / f"{title}_{job['id'][:8]}.mp4"
        progress(91.0, "Encoding final MP4 and restoring original audio", None)
        self._finalize(raw_joined, prepared_audio, destination, profile, int(job.get("fps", 25)))
        progress(97.0, "Running technical sync and freeze checks", None)
        quality = self._quality_report(destination, audio_duration)
        shutil.rmtree(work, ignore_errors=True)
        progress(100.0, "Video completed", 0)
        return {
            "output_filename": destination.name,
            "output_size": destination.stat().st_size,
            "backend": backend.id,
            "backend_label": backend.label,
            "segments": total,
            "effective_render_mode": render_mode,
            "auto_checkpointed": auto_checkpointed,
            "gpu_profile": "a100_40gb" if 0 < memory_mb < 48_000 else "standard",
            "generation_size": list(profile.generation_size),
            "native_resolution": (
                "480p" if "480p" in actual_native_resolutions else native_resolution
            ),
            "motion_style": str(job.get("motion_style", "natural")),
            "duration": quality["video_duration"],
            "quality_report": quality,
            "log_filename": log_path.name,
        }
