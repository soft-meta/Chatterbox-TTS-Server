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
class DittoBackend:
    id: str
    label: str
    data_root: Path
    config_path: Path


@dataclass(frozen=True)
class RenderProfile:
    generation_size: tuple[int, int]
    final_size: tuple[int, int]
    crf: int
    preset: str


class AvatarEngineService:
    """External-worker adapter for long-form talking-avatar rendering.

    The server remains dependency-light. Ditto is installed in a separate Python
    environment and invoked through its official inference.py CLI. This prevents
    TensorRT/PyTorch requirements from changing the working TTS environment.
    """

    def __init__(self, storage: Storage, config: dict[str, Any]) -> None:
        avatar_cfg = config.get("avatar", {})
        self.storage = storage
        self.python = Path(
            os.getenv("SOFTMETA_AVATAR_PYTHON")
            or avatar_cfg.get("python")
            or sys.executable
        ).expanduser()
        self.ditto_dir = Path(
            os.getenv("SOFTMETA_DITTO_DIR")
            or avatar_cfg.get("ditto_dir")
            or "/content/ditto-talkinghead"
        ).expanduser()
        self.checkpoints = Path(
            os.getenv("SOFTMETA_DITTO_CHECKPOINTS")
            or avatar_cfg.get("ditto_checkpoints")
            or self.ditto_dir / "checkpoints"
        ).expanduser()
        self.ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe = shutil.which("ffprobe") or "ffprobe"
        self._active_process: subprocess.Popen[str] | None = None

    @staticmethod
    def _run_capture(command: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

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

    def _backends(self) -> list[DittoBackend]:
        return [
            DittoBackend(
                id="ditto_trt",
                label="Ditto TensorRT",
                data_root=self.checkpoints / "ditto_trt_Ampere_Plus",
                config_path=self.checkpoints / "ditto_cfg" / "v0.4_hubert_cfg_trt.pkl",
            ),
            DittoBackend(
                id="ditto_pytorch",
                label="Ditto PyTorch",
                data_root=self.checkpoints / "ditto_pytorch",
                config_path=self.checkpoints / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl",
            ),
        ]

    def backend_status(self) -> list[dict[str, Any]]:
        inference = self.ditto_dir / "inference.py"
        python_ok = self.python.is_file() or shutil.which(str(self.python)) is not None
        return [
            {
                "id": item.id,
                "label": item.label,
                "ready": bool(
                    python_ok
                    and inference.is_file()
                    and item.data_root.is_dir()
                    and item.config_path.is_file()
                ),
                "data_root": str(item.data_root),
                "config": str(item.config_path),
            }
            for item in self._backends()
        ]

    def status(self) -> dict[str, Any]:
        backends = self.backend_status()
        ready = any(item["ready"] for item in backends)
        return {
            "ready": ready,
            "gpu": self.gpu_info(),
            "python": str(self.python),
            "ditto_dir": str(self.ditto_dir),
            "backends": backends,
            "recommended": "ditto_trt" if any(
                item["id"] == "ditto_trt" and item["ready"] for item in backends
            ) else "ditto_pytorch",
            "message": (
                "Ditto is ready for long-form avatar rendering."
                if ready
                else "Install Ditto and its checkpoints with the v0.9.0 A100 notebook."
            ),
        }

    def _resolve_backend(self, requested: str) -> DittoBackend:
        available = {item.id: item for item in self._backends()}
        if requested == "auto":
            order = ("ditto_trt", "ditto_pytorch")
        else:
            order = (requested,)
        inference = self.ditto_dir / "inference.py"
        for backend_id in order:
            item = available.get(backend_id)
            if not item:
                continue
            if inference.is_file() and item.data_root.is_dir() and item.config_path.is_file():
                return item
        status = self.status()
        raise AvatarEngineError(
            "No requested Ditto backend is ready. Run the avatar installation cells first. "
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

    @staticmethod
    def _profile(aspect_ratio: str, resolution: str, quality: str) -> RenderProfile:
        final_sizes = {
            ("9:16", "720p"): (720, 1280),
            ("9:16", "1080p"): (1080, 1920),
            ("16:9", "720p"): (1280, 720),
            ("16:9", "1080p"): (1920, 1080),
            ("1:1", "720p"): (720, 720),
            ("1:1", "1080p"): (1080, 1080),
        }
        final_size = final_sizes[(aspect_ratio, resolution)]
        # 720-class source gives Ditto a practical A100 long-video workload.
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
    def _expected_render_seconds(audio_seconds: float, backend_id: str) -> float:
        factor = 0.75 if backend_id == "ditto_trt" else 1.45
        return max(35.0, audio_seconds * factor + 30.0)

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
    ) -> None:
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("\n$ " + shlex.join(command) + "\n")
            log.flush()
            self._active_process = subprocess.Popen(
                command,
                cwd=self.ditto_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                while self._active_process.poll() is None:
                    if cancelled():
                        self.cancel_active()
                        raise AvatarCancelled("Video generation was cancelled.")
                    elapsed = time.monotonic() - started
                    fraction = min(0.94, elapsed / max(expected_seconds, 1.0))
                    percent = progress_start + (progress_end - progress_start) * fraction
                    eta = max(1.0, expected_seconds - elapsed)
                    progress(percent, stage, eta)
                    time.sleep(1.5)
                code = self._active_process.returncode
            finally:
                self._active_process = None
        if code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise AvatarEngineError(f"Ditto exited with code {code}. Recent log:\n{tail}")

    def _render_segment(
        self,
        backend: DittoBackend,
        avatar: Path,
        audio: Path,
        output: Path,
        log_path: Path,
        progress_start: float,
        progress_end: float,
        label: str,
        progress: ProgressCallback,
        cancelled: CancelCallback,
    ) -> None:
        command = [
            str(self.python),
            str(self.ditto_dir / "inference.py"),
            "--data_root",
            str(backend.data_root),
            "--cfg_pkl",
            str(backend.config_path),
            "--audio_path",
            str(audio),
            "--source_path",
            str(avatar),
            "--output_path",
            str(output),
        ]
        self._run_process(
            command=command,
            log_path=log_path,
            expected_seconds=self._expected_render_seconds(self.media_duration(audio), backend.id),
            progress_start=progress_start,
            progress_end=progress_end,
            stage=label,
            progress=progress,
            cancelled=cancelled,
        )
        if not output.is_file() or output.stat().st_size < 50_000:
            raise AvatarEngineError("Ditto did not produce a valid MP4 file.")

    def _concat_segments(self, clips: list[Path], destination: Path, work: Path) -> None:
        if len(clips) == 1:
            shutil.copy2(clips[0], destination)
            return
        list_path = work / "concat.txt"
        lines = [f"file '{clip.as_posix().replace("'", "'\\''")}'" for clip in clips]
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
        if render_mode == "checkpointed":
            boundaries = self._segment_boundaries(
                prepared_audio,
                int(job.get("segment_seconds", 180)),
            )
        else:
            boundaries = [(0.0, audio_duration)]

        clips: list[Path] = []
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
            self._render_segment(
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
            )
            clips.append(clip)

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
        progress(100.0, "Video completed", 0)
        return {
            "output_filename": destination.name,
            "output_size": destination.stat().st_size,
            "backend": backend.id,
            "backend_label": backend.label,
            "segments": total,
            "duration": quality["video_duration"],
            "quality_report": quality,
            "log_filename": log_path.name,
        }
