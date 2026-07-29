from __future__ import annotations

import json
import mimetypes
import shutil
import time
from pathlib import Path
from threading import RLock
from typing import Any

from config import ROOT, load_config
from utils import resolve_inside

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


class Storage:
    def __init__(self) -> None:
        config = load_config()
        self.voices = ROOT / config["tts_engine"]["predefined_voices_path"]
        self.references = ROOT / config["tts_engine"]["reference_audio_path"]
        self.generated = ROOT / config["tts_engine"].get("generated_voices_path", "generated_voices")
        self.outputs = ROOT / config["storage"]["outputs_path"]
        self.data = ROOT / config["storage"]["data_path"]
        self.logs = ROOT / config["storage"]["logs_path"]
        self.voice_candidates = self.data / "voice_candidates"
        for directory in (
            self.voices,
            self.references,
            self.generated,
            self.outputs,
            self.data,
            self.logs,
            self.voice_candidates,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.data / "jobs.json"
        self._lock = RLock()

    def list_audio(self, directory: Path) -> list[dict[str, Any]]:
        items = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                metadata_path = path.with_suffix(".json")
                metadata: dict[str, Any] = {}
                if metadata_path.exists():
                    try:
                        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            metadata = loaded
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                items.append(
                    {
                        "filename": path.name,
                        "size": path.stat().st_size,
                        "display_name": metadata.get("name") or path.stem,
                        "age": metadata.get("age"),
                        "gender": metadata.get("gender"),
                        "accent": metadata.get("accent"),
                    }
                )
        return items

    def voice_path(self, kind: str, filename: str) -> Path:
        directories = {
            "predefined": self.voices,
            "clone": self.references,
            "generated": self.generated,
        }
        if kind not in directories:
            raise FileNotFoundError(filename)
        directory = directories[kind]
        path = resolve_inside(directory, filename)
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise FileNotFoundError(filename)
        return path

    def candidate_session_path(self, session_id: str) -> Path:
        safe = "".join(character for character in session_id if character.isalnum() or character in "-_")
        if not safe or safe != session_id:
            raise FileNotFoundError(session_id)
        path = resolve_inside(self.voice_candidates, safe)
        if not path.is_dir():
            raise FileNotFoundError(session_id)
        return path

    def candidate_path(self, session_id: str, filename: str) -> Path:
        session = self.candidate_session_path(session_id)
        path = resolve_inside(session, filename)
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise FileNotFoundError(filename)
        return path

    def cleanup_voice_candidates(self, max_age_hours: float = 24.0) -> None:
        cutoff = time.time() - max_age_hours * 3600
        for path in self.voice_candidates.iterdir():
            if not path.is_dir():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                continue

    def output_path(self, filename: str) -> Path:
        return resolve_inside(self.outputs, filename)

    def save_jobs(self, jobs: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            temp = self.jobs_file.with_suffix(".tmp")
            temp.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
            temp.replace(self.jobs_file)

    def load_jobs(self) -> dict[str, dict[str, Any]]:
        if not self.jobs_file.exists():
            return {}
        try:
            data = json.loads(self.jobs_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def delete_output_artifacts(self, filename: str) -> None:
        path = self.output_path(filename)
        path.unlink(missing_ok=True)
        for cache in self.outputs.glob(f"{path.stem}.*.peaks.json"):
            cache.unlink(missing_ok=True)

    def clear_outputs(self) -> None:
        for path in self.outputs.iterdir():
            if path.is_file() and path.name != "README.md":
                path.unlink(missing_ok=True)

    @staticmethod
    def media_type(path: Path) -> str:
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
