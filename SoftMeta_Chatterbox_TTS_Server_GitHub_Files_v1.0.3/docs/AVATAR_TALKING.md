# Avatar Talking in SoftMeta v1.0.3

## Engine

SoftMeta uses the official **LongCat Video Avatar 1.5** single-audio
image-to-video pipeline. The isolated worker uses the INT8 DiT, distilled
eight-step inference, Whisper-Large-v3 audio conditioning and the upstream
continuation mechanism. The server never mixes LongCat packages into the
working Chatterbox environment.

The upstream examples use context-parallel multi-GPU commands. SoftMeta's
single-A100 path keeps the large T5 text encoder on CPU, loads the quantized DiT
before T5 to reduce peak host RAM, and runs context parallelism at one process.

## Rendering flow

```text
Permitted portrait + completed or uploaded audio
  -> identity-preserving image framing
  -> mono 16 kHz speech preparation
  -> LongCat audio-driven video diffusion and continuation
  -> optional silence-aware checkpoint joining
  -> original audio restoration
  -> H.264/AAC delivery encoding
  -> duration and freeze checks
```

Continuous mode gives the best gaze, face and body continuity for up to ten
minutes. Audio longer than ten minutes automatically switches to silence-aware
checkpoints of at most five minutes so a 10–30 minute render remains recoverable
and does not retain every decoded frame in memory. Each checkpoint starts from
the portrait, so inspect joins for a small posture reset.

## A100 defaults

- Engine: LongCat Video Avatar 1.5
- Model path: official INT8 and eight-step distilled inference
- Native generation: 720p, with automatic 480p retry after CUDA OOM
- Final delivery: 1080p portrait, 25 fps, H.264/AAC
- Mode: continuous up to ten minutes
- Checkpoint length: about five minutes
- Motion: natural lip, jaw, cheek, eye, head and shoulder movement

The final 1080p option is delivery encoding. It does not invent detail beyond
the selected native 720p or 480p AI render.

## API

- `GET /api/video/status`
- `POST /api/video/avatar-upload`
- `GET /api/video/avatar/{filename}`
- `POST /api/video/audio-upload`
- `GET /api/video/audio/{filename}`
- `GET /api/video/jobs`
- `POST /api/video/jobs`
- `GET /api/video/jobs/{job_id}`
- `POST /api/video/jobs/{job_id}/cancel`
- `DELETE /api/video/jobs/{job_id}`
- `DELETE /api/video/jobs`
- `GET /api/video/jobs/{job_id}/log`
- `GET /api/video/jobs/{job_id}/file`

## Storage

```text
data/avatar_images   uploaded avatar images
data/video_audio     separately uploaded audio
data/video_work      temporary render sections
video_outputs        completed MP4 files
logs/avatar_*.log    per-job engine output
```

## Portrait and prompt guidance

Use one clear person with visible eyes and mouth, natural lighting and at least
768 px on the shorter side. An upper-body portrait gives the model useful head,
neck and shoulder context. The Natural preset is the best default. Use a custom
motion prompt only to describe movement and camera behavior; avoid changing the
person's identity, clothing, lighting or background.

## Performance policy

The installer selectively downloads only the LongCat components used by this
pipeline and reuses a verified installation. SoftMeta's pinned runtime patch
changes only operational details: each job gets an explicit seed, the large T5
encoder can stay on CPU for one-GPU inference, and long output frames are
encoded as constant-memory chunks instead of accumulating the entire video in
RAM or repeatedly re-encoding a growing result.

## Safety

Technical checks can detect duration drift and long frozen sections but cannot
guarantee photorealism. Watch the full result. Use only media you have the right
to animate, do not impersonate people, and disclose synthetic footage where
required.
