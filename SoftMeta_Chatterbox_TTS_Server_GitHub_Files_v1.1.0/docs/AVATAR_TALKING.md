# Avatar Talking in SoftMeta v1.1.0

## Engine

SoftMeta uses the official **EchoMimicV3 Flash** image-and-audio pipeline. The
isolated worker uses the 1.3B Wan base, eight-step inference, TeaCache and an
audio-conditioned Flash transformer. Its packages never enter Chatterbox.

## Rendering flow

```text
Permitted portrait + completed or uploaded audio
  -> identity-preserving image framing
  -> mono 16 kHz speech preparation
  -> EchoMimicV3 audio-driven Flash chunks in one loaded worker
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

- Engine: EchoMimicV3 Flash
- Model path: official 1.3B base and eight-step Flash transformer
- Native generation budget: 768 px, with automatic 512 px retry after CUDA OOM
- Final delivery: 1080p portrait, 25 fps, H.264/AAC
- Mode: continuous up to ten minutes
- Checkpoint length: about five minutes
- Motion: natural lip, jaw, cheek, eye, head and shoulder movement

The final 1080p option is delivery encoding. It does not invent native detail.

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

The installer downloads pinned EchoMimicV3 Flash components and reuses a verified
installation. SoftMeta's guarded patch accepts a batch manifest and emits chunk
heartbeats, allowing one model load per long-form job and meaningful progress.

## Safety

Technical checks can detect duration drift and long frozen sections but cannot
guarantee photorealism. Watch the full result. Use only media you have the right
to animate, do not impersonate people, and disclose synthetic footage where
required.
