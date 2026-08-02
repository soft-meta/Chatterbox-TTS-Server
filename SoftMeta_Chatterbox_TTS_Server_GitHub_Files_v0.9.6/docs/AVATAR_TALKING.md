# Avatar Talking in SoftMeta v0.9.6

## UI placement

`Generate Video` is not counted as one of the five audio workspaces. The browser
rebuilds the tab row in this order:

1. Every existing Audio workspace
2. Generate Video
3. The plus button, while fewer than five Audio workspaces exist

This guarantees that adding Audio 2–5 moves Generate Video to the right.

## Rendering flow

```text
Avatar image + completed or uploaded audio
  -> image normalization and aspect-ratio framing
  -> mono 16 kHz speech preparation for Ditto
  -> continuous or silence-aware checkpointed rendering
  -> section joining
  -> original audio restoration
  -> H.264/AAC final encoding
  -> duration and freeze quality checks
```

Continuous mode sends the whole audio to Ditto and gives the best motion
continuity. Checkpointed mode is more resilient for very long jobs. It searches
for natural silence near the selected checkpoint length and renders each section
independently. Because a still image initializes each section, inspect the joins
before publishing.

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

## A100 defaults

- Engine: Auto, preferring Ditto TensorRT
- Fallback: Ditto PyTorch
- Mode: Continuous
- Final delivery: 1080p portrait, 25 fps, H.264/AAC
- Checkpoint length: about 3 minutes when checkpointed mode is selected

The engine is isolated from Chatterbox and MOSS because their CUDA, PyTorch and
Transformers requirements differ.

## Realism guidance

Use one clear person, a mostly forward face, visible eyes and mouth, natural room
lighting and enough resolution. Upper-body portraits usually give better
results than distant full-body images. The system can detect technical failures,
but it cannot reliably detect every unnatural blink, tooth shape, lip shape or
background deformation. Watch the complete result.

## Commercial deployment warning

Ditto code is Apache-2.0. Its official checkpoint bundle includes face-detection
assets named `insightface_det` or `det_10g`. InsightFace states that its supplied
pretrained models are for non-commercial research unless separately licensed.
SoftMeta does not change or grant those rights. Obtain appropriate permission or
replace the restricted detector assets before monetized deployment.

## v0.9.6 Colab backend policy

Current Colab A100 images use a newer CUDA/Python stack than the legacy
TensorRT 8.6.1 wheel used by the original Ditto test environment. SoftMeta
therefore installs and selects the official Ditto PyTorch checkpoint by
default. This avoids a false installation failure and keeps output quality
unchanged. TensorRT remains an opt-in advanced path for matching custom images.

Set `SOFTMETA_TRY_TENSORRT=1` during installation and
`SOFTMETA_ENABLE_TENSORRT=1` at runtime only when TensorRT imports successfully.
