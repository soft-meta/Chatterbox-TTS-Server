# SoftMeta Chatterbox TTS Server v0.9.0

This release adds a complete **Generate Video** workspace for long-form talking
avatar production while preserving the existing TTS and Generate Voice system.

## Generate Video tab placement

- Generate Video always appears immediately after the last Audio workspace
- Initial order: Audio 1, Generate Video, plus
- Adding Audio 2–5 automatically moves Generate Video to the right
- Generate Video does not consume one of the five Audio workspace slots

## Avatar Talking workflow

- Upload PNG, JPG or WebP avatar images
- Select any completed Audio 1–5 output directly
- Upload separate WAV, MP3, M4A, FLAC, OGG or AAC audio
- Preview uploaded image and audio assets
- Choose portrait, landscape or square output
- Choose 720p or 1080p delivery and 25 or 30 fps
- Choose head, upper-body or medium framing
- Choose cover or blurred-background contain fitting
- Continuous mode for best motion continuity
- Silence-aware checkpointed mode for restart-friendly 10–30 minute jobs
- Persistent queue, progress, stage, elapsed time and ETA
- Cancel, inspect log, preview, download, remove or clear video jobs

## A100 worker

- Adds an isolated Python 3.10 Ditto worker
- Prefers the official Ampere+ TensorRT checkpoint
- Falls back to the official Ditto PyTorch checkpoint
- Adds an A100 installation script and Colab setup
- Automatically unloads the TTS model before video rendering and restores it
  afterward to avoid GPU memory competition

## Long-video processing

- Normalizes avatar images to the selected generation frame
- Converts the engine input to clean mono 16 kHz speech audio
- Locates natural silence near checkpoint boundaries
- Joins rendered sections and restores the original complete audio
- Encodes browser-compatible H.264 video and AAC audio with fast-start metadata
- Runs duration-drift and long-freeze technical checks
- Keeps per-job engine logs and persistent job metadata

## API additions

Adds upload, status, queue, cancel, delete, log and MP4 endpoints under
`/api/video`.

## Validation

- Python compilation passed
- JavaScript syntax validation passed
- HTML parsing passed
- 35 automated tests passed

## Important limitation

The complete Ditto model was not executed in the packaging environment. Run the
A100 Colab notebook and test short audio before committing a 10–30 minute job.
No model can guarantee that generated video will be indistinguishable from a
camera recording.

## Licence warning

Ditto code is Apache-2.0, but its official checkpoint bundle contains
third-party InsightFace detector assets with separate non-commercial model
terms. Review `THIRD_PARTY_NOTICES.md` before monetized use.

## Required engine

Keep:

`soft-meta/chatterbox-v2@v0.2.1`
