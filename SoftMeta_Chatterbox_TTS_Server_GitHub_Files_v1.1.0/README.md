# SoftMeta Chatterbox TTS Server

A self-hosted speech and realistic long-form avatar studio maintained by
**SoftMeta**. The server, persistent queues, browser UI, waveform editor,
voice-candidate workflow and avatar orchestration are SoftMeta code.

## v1.1.0: EchoMimicV3 Flash avatar engine

- Replaces LongCat with the pinned official **EchoMimicV3 Flash** 1.3B engine.
- Uses official eight-step generation with TeaCache on an A100 40 GB.
- Loads the model once per job and renders all long-form chunks in one guarded
  batch instead of reloading weights for every chunk.
- Emits real chunk heartbeat markers so progress reflects completed inference
  work instead of stopping at an artificial percentage.
- Defaults to a 768 px generation budget and automatically retries at 512 px
  after CUDA out-of-memory errors.
- Adds Calm, Natural and Expressive motion presets plus an optional prompt.
- Uses 137-frame Flash chunks and restores the original audio after joining.
- Uses silence-aware, restart-safe checkpoints for 10–30 minute jobs.
- Streams continuation output in constant-memory video chunks, avoiding both
  repeated full-video encoding and a many-gigabyte RGB-frame accumulator.
- Keeps EchoMimicV3 in an isolated Python 3.10 environment, so Chatterbox and MOSS
  dependencies remain stable.
- Keeps MOSS and Avatar installation optional. A TTS-only Colab start does not
  download or verify either large runtime.
- Reuses verified environments and model files on notebook reruns.
- Pins the EchoMimicV3 source, Wan base, audio encoder and Flash revisions.

## Studio features

- Audio 1 by default; add removable Audio 2–5 workspaces
- Sequential audio queue with Generate All and Queue Monitor
- MOSS fictional voice generation with reusable reference voices
- Five bundled predefined American male voices
- Import additional predefined or cloning WAV files
- Main waveform, live playhead, zoom, pan and audio cutter
- Download full audio, selected audio, Part One or Part Two
- Realistic avatar video with generated or uploaded audio
- FastAPI documentation at `/docs`
- Responsive light and dark interface

## Google Colab A100

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.1.0.ipynb`, select an **A100 GPU**,
configure the three install switches and run all cells. The optional runtimes
are isolated:

- Python 3.11 Chatterbox server
- Python 3.12 MOSS VoiceGenerator
- Python 3.10 EchoMimicV3 Flash

The first Avatar installation needs at least 55 GB of free Colab disk and a
large model download. Later runs reuse the verified files. Diffusion video is
computationally expensive; an A100 improves it substantially but does not make
it real-time.

## Local avatar installation

After preparing the main server and micromamba, run:

```bash
SOFTMETA_SERVER_DIR=/path/to/Chatterbox-TTS-Server \
bash scripts/install_echomimic_v3_flash_a100.sh /path/to/micromamba
```

Then configure the server:

```text
SOFTMETA_AVATAR_PYTHON=/path/to/mamba/envs/echo310/bin/python
SOFTMETA_ECHOMIMIC_DIR=/content/EchoMimicV3
SOFTMETA_ECHOMIMIC_MODELS=/content/echomimic_v3_models
```

See `docs/AVATAR_TALKING.md` for the render behavior and API.

## Docker

The main image does not bake in the EchoMimicV3 models or its separate GPU
environment. Mount externally prepared EchoMimicV3 code, models and avatar
Python environment as shown in `docker-compose.yml`.

## Rights and disclosure

Use only avatar images and voices that you own or have permission to use. Do
not impersonate a real person or misrepresent synthetic footage as authentic.
Follow platform rules that require synthetic-media disclosure.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox and EchoMimicV3 are
MIT licensed; MOSS VoiceGenerator is Apache-2.0. Review
`THIRD_PARTY_NOTICES.md` before deployment.
