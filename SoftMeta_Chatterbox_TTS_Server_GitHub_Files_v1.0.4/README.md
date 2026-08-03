# SoftMeta Chatterbox TTS Server

A self-hosted speech and realistic long-form avatar studio maintained by
**SoftMeta**. The server, persistent queues, browser UI, waveform editor,
voice-candidate workflow and avatar orchestration are SoftMeta code.

## v1.0.4: LongCat realistic avatar engine

- Verifies every INT8 safetensors shard referenced by the model index instead
  of declaring the engine ready when only the index file was downloaded.
- Automatically resumes any missing Hugging Face shard and installs Accelerate
  for faster, lower-memory model loading.
- Removes the incorrect requirement for `lora/dmd_lora.safetensors`. The pinned
  official Avatar 1.5 revision has no `lora/` directory, and upstream inference
  already treats a standalone DMD LoRA as optional.
- Fixes the second upstream packaging failure by excluding the unavailable
  `tritonserverclient==0.0.6` entry, which is not imported by Avatar inference.
- Fixes the Colab installer failure caused by LongCat upstream listing the
  Ubuntu `libsndfile1` library as a nonexistent PyPI package. The installer now
  keeps the installed system library and filters only that invalid pip line.
- Safely resumes a partially created `longcat310` environment, so rerunning the
  Avatar cell does not delete the downloaded Torch packages.
- Replaces Ditto with **LongCat Video Avatar 1.5** for realistic lip, jaw,
  cheek, eye, head, shoulder and upper-body motion.
- Uses the official distilled eight-step, INT8 inference path on an A100 40 GB.
- Defaults to native 720p generation and automatically retries at 480p if a
  Colab runtime runs out of GPU memory.
- Adds Calm, Natural and Expressive motion presets plus an optional prompt.
- Keeps continuous identity and motion for videos up to ten minutes.
- Uses silence-aware, restart-safe checkpoints for 10–30 minute jobs.
- Streams continuation output in constant-memory video chunks, avoiding both
  repeated full-video encoding and a many-gigabyte RGB-frame accumulator.
- Keeps LongCat in an isolated Python 3.10 environment, so Chatterbox and MOSS
  dependencies remain stable.
- Keeps MOSS and Avatar installation optional. A TTS-only Colab start does not
  download or verify either large runtime.
- Reuses verified environments and model files on notebook reruns.
- Pins the LongCat source and Avatar 1.5 model revisions for repeatability.

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

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.0.4.ipynb`, select an **A100 GPU**,
configure the three install switches and run all cells. The optional runtimes
are isolated:

- Python 3.11 Chatterbox server
- Python 3.12 MOSS VoiceGenerator
- Python 3.10 LongCat Video Avatar 1.5

The first Avatar installation needs at least 55 GB of free Colab disk and a
large model download. Later runs reuse the verified files. Diffusion video is
computationally expensive; an A100 improves it substantially but does not make
it real-time.

## Local avatar installation

After preparing the main server and micromamba, run:

```bash
SOFTMETA_SERVER_DIR=/path/to/Chatterbox-TTS-Server \
bash scripts/install_longcat_avatar_a100.sh /path/to/micromamba
```

Then configure the server:

```text
SOFTMETA_AVATAR_PYTHON=/path/to/mamba/envs/longcat310/bin/python
SOFTMETA_LONGCAT_DIR=/content/LongCat-Video
SOFTMETA_LONGCAT_MODELS=/content/longcat_models
```

See `docs/AVATAR_TALKING.md` for the render behavior and API.

## Docker

The main image does not bake in the very large LongCat models or its separate
GPU environment. Mount externally prepared LongCat code, models and avatar
Python environment as shown in `docker-compose.yml`.

## Rights and disclosure

Use only avatar images and voices that you own or have permission to use. Do
not impersonate a real person or misrepresent synthetic footage as authentic.
Follow platform rules that require synthetic-media disclosure.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox and LongCat Video are
MIT licensed; MOSS VoiceGenerator is Apache-2.0. Review
`THIRD_PARTY_NOTICES.md` before deployment.
