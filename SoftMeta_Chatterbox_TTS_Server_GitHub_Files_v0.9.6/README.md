# SoftMeta Chatterbox TTS Server

A self-hosted speech and long-form avatar studio maintained by **SoftMeta**.
The server, persistent queues, browser UI, waveform editor, voice-candidate
workflow and avatar orchestration are SoftMeta code.

## v0.9.6: reliable fast-start Colab build

- Installs the missing ONNX Runtime GPU dependency required by Ditto's face detector.
- Verifies ONNX Runtime before the UI reports that the Avatar engine is ready.
- Detects the newest complete server folder when GitHub uploads accidentally create a nested directory.
- Ignores stale v0.9.1 and v0.9.2 root copies when a newer uploaded build is available.
- Fixes the Python 3.11 Avatar segment-concatenation syntax error.
- Removes the huge embedded notebook patch and clones the repository `main` branch directly.
- Starts the normal Chatterbox server without installing MOSS or Ditto unless their optional switches are enabled.
- Reuses verified environments on reruns instead of deleting and rebuilding them.
- Downloads only Ditto PyTorch checkpoints, not the unused TensorRT and ONNX bundles.

- Removes the incompatible SciPy override from the isolated MOSS environment.
- Uses the official MOSS-TTS `torch-runtime` dependency set.
- Updates the speaker checker to SpeechBrain 1.1.0 and SoundFile audio I/O.
- Adds a compatibility guard for stale cached SpeechBrain wheels.
- Uses Ditto PyTorch as the stable Colab A100 40GB avatar backend.
- Skips legacy TensorRT 8.6.1 by default instead of printing a failed build.
- Verifies every isolated environment with `pip check` before starting the server.
- Keeps checkpointed 10-30 minute video rendering and reduced working resolution.


## v0.8.0: MOSS unique voice generation

The **Generate Voice** workflow uses `OpenMOSS-Team/MOSS-VoiceGenerator`:

- Creates fictional American speaker timbres from text instructions
- Builds speaker identity before age, emotion and delivery style
- Does not globally slow audio or stretch words to imitate age
- Over-generates and screens candidates for identity repetition and audio quality
- Saves the selected candidate as a reusable Chatterbox reference voice

## Other studio features

- Audio 1 only by default; add removable Audio 2–5 workspaces
- Sequential audio queue with Generate All and Queue Monitor
- Stable generated-voice preview players
- Five bundled predefined American male voices
- Import additional predefined or cloning WAV files
- Main waveform, live playhead, zoom, pan and audio cutter
- Download full audio, selected audio, Part One or Part Two
- FastAPI documentation at `/docs`
- Responsive light and dark interface

## Repository relationship

```text
soft-meta/chatterbox-v2@v0.2.1
        ↓
soft-meta/Chatterbox-TTS-Server main branch
        ↓
Chatterbox + MOSS VoiceGenerator + isolated Ditto avatar worker
```

This project has no runtime dependency on Devnen repositories.

## Google Colab A100

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.9.6.ipynb`, select an **A100 GPU**
and run all cells. The notebook creates three isolated environments:

- Python 3.11 Chatterbox server environment
- Python 3.12 MOSS VoiceGenerator environment
- Python 3.10 Ditto avatar environment

The first setup downloads several gigabytes of voice and avatar checkpoints.
Long video generation can take substantial time and storage even on A100.

## Local avatar installation

After preparing the main server and micromamba, run:

```bash
bash scripts/install_ditto_a100.sh /path/to/micromamba
```

Then export the paths printed by the script:

```text
SOFTMETA_AVATAR_PYTHON=/path/to/avatar310/bin/python
SOFTMETA_DITTO_DIR=/path/to/ditto-talkinghead
SOFTMETA_DITTO_CHECKPOINTS=/path/to/ditto-talkinghead/checkpoints
```

## Docker

The main container intentionally does not bake the very large Ditto checkpoints
into the image. Mount an externally prepared Ditto directory and avatar Python
environment, or run the avatar worker on the GPU host. See
`docs/AVATAR_TALKING.md`.

## Rights and disclosure

Use only avatar images and reference voices that you own or have permission to
use. Do not present a fictional avatar as a real named person. Follow platform
rules that require synthetic-media disclosure.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox is MIT licensed. MOSS
VoiceGenerator and Ditto code are open source under their upstream licences.
Ditto's published checkpoint bundle contains third-party face-analysis assets
whose commercial terms require separate review. Read `THIRD_PARTY_NOTICES.md`
before monetized deployment.
