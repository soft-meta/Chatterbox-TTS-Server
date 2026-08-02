# SoftMeta Chatterbox TTS Server

A self-hosted speech and long-form avatar studio maintained by **SoftMeta**.
The server, persistent queues, browser UI, waveform editor, voice-candidate
workflow and avatar orchestration are SoftMeta code.

## v0.9.1: A100 40GB Avatar installer hotfix

- Repairs the missing Ditto installer error from the v0.9.0 GitHub tag.
- Detects the common Colab A100 40GB GPU profile.
- Defaults 10–30 minute videos to two-minute checkpointed sections.
- Uses a lower internal working resolution on 40GB VRAM, then exports 720p or 1080p.
- The notebook contains an embedded installer fallback, so the cell is not dependent on a repository script being present.


The audio tab row now always places **Generate Video** after the last existing
audio workspace:

```text
Audio 1 | Generate Video | +
Audio 1 | Audio 2 | Generate Video | +
Audio 1 | Audio 2 | Audio 3 | Generate Video | +
```

The button therefore moves to the right when Audio 2, Audio 3, Audio 4 or Audio
5 is added. Audio 1 remains the only default workspace.

### Complete Avatar Talking workflow

- Upload a permitted PNG, JPG or WebP portrait
- Use any completed Audio 1–5 result without downloading and uploading it again
- Or upload a separate WAV, MP3, M4A, FLAC, OGG or AAC file
- Render through an isolated Ditto TensorRT worker on A100
- Automatically fall back to the official Ditto PyTorch checkpoint
- Choose 9:16, 16:9 or 1:1 output
- Choose portrait framing, image fit, 720p or 1080p delivery and 25 or 30 fps
- Use continuous rendering for maximum visual continuity
- Use checkpointed rendering for restart-friendly 10–30 minute jobs
- Split checkpointed jobs near natural audio silence rather than fixed hard cuts
- Persist video jobs and progress across page refreshes
- Cancel safely, inspect logs, preview the result and download the MP4
- Preserve the original full-quality audio in the final H.264/AAC file
- Run duration-drift and long-freeze technical checks after rendering
- Automatically unload the TTS model before avatar rendering and restore it after
  the video finishes, preventing the two GPU workloads from competing

No avatar model can guarantee that every viewer will believe a generated video
is a camera recording. Use clear source images, review the entire output, and
regenerate any segment with unnatural eyes, teeth, lips, hair or background
motion.

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
soft-meta/Chatterbox-TTS-Server@v0.9.1
        ↓
Chatterbox + MOSS VoiceGenerator + isolated Ditto avatar worker
```

This project has no runtime dependency on Devnen repositories.

## Google Colab A100

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.9.1.ipynb`, select an **A100 GPU**
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
