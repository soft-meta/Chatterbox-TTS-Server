# SoftMeta Chatterbox TTS Server

A self-hosted speech studio maintained by **SoftMeta**. The server, persistent
audio queue, browser UI, waveform editor and voice-candidate workflow are
SoftMeta code.

## Studio features

- Audio 1 by default; add removable Audio 2–5 workspaces
- Sequential audio queue with Generate All and Queue Monitor
- Five bundled predefined American male voices
- Import additional predefined or cloning WAV files
- Main waveform, live playhead, zoom, pan and audio cutter
- Download full audio, selected audio, Part One or Part Two
- FastAPI documentation at `/docs`
- Responsive light and dark interface

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.1.0.ipynb`, select a GPU,
run all cells. Turbo is the default model for faster English long-form generation.

- Python 3.11 Chatterbox server

The notebook reuses verified environments and downloaded model files on later
runs.

## Docker

Build with `docker compose up --build`. Generated audio, reference voices,
predefined voices and saved generated voices are mounted from the local
directories shown in `docker-compose.yml`.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox is MIT licensed and
