# soft-meta/Chatterbox-TTS-Server

An independent FastAPI server and browser UI for the official Resemble AI Chatterbox TTS package.

This repository does **not** clone or patch `devnen/Chatterbox-TTS-Server`, and it does not depend on `devnen/chatterbox-v2`.

## Included in v0.1.0

- Chatterbox Original English as the default model
- Motivational Speech as the default preset
- Up to five Audio tabs
- Video Title field inside each Audio tab
- Generate Audio and Generate All
- Sequential server-side GPU queue
- Queue Monitor with selectable title, live percentage, remaining percentage, words, ETA and audio-length estimate
- Minimisable progress monitor and bottom-right live status
- Predefined voice and uploaded clone voice preview
- Generated audio preview and WAV download
- Normal player fallback when waveform loading fails
- Server-generated lightweight waveform peaks
- Waveform zoom, horizontal scrolling, hover time and click-to-set Start/End
- Quick End buttons: 2, 2:30, 3, 3:30, 4, 4:30 and 5 minutes
- Preview Selected, Download Selected WAV, Part One and Part Two downloads
- Original generated audio is never modified
- OpenAI-style `/v1/audio/speech` compatibility endpoint
- Google Colab launcher

## Repository pair

1. `soft-meta/chatterbox-v2` — runtime adapter around the official Resemble AI package
2. `soft-meta/Chatterbox-TTS-Server` — server, queue, UI and tools

## Local installation

```bash
git clone https://github.com/soft-meta/chatterbox-v2.git
cd chatterbox-v2
pip install -e .

cd ..
git clone https://github.com/soft-meta/Chatterbox-TTS-Server.git
cd Chatterbox-TTS-Server
pip install -r requirements-colab.txt
python start.py
```

Open `http://localhost:8004`.

## GitHub upload order

Upload and release the engine first, because the server's `requirements.txt` points to `soft-meta/chatterbox-v2@v0.1.0`.

```bash
# In chatterbox-v2
git init
git add .
git commit -m "Initial Soft Meta engine adapter"
git branch -M main
git remote add origin https://github.com/soft-meta/chatterbox-v2.git
git push -u origin main
git tag -a v0.1.0 -m "First stable engine adapter"
git push origin v0.1.0

# In Chatterbox-TTS-Server
git init
git add .
git commit -m "Initial independent Soft Meta TTS server"
git branch -M main
git remote add origin https://github.com/soft-meta/Chatterbox-TTS-Server.git
git push -u origin main
git tag -a v0.1.0 -m "First server release"
git push origin v0.1.0
```

## Voice cloning

Only clone a voice you own or have permission to use. A clear 10–20 second reference clip with one speaker, no music and low background noise is recommended.

## Licence

Soft Meta application code is MIT licensed. Chatterbox remains an MIT-licensed third-party technology by Resemble AI; retain all upstream notices.
