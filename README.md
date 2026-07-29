# SoftMeta Chatterbox TTS Server

A professional, self-hosted Chatterbox speech studio maintained by **SoftMeta**.
The server and browser UI are SoftMeta code. Model inference is provided by the
official MIT-licensed Chatterbox package from Resemble AI through
`soft-meta/chatterbox-v2`.

## v0.2.0

This release replaces the v0.1 prototype UI and makes the uploaded Azad Colab
workspace the product reference for layout and behaviour.

### Studio interface

- Professional light and dark interface
- Chatterbox Original English as the default model
- Motivational Speech as the default preset
- Audio 1 and Audio 2 workspaces by default
- Add tabs up to Audio 5
- Video Title stays inside each Audio workspace
- Generate Audio buttons and Generate All
- No Server Configuration block in the UI

### Voice tools

- Model default voice
- Predefined voice files
- Uploaded reference voice cloning
- Voice upload, refresh and preview
- Per-tab voice and generation settings

### Queue and progress

- Server-side sequential GPU queue
- A tab can be edited while another audio is generating
- Queue Monitor with selectable video titles
- Live generated and remaining percentages
- Live word progress, elapsed time and ETA
- Estimated final audio duration
- Completed-audio preview and download
- Minimise the monitor to a bottom-right progress widget

### Generated audio

- Native browser player is always available
- Waveform is loaded from server-generated peaks, avoiding full-WAV browser decoding
- Graceful waveform failure with Download WAV preserved
- Mouse time and click-to-set Start or End
- Zoom, fit, horizontal wheel scrolling and drag-to-pan
- Start and End selection
- Quick first 2, 2:30, 3, 3:30, 4, 4:30 and 5 minute selections
- Preview Selected and Download Selected WAV
- Part One and Part Two downloads
- `Part_One_Video_Title.wav` and `Part_Two_Video_Title.wav` filenames
- Original generated WAV is never changed by cutting

## Repository relationship

```text
soft-meta/chatterbox-v2
        ↓
soft-meta/Chatterbox-TTS-Server
        ↓
Google Colab L4 or local/Docker runtime
```

This repository has no runtime dependency on `devnen/Chatterbox-TTS-Server` or
`devnen/chatterbox-v2`.

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.2.0.ipynb`, select an L4 GPU and
run every cell from top to bottom. The notebook clones the `v0.2.0` releases of
both SoftMeta repositories.

## Local installation

Python 3.11 and a compatible PyTorch installation are recommended.

```bash
git clone --branch v0.2.0 https://github.com/soft-meta/chatterbox-v2.git
git clone --branch v0.2.0 https://github.com/soft-meta/Chatterbox-TTS-Server.git

python -m pip install chatterbox-tts==0.1.7
python -m pip install --no-deps -e ./chatterbox-v2
python -m pip install -r ./Chatterbox-TTS-Server/requirements-colab.txt

cd Chatterbox-TTS-Server
python start.py
```

Open `http://127.0.0.1:8004`.

## Reference voice guidance

For natural cloning, use a clean 10–20 second recording from one speaker with
no music, echo or strong background noise. A preset cannot reliably replace the
accent in a reference voice. Only clone voices you own or have permission to use.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox software and model
technology remain the work of Resemble AI and retain their own copyright and
licence notices. See `THIRD_PARTY_NOTICES.md`.


## v0.2.1 Colab compatibility

The Colab launcher pins `setuptools==80.9.0` because the current official PerTh package still imports `pkg_resources`. This preserves the official neural watermarker and prevents the model-loading error `TypeError: 'NoneType' object is not callable`.
