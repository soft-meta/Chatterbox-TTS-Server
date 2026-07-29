## SoftMeta Chatterbox TTS Server v0.3.1

This maintenance release fixes Generate Voice in Google Colab and Docker.

### Fixed

- Fixed `Generate Voice requires parler-tts==0.2.3` errors when Parler-TTS was installed but could not import
- Separated Parler-TTS from the Chatterbox runtime because the two engines require incompatible Transformers versions
- Added a dedicated isolated voice-generation process
- Added clear worker logs, environment checks and a 30-minute safety timeout
- Preserved the main CUDA PyTorch installation through a lightweight `--system-site-packages` virtual environment

### Colab Improvements

- Creates `/content/softmeta_voice_env` for Parler-TTS
- Installs all official Parler-TTS dependencies instead of using `--no-deps`
- Verifies `parler-tts`, Transformers 4.46.1 and CUDA before starting the server
- Passes the isolated Python executable to the SoftMeta server

### Existing Features Preserved

- Professional SoftMeta UI
- Main-waveform playback controls
- Multi-audio queue and Generate All
- Remove All and removable Audio tabs
- Voice cloning, predefined voices and generated voices
- Queue Monitor, live words, percentage and ETA
- Waveform cutter, Start/End selection and split downloads

### Required Engine

Use this server with `soft-meta/chatterbox-v2@v0.2.1`.
