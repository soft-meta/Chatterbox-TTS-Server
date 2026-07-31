## SoftMeta Chatterbox TTS Server v0.5.0

This release rebuilds **Generate Voice** around the official Qwen3-TTS
VoiceDesign model for more natural, varied and age-aware fictional voices.

### Qwen3-TTS Voice Designer

- Removed Parler-TTS from Generate Voice
- Added Qwen3-TTS 1.7B VoiceDesign in an isolated environment
- Generate 2, 3 or 4 distinct candidates per request
- Added age, gender, US English, General American accent and emotion control
- Added seed-driven variation across pitch, resonance, vocal texture,
  articulation, personality and cadence
- Added age-aware phrase and pause instructions without globally slowing audio
- Added candidate preview and sample download
- Added Save and Use Voice for the selected candidate
- Saved voices remain available for Chatterbox long-form cloning

### Voice difference checking

- Added optional SpeechBrain ECAPA speaker embeddings
- Compares candidates with saved generated voices and other candidates
- Shows difference score, closest saved voice and Unique/Review/Too Similar status
- Embeddings are cached beside saved voices for faster future comparisons
- The score is guidance, not a guarantee of platform originality

### Existing features preserved

- Audio 1 to Audio 5 workspaces
- Generate All and sequential queue
- Queue Monitor with live words, percentage and ETA
- Voice cloning and predefined voices
- Main waveform playback and browser fallback
- Start/End selection, zoom, scrolling and drag
- Audio cutter, Selected WAV, Part One and Part Two downloads
- Remove All and removable Audio 3–5 tabs
- Light and dark modes

### Required engine

`soft-meta/chatterbox-v2@v0.2.1`

### Colab

Use `SoftMeta_Chatterbox_TTS_Colab_v0.5.0.ipynb` in a fresh L4 runtime. The
first voice-design request downloads Qwen3-TTS VoiceDesign and the optional
speaker-embedding model.
