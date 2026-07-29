# Changelog

## v0.5.0

- Replaced Parler-TTS Generate Voice with official Qwen3-TTS VoiceDesign
- Added 2–4 fictional voice candidates per request
- Added age-aware natural phrase and pause instructions without global slowdown
- Added seed-driven identity variation across pitch, resonance, texture, articulation, personality and cadence
- Added candidate preview, download, save and reuse workflow
- Added optional SpeechBrain ECAPA voice-difference checking and embedding cache
- Added isolated Qwen3-TTS Colab environment
- Updated server and UI version to 0.5.0

## v0.4.0

- Added explicit speaker age, gender, US English accent and emotion fields
- Added age-based pacing from mature adult through 90+ elderly delivery
- Added a Natural Human Voice Formula and UI profile preview
- Added stable gender-matched Parler speaker identity selection by seed
- Added gentle pitch-preserving age tempo correction
- Added automatic recommended final Chatterbox speed
- Added generated voice JSON profile metadata
- Updated server and UI version to 0.4.0

## v0.3.1

- Fixed Generate Voice import failures caused by incompatible Transformers requirements
- Added an isolated Parler-TTS virtual environment that shares the main CUDA PyTorch installation
- Added a dedicated voice worker process with detailed diagnostics and timeout handling
- Kept Chatterbox and Parler-TTS dependencies separated without duplicating the GPU job queue
- Updated Colab and Docker installation flows

## v0.3.0

- Replaced the duplicate visible browser audio player with a hidden audio engine
  and custom playback controls directly below the main waveform
- Added a live waveform playhead, current/total time and seek slider
- Removed fixed “Keep first” quick-time buttons from Cut Generated Audio
- Added a linked SoftMeta Chatterbox TTS footer credit
- Increased typography throughout the studio, queue monitor and editor
- Added Remove All for completed jobs, generated output files, titles and scripts
- Added removable Audio 3–5 tabs with a minus button
- Removed Model Default from the visible voice-mode choices
- Added Generate Voice with speaker description, sample text, seed, preview,
  download and reusable saved voice references
- Added lazy Parler-TTS Mini v1.1 integration for text-described reference WAVs
- Added generated voice storage and Chatterbox cloning support
- Updated the server to `0.3.0` and kept the engine pinned to `v0.2.1`

## v0.2.1

- Pinned `setuptools<81` for compatibility with the current official PerTh package
- Verified that `PerthImplicitWatermarker` is callable before launching the server
- Preserved the safe Colab launcher that does not stop the server during Run all
- Showed the real startup error inside Colab before opening the proxy URL

## v0.2.0

- Rebuilt the browser studio to match the Azad multi-audio workflow
- Added server-side sequential Audio 1–5 queue
- Added live word progress, ETA and audio-length estimates
- Added completed-audio preview in Queue Monitor
- Added server waveform peaks, zoom, pan, mouse time and Start/End selection
- Added selected, Part One and Part Two downloads with title-based filenames

## v0.1.0

- Initial prototype
