# Changelog

## v0.7.0

- Added identity-first speaker prompting so a new human identity is established before age and emotion
- Added stable identity codes and expanded vocal anatomy, spectral colour, nasality, vowel, consonant and speaking-habit dimensions
- Added strict over-generation: up to 12 attempts are searched to fill the requested 1–4 candidate slots
- Added immediate SpeechBrain comparison against saved voices and every earlier attempt in the current batch
- Added automatic rejection of candidates above the speaker-similarity threshold
- Added review fallback only when the strict search cannot fill all requested slots
- Replaced first-candidate “100% different” output with a truthful baseline status
- Changed the UI to show closest speaker similarity instead of a misleading difference percentage
- Added comparison count, identity code and richer identity traits to every candidate card
- Raised the default repeated-identity threshold from 0.68 to 0.72
- Updated server, UI cache keys and Colab release references to v0.7.0

## v0.6.0

- Fixed Qwen candidate preview players being rebuilt and stopped by background queue polling
- Added one-at-a-time candidate playback without refreshing the other preview players
- Added stronger identity families, age-conditioned vocal character and varied Qwen sampling profiles
- Tightened generated-voice similarity screening and blocked saving candidates marked too similar
- Auto-advanced the variation seed after each generated batch
- Changed the default workspace from Audio 1 + Audio 2 to Audio 1 only
- Added removable minus controls to Audio 2 through Audio 5
- Added a second API link labelled Chatterbox TTS API
- Refined the interface with a professional header, visual hierarchy, spacing, form controls and candidate cards
- Added five bundled user-provided predefined American male reference voices
- Added predefined voice importing directly from the Predefined Voices tab
- Updated server and UI version to 0.6.0


## v0.5.2

- Pinned a complete official Qwen3-TTS VoiceDesign model revision
- Added dedicated local snapshot download and required-file validation
- Fixed stale speech-tokenizer feature extractor cache failures
- Installed SoX in the Colab environment
- Fixed the current server-log path


## v0.5.1

- Fixed an unterminated Python string in the Qwen3-TTS Colab environment verification cell.
- Added a clean verification heading using two valid `print()` calls.
- Updated Colab release references and cache-busting UI version strings.
- No changes to voice design, queue processing, or Chatterbox generation.

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
