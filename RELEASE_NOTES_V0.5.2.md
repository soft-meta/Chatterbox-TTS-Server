## SoftMeta Chatterbox TTS Server v0.5.2

This maintenance release fixes Qwen3-TTS VoiceDesign model loading when an older or incomplete Hugging Face cache snapshot is selected in Google Colab.

### Fixed

- Fixed `Can't load feature extractor` for the Qwen VoiceDesign speech tokenizer
- Pinned the verified official VoiceDesign model revision `fa0251e3279a10b4936dc49d69a59c41b07cbfc0`
- Downloads the model into a dedicated local SoftMeta model directory
- Validates all required model, processor and speech-tokenizer files before inference
- Avoids stale or partial Hugging Face snapshot folders
- Added clearer download and incomplete-model errors
- Added the system SoX package required by the Qwen audio stack
- Fixed the Recent Server Log cell to use the current release log file

### Existing Features Preserved

- Multiple unique fictional voice candidates
- Age, gender, General American accent and emotion controls
- Natural age-aware cadence without global audio slowdown
- Candidate preview, download and Save and Use Voice
- Optional speaker-difference checking
- Professional multi-audio queue and editing workflow

### Required Engine

Use this release with `soft-meta/chatterbox-v2@v0.2.1`.

### Colab

Use `SoftMeta_Chatterbox_TTS_Colab_v0.5.2.ipynb` in a fresh L4 GPU runtime. The first Generate Voice request downloads the verified Qwen3-TTS VoiceDesign snapshot and may take several minutes.
