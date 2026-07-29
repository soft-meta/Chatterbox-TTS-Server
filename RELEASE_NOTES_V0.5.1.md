## SoftMeta Chatterbox TTS Server v0.5.1

This maintenance release fixes the Google Colab verification-cell syntax error introduced in v0.5.0.

### Fixed

- Fixed `SyntaxError: unterminated string literal` in the Qwen3-TTS environment verification cell
- Replaced the broken multiline print statement with valid Python
- Restored successful verification of the isolated Qwen3-TTS environment
- Updated the Colab notebook to clone the `v0.5.1` server release
- Updated server and UI version labels to `v0.5.1`

### Preserved

- Qwen3-TTS VoiceDesign candidate generation
- Age-aware American voice instructions
- Male and female voice controls
- Candidate preview, download, save and reuse
- Optional speaker-difference checking
- Multi-audio queue, waveform, cutter and split downloads

### Required Engine

Use this release with:

`soft-meta/chatterbox-v2@v0.2.1`

### Colab

Use `SoftMeta_Chatterbox_TTS_Colab_v0.5.1.ipynb` in a fresh L4 GPU runtime.
