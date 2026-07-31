# Third-party notices

## Chatterbox

SoftMeta Chatterbox TTS Server uses the official open-source Chatterbox package
and model technology developed by Resemble AI.

- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT

## Qwen3-TTS

The optional **Generate Voice** feature uses Qwen3-TTS VoiceDesign to create new
fictional reference voices from natural-language descriptions.

- Upstream: https://github.com/QwenLM/Qwen3-TTS
- Model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- Licence: Apache-2.0

## SpeechBrain speaker embeddings

The optional candidate difference check uses SpeechBrain ECAPA-TDNN speaker
embeddings. It is a similarity aid, not a guarantee that two voices will always
sound different to every listener.

- Upstream: https://github.com/speechbrain/speechbrain
- Model: speechbrain/spkrec-ecapa-voxceleb
- Licence: Apache-2.0

## SoftMeta code

The server, queue, web UI, waveform tools, audio cutter, voice-candidate workflow
and Colab launcher in this repository are maintained by SoftMeta under the
repository MIT licence.
