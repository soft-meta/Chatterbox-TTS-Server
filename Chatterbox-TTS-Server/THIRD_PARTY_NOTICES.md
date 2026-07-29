# Third-party notices

## Chatterbox

SoftMeta Chatterbox TTS Server uses the official open-source Chatterbox package
and model technology developed by Resemble AI.

- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT

The original Chatterbox copyright and licence remain with their respective
owners. SoftMeta does not claim authorship of the underlying Chatterbox model.

## SoftMeta code

The server, queue, web interface, waveform tools, audio cutter and Colab launcher
in this repository are maintained by SoftMeta under the repository MIT licence.


## Parler-TTS

The optional **Generate Voice** feature uses the open-source Parler-TTS project
from Hugging Face to create a short text-described reference WAV.

- Upstream: https://github.com/huggingface/parler-tts
- Model: https://huggingface.co/parler-tts/parler-tts-mini-v1.1
- Licence: Apache-2.0

The generated reference is then used by Chatterbox voice cloning. A text
description controls broad characteristics and style, but it does not guarantee
a precise age, accent, identity or repeatable real-world person.
