# Third-party notices

## Chatterbox

- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT

## Faster-Whisper

Production transcript verification uses Faster-Whisper as a lazy quality-control dependency.

- Upstream: https://github.com/SYSTRAN/faster-whisper
- Licence: MIT

## SpeechBrain speaker verification

Reference-speaker consistency uses SpeechBrain's ECAPA-TDNN verifier with the `speechbrain/spkrec-ecapa-voxceleb` model. Similarity screening is a production consistency aid, not an identity guarantee.

- Upstream: https://github.com/speechbrain/speechbrain
- Model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Licence: Apache-2.0

## FFmpeg

SoftMeta uses the locally installed FFmpeg executable for audio decoding, pitch-preserving tempo changes, professional mastering, 48 kHz video-master export and waveform/audio utilities. FFmpeg licensing depends on the build installed by the user.

## SoftMeta code

The server, queue, browser UI, text direction, quality orchestration, caption export and audio workflow are maintained by SoftMeta under the repository MIT licence.
