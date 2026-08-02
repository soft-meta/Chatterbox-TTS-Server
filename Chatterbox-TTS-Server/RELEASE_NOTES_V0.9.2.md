# SoftMeta Chatterbox TTS Server v0.9.2

This release fixes the Colab A100 40GB installation failures shown in the v0.9.1 notebook.

## Fixed

- Replaced SpeechBrain 1.0.3 with SpeechBrain 1.1.0
- Removed the incompatible TorchAudio backend import failure
- Removed the SoftMeta SciPy pin that conflicted with the official MOSS-TTS dependency set
- Added a dedicated MOSS installer that follows the upstream `torch-runtime` environment
- Added `pip check` and import verification for the MOSS and Avatar environments
- Added a compatibility guard for stale cached SpeechBrain packages
- Changed Colab Avatar Talking to the official Ditto PyTorch backend by default
- Stopped automatically building legacy TensorRT 8.6.1 on current Colab images
- Prevented the server from selecting TensorRT when the Python runtime cannot import it
- Preserved the A100 40GB checkpointed long-video profile for 10, 20 and 30-minute renders

## Runtime profile

- GPU: Colab NVIDIA A100 40GB
- TTS: Chatterbox plus isolated MOSS VoiceGenerator
- Avatar: Ditto PyTorch
- Long video: checkpointed mode
- Working resolution: reduced internally for stability
- Export: 720p or 1080p

## Required engine

`soft-meta/chatterbox-v2@v0.2.1`

Run `colab/SoftMeta_Chatterbox_TTS_Colab_v0.9.2.ipynb` in a fresh A100 runtime.
