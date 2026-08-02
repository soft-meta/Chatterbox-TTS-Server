# SoftMeta Chatterbox TTS Server v0.9.1

This hotfix repairs the A100 Avatar Talking installer and tunes long-video rendering for the 40GB A100 commonly provided by Google Colab.

## Fixed

- Fixed `No such file or directory: scripts/install_ditto_a100.sh`
- Added a complete embedded installer fallback to the Colab notebook
- Added explicit required-file checks after cloning the server
- Simplified the TensorRT 8.6.1 installation to match the upstream Ditto instructions

## A100 40GB long-video profile

- Detects GPU name and total VRAM with `nvidia-smi`
- Uses 576×1024 portrait, 1024×576 landscape, or 640×640 square internal rendering on GPUs below 48GB
- Keeps final export options at 720p and 1080p
- Defaults to checkpointed rendering with approximately two-minute sections
- Automatically changes continuous videos longer than five minutes to checkpointed mode on 40GB GPUs
- Unloads the Chatterbox model before Avatar Talking and restores it after the video job

## Required engine

Keep `soft-meta/chatterbox-v2@v0.2.1`.

## Test first

Run a 10–20 second avatar test before a 10–30 minute job. Full Ditto inference still needs validation in the target Colab A100 runtime.
