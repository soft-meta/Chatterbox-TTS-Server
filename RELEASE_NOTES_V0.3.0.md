## SoftMeta Chatterbox TTS Server v0.3.0

This release improves the completed-audio workflow, workspace management,
readability and voice creation tools.

### Main waveform playback

- Removed the duplicate visible browser music player
- Added Play/Pause directly below the first waveform
- Added current time, total duration and a seek slider
- Added a live playhead to the waveform while audio is playing
- Preserved Download WAV and playback when waveform loading fails

### Audio cutter

- Removed fixed Keep First buttons
- Kept custom Start and End times
- Kept Preview Selected, Download Selected WAV, Part One and Part Two
- Original generated audio remains unchanged

### Workspace controls

- Added Remove All beside Generate All
- Remove All clears completed jobs, generated output files, titles and scripts
- Added a minus button for removing accidentally added Audio 3, 4 or 5 tabs
- Active queued or running tabs remain protected from deletion

### Generate Voice

- Removed Model Default from the visible voice-mode tabs
- Added Generate Voice with speaker description and sample text
- Added seed control for creating alternative samples
- Added preview, download and reusable saved generated voices
- Generated reference WAVs can be selected for Chatterbox long-form cloning
- Uses Parler-TTS Mini v1.1 lazily and unloads Chatterbox during voice design to
  reduce GPU pressure on Colab L4

Text-described age, accent and voice traits are approximate and can vary between
seeds. This feature does not create or verify a real person’s identity.

### Interface polish

- Increased font sizes across the studio and Queue Monitor
- Added a linked SoftMeta Chatterbox TTS footer credit
- Updated server/UI version to v0.3.0

### Required engine

Use with:

`soft-meta/chatterbox-v2@v0.2.1`
