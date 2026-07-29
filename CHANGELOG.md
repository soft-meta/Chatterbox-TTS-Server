# Changelog

## v0.2.1

- Pin `setuptools<81` for compatibility with the current official PerTh package.
- Verify that `PerthImplicitWatermarker` is callable before launching the server.
- Preserve the safe Colab launcher that does not stop the server during Run all.
- Show the real startup error inside Colab before opening the proxy URL.


## 0.2.0

- Rebuilt the browser studio to match the Azad multi-audio workflow
- Added a professional Devnen-inspired visual structure without using Devnen UI code
- Added robust model-loading status and readable server errors
- Added server-side sequential Audio 1–5 queue
- Added live word progress, ETA and audio-length estimates
- Added completed-audio preview in Queue Monitor
- Added custom server waveform peaks and native-player fallback
- Added waveform zoom, pan, mouse time and Start/End selection
- Added selected, Part One and Part Two downloads with title-based filenames
- Added SoftMeta-only Colab launcher and v0.2.0 release pins

## 0.1.0

- Initial prototype
