# SoftMeta Chatterbox TTS Server v0.6.0

This release improves voice candidate playback, speaker diversity, workspace management, API discoverability, interface quality and predefined voice support.

## Stable generated-voice previews

- Fixed candidate previews stopping after a few seconds during background queue polling
- Candidate cards are no longer rebuilt when their data has not changed
- Playing one candidate pauses only the other candidate players
- Added a clear playing state around the active candidate card
- Candidate audio uses full preview preloading for smoother playback

## Stronger voice identity diversity

- Expanded Qwen3-TTS voice identity families, pitch regions, resonance, vocal weight, textures, articulation, personality, melody, breathing and cadence
- Added age-conditioned vocal character from adult through 90+ delivery
- Reinforced that age must come from vocal texture, physical energy, breath placement and phrase planning, not global slowdown
- Varied sampling settings across candidates while keeping generation stable
- Tightened the similarity threshold from 0.78 to 0.68
- Candidates marked too similar cannot be saved
- The Base Variation Seed automatically advances after each successful batch
- Added visible voice-family labels to candidate cards

## Audio workspace changes

- The studio now opens with Audio 1 only
- Audio 2 through Audio 5 receive a removable minus control
- Removing a background tab no longer changes the current tab unnecessarily
- Remove All now resets the studio to one clean Audio 1 workspace

## API and interface improvements

- Added a Chatterbox TTS API link under API Docs
- Added a professional SoftMeta Audio Studio brand header
- Improved hierarchy, spacing, typography, form controls, buttons, tabs, voice cards, dark mode and responsive layout
- Preserved the existing queue, waveform, cutter and generation functionality

## Predefined voices

- Added five user-provided predefined American male WAV references
- Added readable display names and metadata
- Predefined Voices can now import additional WAV files directly from the interface

## Required engine

Use this release with:

`soft-meta/chatterbox-v2@v0.2.1`
