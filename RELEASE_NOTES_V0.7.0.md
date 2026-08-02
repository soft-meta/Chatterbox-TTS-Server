# SoftMeta Chatterbox TTS Server v0.7.0

This release changes Generate Voice from simple performance variation to an identity-first speaker search.

## Identity-first voice creation

- Speaker identity is established before age, emotion and acting style
- Explicitly prevents the same default speaker from returning with only a new pitch, tempo or tune
- Adds vocal anatomy, resonance, spectral colour, nasal balance, vocal weight, vowel shape, consonant attack, texture, personality, cadence and speaking habits
- Adds a stable identity code to each candidate
- Keeps General American English while allowing genuinely different human timbres

## Strict candidate search

- Generates up to 12 internal attempts to find the requested 1–4 candidates
- Compares every attempt with all saved generated voices
- Compares every attempt with previous attempts in the same batch
- Automatically rejects candidates that are too similar
- Uses review candidates only when the strict search cannot fill every requested slot
- Reports attempted and rejected candidate counts

## Clearer speaker comparison

- Removes the misleading “100% different” label for the first candidate
- First candidate is shown as a baseline when no previous voice exists
- Candidate cards show closest speaker similarity and comparison count
- Candidates above the strict similarity limit cannot be saved
- Review candidates require confirmation before saving

## Natural age behaviour

- Age is applied after the unique speaker identity is established
- Age changes vocal texture, projection, breath support, thought grouping and phrase planning
- No global slowdown or artificial vowel stretching
- Existing speed remains 1.00× unless the user changes it manually

## Existing features preserved

- Professional SoftMeta UI
- Audio 1–5 sequential queue
- Generate All and Remove All
- Five predefined American male voices
- Voice cloning and saved generated voices
- Queue Monitor, live progress and ETA
- Waveform playback, Start/End selection, cutter and split downloads

## Required engine

Use with `soft-meta/chatterbox-v2@v0.2.1`.
