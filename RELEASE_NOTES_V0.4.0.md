## SoftMeta Chatterbox TTS Server v0.4.0

This release introduces a deeply improved age-aware Natural Human Voice Designer for more believable American male and female voice references.

### Age-Aware Voice Creation

- Added a dedicated Speaker Age field from 18 to 110
- Added explicit Male and Female selection
- Added US English with American accent as the default and enforced voice-design language
- Added selectable emotional delivery: warm, calm, reflective, concerned, serious and hopeful
- Added age-specific pacing, vocal energy, breath, pause and texture instructions
- Added progressively slower recommended final speed for speakers in their 50s, 60s, 70s, 80s and 90s
- Added gentle pitch-preserving age-tempo correction to generated reference samples

### Natural Human Voice Formula

- Added an automatic professional voice formula based on age, gender, emotion and user notes
- Added private one-to-one conversational delivery guidance
- Added subtle human variation in timing, pitch, breath, energy and emphasis
- Added guidance to avoid AI-assistant, audiobook, advertisement, newsreader, radio-host and customer-service delivery
- Added an on-screen age profile, pacing summary and Natural Human Voice Formula preview

### Voice Consistency

- Uses high-consistency named speakers supported by Parler-TTS Mini
- Selects a stable male or female identity from the Voice Variation Seed
- Saves a JSON profile beside every generated reference WAV
- Automatically applies the recommended Chatterbox final speed after creating a voice

### Existing Features Preserved

- Professional SoftMeta light and dark interface
- Audio 1 to Audio 5 workspaces
- Generate All and sequential queue
- Queue Monitor with live percentage, word progress and ETA
- Voice cloning and predefined voice preview
- Main waveform playback and fallback
- Start and End selection
- Audio cutter, selected download, Part One and Part Two
- Remove All and removable Audio 3 to Audio 5 tabs

### Important Limitation

Text-described age and accent are guided characteristics, not exact guarantees. Generate several seeds and keep the most natural sample for your project.

### Required Engine

Use this release with:

`soft-meta/chatterbox-v2@v0.2.1`
