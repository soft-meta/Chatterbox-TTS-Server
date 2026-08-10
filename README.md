# SoftMeta Chatterbox TTS Server

## v1.5.7 Single Professional Generate Audio

Chatterbox Turbo remains the production default. The former Standard-vs-Advanced A/B experiment is complete: the professional pipeline is now the single **Generate Audio** workflow. Chatterbox Original remains selectable and its existing behavior is intentionally frozen for now.

### Generate Audio

Generate Audio includes the professional long-form pipeline: pronunciation preparation, Senior Clear Speech, semantic/chunk planning, natural micro-pauses, senior pace profiles, Production QC, speaker consistency advisory, dynamics-preserving mastering, Prosody reporting, 48 kHz video master and SRT/VTT captions.

### Optional Auto Emotion

Auto Emotion is **OFF by default**. When enabled, the server reads the full Turbo script and may insert only four audible native-event controls when the wording genuinely supports them: `[chuckle]`, `[laugh]`, `[sigh]`, and `[gasp]`. A typical seven-minute script with enough real emotional moments can reach roughly twelve well-spaced events, with more attention in the first five-minute avatar section and a calmer B-roll tail. Neutral instructional text receives no fabricated event just to hit a count.

Manual Turbo event insertion in the UI is limited to the same four controls. Event brackets are protected through pronunciation and clear-speech normalization so the token reaches Turbo intact.

### Queue workflow

Each finished Queue Monitor card has an × control that dismisses only that history card while preserving the generated audio and Audio workspace. The minimized queue card now shows a live percentage bar and can be dragged to a convenient screen position; its position is remembered in the browser.

