(() => {
  'use strict';

  const MAX_TABS = 5;
  const VIDEO_VIEW_ID = 'generate-video';
  const STORAGE_KEY = 'softMetaChatterboxTabsV9';
  const THEME_KEY = 'softMetaChatterboxTheme';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const state = {
    initial: null,
    tabs: [],
    activeId: '',
    jobs: new Map(),
    videoJobs: new Map(),
    video: {
      panel: null,
      avatar_filename: '',
      avatar_preview_url: '',
      audio_mode: 'audio_job',
      audio_job_id: '',
      audio_filename: '',
      audio_preview_url: '',
      title: '',
      engine: 'auto',
      render_mode: 'continuous',
      segment_seconds: 180,
      aspect_ratio: '9:16',
      resolution: '1080p',
      fps: 25,
      framing: 'upper',
      image_fit: 'cover',
      quality: 'high',
      consent: false,
      active_job_id: null,
    },
    pollTimer: null,
    modelPollTimer: null,
    monitorMinimised: false,
    monitorOpen: false,
  };

  function createId() {
    return globalThis.crypto?.randomUUID?.() || `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function defaultTab(number) {
    const defaults = state.initial?.defaults || {};
    return {
      id: createId(),
      number,
      title: '',
      text: '',
      preset: defaults.preset || 'Motivational Speech',
      language: defaults.language || 'en',
      voice_mode: 'clone',
      voice_filename: '',
      generated_voice_name: '',
      generated_voice_age: 50,
      generated_voice_gender: 'male',
      generated_voice_language: 'en-US',
      generated_voice_emotion: 'warm',
      generated_voice_description: 'Warm, confident, natural, intimate, and emotionally subtle.',
      generated_voice_text: 'Today, I want to share a simple lesson that can make life feel calmer and more meaningful.',
      generated_voice_seed: 2025,
      generated_voice_candidate_count: 3,
      generated_voice_filename: '',
      generated_voice_candidates: [],
      generated_voice_session_id: '',
      temperature: defaults.temperature ?? 0.8,
      exaggeration: defaults.exaggeration ?? 0.65,
      cfg_weight: defaults.cfg_weight ?? 0.35,
      repetition_penalty: defaults.repetition_penalty ?? 1.2,
      min_p: defaults.min_p ?? 0.05,
      top_p: defaults.top_p ?? 1.0,
      top_k: defaults.top_k ?? 1000,
      speed_factor: defaults.speed_factor ?? 1.0,
      seed: defaults.seed ?? 2025,
      split_text: defaults.split_text ?? true,
      chunk_words: defaults.chunk_words ?? 90,
      output_format: defaults.output_format || 'wav',
      cut_start: '0:00',
      cut_end: '',
      job_id: null,
      panel: null,
      waveform: null,
    };
  }

  async function api(url, options = {}) {
    const response = await fetch(url, {
      cache: 'no-store',
      ...options,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const data = await response.json();
        detail = data.detail || data.message || detail;
      } catch (_) {
        // Keep the HTTP status message.
      }
      throw new Error(detail);
    }
    const contentType = response.headers.get('content-type') || '';
    return contentType.includes('application/json') ? response.json() : response;
  }

  function toast(message, type = '') {
    const item = document.createElement('div');
    item.className = `toast ${type}`;
    item.textContent = message;
    $('#toast-root').append(item);
    setTimeout(() => item.remove(), 5200);
  }

  function countWords(text) {
    return (String(text || '').match(/\b[\w’'-]+\b/gu) || []).length;
  }

  function formatTime(seconds, precise = false) {
    const value = Number.isFinite(Number(seconds)) ? Math.max(0, Number(seconds)) : 0;
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const remain = value % 60;
    const secondText = precise
      ? remain.toFixed(1).padStart(4, '0')
      : Math.floor(remain).toString().padStart(2, '0');
    if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${secondText}`;
    return `${minutes}:${secondText}`;
  }

  function humanDuration(seconds) {
    if (seconds == null || !Number.isFinite(Number(seconds))) return 'calculating...';
    const value = Math.max(0, Math.round(Number(seconds)));
    if (value < 60) return `${value} sec`;
    const minutes = Math.floor(value / 60);
    const remain = value % 60;
    return remain ? `${minutes} min ${remain} sec` : `${minutes} min`;
  }

  function parseTime(value) {
    const parts = String(value || '').trim().split(':').map(Number);
    if (!parts.length || parts.some(Number.isNaN)) return NaN;
    if (parts.length === 1) return parts[0];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    return NaN;
  }

  function safeText(value) {
    return String(value ?? '');
  }

  function generatedAgeProfile(ageValue) {
    const age = Math.max(18, Math.min(110, Number(ageValue) || 50));
    if (age < 40) return { label: 'Adult', pace: 'Natural conversational pace with varied thought groups and short pauses.', speed: 1.00 };
    if (age < 50) return { label: 'Mature adult', pace: 'Unhurried, mature delivery with natural breathing.', speed: 1.00 };
    if (age < 60) return { label: 'Mature and experienced', pace: 'Thoughtful delivery with small pauses before meaningful ideas.', speed: 1.00 };
    if (age < 70) return { label: 'Older and experienced', pace: 'Clear, grounded delivery using shorter thought groups and gentle pauses.', speed: 1.00 };
    if (age < 80) return { label: 'Elderly but mentally clear', pace: 'Measured delivery using short thought groups, variable pauses and natural breaths.', speed: 1.00 };
    if (age < 90) return { label: 'Very elderly and thoughtful', pace: 'Careful, spacious delivery with softer projection and clear natural pauses.', speed: 1.00 };
    return { label: 'Very elderly and mentally present', pace: 'Very careful, spacious delivery with low physical energy and meaningful pauses.', speed: 1.00 };
  }

  function updateVoiceProfilePreview(tab) {
    if (!tab?.panel) return;
    const panel = tab.panel;
    const age = Number(field(panel, 'generated_voice_age')?.value || tab.generated_voice_age || 50);
    const gender = field(panel, 'generated_voice_gender')?.value || tab.generated_voice_gender || 'male';
    const emotion = field(panel, 'generated_voice_emotion')?.value || tab.generated_voice_emotion || 'warm';
    const profile = generatedAgeProfile(age);
    const genderText = gender === 'female' ? 'woman' : 'man';
    const title = panel.querySelector('[data-role="age-profile-title"]');
    const summary = panel.querySelector('[data-role="age-profile-summary"]');
    const speed = panel.querySelector('[data-role="age-speed-value"]');
    const formula = panel.querySelector('[data-role="voice-formula-preview"]');
    if (title) title.textContent = `Age ${Math.round(age)} · ${profile.label}`;
    if (summary) summary.textContent = profile.pace;
    if (speed) speed.textContent = `${profile.speed.toFixed(2)}×`;
    if (formula) {
      formula.textContent = `Create one completely original fictional American ${genderText}. Speaker identity comes first: use a genuinely different perceived vocal anatomy, resonance, texture, articulation, sentence melody and personality. Do not reuse the same default person with only a different pitch, age performance or emotional tune. Then make the speaker ${Math.round(age)} years old through natural vocal texture, projection, breathing and thought grouping. ${profile.pace} Do not stretch words or globally slow the recording. Use General American English, intimate one-to-one delivery and naturally imperfect human timing. Avoid narrator, announcer, customer-service and synthetic AI rhythms.`;
    }
  }

  function saveTabs() {
    const tabs = state.tabs.map(({ panel, waveform, generated_voice_candidates, generated_voice_session_id, ...tab }) => tab);
    const { panel, ...video } = state.video;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ tabs, activeId: state.activeId, video }));
  }

  function restoreTabs() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      if (Array.isArray(saved.tabs) && saved.tabs.length) {
        state.tabs = saved.tabs.slice(0, MAX_TABS).map((savedTab, index) => ({
          ...defaultTab(index + 1),
          ...savedTab,
          number: index + 1,
          panel: null,
          waveform: null,
        }));
        state.activeId = saved.activeId === VIDEO_VIEW_ID || state.tabs.some(tab => tab.id === saved.activeId)
          ? saved.activeId
          : state.tabs[0].id;
        if (saved.video && typeof saved.video === 'object') state.video = { ...state.video, ...saved.video, panel: null };
        return;
      }
    } catch (_) {
      localStorage.removeItem(STORAGE_KEY);
    }
    state.tabs = [defaultTab(1)];
    state.activeId = state.tabs[0].id;
  }

  function currentTab() {
    return state.tabs.find(tab => tab.id === state.activeId) || null;
  }

  function videoActive() {
    return state.activeId === VIDEO_VIEW_ID;
  }

  function findTabByJob(jobId) {
    return state.tabs.find(tab => tab.job_id === jobId);
  }

  function field(panel, name) {
    return panel.querySelector(`[data-field="${name}"]`);
  }

  function buildTabs() {
    const bar = $('#audio-tabs');
    const panels = $('#audio-panels');
    bar.innerHTML = '';
    panels.innerHTML = '';

    state.tabs.forEach(tab => {
      const wrap = document.createElement('span');
      wrap.className = 'audio-tab-wrap';

      const button = document.createElement('button');
      button.className = 'audio-tab';
      button.type = 'button';
      button.textContent = `Audio ${tab.number}`;
      button.dataset.tabId = tab.id;
      button.addEventListener('click', () => {
        captureTab(currentTab());
        captureVideoState();
        state.activeId = tab.id;
        renderActive();
        saveTabs();
      });
      wrap.append(button);

      if (tab.number > 1) {
        const remove = document.createElement('button');
        remove.className = 'remove-tab';
        remove.type = 'button';
        remove.textContent = '−';
        remove.title = `Remove Audio ${tab.number}`;
        remove.setAttribute('aria-label', `Remove Audio ${tab.number}`);
        remove.addEventListener('click', event => {
          event.stopPropagation();
          removeTab(tab);
        });
        wrap.append(remove);
      }
      bar.append(wrap);

      const panel = $('#audio-panel-template').content.firstElementChild.cloneNode(true);
      panel.dataset.tabId = tab.id;
      tab.panel = panel;
      wirePanel(tab);
      panels.append(panel);
    });

    const videoButton = document.createElement('button');
    videoButton.className = 'audio-tab video-workspace-tab';
    videoButton.type = 'button';
    videoButton.dataset.tabId = VIDEO_VIEW_ID;
    videoButton.innerHTML = '<span class="video-tab-icon">▶</span> Generate Video';
    videoButton.addEventListener('click', () => {
      captureTab(currentTab());
      state.activeId = VIDEO_VIEW_ID;
      renderActive();
      saveTabs();
    });
    bar.append(videoButton);

    if (state.tabs.length < MAX_TABS) {
      const add = document.createElement('button');
      add.className = 'add-tab';
      add.type = 'button';
      add.textContent = '+';
      add.title = 'Add another audio workspace';
      add.setAttribute('aria-label', 'Add another audio workspace');
      add.addEventListener('click', addTab);
      bar.append(add);
    }

    const videoPanel = $('#video-panel-template').content.firstElementChild.cloneNode(true);
    state.video.panel = videoPanel;
    wireVideoPanel();
    panels.append(videoPanel);
    renderActive();
  }

  function addTab() {
    if (state.tabs.length >= MAX_TABS) {
      toast('You can prepare a maximum of five audio jobs.', 'error');
      return;
    }
    captureTab(currentTab());
    const tab = defaultTab(state.tabs.length + 1);
    state.tabs.push(tab);
    state.activeId = tab.id;
    buildTabs();
    saveTabs();
  }

  async function removeTab(tab) {
    const job = tab.job_id ? state.jobs.get(tab.job_id) : null;
    if (job && ['queued', 'running'].includes(job.status)) {
      toast(`Audio ${tab.number} is active. Cancel or wait for it before removing this tab.`, 'error');
      return;
    }
    if (!confirm(`Remove Audio ${tab.number} and its saved workspace${job ? ' and generated file' : ''}?`)) return;
    try {
      if (job) {
        await api(`/api/jobs/${job.id}?delete_file=true`, { method: 'DELETE' });
        state.jobs.delete(job.id);
      }
      const removedWasActive = state.activeId === tab.id;
      state.tabs = state.tabs.filter(item => item.id !== tab.id);
      state.tabs.forEach((item, index) => { item.number = index + 1; });
      if (removedWasActive || !state.tabs.some(item => item.id === state.activeId)) {
        state.activeId = state.tabs[Math.max(0, state.tabs.length - 1)].id;
      }
      buildTabs();
      saveTabs();
      renderQueue();
      toast('Audio workspace removed.', 'success');
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  function renderActive() {
    $$('.audio-tab').forEach(button => {
      const tab = state.tabs.find(item => item.id === button.dataset.tabId);
      button.classList.toggle('active', button.dataset.tabId === state.activeId);
      button.classList.remove('status-running', 'status-queued', 'status-completed', 'status-failed');
      if (button.dataset.tabId === VIDEO_VIEW_ID) {
        const job = activeVideoJob();
        if (job?.status) button.classList.add(`status-${job.status}`);
        return;
      }
      const job = tab?.job_id ? state.jobs.get(tab.job_id) : null;
      if (job?.status) button.classList.add(`status-${job.status}`);
    });

    state.tabs.forEach(tab => {
      tab.panel.classList.toggle('hidden', tab.id !== state.activeId);
      if (tab.id === state.activeId) fillPanel(tab);
    });
    if (state.video.panel) {
      state.video.panel.classList.toggle('hidden', !videoActive());
      if (videoActive()) {
        fillVideoPanel();
        renderVideoHistory();
      }
    }
  }

  function updateSliderOutput(panel, name) {
    const input = field(panel, name);
    const output = panel.querySelector(`[data-output="${name}"]`);
    if (input && output) output.value = Number(input.value).toFixed(name === 'speed_factor' ? 2 : 2);
  }

  function fillPanel(tab) {
    const panel = tab.panel;
    const stringFields = ['title', 'text', 'preset', 'language', 'voice_filename', 'generated_voice_name', 'generated_voice_gender', 'generated_voice_language', 'generated_voice_emotion', 'generated_voice_description', 'generated_voice_text', 'generated_voice_filename', 'output_format', 'cut_start', 'cut_end'];
    const numberFields = ['temperature', 'exaggeration', 'cfg_weight', 'speed_factor', 'seed', 'chunk_words', 'generated_voice_age', 'generated_voice_seed', 'generated_voice_candidate_count'];
    stringFields.forEach(name => {
      const element = field(panel, name);
      if (element && tab[name] !== undefined) element.value = tab[name];
    });
    numberFields.forEach(name => {
      const element = field(panel, name);
      if (element && tab[name] !== undefined) element.value = tab[name];
    });
    field(panel, 'split_text').checked = Boolean(tab.split_text);
    panel.querySelector('[data-role="word-count"]').textContent = `${countWords(tab.text)} words`;
    panel.querySelector('[data-role="chunk-value"]').textContent = tab.chunk_words;
    panel.querySelector('.audio-number-chip').textContent = `Audio ${tab.number}`;
    panel.querySelector('[data-role="generate-label"]').textContent = `Generate Audio ${tab.number}`;
    ['temperature', 'exaggeration', 'cfg_weight', 'speed_factor'].forEach(name => updateSliderOutput(panel, name));
    updateVoiceControls(tab);
    updateVoiceProfilePreview(tab);
    renderVoiceCandidates(tab);
    updatePresetChips(tab);
    const job = tab.job_id ? state.jobs.get(tab.job_id) : null;
    if (job?.status === 'completed') showGenerated(tab, job);
  }

  function captureTab(tab) {
    if (!tab?.panel) return;
    const panel = tab.panel;
    ['title', 'text', 'preset', 'language', 'voice_filename', 'generated_voice_name', 'generated_voice_gender', 'generated_voice_language', 'generated_voice_emotion', 'generated_voice_description', 'generated_voice_text', 'generated_voice_filename', 'output_format', 'cut_start', 'cut_end'].forEach(name => {
      const element = field(panel, name);
      if (element) tab[name] = element.value;
    });
    ['temperature', 'exaggeration', 'cfg_weight', 'speed_factor', 'seed', 'chunk_words', 'generated_voice_age', 'generated_voice_seed', 'generated_voice_candidate_count'].forEach(name => {
      const element = field(panel, name);
      if (element) tab[name] = Number(element.value);
    });
    tab.split_text = Boolean(field(panel, 'split_text')?.checked);
  }

  function buildPresetButtons(tab) {
    const container = tab.panel.querySelector('[data-role="preset-buttons"]');
    const select = field(tab.panel, 'preset');
    container.innerHTML = '';
    select.innerHTML = '';
    state.initial.presets.forEach(preset => {
      select.add(new Option(preset.name, preset.name));
      const button = document.createElement('button');
      button.className = 'preset-chip';
      button.type = 'button';
      button.textContent = preset.name;
      button.title = preset.description || preset.name;
      button.dataset.preset = preset.name;
      button.addEventListener('click', () => {
        select.value = preset.name;
        applyPreset(tab, preset.name);
      });
      container.append(button);
    });
  }

  function updatePresetChips(tab) {
    $$('[data-preset]', tab.panel).forEach(button => {
      button.classList.toggle('active', button.dataset.preset === tab.preset);
    });
  }

  function applyPreset(tab, requestedName = null) {
    const name = requestedName || field(tab.panel, 'preset').value;
    const preset = state.initial.presets.find(item => item.name === name);
    if (!preset) return;
    tab.preset = name;
    for (const [key, value] of Object.entries(preset)) {
      if (key in tab) tab[key] = value;
    }
    fillPanel(tab);
    saveTabs();
    toast(`${name} preset applied.`, 'success');
  }

  function setVoiceMode(tab, mode) {
    tab.voice_mode = mode;
    updateVoiceControls(tab);
    saveTabs();
  }

  function updateVoiceControls(tab) {
    const panel = tab.panel;
    const mode = tab.voice_mode || 'clone';
    $$('[data-voice-mode]', panel).forEach(button => {
      button.classList.toggle('active', button.dataset.voiceMode === mode);
    });

    const standardTools = panel.querySelector('[data-role="standard-voice-tools"]');
    const designer = panel.querySelector('[data-role="voice-designer"]');
    standardTools.classList.toggle('hidden', mode === 'generated');
    designer.classList.toggle('hidden', mode !== 'generated');

    const generatedSelect = field(panel, 'generated_voice_filename');
    generatedSelect.innerHTML = '';
    const generatedList = state.initial.generated_voices || [];
    if (!generatedList.length) {
      generatedSelect.add(new Option('Generate your first voice', ''));
    } else {
      generatedList.forEach(item => {
        const details = [item.display_name || item.filename, item.age ? `Age ${item.age}` : '', item.gender || ''].filter(Boolean).join(' · ');
        generatedSelect.add(new Option(details, item.filename));
      });
    }
    if (tab.generated_voice_filename && generatedList.some(item => item.filename === tab.generated_voice_filename)) {
      generatedSelect.value = tab.generated_voice_filename;
    } else if (mode === 'generated') {
      tab.generated_voice_filename = generatedSelect.value || '';
    }
    updateGeneratedVoiceDownload(tab);

    if (mode === 'generated') {
      tab.voice_filename = tab.generated_voice_filename || '';
      return;
    }

    const select = field(panel, 'voice_filename');
    const upload = panel.querySelector('[data-role="voice-upload-button"]');
    const preview = panel.querySelector('[data-action="preview-voice"]');
    const label = panel.querySelector('[data-role="voice-label"]');
    select.innerHTML = '';

    const list = mode === 'predefined'
      ? state.initial.predefined_voices
      : state.initial.reference_voices;
    select.disabled = false;
    upload.classList.remove('hidden');
    upload.title = mode === 'predefined' ? 'Import a reusable predefined voice' : 'Import a reference voice for cloning';
    preview.disabled = false;
    label.textContent = mode === 'predefined' ? 'Select Predefined Voice' : 'Reference Audio File';

    if (!list.length) {
      select.add(new Option(mode === 'clone' ? 'Upload a reference voice' : 'No predefined voices found', ''));
    } else {
      list.forEach(item => {
        const details = [
          item.display_name || item.filename,
          item.gender ? String(item.gender).replace(/^./, value => value.toUpperCase()) : '',
          item.accent || '',
        ].filter(Boolean).join(' · ');
        select.add(new Option(details, item.filename));
      });
    }
    if (tab.voice_filename && list.some(item => item.filename === tab.voice_filename)) {
      select.value = tab.voice_filename;
    } else {
      tab.voice_filename = select.value || '';
    }
  }

  async function refreshVoices(tab, quiet = false) {
    try {
      const voices = await api('/api/voices');
      state.initial.predefined_voices = voices.predefined;
      state.initial.reference_voices = voices.clone;
      state.initial.generated_voices = voices.generated || [];
      state.tabs.forEach(updateVoiceControls);
      if (!quiet) toast('Voice list refreshed.', 'success');
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  async function uploadVoice(tab, file) {
    if (!file) return;
    const data = new FormData();
    data.append('file', file);
    const localPlayer = tab.panel.querySelector('[data-role="voice-player"]');
    const localUrl = URL.createObjectURL(file);
    localPlayer.src = localUrl;
    localPlayer.classList.remove('hidden');
    try {
      const kind = tab.voice_mode === 'predefined' ? 'predefined' : 'clone';
      const result = await api(`/api/voices/upload?kind=${kind}`, { method: 'POST', body: data });
      await refreshVoices(tab, true);
      tab.voice_filename = result.filename;
      updateVoiceControls(tab);
      field(tab.panel, 'voice_filename').value = result.filename;
      saveTabs();
      toast(kind === 'predefined' ? 'Predefined voice imported.' : 'Reference voice uploaded.', 'success');
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  function previewVoice(tab) {
    captureTab(tab);
    if (!tab.voice_filename) {
      toast('Select or upload a voice first.', 'error');
      return;
    }
    const player = tab.panel.querySelector('[data-role="voice-player"]');
    player.src = `/api/voices/${encodeURIComponent(tab.voice_mode)}/${encodeURIComponent(tab.voice_filename)}`;
    player.classList.remove('hidden');
    player.play().catch(() => toast('The browser could not play this voice preview.', 'error'));
  }

  function updateGeneratedVoiceDownload(tab) {
    if (!tab?.panel) return;
    const filename = field(tab.panel, 'generated_voice_filename')?.value || tab.generated_voice_filename || '';
    const link = tab.panel.querySelector('[data-role="download-generated-voice"]');
    if (!filename) {
      link.href = '#';
      link.classList.add('disabled');
      return;
    }
    link.href = `/api/voices/generated/${encodeURIComponent(filename)}?download=true`;
    link.download = filename;
    link.classList.remove('disabled');
  }

  function previewGeneratedVoice(tab) {
    captureTab(tab);
    const filename = tab.generated_voice_filename;
    if (!filename) {
      toast('Generate or select a saved voice first.', 'error');
      return;
    }
    const player = tab.panel.querySelector('[data-role="generated-voice-player"]');
    player.src = `/api/voices/generated/${encodeURIComponent(filename)}`;
    player.classList.remove('hidden');
    player.play().catch(() => toast('The browser could not play this generated voice.', 'error'));
  }

  function uniquenessLabel(candidate) {
    const uniqueness = candidate?.uniqueness || {};
    if (!uniqueness.checked) return { text: 'Speaker comparison unavailable', className: 'not-checked' };
    if (uniqueness.status === 'baseline') return { text: 'Baseline candidate', className: 'baseline' };
    const similarity = Number(uniqueness.similarity_percent ?? 0).toFixed(1);
    if (uniqueness.status === 'too_similar') return { text: `${similarity}% similar · rejected`, className: 'too-similar' };
    if (uniqueness.status === 'review') return { text: `${similarity}% similar · review`, className: 'review' };
    return { text: `${similarity}% similar · unique`, className: 'unique' };
  }


  function renderVoiceCandidates(tab, force = false) {
    if (!tab?.panel) return;
    const container = tab.panel.querySelector('[data-role="voice-candidate-list"]');
    const candidates = Array.isArray(tab.generated_voice_candidates) ? tab.generated_voice_candidates : [];
    const signature = JSON.stringify({
      session: tab.generated_voice_session_id || '',
      candidates: candidates.map(candidate => ({
        filename: candidate.filename,
        seed: candidate.seed,
        difference: candidate.uniqueness?.difference_score ?? null,
        status: candidate.uniqueness?.status ?? null,
        quality: candidate.quality?.score ?? null,
        qualityStatus: candidate.quality?.status ?? null,
      })),
    });

    if (!force && container.dataset.renderSignature === signature) return;
    container.dataset.renderSignature = signature;
    container.innerHTML = '';
    container.classList.toggle('hidden', !candidates.length);

    candidates.forEach(candidate => {
      const card = document.createElement('article');
      card.className = 'voice-candidate-card';
      const label = uniquenessLabel(candidate);
      const traits = candidate.identity_traits || {};
      const closest = candidate.uniqueness?.closest_voice
        ? `Closest comparison: ${candidate.uniqueness.closest_voice}`
        : candidate.uniqueness?.status === 'baseline'
          ? 'First comparison baseline'
          : 'No close comparison found';
      const compared = Number(candidate.uniqueness?.reference_count || 0);
      const family = traits.voice_family ? `<span class="voice-family-chip">${safeText(traits.voice_family)}</span>` : '';
      const quality = candidate.quality || {};
      const qualityScore = Number(quality.score ?? 0).toFixed(1);
      const qualityStatus = quality.status || 'not-checked';
      const qualityText = quality.checked ? `Naturalness ${qualityScore}` : 'Quality not checked';
      const tooSimilar = candidate.uniqueness?.status === 'too_similar';
      const qualityRejected = qualityStatus === 'reject';
      const needsReview = candidate.uniqueness?.status === 'review' || qualityStatus === 'review';
      const rejected = tooSimilar || qualityRejected;
      const saveLabel = rejected ? 'Rejected — Generate Again' : needsReview ? 'Review and Save Voice' : 'Save and Use Voice';
      const saveDisabled = rejected ? 'disabled aria-disabled="true"' : '';
      card.innerHTML = `
        <div class="voice-candidate-head">
          <div>
            <strong>Candidate ${Number(candidate.candidate_number || 0)}</strong>
            <span>Seed ${Number(candidate.seed || 0)} · ${safeText(candidate.age_label || '')}</span>
          </div>
          <div class="voice-score-stack">
            <span class="voice-uniqueness ${label.className}">${label.text}</span>
            <span class="voice-quality ${safeText(qualityStatus)}">${safeText(qualityText)}</span>
          </div>
        </div>
        ${family}
        <p class="voice-candidate-code">${safeText(candidate.identity_code || '')}</p>
        <p class="voice-candidate-traits">${safeText(traits.pitch || '')} · ${safeText(traits.vocal_anatomy || '')} · ${safeText(traits.spectral_colour || '')}</p>
        <p class="voice-candidate-traits">${safeText(traits.texture || '')} · ${safeText(traits.personality || '')} · ${safeText(traits.speaking_habit || '')}</p>
        <p class="voice-candidate-closest">${safeText(closest)}${compared ? ` · compared with ${compared} voice${compared === 1 ? '' : 's'}` : ''}</p>
        <p class="voice-candidate-quality">Pause ${(Number(quality.silence_ratio || 0) * 100).toFixed(1)}% · level ${Number(quality.rms_db || 0).toFixed(1)} dBFS · dynamics ${Number(quality.dynamic_range_db || 0).toFixed(1)} dB</p>
        <audio controls preload="auto" src="${candidate.preview_url}"></audio>
        <div class="voice-candidate-actions">
          <a class="button button-secondary" href="${candidate.download_url}" download>Download Sample</a>
          <button class="button button-primary" type="button" data-save-candidate="${safeText(candidate.filename)}" ${saveDisabled}>${saveLabel}</button>
        </div>
      `;

      const player = card.querySelector('audio');
      player.addEventListener('play', () => {
        $$('audio', container).forEach(other => {
          if (other !== player && !other.paused) other.pause();
        });
        $$('.voice-candidate-card', container).forEach(item => item.classList.remove('is-playing'));
        card.classList.add('is-playing');
      });
      player.addEventListener('pause', () => card.classList.remove('is-playing'));
      player.addEventListener('ended', () => card.classList.remove('is-playing'));
      const saveButton = card.querySelector('[data-save-candidate]');
      if (!rejected) saveButton.addEventListener('click', () => {
        if (needsReview && !confirm('This candidate is closer to another saved or batch voice than recommended. Save it anyway?')) return;
        saveVoiceCandidate(tab, candidate);
      });
      container.append(card);
    });
  }

  async function saveVoiceCandidate(tab, candidate) {
    const button = tab.panel.querySelector(`[data-save-candidate="${CSS.escape(candidate.filename)}"]`);
    if (button) {
      button.disabled = true;
      button.textContent = 'Saving...';
    }
    try {
      const suffix = Number(candidate.candidate_number || 1);
      const voiceName = `${tab.generated_voice_name.trim()} ${suffix}`.trim();
      const result = await api('/api/voice-designer/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: tab.generated_voice_session_id,
          filename: candidate.filename,
          voice_name: voiceName,
        }),
      });
      await refreshVoices(tab, true);
      tab.voice_mode = 'generated';
      tab.generated_voice_filename = result.filename;
      tab.voice_filename = result.filename;
      tab.speed_factor = 1.0;
      updateVoiceControls(tab);
      field(tab.panel, 'generated_voice_filename').value = result.filename;
      field(tab.panel, 'speed_factor').value = '1';
      updateSliderOutput(tab.panel, 'speed_factor');
      updateGeneratedVoiceDownload(tab);
      saveTabs();
      previewGeneratedVoice(tab);
      toast('Voice saved and selected for audio generation.', 'success');
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = 'Save and Use Voice';
      }
    }
  }

  async function generateDesignedVoice(tab) {
    captureTab(tab);
    if (!tab.generated_voice_name.trim()) {
      toast('Add a Voice Name.', 'error');
      return;
    }
    if (tab.generated_voice_text.trim().length < 4) {
      toast('Add a short Voiceover Sample Text.', 'error');
      return;
    }
    const activeJobs = [...state.jobs.values()].some(job => ['queued', 'running'].includes(job.status));
    if (activeJobs) {
      toast('Wait for the audio queue to finish before generating new voices.', 'error');
      return;
    }

    const button = tab.panel.querySelector('[data-action="generate-voice"]');
    const status = tab.panel.querySelector('[data-role="voice-designer-status"]');
    button.disabled = true;
    button.textContent = 'Generating Candidates...';
    status.textContent = 'Building identity-first MOSS speakers, over-generating candidates, screening acoustic quality, and rejecting repeated identities. Up to 12 attempts may be checked.';
    try {
      const result = await api('/api/voice-designer/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: tab.generated_voice_name,
          age: Number(tab.generated_voice_age || 50),
          gender: tab.generated_voice_gender || 'male',
          language: tab.generated_voice_language || 'en-US',
          emotion: tab.generated_voice_emotion || 'warm',
          description: tab.generated_voice_description,
          sample_text: tab.generated_voice_text,
          seed: Number(tab.generated_voice_seed || 2025),
          candidate_count: Number(tab.generated_voice_candidate_count || 3),
          uniqueness_threshold: 0.72,
        }),
      });
      tab.generated_voice_session_id = result.session_id;
      tab.generated_voice_candidates = result.candidates || [];
      tab.generated_voice_seed = (Number(tab.generated_voice_seed || 2025) + 1000003) % 2147483647;
      field(tab.panel, 'generated_voice_seed').value = tab.generated_voice_seed;
      renderVoiceCandidates(tab, true);
      saveTabs();
      const checked = tab.generated_voice_candidates.filter(item => item.uniqueness?.checked).length;
      const duplicateRejected = Number(result.duplicate_rejected_count || 0);
      const qualityRejected = Number(result.quality_rejected_count || 0);
      status.textContent = `Selected ${tab.generated_voice_candidates.length} voice${tab.generated_voice_candidates.length === 1 ? '' : 's'} from ${Number(result.attempted_count || tab.generated_voice_candidates.length)} attempt${Number(result.attempted_count || 0) === 1 ? '' : 's'}. Rejected ${duplicateRejected} repeated identit${duplicateRejected === 1 ? 'y' : 'ies'} and ${qualityRejected} low-quality sample${qualityRejected === 1 ? '' : 's'}.${result.search_exhausted ? ' The strict search returned fewer voices than requested.' : ''}`;
      toast('Voice candidates are ready to preview.', 'success');
    } catch (error) {
      status.textContent = error.message;
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
      button.textContent = 'Generate Candidates';
    }
  }

  function wirePanel(tab) {
    const panel = tab.panel;
    buildPresetButtons(tab);
    $$('[data-voice-mode]', panel).forEach(button => {
      button.addEventListener('click', () => setVoiceMode(tab, button.dataset.voiceMode));
    });

    panel.addEventListener('input', event => {
      if (event.target.matches('[data-field="text"]')) {
        panel.querySelector('[data-role="word-count"]').textContent = `${countWords(event.target.value)} words`;
      }
      if (event.target.matches('[data-field="chunk_words"]')) {
        panel.querySelector('[data-role="chunk-value"]').textContent = event.target.value;
      }
      if (event.target.matches('[data-field="temperature"], [data-field="exaggeration"], [data-field="cfg_weight"], [data-field="speed_factor"]')) {
        updateSliderOutput(panel, event.target.dataset.field);
      }
      if (event.target.matches('[data-field="generated_voice_age"], [data-field="generated_voice_gender"], [data-field="generated_voice_language"], [data-field="generated_voice_emotion"]')) {
        updateVoiceProfilePreview(tab);
      }
      captureTab(tab);
      saveTabs();
      updateCutSummary(tab);
    });

    panel.querySelector('[data-action="generate"]').addEventListener('click', () => generateOne(tab));
    panel.querySelector('[data-role="voice-upload"]').addEventListener('change', event => uploadVoice(tab, event.target.files[0]));
    panel.querySelector('[data-action="refresh-voices"]').addEventListener('click', () => refreshVoices(tab));
    panel.querySelector('[data-action="preview-voice"]').addEventListener('click', () => previewVoice(tab));
    panel.querySelector('[data-action="generate-voice"]').addEventListener('click', () => generateDesignedVoice(tab));
    panel.querySelector('[data-action="preview-generated-voice"]').addEventListener('click', () => previewGeneratedVoice(tab));
    field(panel, 'generated_voice_filename').addEventListener('change', event => {
      tab.generated_voice_filename = event.target.value;
      tab.voice_filename = event.target.value;
      updateGeneratedVoiceDownload(tab);
      saveTabs();
    });
    panel.querySelector('[data-action="toggle-playback"]').addEventListener('click', () => togglePlayback(tab));
    panel.querySelector('[data-role="playback-progress"]').addEventListener('input', event => seekPlayback(tab, Number(event.target.value) / 1000));

    panel.querySelector('[data-action="set-start-mode"]').addEventListener('click', () => setClickMode(tab, 'start'));
    panel.querySelector('[data-action="set-end-mode"]').addEventListener('click', () => setClickMode(tab, 'end'));
    panel.querySelector('[data-action="pan-wave"]').addEventListener('click', () => setClickMode(tab, 'pan'));
    panel.querySelector('[data-action="zoom-in"]').addEventListener('click', () => zoomWave(tab, 1.5));
    panel.querySelector('[data-action="zoom-out"]').addEventListener('click', () => zoomWave(tab, 1 / 1.5));
    panel.querySelector('[data-action="fit-wave"]').addEventListener('click', () => {
      if (!tab.waveform) return;
      tab.waveform.zoom = 1;
      drawWave(tab);
      panel.querySelector('[data-role="wave-scroll"]').scrollLeft = 0;
    });
    panel.querySelector('[data-action="use-end-start"]').addEventListener('click', () => {
      field(panel, 'cut_start').value = field(panel, 'cut_end').value;
      captureTab(tab);
      updateCutSummary(tab);
      saveTabs();
    });
    panel.querySelector('[data-action="preview-selected"]').addEventListener('click', () => previewSelected(tab));
    panel.querySelector('[data-action="download-selected"]').addEventListener('click', () => {
      cutDownload(tab, 'Selected', parseTime(field(panel, 'cut_start').value), parseTime(field(panel, 'cut_end').value));
    });
    panel.querySelector('[data-action="download-part-one"]').addEventListener('click', () => {
      cutDownload(tab, 'Part_One', 0, parseTime(field(panel, 'cut_end').value));
    });
    panel.querySelector('[data-action="download-part-two"]').addEventListener('click', () => {
      cutDownload(tab, 'Part_Two', parseTime(field(panel, 'cut_end').value), tab.waveform?.duration);
    });
  }

  function optionsFor(tab) {
    return {
      model: $('#active-model').value,
      language: tab.language,
      temperature: tab.temperature,
      exaggeration: tab.exaggeration,
      cfg_weight: tab.cfg_weight,
      repetition_penalty: tab.repetition_penalty,
      min_p: tab.min_p,
      top_p: tab.top_p,
      top_k: tab.top_k,
      speed_factor: tab.speed_factor,
      seed: tab.seed,
      split_text: tab.split_text,
      chunk_words: tab.chunk_words,
      output_format: tab.output_format,
    };
  }

  function jobPayload(tab) {
    captureTab(tab);
    return {
      audio_number: tab.number,
      title: tab.title,
      text: tab.text,
      voice_mode: tab.voice_mode,
      voice_filename: tab.voice_mode === 'generated' ? tab.generated_voice_filename : tab.voice_filename,
      options: optionsFor(tab),
    };
  }

  function validateTab(tab) {
    captureTab(tab);
    if (!tab.text.trim()) return `Audio ${tab.number} has no script.`;
    const selectedVoice = tab.voice_mode === 'generated' ? tab.generated_voice_filename : tab.voice_filename;
    if (!selectedVoice) {
      return `Audio ${tab.number} needs a selected voice.`;
    }
    return '';
  }

  async function generateOne(tab) {
    const errorMessage = validateTab(tab);
    if (errorMessage) {
      toast(errorMessage, 'error');
      return;
    }
    const button = tab.panel.querySelector('[data-action="generate"]');
    button.disabled = true;
    try {
      const job = await api('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jobPayload(tab)),
      });
      tab.job_id = job.id;
      state.jobs.set(job.id, job);
      saveTabs();
      renderActive();
      openMonitor();
      schedulePoll(150);
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  async function generateAll() {
    state.tabs.forEach(captureTab);
    const ready = state.tabs.filter(tab => tab.text.trim());
    if (!ready.length) {
      toast('Add a script to at least one Audio tab.', 'error');
      return;
    }
    for (const tab of ready) {
      const message = validateTab(tab);
      if (message) {
        toast(message, 'error');
        state.activeId = tab.id;
        renderActive();
        return;
      }
    }

    const button = $('#generate-all');
    button.disabled = true;
    button.textContent = 'Adding to queue...';
    try {
      const jobs = await api('/api/jobs/generate-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobs: ready.map(jobPayload) }),
      });
      jobs.forEach((job, index) => {
        const tab = ready[index];
        tab.job_id = job.id;
        state.jobs.set(job.id, job);
      });
      saveTabs();
      renderActive();
      openMonitor();
      schedulePoll(150);
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
      button.textContent = 'Generate All';
    }
  }

  async function removeAll() {
    const active = [...state.jobs.values()].some(job => ['queued', 'running'].includes(job.status));
    if (active) {
      toast('Wait for all queued audio jobs to finish before using Remove All.', 'error');
      return;
    }
    if (!confirm('Remove all titles, scripts, completed jobs and generated audio files? Your saved voice files will remain.')) return;
    const button = $('#remove-all');
    button.disabled = true;
    button.textContent = 'Removing...';
    try {
      await api('/api/jobs', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delete_files: true }),
      });
      state.jobs.clear();
      state.tabs = [defaultTab(1)];
      state.activeId = state.tabs[0].id;
      localStorage.removeItem(STORAGE_KEY);
      buildTabs();
      renderQueue();
      updateFloating();
      toast('All audio workspaces are ready for new scripts.', 'success');
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
      button.textContent = 'Remove All';
    }
  }

  function setModelState(status) {
    const modelState = $('#model-state');
    const label = $('#model-status');
    const loadButton = $('#load-model');
    const select = $('#active-model');
    const info = state.initial.models.find(model => model.id === (status.model_name || select.value));
    $('#model-badge').textContent = info?.badge || 'Model';

    if (status.loading) {
      modelState.dataset.state = 'loading';
      label.textContent = `Loading ${info?.name || select.value}...`;
      loadButton.disabled = true;
      return;
    }
    if (status.error) {
      modelState.dataset.state = 'error';
      label.textContent = 'Model load failed';
      modelState.title = status.error;
      loadButton.disabled = false;
      loadButton.textContent = 'Retry Load';
      return;
    }
    if (status.model_name) {
      modelState.dataset.state = 'ready';
      label.textContent = `${info?.name || status.model_name} loaded on ${status.device}`;
      select.value = status.model_name;
      loadButton.disabled = false;
      loadButton.textContent = 'Reload Model';
      return;
    }
    modelState.dataset.state = 'loading';
    label.textContent = `Model not loaded • ${status.device}`;
    loadButton.disabled = false;
    loadButton.textContent = 'Load Model';
  }

  async function loadModel() {
    const button = $('#load-model');
    button.disabled = true;
    setModelState({ loading: true, model_name: $('#active-model').value });
    try {
      const status = await api('/api/model/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: $('#active-model').value }),
      });
      setModelState(status);
      toast('Model loaded successfully.', 'success');
    } catch (error) {
      setModelState({ loading: false, model_name: null, device: state.initial.engine.device, error: error.message });
      toast(error.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  async function refreshModelStatus() {
    try {
      const status = await api('/api/model-info');
      state.initial.engine = status;
      setModelState(status);
      if (status.loading) {
        state.modelPollTimer = setTimeout(refreshModelStatus, 900);
      }
    } catch (_) {
      state.modelPollTimer = setTimeout(refreshModelStatus, 1800);
    }
  }

  async function refreshJobs() {
    try {
      const [jobs, videoJobs] = await Promise.all([api('/api/jobs'), api('/api/video/jobs')]);
      state.jobs = new Map(jobs.map(job => [job.id, job]));
      state.videoJobs = new Map(videoJobs.map(job => [job.id, job]));
      state.tabs.forEach(tab => {
        const job = tab.job_id ? state.jobs.get(tab.job_id) : null;
        if (job?.status === 'completed') showGenerated(tab, job);
      });
      renderVideoAudioOptions();
      renderCurrentVideoJob();
      renderVideoHistory();
      renderActive();
      renderQueue();
      updateFloating();
      const active = jobs.some(job => ['queued', 'running'].includes(job.status))
        || videoJobs.some(job => ['queued', 'running'].includes(job.status));
      schedulePoll(active ? 750 : 3000);
    } catch (error) {
      schedulePoll(3000);
    }
  }

  function schedulePoll(delay) {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(refreshJobs, delay);
  }

  function monitorJobList() {
    return [...state.jobs.values()]
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      .slice(0, 20);
  }

  function createActionButton(label, handler, primary = false) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `button ${primary ? 'button-primary' : 'button-secondary'}`;
    button.textContent = label;
    button.addEventListener('click', handler);
    return button;
  }

  function renderQueue() {
    const list = $('#queue-list');
    list.innerHTML = '';
    const jobs = monitorJobList();
    if (!jobs.length) {
      const empty = document.createElement('div');
      empty.className = 'queue-empty';
      empty.textContent = 'No audio jobs yet. Prepare an Audio tab and click Generate.';
      list.append(empty);
      return;
    }

    jobs.forEach(job => {
      const card = document.createElement('article');
      card.className = 'queue-card';

      const head = document.createElement('div');
      head.className = 'queue-card-head';
      const titleBox = document.createElement('div');
      const audioLabel = document.createElement('div');
      audioLabel.className = 'queue-audio-label';
      audioLabel.textContent = `Audio ${job.audio_number}`;
      const title = document.createElement('div');
      title.className = 'queue-title';
      title.textContent = job.title || `Audio ${job.audio_number}`;
      titleBox.append(audioLabel, title);
      const status = document.createElement('span');
      status.className = `queue-status ${job.status}`;
      status.textContent = job.status;
      head.append(titleBox, status);

      const meta = document.createElement('div');
      meta.className = 'queue-meta';
      const wordValue = job.display_words ?? job.completed_words ?? 0;
      meta.innerHTML = `
        <span>Words: <strong>${wordValue} of ${job.total_words || 0}</strong></span>
        <span>Progress: <strong>${Number(job.percent || 0).toFixed(1)}%</strong></span>
        <span>Remaining: <strong>${Number(job.remaining_percent ?? 100).toFixed(1)}%</strong></span>
        <span>ETA: <strong>${humanDuration(job.eta_seconds)}</strong></span>
        <span>Elapsed: <strong>${humanDuration(job.elapsed_seconds || 0)}</strong></span>
        <span>Audio length: <strong>${humanDuration(job.actual_audio_seconds ?? job.estimated_audio_seconds)}</strong></span>
      `;

      const track = document.createElement('div');
      track.className = 'progress-track';
      const fill = document.createElement('div');
      fill.className = 'progress-fill';
      fill.style.width = `${Math.max(0, Math.min(100, Number(job.percent || 0)))}%`;
      track.append(fill);

      const stage = document.createElement('div');
      stage.className = 'queue-meta';
      stage.innerHTML = `<span>${safeText(job.stage || '')}${job.queue_position ? ` • Queue position ${job.queue_position}` : ''}</span>`;

      const actions = document.createElement('div');
      actions.className = 'queue-actions';
      if (job.status === 'completed') {
        actions.append(
          createActionButton('Preview audio', () => previewMonitor(job)),
          createActionButton(`Open Audio ${job.audio_number}`, () => openJobTab(job)),
        );
        const download = document.createElement('a');
        download.className = 'button button-secondary';
        download.textContent = 'Download WAV';
        download.href = `/api/jobs/${job.id}/audio?download=true`;
        download.download = '';
        actions.append(download);
      } else if (!['failed', 'cancelled', 'interrupted'].includes(job.status)) {
        actions.append(createActionButton('Cancel', () => cancelJob(job.id)));
      }

      card.append(head, meta, track, stage, actions);
      if (job.error) {
        const error = document.createElement('div');
        error.className = 'queue-error';
        error.textContent = job.error;
        card.append(error);
      }
      list.append(card);
    });
  }

  function previewMonitor(job) {
    const box = $('#monitor-preview-box');
    const player = $('#monitor-player');
    box.classList.remove('hidden');
    player.src = `/api/jobs/${job.id}/audio`;
    player.play().catch(() => toast('The browser could not play this audio preview.', 'error'));
  }

  function openJobTab(job) {
    const tab = findTabByJob(job.id);
    if (!tab) {
      toast('This completed job is not attached to a current Audio tab. Use its Preview or Download button.');
      return;
    }
    state.activeId = tab.id;
    closeMonitor();
    renderActive();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  async function cancelJob(jobId) {
    try {
      await api(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      schedulePoll(100);
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  function openMonitor() {
    state.monitorOpen = true;
    state.monitorMinimised = false;
    $('#progress-modal').classList.remove('hidden');
    $('#floating-progress').classList.add('hidden');
    renderQueue();
  }

  function closeMonitor() {
    state.monitorOpen = false;
    $('#progress-modal').classList.add('hidden');
  }

  function minimiseMonitor() {
    state.monitorMinimised = true;
    closeMonitor();
    updateFloating();
  }

  function updateFloating() {
    const jobs = [...state.jobs.values()];
    const active = jobs.find(job => job.status === 'running') || jobs.find(job => job.status === 'queued');
    const button = $('#floating-progress');
    if (!state.monitorMinimised || !active) {
      button.classList.add('hidden');
      return;
    }
    button.classList.remove('hidden');
    button.innerHTML = `
      <strong>Audio ${active.audio_number} • ${active.status}</strong>
      <span>${Number(active.percent || 0).toFixed(1)}% generated • Words ${active.display_words || 0} of ${active.total_words || 0}</span>
      <span>ETA ${humanDuration(active.eta_seconds)}</span>
    `;
  }

  async function showGenerated(tab, job) {
    if (!tab.panel || tab.panel.dataset.generatedJob === job.id) return;
    const panel = tab.panel;
    const section = panel.querySelector('[data-role="generated"]');
    section.classList.remove('hidden');
    panel.dataset.generatedJob = job.id;
    panel.querySelector('[data-role="generated-title"]').textContent = tab.title || `Audio ${tab.number}`;

    const audioUrl = `/api/jobs/${job.id}/audio`;
    const player = panel.querySelector('[data-role="main-player"]');
    player.src = audioUrl;
    wirePlayback(tab);
    const download = panel.querySelector('[data-role="download-original"]');
    download.href = `${audioUrl}?download=true`;
    download.download = '';
    panel.querySelector('[data-role="generation-meta"]').textContent = `Generation time ${humanDuration(job.elapsed_seconds)} • Duration ${formatTime(job.actual_audio_seconds || 0)}`;

    if (!tab.waveform || tab.waveform.jobId !== job.id) {
      await loadWaveform(tab, job.id);
    }
  }

  async function loadWaveform(tab, jobId) {
    const panel = tab.panel;
    const status = panel.querySelector('[data-role="wave-status"]');
    status.textContent = 'Loading waveform...';
    try {
      const data = await api(`/api/jobs/${jobId}/waveform?points=6000`);
      tab.waveform = {
        jobId,
        mins: data.mins,
        maxs: data.maxs,
        duration: data.duration,
        zoom: 1,
        mode: 'end',
        selected: 0,
        playhead: 0,
        dragStartX: 0,
        dragStartScroll: 0,
      };
      field(panel, 'cut_start').value = tab.cut_start || '0:00';
      field(panel, 'cut_end').value = tab.cut_end || formatTime(data.duration, true);
      tab.cut_end = field(panel, 'cut_end').value;
      wireWave(tab);
      drawWave(tab);
      updateCutSummary(tab);
      status.textContent = 'Move the mouse to see time. Click to set Start or End. Zoom and drag left or right for a clearer view.';
      saveTabs();
    } catch (error) {
      status.textContent = 'Could not load waveform. The playback controls and Download WAV still work.';
    }
  }

  function wirePlayback(tab) {
    const panel = tab.panel;
    const player = panel.querySelector('[data-role="main-player"]');
    if (player.dataset.wired === 'true') return;
    player.dataset.wired = 'true';
    const button = panel.querySelector('[data-action="toggle-playback"]');
    const progress = panel.querySelector('[data-role="playback-progress"]');
    const time = panel.querySelector('[data-role="playback-time"]');

    const update = () => {
      const duration = Number.isFinite(player.duration) ? player.duration : (tab.waveform?.duration || 0);
      const current = Number.isFinite(player.currentTime) ? player.currentTime : 0;
      time.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
      progress.value = duration > 0 ? Math.round((current / duration) * 1000) : 0;
      if (tab.waveform) {
        tab.waveform.playhead = current;
        drawWave(tab);
      }
    };
    player.addEventListener('loadedmetadata', update);
    player.addEventListener('timeupdate', update);
    player.addEventListener('play', () => { button.textContent = 'Pause'; update(); });
    player.addEventListener('pause', () => { button.textContent = 'Play'; update(); });
    player.addEventListener('ended', () => { button.textContent = 'Play'; update(); });
    update();
  }

  function togglePlayback(tab) {
    const player = tab.panel.querySelector('[data-role="main-player"]');
    if (!player.src) return;
    if (player.paused) player.play().catch(() => toast('The browser could not play this audio.', 'error'));
    else player.pause();
  }

  function seekPlayback(tab, fraction) {
    const player = tab.panel.querySelector('[data-role="main-player"]');
    const duration = Number.isFinite(player.duration) ? player.duration : tab.waveform?.duration;
    if (!duration) return;
    player.currentTime = Math.max(0, Math.min(duration, duration * fraction));
  }

  function waveTimeFromEvent(tab, event) {
    const canvas = tab.panel.querySelector('[data-role="wave-canvas"]');
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    return (x / Math.max(rect.width, 1)) * tab.waveform.duration;
  }

  function wireWave(tab) {
    const panel = tab.panel;
    const scroll = panel.querySelector('[data-role="wave-scroll"]');
    const canvas = panel.querySelector('[data-role="wave-canvas"]');
    const tooltip = panel.querySelector('[data-role="wave-tooltip"]');

    canvas.onmousemove = event => {
      if (!tab.waveform) return;
      if (tab.waveform.mode === 'pan' && event.buttons === 1) {
        const delta = event.clientX - tab.waveform.dragStartX;
        scroll.scrollLeft = tab.waveform.dragStartScroll - delta;
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const time = waveTimeFromEvent(tab, event);
      panel.querySelector('[data-role="mouse-time"]').textContent = formatTime(time, true);
      tooltip.textContent = formatTime(time, true);
      tooltip.style.left = `${x}px`;
      tooltip.classList.remove('hidden');
    };
    canvas.onmouseleave = () => tooltip.classList.add('hidden');
    canvas.onmousedown = event => {
      if (!tab.waveform || tab.waveform.mode !== 'pan') return;
      tab.waveform.dragStartX = event.clientX;
      tab.waveform.dragStartScroll = scroll.scrollLeft;
      scroll.classList.add('dragging');
    };
    window.addEventListener('mouseup', () => scroll.classList.remove('dragging'));
    canvas.onclick = event => {
      if (!tab.waveform || tab.waveform.mode === 'pan') return;
      const time = waveTimeFromEvent(tab, event);
      tab.waveform.selected = time;
      panel.querySelector('[data-role="selected-time"]').textContent = formatTime(time, true);
      const target = tab.waveform.mode === 'start' ? 'cut_start' : 'cut_end';
      field(panel, target).value = formatTime(time, true);
      captureTab(tab);
      updateCutSummary(tab);
      saveTabs();
    };
    scroll.onwheel = event => {
      if (Math.abs(event.deltaY) >= Math.abs(event.deltaX)) {
        event.preventDefault();
        scroll.scrollLeft += event.deltaY;
      }
    };
  }

  function drawWave(tab) {
    const wave = tab?.waveform;
    if (!wave || !tab.panel) return;
    const panel = tab.panel;
    const canvas = panel.querySelector('[data-role="wave-canvas"]');
    const scroll = panel.querySelector('[data-role="wave-scroll"]');
    const visibleWidth = Math.max(scroll.clientWidth, 600);
    const width = Math.max(visibleWidth, Math.round(visibleWidth * wave.zoom));
    const height = 176;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext('2d');
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);

    const css = getComputedStyle(document.documentElement);
    const primary = css.getPropertyValue('--primary').trim() || '#5f52e8';
    const border = css.getPropertyValue('--border').trim() || '#dce3ef';
    const success = css.getPropertyValue('--success').trim() || '#16a36a';
    const danger = css.getPropertyValue('--danger').trim() || '#df5362';
    const middle = height / 2;

    context.strokeStyle = border;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(0, middle);
    context.lineTo(width, middle);
    context.stroke();

    context.strokeStyle = primary;
    context.lineWidth = 1;
    context.beginPath();
    const length = wave.mins.length;
    const step = width / Math.max(length, 1);
    for (let index = 0; index < length; index += 1) {
      const x = index * step;
      const top = middle - wave.maxs[index] * (middle - 13);
      const bottom = middle - wave.mins[index] * (middle - 13);
      context.moveTo(x, top);
      context.lineTo(x, bottom);
    }
    context.stroke();

    const start = parseTime(field(panel, 'cut_start').value) || 0;
    const parsedEnd = parseTime(field(panel, 'cut_end').value);
    const end = Number.isFinite(parsedEnd) ? parsedEnd : wave.duration;
    const startX = (start / wave.duration) * width;
    const endX = (end / wave.duration) * width;
    context.fillStyle = 'rgba(95,82,232,.11)';
    context.fillRect(startX, 0, Math.max(0, endX - startX), height);

    context.strokeStyle = success;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(startX, 0);
    context.lineTo(startX, height);
    context.stroke();

    context.strokeStyle = danger;
    context.beginPath();
    context.moveTo(endX, 0);
    context.lineTo(endX, height);
    context.stroke();

    const playhead = Math.max(0, Math.min(wave.duration, Number(wave.playhead || 0)));
    const playX = (playhead / Math.max(wave.duration, 0.001)) * width;
    context.strokeStyle = '#111827';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(playX, 0);
    context.lineTo(playX, height);
    context.stroke();
    context.fillStyle = '#111827';
    context.beginPath();
    context.arc(playX, 7, 5, 0, Math.PI * 2);
    context.fill();
  }

  function setClickMode(tab, mode) {
    if (!tab.waveform) return;
    tab.waveform.mode = mode;
    const panel = tab.panel;
    $$('[data-action="set-start-mode"], [data-action="set-end-mode"], [data-action="pan-wave"]', panel)
      .forEach(button => button.classList.remove('active'));
    const selector = mode === 'pan' ? '[data-action="pan-wave"]' : `[data-action="set-${mode}-mode"]`;
    panel.querySelector(selector).classList.add('active');
    panel.querySelector('[data-role="wave-scroll"]').classList.toggle('drag-mode', mode === 'pan');
  }

  function zoomWave(tab, factor) {
    if (!tab.waveform) return;
    tab.waveform.zoom = Math.max(1, Math.min(18, tab.waveform.zoom * factor));
    drawWave(tab);
  }

  function updateCutSummary(tab) {
    if (!tab.waveform || !tab.panel) return;
    const panel = tab.panel;
    const start = parseTime(field(panel, 'cut_start').value);
    const end = parseTime(field(panel, 'cut_end').value);
    const valid = Number.isFinite(start)
      && Number.isFinite(end)
      && start >= 0
      && end > start
      && end <= tab.waveform.duration + 0.25;
    panel.querySelector('[data-role="selected-duration"]').textContent = valid
      ? `Selected: ${formatTime(end - start, true)}`
      : 'Selected: invalid range';
    panel.querySelector('[data-role="removed-duration"]').textContent = valid
      ? `Removed: ${formatTime(Math.max(0, tab.waveform.duration - (end - start)), true)}`
      : '';
    drawWave(tab);
  }

  function previewSelected(tab) {
    const panel = tab.panel;
    const player = panel.querySelector('[data-role="main-player"]');
    const start = parseTime(field(panel, 'cut_start').value);
    const end = parseTime(field(panel, 'cut_end').value);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      toast('Enter a valid Start and End time.', 'error');
      return;
    }
    player.currentTime = start;
    player.play().catch(() => toast('The browser could not play this selection.', 'error'));
    const stopAtEnd = () => {
      if (player.currentTime >= end) {
        player.pause();
        player.removeEventListener('timeupdate', stopAtEnd);
      }
    };
    player.addEventListener('timeupdate', stopAtEnd);
  }

  async function cutDownload(tab, prefix, start, end) {
    if (!tab.job_id || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      toast('Enter a valid audio range.', 'error');
      return;
    }
    try {
      const data = await api(`/api/jobs/${tab.job_id}/cut`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_seconds: start,
          end_seconds: end,
          filename_prefix: prefix,
        }),
      });
      const anchor = document.createElement('a');
      anchor.href = data.url;
      anchor.download = data.filename;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      toast(`${data.filename} is ready.`, 'success');
    } catch (error) {
      toast(error.message, 'error');
    }
  }


  function videoField(name) {
    return state.video.panel?.querySelector(`[data-field="${name}"]`);
  }

  function activeVideoJob() {
    if (state.video.active_job_id && state.videoJobs.has(state.video.active_job_id)) {
      return state.videoJobs.get(state.video.active_job_id);
    }
    return [...state.videoJobs.values()]
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0] || null;
  }

  function captureVideoState() {
    const panel = state.video.panel;
    if (!panel) return;
    const values = {
      title: 'video_title',
      audio_job_id: 'video_audio_job_id',
      engine: 'video_engine',
      render_mode: 'video_render_mode',
      segment_seconds: 'video_segment_seconds',
      aspect_ratio: 'video_aspect_ratio',
      resolution: 'video_resolution',
      fps: 'video_fps',
      image_fit: 'video_image_fit',
      quality: 'video_quality',
      framing: 'video_framing',
    };
    Object.entries(values).forEach(([key, fieldName]) => {
      const input = videoField(fieldName);
      if (!input) return;
      state.video[key] = ['segment_seconds', 'fps'].includes(key) ? Number(input.value) : input.value;
    });
    state.video.consent = Boolean(videoField('video_consent')?.checked);
  }

  function fillVideoPanel() {
    const panel = state.video.panel;
    if (!panel) return;
    const values = {
      video_title: state.video.title,
      video_engine: state.video.engine,
      video_render_mode: state.video.render_mode,
      video_segment_seconds: state.video.segment_seconds,
      video_aspect_ratio: state.video.aspect_ratio,
      video_resolution: state.video.resolution,
      video_fps: state.video.fps,
      video_image_fit: state.video.image_fit,
      video_quality: state.video.quality,
      video_framing: state.video.framing,
    };
    Object.entries(values).forEach(([name, value]) => {
      const input = videoField(name);
      if (input && value !== undefined) input.value = value;
    });
    if (videoField('video_consent')) videoField('video_consent').checked = Boolean(state.video.consent);
    renderAvatarAsset();
    setVideoAudioMode(state.video.audio_mode, false);
    renderVideoAudioOptions();
    renderVideoEngineStatus(state.initial?.avatar);
    renderCurrentVideoJob();
  }

  function renderAvatarAsset() {
    const panel = state.video.panel;
    if (!panel) return;
    const preview = panel.querySelector('[data-role="avatar-preview"]');
    const placeholder = panel.querySelector('[data-role="avatar-placeholder"]');
    const filename = panel.querySelector('[data-role="avatar-filename"]');
    const remove = panel.querySelector('[data-action="remove-avatar"]');
    if (state.video.avatar_filename && state.video.avatar_preview_url) {
      preview.src = `${state.video.avatar_preview_url}?v=${Date.now()}`;
      preview.classList.remove('hidden');
      placeholder.classList.add('hidden');
      filename.textContent = state.video.avatar_filename;
      remove.classList.remove('hidden');
    } else {
      preview.removeAttribute('src');
      preview.classList.add('hidden');
      placeholder.classList.remove('hidden');
      filename.textContent = 'No avatar selected';
      remove.classList.add('hidden');
    }
  }

  function setVideoAudioMode(mode, persist = true) {
    state.video.audio_mode = mode;
    const panel = state.video.panel;
    if (!panel) return;
    $$('[data-video-audio-mode]', panel).forEach(button => {
      button.classList.toggle('active', button.dataset.videoAudioMode === mode);
    });
    panel.querySelector('[data-role="video-generated-audio-tools"]').classList.toggle('hidden', mode !== 'audio_job');
    panel.querySelector('[data-role="video-upload-audio-tools"]').classList.toggle('hidden', mode !== 'upload');
    if (persist) saveTabs();
  }

  function completedAudioJobs() {
    return [...state.jobs.values()]
      .filter(job => job.status === 'completed' && job.output_filename)
      .sort((a, b) => (a.audio_number || 0) - (b.audio_number || 0));
  }

  function renderVideoAudioOptions() {
    const select = videoField('video_audio_job_id');
    if (!select) return;
    const selected = state.video.audio_job_id || select.value;
    select.innerHTML = '';
    const jobs = completedAudioJobs();
    if (!jobs.length) {
      select.add(new Option('No completed audio yet', ''));
      select.disabled = true;
    } else {
      select.disabled = false;
      jobs.forEach(job => {
        const title = job.title || job.output_filename;
        select.add(new Option(`Audio ${job.audio_number} · ${title}`, job.id));
      });
      select.value = jobs.some(job => job.id === selected) ? selected : jobs[0].id;
      state.video.audio_job_id = select.value;
    }
    const help = state.video.panel?.querySelector('[data-role="video-audio-help"]');
    if (help) help.textContent = jobs.length
      ? `${jobs.length} completed audio ${jobs.length === 1 ? 'track is' : 'tracks are'} ready.`
      : 'Complete an Audio workspace first, then it appears here automatically.';
  }

  function renderUploadedVideoAudio() {
    const panel = state.video.panel;
    if (!panel) return;
    const player = panel.querySelector('[data-role="video-upload-audio-player"]');
    const name = panel.querySelector('[data-role="video-audio-filename"]');
    const remove = panel.querySelector('[data-action="remove-video-audio"]');
    if (state.video.audio_filename && state.video.audio_preview_url) {
      player.src = state.video.audio_preview_url;
      player.classList.remove('hidden');
      name.textContent = state.video.audio_filename;
      remove.classList.remove('hidden');
    } else {
      player.pause();
      player.removeAttribute('src');
      player.classList.add('hidden');
      name.textContent = 'No audio uploaded';
      remove.classList.add('hidden');
    }
  }

  function renderVideoEngineStatus(status) {
    const box = state.video.panel?.querySelector('[data-role="avatar-engine-state"]');
    if (!box) return;
    const strong = box.querySelector('strong');
    const small = box.querySelector('small');
    const gpu = status?.gpu;
    if (status?.ready) {
      box.dataset.state = 'ready';
      strong.textContent = 'Avatar engine ready';
      small.textContent = `${gpu?.name || 'NVIDIA GPU'} · ${status.recommended === 'ditto_trt' ? 'TensorRT preferred' : 'PyTorch ready'}`;
    } else {
      box.dataset.state = 'error';
      strong.textContent = 'Avatar engine not installed';
      small.textContent = status?.message || 'Run the v0.9.0 A100 installation cells.';
    }
  }

  async function uploadAvatarImage(file) {
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const status = state.video.panel.querySelector('[data-role="video-action-status"]');
    status.textContent = 'Uploading and validating avatar image...';
    try {
      const data = await api('/api/video/avatar-upload', { method: 'POST', body: form });
      state.video.avatar_filename = data.filename;
      state.video.avatar_preview_url = data.preview_url;
      renderAvatarAsset();
      saveTabs();
      toast('Avatar image uploaded.', 'success');
      status.textContent = 'Avatar ready. Choose audio and video settings.';
    } catch (error) {
      toast(error.message, 'error');
      status.textContent = error.message;
    }
  }

  async function uploadVideoAudio(file) {
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const status = state.video.panel.querySelector('[data-role="video-action-status"]');
    status.textContent = 'Uploading video audio...';
    try {
      const data = await api('/api/video/audio-upload', { method: 'POST', body: form });
      state.video.audio_filename = data.filename;
      state.video.audio_preview_url = data.preview_url;
      renderUploadedVideoAudio();
      saveTabs();
      toast('Video audio uploaded.', 'success');
      status.textContent = 'Uploaded audio is ready.';
    } catch (error) {
      toast(error.message, 'error');
      status.textContent = error.message;
    }
  }

  function videoPayload() {
    captureVideoState();
    return {
      title: state.video.title || 'Avatar Video',
      avatar_filename: state.video.avatar_filename,
      audio_source: state.video.audio_mode,
      audio_job_id: state.video.audio_mode === 'audio_job' ? state.video.audio_job_id : null,
      audio_filename: state.video.audio_mode === 'upload' ? state.video.audio_filename : null,
      engine: state.video.engine,
      render_mode: state.video.render_mode,
      segment_seconds: Number(state.video.segment_seconds),
      aspect_ratio: state.video.aspect_ratio,
      resolution: state.video.resolution,
      fps: Number(state.video.fps),
      framing: state.video.framing,
      image_fit: state.video.image_fit,
      quality: state.video.quality,
      consent: Boolean(state.video.consent),
    };
  }

  function validateVideoPayload(payload) {
    if (!payload.avatar_filename) throw new Error('Upload an avatar image first.');
    if (payload.audio_source === 'audio_job' && !payload.audio_job_id) throw new Error('Select a completed audio track.');
    if (payload.audio_source === 'upload' && !payload.audio_filename) throw new Error('Upload an audio file.');
    if (!payload.consent) throw new Error('Confirm that you own or have permission to animate the avatar image.');
    if (!state.initial?.avatar?.ready) throw new Error('Avatar engine is not installed. Run the v0.9.0 A100 notebook installation cells.');
  }

  async function generateVideo() {
    const button = state.video.panel.querySelector('[data-action="generate-video"]');
    const status = state.video.panel.querySelector('[data-role="video-action-status"]');
    try {
      const payload = videoPayload();
      validateVideoPayload(payload);
      button.disabled = true;
      status.textContent = 'Creating the avatar video job...';
      const job = await api('/api/video/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      state.videoJobs.set(job.id, job);
      state.video.active_job_id = job.id;
      saveTabs();
      renderCurrentVideoJob();
      renderVideoHistory();
      renderActive();
      toast('Avatar video added to the GPU queue.', 'success');
      schedulePoll(250);
    } catch (error) {
      toast(error.message, 'error');
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  function videoStatusLabel(status) {
    return ({ queued: 'Queued', running: 'Generating', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted' })[status] || status;
  }

  function renderCurrentVideoJob() {
    const panel = state.video.panel;
    if (!panel) return;
    const job = activeVideoJob();
    const progressCard = panel.querySelector('[data-role="video-progress-card"]');
    const resultCard = panel.querySelector('[data-role="video-result-card"]');
    if (!job) {
      progressCard.classList.add('hidden');
      resultCard.classList.add('hidden');
      return;
    }
    state.video.active_job_id = job.id;
    if (['queued', 'running'].includes(job.status)) {
      progressCard.classList.remove('hidden');
      resultCard.classList.add('hidden');
      const pill = panel.querySelector('[data-role="video-job-status"]');
      pill.textContent = videoStatusLabel(job.status);
      pill.dataset.status = job.status;
      panel.querySelector('[data-role="video-job-title"]').textContent = job.title || 'Avatar video';
      panel.querySelector('[data-role="video-job-stage"]').textContent = job.stage || 'Working...';
      panel.querySelector('[data-role="video-progress-bar"]').style.width = `${Math.max(0, Math.min(100, job.percent || 0))}%`;
      panel.querySelector('[data-role="video-progress-percent"]').textContent = `${Math.round(job.percent || 0)}%`;
      panel.querySelector('[data-role="video-progress-time"]').textContent = `Elapsed ${humanDuration(job.elapsed_seconds || 0)}`;
      panel.querySelector('[data-role="video-progress-eta"]').textContent = `ETA ${humanDuration(job.eta_seconds)}`;
      panel.querySelector('[data-action="cancel-video"]').disabled = false;
      return;
    }
    progressCard.classList.toggle('hidden', !['failed', 'cancelled', 'interrupted'].includes(job.status));
    if (!progressCard.classList.contains('hidden')) {
      panel.querySelector('[data-role="video-job-status"]').textContent = videoStatusLabel(job.status);
      panel.querySelector('[data-role="video-job-stage"]').textContent = job.error || job.stage || videoStatusLabel(job.status);
      panel.querySelector('[data-role="video-progress-bar"]').style.width = `${Math.round(job.percent || 0)}%`;
      panel.querySelector('[data-action="cancel-video"]').disabled = true;
    }
    if (job.status === 'completed') {
      resultCard.classList.remove('hidden');
      panel.querySelector('[data-role="video-result-title"]').textContent = job.title || job.output_filename;
      panel.querySelector('[data-role="video-result-summary"]').textContent = `${job.backend_label || job.backend} · ${job.segments || 1} section${job.segments === 1 ? '' : 's'} · ${humanDuration(job.duration)}`;
      const url = `/api/video/jobs/${job.id}/file`;
      panel.querySelector('[data-role="video-result-player"]').src = url;
      const download = panel.querySelector('[data-role="download-video"]');
      download.href = `${url}?download=true`;
      download.download = job.output_filename || 'avatar-video.mp4';
      const report = job.quality_report || {};
      const qualityBox = panel.querySelector('[data-role="video-quality-report"]');
      qualityBox.dataset.state = report.passed ? 'passed' : 'review';
      qualityBox.innerHTML = '';
      const items = [
        ['Technical status', report.passed ? 'Passed' : 'Review'],
        ['Audio / video drift', `${Number(report.duration_drift || 0).toFixed(2)} sec`],
        ['Long freeze time', `${Number(report.freeze_seconds || 0).toFixed(1)} sec`],
        ['File size', `${((job.output_size || 0) / 1024 / 1024).toFixed(1)} MB`],
      ];
      items.forEach(([label, value]) => {
        const item = document.createElement('div');
        const small = document.createElement('small');
        const strong = document.createElement('strong');
        small.textContent = label;
        strong.textContent = value;
        item.append(small, strong);
        qualityBox.append(item);
      });
    } else {
      resultCard.classList.add('hidden');
    }
  }

  async function cancelVideoJob() {
    const job = activeVideoJob();
    if (!job || !['queued', 'running'].includes(job.status)) return;
    try {
      const updated = await api(`/api/video/jobs/${job.id}/cancel`, { method: 'POST' });
      state.videoJobs.set(updated.id, updated);
      renderCurrentVideoJob();
      toast('Video cancellation requested.', 'success');
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  function renderVideoHistory() {
    const list = state.video.panel?.querySelector('[data-role="video-history-list"]');
    if (!list) return;
    const jobs = [...state.videoJobs.values()].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    list.innerHTML = '';
    if (!jobs.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-video-history';
      empty.textContent = 'No avatar videos yet.';
      list.append(empty);
      return;
    }
    jobs.slice(0, 20).forEach(job => {
      const card = document.createElement('article');
      card.className = 'video-history-item';
      card.dataset.status = job.status;
      const body = document.createElement('div');
      const title = document.createElement('strong');
      const meta = document.createElement('span');
      title.textContent = job.title || 'Avatar video';
      meta.textContent = `${videoStatusLabel(job.status)} · ${job.audio_label || 'audio'}${job.duration ? ` · ${humanDuration(job.duration)}` : ''}`;
      body.append(title, meta);
      const actions = document.createElement('div');
      if (job.status === 'completed') {
        const open = document.createElement('button');
        open.type = 'button';
        open.className = 'button button-secondary';
        open.textContent = 'Open';
        open.addEventListener('click', () => {
          state.video.active_job_id = job.id;
          renderCurrentVideoJob();
          state.video.panel.querySelector('[data-role="video-result-card"]').scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        actions.append(open);
      }
      if (!['queued', 'running'].includes(job.status)) {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'button button-danger-soft';
        remove.textContent = 'Remove';
        remove.addEventListener('click', async () => {
          if (!confirm('Remove this video job and its generated MP4?')) return;
          try {
            await api(`/api/video/jobs/${job.id}?delete_file=true`, { method: 'DELETE' });
            state.videoJobs.delete(job.id);
            if (state.video.active_job_id === job.id) state.video.active_job_id = null;
            renderVideoHistory();
            renderCurrentVideoJob();
            saveTabs();
          } catch (error) { toast(error.message, 'error'); }
        });
        actions.append(remove);
      }
      card.append(body, actions);
      list.append(card);
    });
  }

  async function clearVideoJobs() {
    if ([...state.videoJobs.values()].some(job => ['queued', 'running'].includes(job.status))) {
      toast('Cancel or finish the active video before clearing history.', 'error');
      return;
    }
    if (!confirm('Clear all video history and generated MP4 files?')) return;
    try {
      await api('/api/video/jobs', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ delete_files: true }) });
      state.videoJobs.clear();
      state.video.active_job_id = null;
      renderVideoHistory();
      renderCurrentVideoJob();
      saveTabs();
      toast('Video history cleared.', 'success');
    } catch (error) { toast(error.message, 'error'); }
  }

  async function refreshVideoStatus() {
    try {
      state.initial.avatar = await api('/api/video/status');
      renderVideoEngineStatus(state.initial.avatar);
    } catch (_) {
      // The shared polling loop will retry.
    }
  }

  function wireVideoPanel() {
    const panel = state.video.panel;
    if (!panel) return;
    const avatarInput = panel.querySelector('[data-role="avatar-upload"]');
    avatarInput.addEventListener('change', () => uploadAvatarImage(avatarInput.files?.[0]));
    panel.querySelector('[data-action="remove-avatar"]').addEventListener('click', () => {
      state.video.avatar_filename = '';
      state.video.avatar_preview_url = '';
      avatarInput.value = '';
      renderAvatarAsset();
      saveTabs();
    });
    $$('[data-video-audio-mode]', panel).forEach(button => button.addEventListener('click', () => setVideoAudioMode(button.dataset.videoAudioMode)));
    videoField('video_audio_job_id').addEventListener('change', event => { state.video.audio_job_id = event.target.value; saveTabs(); });
    const audioInput = panel.querySelector('[data-role="video-audio-upload"]');
    audioInput.addEventListener('change', () => uploadVideoAudio(audioInput.files?.[0]));
    panel.querySelector('[data-action="remove-video-audio"]').addEventListener('click', () => {
      state.video.audio_filename = '';
      state.video.audio_preview_url = '';
      audioInput.value = '';
      renderUploadedVideoAudio();
      saveTabs();
    });
    $$('[data-field^="video_"]', panel).forEach(input => input.addEventListener('change', () => { captureVideoState(); saveTabs(); }));
    panel.querySelector('[data-action="generate-video"]').addEventListener('click', generateVideo);
    panel.querySelector('[data-action="cancel-video"]').addEventListener('click', cancelVideoJob);
    panel.querySelector('[data-action="clear-video-jobs"]').addEventListener('click', clearVideoJobs);
    panel.querySelector('[data-action="refresh-video-jobs"]').addEventListener('click', async () => { await refreshJobs(); toast('Video history refreshed.', 'success'); });
    renderUploadedVideoAudio();
    fillVideoPanel();
  }

  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'dark') document.documentElement.classList.add('dark');
    const updateIcon = () => {
      $('#theme-toggle .theme-icon').textContent = document.documentElement.classList.contains('dark') ? '☾' : '☀';
    };
    updateIcon();
    $('#theme-toggle').addEventListener('click', () => {
      document.documentElement.classList.toggle('dark');
      localStorage.setItem(THEME_KEY, document.documentElement.classList.contains('dark') ? 'dark' : 'light');
      updateIcon();
      state.tabs.forEach(drawWave);
    });
  }

  async function init() {
    initTheme();
    try {
      state.initial = await api('/api/ui/initial-data');
    } catch (error) {
      $('#model-state').dataset.state = 'error';
      $('#model-status').textContent = 'Server unavailable';
      toast(error.message, 'error');
      return;
    }

    const modelSelect = $('#active-model');
    state.initial.models.forEach(model => modelSelect.add(new Option(model.name, model.id)));
    modelSelect.value = state.initial.active_model;
    modelSelect.addEventListener('change', () => {
      const info = state.initial.models.find(model => model.id === modelSelect.value);
      $('#model-badge').textContent = info?.badge || 'Model';
    });
    setModelState(state.initial.engine);

    restoreTabs();
    buildTabs();
    state.initial.jobs.forEach(job => state.jobs.set(job.id, job));
    (state.initial.video_jobs || []).forEach(job => state.videoJobs.set(job.id, job));
    renderVideoEngineStatus(state.initial.avatar);
    renderActive();
    renderQueue();

    $('#load-model').addEventListener('click', loadModel);
    $('#generate-all').addEventListener('click', generateAll);
    $('#remove-all').addEventListener('click', removeAll);
    $('#open-monitor').addEventListener('click', openMonitor);
    $('#close-monitor').addEventListener('click', closeMonitor);
    $('#minimise-monitor').addEventListener('click', minimiseMonitor);
    $('#floating-progress').addEventListener('click', openMonitor);
    $('#progress-modal').addEventListener('click', event => {
      if (event.target === $('#progress-modal')) closeMonitor();
    });

    if (state.initial.engine.loading) refreshModelStatus();
    refreshVideoStatus();
    schedulePoll(200);
  }

  window.addEventListener('resize', () => state.tabs.forEach(drawWave));
  window.addEventListener('DOMContentLoaded', init);
})();
