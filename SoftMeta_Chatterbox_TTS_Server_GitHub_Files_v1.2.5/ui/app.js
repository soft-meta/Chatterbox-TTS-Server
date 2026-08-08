(() => {
  'use strict';

  const MAX_TABS = 5;
  const STORAGE_KEY = 'softMetaChatterboxTabsV10';
  const THEME_KEY = 'softMetaChatterboxTheme';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const state = {
    initial: null,
    tabs: [],
    activeId: '',
    jobs: new Map(),
    pollTimer: null,
    modelPollTimer: null,
    monitorMinimised: false,
    monitorOpen: false,
    emotionTimers: new Map(),
    emotionRequestSerial: 0,
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
      emotion_summary: null,
      preset: defaults.preset || 'Motivational Speech',
      language: defaults.language || 'en',
      voice_mode: 'clone',
      voice_filename: '',
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
      chunk_words: defaults.chunk_words ?? 85,
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


  function shouldAutoEmotion(tab) {
    const model = $('#active-model')?.value || state.initial?.active_model || '';
    return model === 'chatterbox-turbo'
      && tab?.preset === 'Motivational Speech'
      && countWords(tab?.text || '') >= 25;
  }

  function emotionSummaryText(summary) {
    if (!summary) return 'Analyzing serious, natural expression...';
    const labels = Object.entries(summary.labels || {})
      .filter(([, count]) => Number(count) > 0)
      .map(([label, count]) => `${count} ${label}`);
    const headingText = summary.protected_headings
      ? `${summary.protected_headings} heading${summary.protected_headings === 1 ? '' : 's'} protected`
      : 'headings protected automatically';
    if (!summary.applied_count) {
      return `Calm narration kept • no extra emotion needed • ${headingText}`;
    }
    const details = labels.length ? `: ${labels.join(', ')}` : '';
    return `${summary.applied_count} subtle cue${summary.applied_count === 1 ? '' : 's'}${details} • ${headingText}`;
  }

  function renderEmotionStatus(tab, message = null) {
    const box = tab?.panel?.querySelector('[data-role="emotion-status"]');
    const summary = tab?.panel?.querySelector('[data-role="emotion-summary"]');
    if (!box || !summary) return;
    if (!shouldAutoEmotion(tab)) {
      box.classList.add('hidden');
      return;
    }
    box.classList.remove('hidden');
    summary.textContent = message || emotionSummaryText(tab.emotion_summary);
  }

  function scheduleEmotionAnalysis(tab, { immediate = false } = {}) {
    if (!tab) return;
    const existing = state.emotionTimers.get(tab.id);
    if (existing) clearTimeout(existing);
    if (!shouldAutoEmotion(tab)) {
      tab.emotion_summary = null;
      renderEmotionStatus(tab);
      return;
    }
    renderEmotionStatus(tab, 'Analyzing serious, natural expression...');
    const snapshot = tab.text;
    const requestId = ++state.emotionRequestSerial;
    const timer = setTimeout(async () => {
      try {
        const result = await api('/api/emotion/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: snapshot }),
        });
        if (requestId !== state.emotionRequestSerial && tab.text !== snapshot) return;
        if (tab.text !== snapshot) return;
        tab.emotion_summary = result;
        renderEmotionStatus(tab);
        saveTabs();
      } catch (error) {
        if (tab.text === snapshot) {
          renderEmotionStatus(tab, 'Auto Emotion will be applied safely during generation.');
        }
      } finally {
        state.emotionTimers.delete(tab.id);
      }
    }, immediate ? 40 : 850);
    state.emotionTimers.set(tab.id, timer);
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

  function saveTabs() {
    const tabs = state.tabs.map(({ panel, waveform, ...tab }) => tab);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ tabs, activeId: state.activeId }));
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
        state.activeId = state.tabs.some(tab => tab.id === saved.activeId) ? saved.activeId : state.tabs[0].id;
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
      const job = tab?.job_id ? state.jobs.get(tab.job_id) : null;
      if (job?.status) button.classList.add(`status-${job.status}`);
    });

    state.tabs.forEach(tab => {
      tab.panel.classList.toggle('hidden', tab.id !== state.activeId);
      if (tab.id === state.activeId) fillPanel(tab);
    });  }

  function updateSliderOutput(panel, name) {
    const input = field(panel, name);
    const output = panel.querySelector(`[data-output="${name}"]`);
    if (input && output) output.value = Number(input.value).toFixed(name === 'speed_factor' ? 2 : 2);
  }

  function fillPanel(tab) {
    const panel = tab.panel;
    const stringFields = ['title', 'text', 'preset', 'language', 'voice_filename', 'output_format', 'cut_start', 'cut_end'];
    const numberFields = ['temperature', 'exaggeration', 'cfg_weight', 'speed_factor', 'seed', 'chunk_words'];
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
    updatePresetChips(tab);
    renderEmotionStatus(tab);
    const job = tab.job_id ? state.jobs.get(tab.job_id) : null;
    if (job?.status === 'completed') showGenerated(tab, job);
  }

  function captureTab(tab) {
    if (!tab?.panel) return;
    const panel = tab.panel;
    ['title', 'text', 'preset', 'language', 'voice_filename', 'output_format', 'cut_start', 'cut_end'].forEach(name => {
      const element = field(panel, name);
      if (element) tab[name] = element.value;
    });
    ['temperature', 'exaggeration', 'cfg_weight', 'speed_factor', 'seed', 'chunk_words'].forEach(name => {
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
    const activeModel = $('#active-model')?.value || state.initial?.active_model || '';
    if (activeModel === 'chatterbox-turbo' && preset.turbo_overrides) {
      for (const [key, value] of Object.entries(preset.turbo_overrides)) {
        if (key in tab) tab[key] = value;
      }
    }
    fillPanel(tab);
    saveTabs();
    scheduleEmotionAnalysis(tab, { immediate: true });
    toast(`${name} preset applied.`, 'success');
  }

  function setVoiceMode(tab, mode) {
    tab.voice_mode = mode;
    updateVoiceControls(tab);
    saveTabs();
  }

  function updateVoiceControls(tab) {
    const panel = tab.panel;
    const mode = tab.voice_mode === 'predefined' ? 'predefined' : 'clone';
    tab.voice_mode = mode;
    $$('[data-voice-mode]', panel).forEach(button => {
      button.classList.toggle('active', button.dataset.voiceMode === mode);
    });

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
      captureTab(tab);
      saveTabs();
      if (event.target.matches('[data-field="text"]')) scheduleEmotionAnalysis(tab);
      updateCutSummary(tab);
    });

    panel.querySelector('[data-action="generate"]').addEventListener('click', () => generateOne(tab));
    panel.querySelector('[data-role="voice-upload"]').addEventListener('change', event => uploadVoice(tab, event.target.files[0]));
    panel.querySelector('[data-action="refresh-voices"]').addEventListener('click', () => refreshVoices(tab));
    panel.querySelector('[data-action="preview-voice"]').addEventListener('click', () => previewVoice(tab));
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
      preset: tab.preset,
      audio_number: tab.number,
      title: tab.title,
      text: tab.text,
      voice_mode: tab.voice_mode,
      voice_filename: tab.voice_filename,
      options: optionsFor(tab),
    };
  }

  function validateTab(tab) {
    captureTab(tab);
    if (!tab.text.trim()) return `Audio ${tab.number} has no script.`;
    const selectedVoice = tab.voice_filename;
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
      const jobs = await api('/api/jobs');
      state.jobs = new Map(jobs.map(job => [job.id, job]));
      state.tabs.forEach(tab => {
        const job = tab.job_id ? state.jobs.get(tab.job_id) : null;
        if (job?.status === 'completed') showGenerated(tab, job);
      });
      renderActive();
      renderQueue();
      updateFloating();
      const active = jobs.some(job => ['queued', 'running'].includes(job.status));
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
      state.tabs.forEach(tab => scheduleEmotionAnalysis(tab, { immediate: true }));
    });
    setModelState(state.initial.engine);

    restoreTabs();
    buildTabs();
    state.tabs.forEach(tab => scheduleEmotionAnalysis(tab, { immediate: true }));
    state.initial.jobs.forEach(job => state.jobs.set(job.id, job));
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
    schedulePoll(200);
  }

  window.addEventListener('resize', () => state.tabs.forEach(drawWave));
  window.addEventListener('DOMContentLoaded', init);
})();
