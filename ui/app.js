(() => {
  'use strict';

  const MAX_TABS = 5;
  const STORAGE_KEY = 'softMetaChatterboxTabsV1';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const state = {
    initial: null,
    tabs: [],
    activeId: '',
    jobs: new Map(),
    pollTimer: null,
    monitorMinimised: false,
  };

  function defaultTab(number) {
    const d = state.initial?.defaults || {};
    return {
      id: crypto.randomUUID(), number, title: '', text: '', preset: 'Motivational Speech',
      language: d.language || 'en', voice_mode: 'default', voice_filename: '',
      temperature: d.temperature ?? 0.8, exaggeration: d.exaggeration ?? 0.65,
      cfg_weight: d.cfg_weight ?? 0.35, repetition_penalty: d.repetition_penalty ?? 1.2,
      min_p: d.min_p ?? 0.05, top_p: d.top_p ?? 1, top_k: d.top_k ?? 1000,
      speed_factor: d.speed_factor ?? 1, seed: d.seed ?? 2025,
      split_text: d.split_text ?? true, chunk_words: d.chunk_words ?? 90,
      job_id: null, job: null, panel: null, waveform: null,
    };
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { const data = await response.json(); detail = data.detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const type = response.headers.get('content-type') || '';
    return type.includes('application/json') ? response.json() : response;
  }

  function toast(message, type = '') {
    const item = document.createElement('div');
    item.className = `toast ${type}`;
    item.textContent = message;
    $('#toast-root').append(item);
    setTimeout(() => item.remove(), 4500);
  }

  function countWords(text) {
    return (text.match(/\b[\w’'-]+\b/gu) || []).length;
  }

  function formatTime(seconds, precise = false) {
    const value = Number.isFinite(Number(seconds)) ? Math.max(0, Number(seconds)) : 0;
    const minutes = Math.floor(value / 60);
    const remain = value - minutes * 60;
    return `${minutes}:${precise ? remain.toFixed(1).padStart(4, '0') : Math.floor(remain).toString().padStart(2, '0')}`;
  }

  function parseTime(value) {
    const parts = String(value || '').trim().split(':').map(Number);
    if (parts.some(Number.isNaN)) return NaN;
    if (parts.length === 1) return parts[0];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }

  function saveTabs() {
    const serialisable = state.tabs.map(({panel, waveform, job, ...tab}) => tab);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({tabs: serialisable, activeId: state.activeId}));
  }

  function restoreTabs() {
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      if (Array.isArray(data.tabs) && data.tabs.length) {
        state.tabs = data.tabs.slice(0, MAX_TABS).map((tab, index) => ({...defaultTab(index + 1), ...tab, number: index + 1}));
        state.activeId = state.tabs.some(t => t.id === data.activeId) ? data.activeId : state.tabs[0].id;
        return;
      }
    } catch (_) {}
    state.tabs = [defaultTab(1), defaultTab(2)];
    state.activeId = state.tabs[0].id;
  }

  function buildTabs() {
    const bar = $('#audio-tabs');
    const panels = $('#audio-panels');
    bar.innerHTML = '';
    panels.innerHTML = '';
    state.tabs.forEach(tab => {
      const button = document.createElement('button');
      button.className = 'audio-tab'; button.type = 'button'; button.textContent = `Audio ${tab.number}`;
      button.dataset.tabId = tab.id;
      button.addEventListener('click', () => { captureTab(currentTab()); state.activeId = tab.id; renderActive(); saveTabs(); });
      bar.append(button);

      const panel = $('#audio-panel-template').content.firstElementChild.cloneNode(true);
      panel.dataset.tabId = tab.id;
      tab.panel = panel;
      wirePanel(tab);
      panels.append(panel);
    });
    if (state.tabs.length < MAX_TABS) {
      const add = document.createElement('button');
      add.className = 'add-tab'; add.type = 'button'; add.textContent = '+'; add.title = 'Add audio tab';
      add.addEventListener('click', addTab);
      bar.append(add);
    }
    renderActive();
  }

  function currentTab() { return state.tabs.find(tab => tab.id === state.activeId) || state.tabs[0]; }
  function findTabByJob(jobId) { return state.tabs.find(tab => tab.job_id === jobId); }

  function renderActive() {
    $$('.audio-tab').forEach(button => {
      const tab = state.tabs.find(item => item.id === button.dataset.tabId);
      button.classList.toggle('active', button.dataset.tabId === state.activeId);
      button.classList.remove('status-running', 'status-queued', 'status-completed');
      if (tab?.job?.status) button.classList.add(`status-${tab.job.status}`);
    });
    state.tabs.forEach(tab => {
      tab.panel.classList.toggle('hidden', tab.id !== state.activeId);
      if (tab.id === state.activeId) fillPanel(tab);
    });
  }

  function addTab() {
    if (state.tabs.length >= MAX_TABS) return;
    captureTab(currentTab());
    const tab = defaultTab(state.tabs.length + 1);
    state.tabs.push(tab); state.activeId = tab.id; buildTabs(); saveTabs();
  }

  function field(panel, name) { return panel.querySelector(`[data-field="${name}"]`); }

  function fillPanel(tab) {
    const p = tab.panel;
    ['title','text','preset','language','voice_mode','voice_filename','temperature','exaggeration','cfg_weight','speed_factor','seed','chunk_words','cut_start','cut_end'].forEach(name => {
      const element = field(p, name); if (element && tab[name] !== undefined) element.value = tab[name];
    });
    field(p, 'split_text').checked = Boolean(tab.split_text);
    $('[data-role="word-count"]', p).textContent = `${countWords(tab.text)} words`;
    updateVoiceControls(tab);
    $('[data-action="generate"]', p).textContent = `Generate Audio ${tab.number}`;
    if (tab.job?.status === 'completed') showGenerated(tab, tab.job);
  }

  function captureTab(tab) {
    if (!tab?.panel) return;
    const p = tab.panel;
    ['title','text','preset','language','voice_mode','voice_filename'].forEach(name => { tab[name] = field(p, name)?.value || ''; });
    ['temperature','exaggeration','cfg_weight','speed_factor','seed','chunk_words'].forEach(name => { tab[name] = Number(field(p, name)?.value); });
    tab.split_text = Boolean(field(p, 'split_text')?.checked);
  }

  function optionsFor(tab) {
    return {
      model: $('#active-model').value, language: tab.language,
      temperature: tab.temperature, exaggeration: tab.exaggeration, cfg_weight: tab.cfg_weight,
      repetition_penalty: tab.repetition_penalty, min_p: tab.min_p, top_p: tab.top_p,
      top_k: tab.top_k, speed_factor: tab.speed_factor, seed: tab.seed,
      split_text: tab.split_text, chunk_words: tab.chunk_words,
    };
  }

  function jobPayload(tab) {
    captureTab(tab);
    return {
      audio_number: tab.number, title: tab.title, text: tab.text,
      voice_mode: tab.voice_mode, voice_filename: tab.voice_mode === 'default' ? null : tab.voice_filename,
      options: optionsFor(tab),
    };
  }

  function wirePanel(tab) {
    const p = tab.panel;
    const preset = field(p, 'preset');
    state.initial.presets.forEach(item => preset.add(new Option(item.name, item.name)));
    p.addEventListener('input', event => {
      if (event.target.matches('[data-field="text"]')) $('[data-role="word-count"]', p).textContent = `${countWords(event.target.value)} words`;
      captureTab(tab); saveTabs(); updateCutSummary(tab);
    });
    field(p, 'preset').addEventListener('change', () => applyPreset(tab));
    field(p, 'voice_mode').addEventListener('change', () => updateVoiceControls(tab));
    $('[data-action="generate"]', p).addEventListener('click', () => generateOne(tab));
    $('[data-role="voice-upload"]', p).addEventListener('change', event => uploadVoice(tab, event.target.files[0]));
    $('[data-action="preview-voice"]', p).addEventListener('click', () => previewVoice(tab));
    $$('[data-quick]', p).forEach(button => button.addEventListener('click', () => {
      if (!tab.waveform) return;
      field(p, 'cut_start').value = '0:00';
      field(p, 'cut_end').value = formatTime(Math.min(Number(button.dataset.quick), tab.waveform.duration), true);
      updateCutSummary(tab);
    }));
    $('[data-action="set-start-mode"]', p).addEventListener('click', () => setClickMode(tab, 'start'));
    $('[data-action="set-end-mode"]', p).addEventListener('click', () => setClickMode(tab, 'end'));
    $('[data-action="zoom-in"]', p).addEventListener('click', () => zoomWave(tab, 1.5));
    $('[data-action="zoom-out"]', p).addEventListener('click', () => zoomWave(tab, 1 / 1.5));
    $('[data-action="fit-wave"]', p).addEventListener('click', () => { if (tab.waveform) { tab.waveform.zoom = 1; drawWave(tab); } });
    $('[data-action="use-end-start"]', p).addEventListener('click', () => { field(p, 'cut_start').value = field(p, 'cut_end').value; updateCutSummary(tab); });
    $('[data-action="preview-selected"]', p).addEventListener('click', () => previewSelected(tab));
    $('[data-action="download-selected"]', p).addEventListener('click', () => cutDownload(tab, 'Selected', parseTime(field(p,'cut_start').value), parseTime(field(p,'cut_end').value)));
    $('[data-action="download-part-one"]', p).addEventListener('click', () => cutDownload(tab, 'Part_One', 0, parseTime(field(p,'cut_end').value)));
    $('[data-action="download-part-two"]', p).addEventListener('click', () => cutDownload(tab, 'Part_Two', parseTime(field(p,'cut_end').value), tab.waveform?.duration));
  }

  function applyPreset(tab) {
    const name = field(tab.panel, 'preset').value;
    const preset = state.initial.presets.find(item => item.name === name);
    if (!preset) return;
    Object.entries(preset).forEach(([key, value]) => {
      if (key in tab) tab[key] = value;
    });
    fillPanel(tab); saveTabs();
  }

  function updateVoiceControls(tab) {
    const p = tab.panel; const mode = field(p, 'voice_mode').value;
    const select = field(p, 'voice_filename'); const upload = $('[data-role="voice-upload"]', p).closest('.upload-button');
    const list = mode === 'predefined' ? state.initial.predefined_voices : state.initial.reference_voices;
    select.innerHTML = '';
    if (mode === 'default') {
      select.add(new Option('Model default voice', '')); select.disabled = true; upload.classList.add('hidden');
    } else {
      select.disabled = false; upload.classList.toggle('hidden', mode !== 'clone');
      if (!list.length) select.add(new Option('No voices found', ''));
      list.forEach(item => select.add(new Option(item.filename, item.filename)));
      if (tab.voice_filename && list.some(item => item.filename === tab.voice_filename)) select.value = tab.voice_filename;
    }
  }

  async function uploadVoice(tab, file) {
    if (!file) return;
    const data = new FormData(); data.append('file', file);
    try {
      const result = await api('/api/voices/upload?kind=clone', {method:'POST', body:data});
      const voices = await api('/api/voices');
      state.initial.reference_voices = voices.clone;
      tab.voice_mode = 'clone'; tab.voice_filename = result.filename; fillPanel(tab); saveTabs();
      toast('Voice uploaded.', 'success');
    } catch (error) { toast(error.message, 'error'); }
  }

  function previewVoice(tab) {
    captureTab(tab);
    if (tab.voice_mode === 'default' || !tab.voice_filename) return toast('Select a voice file first.');
    const player = $('[data-role="voice-player"]', tab.panel);
    player.src = `/api/voices/${encodeURIComponent(tab.voice_mode)}/${encodeURIComponent(tab.voice_filename)}`;
    player.play().catch(() => toast('The browser could not play this voice preview.', 'error'));
  }

  async function loadModel() {
    const button = $('#load-model'); button.disabled = true; $('#model-status').textContent = 'Loading model...';
    try {
      const data = await api('/api/model/load', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model:$('#active-model').value})});
      $('#model-status').textContent = 'Model ready'; $('#device-status').textContent = `${data.loaded_model} on ${data.device}`; $('#model-dot').classList.add('ready');
      toast('Model loaded.', 'success');
    } catch (error) { $('#model-status').textContent = 'Model not loaded'; toast(error.message, 'error'); }
    finally { button.disabled = false; }
  }

  async function generateOne(tab) {
    const payload = jobPayload(tab);
    if (!payload.text.trim()) return toast(`Audio ${tab.number} has no script.`, 'error');
    try {
      const job = await api('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      bindJob(tab, job); openMonitor(); ensurePolling(); toast(`Audio ${tab.number} added to queue.`, 'success');
    } catch (error) { toast(error.message, 'error'); }
  }

  async function generateAll() {
    captureTab(currentTab());
    const ready = state.tabs.filter(tab => tab.text.trim()).map(jobPayload);
    if (!ready.length) return toast('Add a script to at least one Audio tab.', 'error');
    $('#generate-all').disabled = true;
    try {
      const jobs = await api('/api/jobs/generate-all', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({jobs:ready})});
      jobs.forEach(job => { const tab = state.tabs.find(item => item.number === job.audio_number); if (tab) bindJob(tab, job); });
      openMonitor(); ensurePolling(); toast(`${jobs.length} audio jobs scheduled.`, 'success');
    } catch (error) { toast(error.message, 'error'); }
    finally { $('#generate-all').disabled = false; }
  }

  function bindJob(tab, job) { tab.job_id = job.id; tab.job = job; state.jobs.set(job.id, job); saveTabs(); renderActive(); renderQueue(); }

  function ensurePolling() {
    if (state.pollTimer) return;
    state.pollTimer = setInterval(refreshJobs, 650);
    refreshJobs();
  }

  async function refreshJobs() {
    try {
      const jobs = await api('/api/jobs');
      jobs.forEach(job => {
        state.jobs.set(job.id, job);
        const tab = findTabByJob(job.id);
        if (tab) {
          const wasComplete = tab.job?.status === 'completed';
          tab.job = job;
          if (!wasComplete && job.status === 'completed') showGenerated(tab, job);
        }
      });
      renderActive(); renderQueue(); updateFloating();
    } catch (_) {}
  }

  function renderQueue() {
    const list = $('#queue-list'); list.innerHTML = '';
    const jobs = [...state.jobs.values()].sort((a,b) => a.created_at - b.created_at).slice(-20);
    if (!jobs.length) { list.innerHTML = '<p class="hint">No audio jobs yet.</p>'; return; }
    jobs.forEach(job => {
      const card = document.createElement('div'); card.className = 'queue-card';
      const title = job.title || `Audio ${job.audio_number}`;
      card.innerHTML = `<div class="queue-card-head"><div><div><strong>Audio ${job.audio_number}</strong></div><div class="queue-title"></div><div class="queue-meta stage"></div></div><strong class="status"></strong></div><div class="progress-track"><div class="progress-fill"></div></div><div class="queue-meta metrics"></div><div class="queue-actions"></div>`;
      $('.queue-title', card).textContent = title;
      $('.stage', card).textContent = job.stage || '';
      $('.status', card).textContent = job.status;
      $('.progress-fill', card).style.width = `${job.percent || 0}%`;
      const eta = job.eta_seconds == null ? 'calculating' : formatTime(job.eta_seconds);
      const duration = job.actual_audio_seconds ?? job.estimated_audio_seconds;
      $('.metrics', card).textContent = `Words: ${Math.min(job.display_words || 0, job.total_words)} of ${job.total_words}  •  ${job.percent || 0}% generated  •  ${job.remaining_percent ?? 100}% remaining  •  ETA ${eta}  •  Audio ${formatTime(duration)}`;
      const actions = $('.queue-actions', card);
      if (job.status === 'completed') {
        actions.append(actionButton('Preview audio', () => previewMonitor(job)));
        const download = document.createElement('a'); download.className = 'secondary-button link-button'; download.textContent = 'Download WAV'; download.href = `/api/jobs/${job.id}/audio`; download.download = ''; actions.append(download);
        actions.append(actionButton(`Open Audio ${job.audio_number}`, () => openJobTab(job)));
      } else if (!['failed','cancelled'].includes(job.status)) {
        actions.append(actionButton('Cancel', () => cancelJob(job.id)));
      }
      list.append(card);
    });
  }

  function actionButton(label, handler) { const b=document.createElement('button'); b.type='button'; b.className='secondary-button'; b.textContent=label; b.addEventListener('click',handler); return b; }
  function previewMonitor(job) { const p=$('#monitor-player'); p.src=`/api/jobs/${job.id}/audio`; p.play().catch(()=>toast('Preview failed.','error')); }
  function openJobTab(job) { const tab=findTabByJob(job.id); if (!tab) return; state.activeId=tab.id; closeMonitor(); renderActive(); }
  async function cancelJob(id) { try { await api(`/api/jobs/${id}/cancel`,{method:'POST'}); refreshJobs(); } catch(error){toast(error.message,'error');} }

  function openMonitor() { state.monitorMinimised=false; $('#progress-modal').classList.remove('hidden'); $('#floating-progress').classList.add('hidden'); renderQueue(); }
  function closeMonitor() { $('#progress-modal').classList.add('hidden'); }
  function minimiseMonitor() { state.monitorMinimised=true; closeMonitor(); updateFloating(); }
  function updateFloating() {
    const running=[...state.jobs.values()].find(job=>job.status==='running') || [...state.jobs.values()].find(job=>job.status==='queued');
    const button=$('#floating-progress');
    if (!state.monitorMinimised || !running) return button.classList.add('hidden');
    button.classList.remove('hidden'); button.innerHTML=`<strong>Audio ${running.audio_number}: ${running.status}</strong><br><span>${running.percent || 0}% • Words ${running.display_words || 0}/${running.total_words} • ETA ${running.eta_seconds==null?'calculating':formatTime(running.eta_seconds)}</span>`;
  }

  async function showGenerated(tab, job) {
    const p=tab.panel; const section=$('[data-role="generated"]',p); section.classList.remove('hidden');
    $('[data-role="generated-title"]',p).textContent=tab.title || `Audio ${tab.number}`;
    const url=`/api/jobs/${job.id}/audio`; const player=$('[data-role="main-player"]',p); player.src=url;
    const download=$('[data-role="download-original"]',p); download.href=url; download.download='';
    if (!tab.waveform || tab.waveform.jobId !== job.id) await loadWaveform(tab, job.id);
  }

  async function loadWaveform(tab, jobId) {
    const status=$('[data-role="wave-status"]',tab.panel); status.textContent='Loading waveform...';
    try {
      const data=await api(`/api/jobs/${jobId}/waveform?points=5000`);
      tab.waveform={jobId, mins:data.mins, maxs:data.maxs, duration:data.duration, zoom:1, mode:'end', selected:0};
      field(tab.panel,'cut_start').value='0:00'; field(tab.panel,'cut_end').value=formatTime(data.duration,true);
      wireWave(tab); drawWave(tab); updateCutSummary(tab); status.textContent='Move the mouse to view time. Click to set the selected Start or End time.';
    } catch(error) {
      status.textContent='Could not load waveform. The normal audio player and Download WAV still work.';
    }
  }

  function wireWave(tab) {
    const scroll=$('[data-role="wave-scroll"]',tab.panel); const canvas=$('[data-role="wave-canvas"]',tab.panel); const tooltip=$('[data-role="wave-tooltip"]',tab.panel);
    canvas.onmousemove=event=>{
      if(!tab.waveform)return; const rect=canvas.getBoundingClientRect(); const x=event.clientX-rect.left; const time=(x/canvas.clientWidth)*tab.waveform.duration;
      $('[data-role="mouse-time"]',tab.panel).textContent=formatTime(time,true); tooltip.textContent=formatTime(time,true); tooltip.style.left=`${x}px`; tooltip.classList.remove('hidden');
    };
    canvas.onmouseleave=()=>tooltip.classList.add('hidden');
    canvas.onclick=event=>{
      const rect=canvas.getBoundingClientRect(); const time=Math.max(0,Math.min(tab.waveform.duration,((event.clientX-rect.left)/canvas.clientWidth)*tab.waveform.duration));
      tab.waveform.selected=time; $('[data-role="selected-time"]',tab.panel).textContent=formatTime(time,true);
      field(tab.panel,tab.waveform.mode==='start'?'cut_start':'cut_end').value=formatTime(time,true); updateCutSummary(tab); drawWave(tab);
    };
    scroll.onwheel=event=>{ if(Math.abs(event.deltaY)>Math.abs(event.deltaX)){ event.preventDefault(); scroll.scrollLeft+=event.deltaY; } };
  }

  function drawWave(tab) {
    const w=tab.waveform; if(!w)return; const canvas=$('[data-role="wave-canvas"]',tab.panel); const scroll=$('[data-role="wave-scroll"]',tab.panel);
    const width=Math.max(scroll.clientWidth,Math.round(scroll.clientWidth*w.zoom)); canvas.width=width; canvas.height=150; canvas.style.width=`${width}px`;
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,width,150); const styles=getComputedStyle(document.documentElement); const base=styles.getPropertyValue('--primary').trim()||'#6657e8'; const border=styles.getPropertyValue('--border').trim()||'#ddd';
    ctx.strokeStyle=border; ctx.beginPath(); ctx.moveTo(0,75); ctx.lineTo(width,75); ctx.stroke(); ctx.strokeStyle=base; ctx.lineWidth=1;
    const len=w.mins.length; const step=width/Math.max(len,1); ctx.beginPath();
    for(let i=0;i<len;i++){const x=i*step; const y1=75-w.maxs[i]*65; const y2=75-w.mins[i]*65; ctx.moveTo(x,y1); ctx.lineTo(x,y2);} ctx.stroke();
    const start=parseTime(field(tab.panel,'cut_start').value)||0; const end=parseTime(field(tab.panel,'cut_end').value)||w.duration;
    ctx.fillStyle='rgba(102,87,232,.12)'; ctx.fillRect((start/w.duration)*width,0,((end-start)/w.duration)*width,150);
    ctx.strokeStyle='#1c9a67'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo((start/w.duration)*width,0); ctx.lineTo((start/w.duration)*width,150); ctx.stroke();
    ctx.strokeStyle='#d84b5b'; ctx.beginPath(); ctx.moveTo((end/w.duration)*width,0); ctx.lineTo((end/w.duration)*width,150); ctx.stroke();
  }

  function setClickMode(tab, mode) { tab.waveform.mode=mode; $$('[data-action="set-start-mode"],[data-action="set-end-mode"]',tab.panel).forEach(b=>b.classList.remove('active')); $(`[data-action="set-${mode}-mode"]`,tab.panel).classList.add('active'); }
  function zoomWave(tab, factor) { if(!tab.waveform)return; tab.waveform.zoom=Math.max(1,Math.min(12,tab.waveform.zoom*factor)); drawWave(tab); }

  function updateCutSummary(tab) {
    if(!tab.waveform)return; const start=parseTime(field(tab.panel,'cut_start').value); const end=parseTime(field(tab.panel,'cut_end').value);
    const valid=Number.isFinite(start)&&Number.isFinite(end)&&end>start&&start>=0&&end<=tab.waveform.duration+.2;
    $('[data-role="selected-duration"]',tab.panel).textContent=valid?`Selected: ${formatTime(end-start,true)}`:'Selected: invalid range';
    $('[data-role="removed-duration"]',tab.panel).textContent=valid?`Removed: ${formatTime(tab.waveform.duration-(end-start),true)}`:''; drawWave(tab);
  }

  function previewSelected(tab) {
    const player=$('[data-role="main-player"]',tab.panel); const start=parseTime(field(tab.panel,'cut_start').value); const end=parseTime(field(tab.panel,'cut_end').value);
    if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return toast('Enter a valid Start and End time.','error');
    player.currentTime=start; player.play(); const stop=()=>{if(player.currentTime>=end){player.pause();player.removeEventListener('timeupdate',stop);}}; player.addEventListener('timeupdate',stop);
  }

  async function cutDownload(tab, prefix, start, end) {
    if(!tab.job_id||!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return toast('Enter a valid audio range.','error');
    try {
      const data=await api(`/api/jobs/${tab.job_id}/cut`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start_seconds:start,end_seconds:end,filename_prefix:prefix})});
      const a=document.createElement('a'); a.href=data.url; a.download=data.filename; document.body.append(a); a.click(); a.remove();
    } catch(error){toast(error.message,'error');}
  }

  function initTheme() {
    const saved=localStorage.getItem('softMetaTheme'); if(saved==='dark')document.documentElement.classList.add('dark');
    $('#theme-toggle').addEventListener('click',()=>{document.documentElement.classList.toggle('dark');localStorage.setItem('softMetaTheme',document.documentElement.classList.contains('dark')?'dark':'light');state.tabs.forEach(drawWave);});
  }

  async function init() {
    initTheme();
    try { state.initial=await api('/api/ui/initial-data'); }
    catch(error){ $('#model-status').textContent='Server unavailable'; toast(error.message,'error'); return; }
    const model=$('#active-model'); state.initial.models.forEach(item=>model.add(new Option(item.name,item.id))); model.value=state.initial.active_model;
    $('#model-status').textContent=state.initial.model_loaded?'Model ready':'Model not loaded'; $('#device-status').textContent=`${state.initial.active_model} on ${state.initial.device}`; $('#model-dot').classList.toggle('ready',state.initial.model_loaded);
    restoreTabs(); buildTabs();
    state.initial.jobs.forEach(job=>{state.jobs.set(job.id,job); const tab=state.tabs.find(t=>t.job_id===job.id); if(tab)tab.job=job;});
    $('#load-model').addEventListener('click',loadModel); $('#generate-all').addEventListener('click',generateAll); $('#open-monitor').addEventListener('click',openMonitor);
    $('#close-monitor').addEventListener('click',closeMonitor); $('#minimise-monitor').addEventListener('click',minimiseMonitor); $('#floating-progress').addEventListener('click',openMonitor);
    renderQueue(); ensurePolling();
  }

  window.addEventListener('DOMContentLoaded',init);
})();
