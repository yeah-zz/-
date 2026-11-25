const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = {
  meta: null,
  artist: null,
  scene: null,
  mood: 'clear',
  abstraction: 0.0,
  job: null,
  previews: [],
  selected: 0,
  files: [],
  training: false,
};

async function api(path, body) {
  const opts = body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || '请求失败');
  return data;
}

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 2600);
}

function setActiveNav(id) {
  $$('.rail-link').forEach(b => b.classList.toggle('active', b.dataset.go === id));
}

function initNav() {
  $$('[data-go]').forEach(el => el.addEventListener('click', () => {
    const id = el.dataset.go;
    const target = document.getElementById(id);
    if (target) target.scrollIntoView({behavior: 'smooth', block: 'start'});
    setActiveNav(id);
  }));
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) setActiveNav(e.target.id); });
  }, {rootMargin: '-30% 0px -55% 0px'});
  ['masters','workbench','previews','anatomy','training','corpus','tree'].forEach(id => {
    const el = document.getElementById(id); if (el) obs.observe(el);
  });
}

function fingerprintHTML(fp) {
  return Object.entries(fp || {}).map(([k,v]) => `
    <div class="fp"><span>${k}</span><i style="--v:${Math.round(v*100)}%"></i></div>`).join('');
}

function artistCard(a) {
  const sel = state.artist && state.artist.id === a.id ? ' selected' : '';
  return `<button class="artist-card${sel}" data-artist="${a.id}">
    <div class="artist-top">
      <span class="artist-seal">${a.name.slice(0,1)}</span>
      <span class="artist-era">${a.dynasty} · ${a.years}<br>${a.school}</span>
    </div>
    <div class="artist-thumb"><img src="/demo/${a.id}.png" alt="" loading="lazy" onerror="this.parentElement.style.display='none'"></div>
    <h3>${a.name}</h3>
    <div class="style-name">${a.style_name} · ${a.work}</div>
    <div class="work">${a.intro}</div>
    <div class="tags">${(a.tags||[]).map(t=>`<span>${t}</span>`).join('')}</div>
    <div class="fingerprint">${fingerprintHTML(a.fingerprint)}</div>
  </button>`;
}

function renderArtists() {
  $('#artist-grid').innerHTML = state.meta.artists.map(artistCard).join('');
  $$('.artist-card').forEach(card => card.addEventListener('click', () => selectArtist(card.dataset.artist)));
}

function selectArtist(id) {
  state.artist = state.meta.artists.find(a => a.id === id) || state.meta.artists[0];
  state.scene = state.artist.default_scene;
  if (!state.mood) state.mood = 'clear';
  renderArtists();
  renderModel();
  renderScenes();
  renderMoods();
  $('#generate-hint').textContent = `已选 ${state.artist.name} · ${state.meta.scenes.find(s=>s.id===state.scene).name}，可生成预览。`;
}

function renderModel() {
  const a = state.artist;
  if (!a) { $('#model-card').innerHTML = '<span class="kicker">当前模型</span><h3>未选择</h3>'; return; }
  const fp = a.fingerprint || {};
  const vals = Object.values(fp);
  const avg = Math.round(vals.reduce((x,y)=>x+y,0) / Math.max(1,vals.length) * 100);
  $('#model-card').innerHTML = `
    <span class="kicker">当前模型 · ${a.dynasty}</span>
    <h3>${a.name}</h3>
    <div class="model-sub">${a.style_name} · ${a.work}</div>
    <p>${a.intro}</p>
    ${a.theory ? `<p style="border-left:2px solid rgba(202,167,122,.55);padding-left:12px;color:#e3ddcc">画理：${a.theory}</p>` : ''}
    <div class="tags model-tags">${(a.tags||[]).map(t=>`<span>${t}</span>`).join('')}</div>
    <div class="model-metrics">
      <div><b>${avg}%</b><span>风格指纹均值</span></div>
      <div><b>7</b><span>笔法参数组</span></div>
      <div><b>${(a.tags||[]).length}</b><span>核心笔法</span></div>
      <div><b>${a.custom ? '自定义' : '内置'}</b><span>模型来源</span></div>
    </div>`;
}

function updateAbsLabel(v) {
  $('#abs-label').textContent = v < .15 ? '写实' : v < .45 ? '简笔' : '意象';
}

function renderMoods() {
  if (!state.meta || !state.meta.moods) return;
  $('#mood-pills').innerHTML = state.meta.moods.map(m => `
    <button class="pill${state.mood === m.id ? ' selected' : ''}" data-mood="${m.id}" title="${m.desc}">${m.name}</button>`).join('');
  $$('#mood-pills .pill').forEach(p => p.addEventListener('click', () => {
    state.mood = p.dataset.mood;
    renderMoods();
  }));
}

function renderScenes() {
  if (!state.meta) return;
  $('#scene-pills').innerHTML = state.meta.scenes.map(s => `
    <button class="pill${state.scene === s.id ? ' selected' : ''}" data-scene="${s.id}" title="${s.desc}">
      ${s.name}<span style="opacity:.45;font-size:9px;margin-left:6px">${s.sub}</span>
    </button>`).join('');
  $$('#scene-pills .pill').forEach(p => p.addEventListener('click', () => {
    state.scene = p.dataset.scene;
    renderScenes();
  }));
}

function renderProgress(job) {
  const box = $('#progress');
  box.hidden = false;
  const pct = job.total ? Math.round(job.done / job.total * 100) : 0;
  $('#progress-bar').style.width = pct + '%';
  $('#progress-label').textContent = job.stage || '生成中';
  $('#generate').disabled = job.status === 'running';
}

async function generate() {
  if (!state.artist) { toast('请先选择一位名家'); return; }
  const body = {
    artist: state.artist.id,
    scene: state.scene,
    mood: state.mood,
    count: parseInt($('#count').value, 10),
    detail: parseFloat($('#detail').value),
    abstraction: parseFloat($('#abstraction').value),
    width: 1200,
  };
  try {
    $('#progress').hidden = false;
    $('#progress-bar').style.width = '2%';
    $('#progress-label').textContent = '准备笔墨';
    $('#generate').disabled = true;
    const {id} = await api('/api/generate', body);
    state.job = id;
    localStorage.setItem('master_job', id);
    pollJob(id);
  } catch (e) {
    $('#generate').disabled = false;
    toast(e.message);
  }
}

async function pollJob(id) {
  try {
    const job = await api('/api/job?id=' + encodeURIComponent(id));
    renderProgress(job);
    if (job.status === 'running') {
      setTimeout(() => pollJob(id), 1000);
    } else if (job.status === 'error') {
      $('#generate').disabled = false;
      toast('生成失败：' + job.error);
    } else {
      state.previews = job.previews || [];
      if (job.artist && state.meta.artists.find(a => a.id === job.artist)) {
        state.artist = state.meta.artists.find(a => a.id === job.artist);
      }
      if (job.scene) state.scene = job.scene;
      if (job.mood) state.mood = job.mood;
      if (job.abstraction !== undefined) {
        state.abstraction = job.abstraction;
        $('#abstraction').value = job.abstraction;
        updateAbsLabel(job.abstraction);
      }
      renderArtists(); renderModel(); renderScenes(); renderMoods();
      renderPreviews();
      $('#generate').disabled = false;
      $('#progress-bar').style.width = '100%';
      $('#progress-label').textContent = '四幅已就绪';
      toast('预览已生成，选一幅放大查看');
      document.getElementById('previews').scrollIntoView({behavior:'smooth'});
    }
  } catch (e) {
    $('#generate').disabled = false;
    toast(e.message);
  }
}

function renderPreviews() {
  if (!state.previews.length) return;
  const a = state.artist;
  const s = state.meta.scenes.find(x => x.id === state.scene);
  $('#preview-grid').innerHTML = state.previews.map((p, i) => `
    <article class="preview-card${i === state.selected ? ' selected' : ''}" data-preview="${i}">
      <img src="${p.url}" alt="${a.name} ${s.name} ${p.label}" loading="lazy">
      <div class="preview-info"><b>${p.label}</b><span>${p.strokes} 笔</span></div>
      <div class="preview-actions"><small>${p.width} × ${p.height}</small><button data-open="${i}">放大选画 ↗</button></div>
    </article>`).join('');
  $$('.preview-card').forEach(card => {
    card.addEventListener('click', () => openModal(parseInt(card.dataset.preview, 10)));
  });
}

function openModal(i) {
  const p = state.previews[i];
  if (!p) return;
  state.selected = i;
  const a = state.artist;
  const s = state.meta.scenes.find(x => x.id === state.scene);
  $('#modal-img').src = p.url;
  $('#modal-kicker').textContent = `${a.dynasty} · ${a.style_name} · ${s.name}`;
  $('#modal-title').textContent = `${a.name} · ${p.label}`;
  $('#modal-desc').textContent = a.intro;
  $('#modal-stats').innerHTML = `
    <div><b>${p.strokes}</b><span>逐笔轨迹</span></div>
    <div><b>${p.width}×${p.height}</b><span>画布像素</span></div>
    <div><b>${(a.tags||[]).slice(0,2).join(' / ')}</b><span>核心笔法</span></div>
    <div><b>${a.custom ? '自定义' : '内置'}</b><span>模型来源</span></div>`;
  $('#export-title').value = `山河绘 · ${a.name}笔意 · ${s.name}`;
  $('#export-result').hidden = true;
  $('#export-result').innerHTML = '';
  $('#install-btn').hidden = true;
  state.exported = false;
  $('#modal').hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  $('#modal').hidden = true;
  document.body.style.overflow = '';
}

async function exportBundle() {
  const btn = $('#export-btn');
  const p = state.previews[state.selected];
  if (!p || !state.job) return;
  btn.disabled = true;
  btn.textContent = '正在整理笔触…';
  try {
    const r = await api('/api/export', {job: state.job, index: state.selected, title: $('#export-title').value});
    const box = $('#export-result');
    box.hidden = false;
    const t = r.trajectory || {};
    box.innerHTML = `已生成可执行绘画包：<b>${r.black}</b> 条黑线、<b>${r.color}</b> 条色笔、<b>${r.waypoints}</b> 个运动点。<br>轨迹：速度 <b>${t.speed_min ?? '—'}–${t.speed_max ?? '—'}%</b>，压力 <b>${t.press_min ?? '—'}–${t.press_max ?? '—'} mm</b>，共 <b>${t.points ?? '—'}</b> 个中心线点。<br><a href="${r.download}" download>下载 painting.json ↓</a><br><span style="color:#8b9488">放入“待画作品”后，仍需在现有工作台核对纸张与现场参数。</span>`;
    $('#install-btn').hidden = false;
    state.exported = true;
    toast('绘画包已生成');
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '导出机械臂绘画包 <span>→</span>';
  }
}

function initModal() {
  $$('[data-close]').forEach(el => el.addEventListener('click', closeModal));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
  $('#export-btn').addEventListener('click', exportBundle);
}

async function installBundle() {
  if (!state.exported || !state.job) { toast('请先导出绘画包'); return; }
  const btn = $('#install-btn');
  btn.disabled = true;
  btn.textContent = '正在保存…';
  try {
    const r = await api('/api/install', {job: state.job, index: state.selected, title: $('#export-title').value});
    const box = $('#export-result');
    box.hidden = false;
    box.innerHTML += `<br><b>${r.message}</b><br><span style="color:#8b9488">位置：${r.path}</span>`;
    btn.hidden = true;
    toast('已保存到待画作品/新作品');
  } catch (e) {
    toast(e.message);
    btn.disabled = false;
    btn.textContent = '保存到待画作品';
  }
}

function initStrokeDemo() {
  const g = document.getElementById('stroke-demo');
  const speed = document.getElementById('speed-demo');
  if (!g || !speed) return;
  const p0 = [18, 78], p1 = [120, 36], p2 = [330, 104], p3 = [582, 44];
  const N = 44;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N, mt = 1 - t;
    const x = mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0];
    const y = mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1];
    pts.push([x, y]);
  }
  const NS = 'http://www.w3.org/2000/svg';
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const w = 1.4 + 9.5 * Math.pow(Math.sin(Math.PI * t), 0.75);
    const a = 0.35 + 0.65 * Math.pow(Math.sin(Math.PI * t), 0.5);
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('x1', pts[i][0]); line.setAttribute('y1', pts[i][1]);
    line.setAttribute('x2', pts[i+1][0]); line.setAttribute('y2', pts[i+1][1]);
    line.setAttribute('stroke', '#243b33');
    line.setAttribute('stroke-width', w.toFixed(2));
    line.setAttribute('stroke-linecap', 'round');
    line.setAttribute('opacity', a.toFixed(2));
    g.appendChild(line);
  }
  // 速度：两端慢、弯处慢，直段快
  const sp = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const curve = 0.62 + 0.38 * Math.abs(Math.sin(Math.PI * 2 * t));
    const v = Math.max(0.25, 1 - curve * 0.55);
    sp.push((55 + v * 34).toFixed(1) + ',' + (113 - v * 18).toFixed(1));
  }
  speed.setAttribute('points', sp.join(' '));
}

function initControls() {
  $('#generate').addEventListener('click', generate);
  $('#install-btn').addEventListener('click', installBundle);
  $('#detail').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    $('#detail-label').textContent = v < .9 ? '疏朗' : v > 1.1 ? '繁密' : '标准';
  });
  $('#abstraction').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    state.abstraction = v;
    updateAbsLabel(v);
  });
}

function initTrain() {
  const box = $('#upload-box');
  const input = $('#train-files');
  box.addEventListener('click', () => input.click());
  box.addEventListener('dragover', e => { e.preventDefault(); box.classList.add('over'); });
  box.addEventListener('dragleave', () => box.classList.remove('over'));
  box.addEventListener('drop', e => {
    e.preventDefault(); box.classList.remove('over');
    addFiles([...e.dataTransfer.files]);
  });
  input.addEventListener('change', () => addFiles([...input.files]));
  $('#train-btn').addEventListener('click', train);
}

function addFiles(files) {
  state.files = [...state.files, ...files.filter(f => /^image\/(png|jpeg|webp)$/.test(f.type))].slice(0, 12);
  $('#train-files-list').innerHTML = state.files.map(f => `<span>${f.name.slice(0,18)}</span>`).join('');
}

async function train() {
  if (!state.files.length) { toast('请先上传至少一幅原作'); return; }
  const name = $('#train-name').value.trim();
  if (!name) { toast('请先给模型起名'); return; }
  const btn = $('#train-btn');
  btn.disabled = true;
  btn.textContent = '正在读取笔法…';
  try {
    const images = [];
    for (const f of state.files.slice(0, 8)) {
      const data = await fileBase64(f);
      images.push({name: f.name, data});
    }
    const r = await api('/api/train', {name, base: $('#train-base').value, images});
    const a = r.artist;
    state.meta.artists.push(a);
    renderArtists();
    $('#train-result').innerHTML = `<span>训练完成 · ${a.name}</span>
      <p>${a.intro}</p>
      <div class="fingerprint">${fingerprintHTML(a.fingerprint)}</div>
      <button class="btn ghost" style="margin-top:12px" id="use-custom">用这套模型创作 →</button>`;
    $('#use-custom').addEventListener('click', () => {
      selectArtist(a.id);
      document.getElementById('workbench').scrollIntoView({behavior:'smooth'});
    });
    toast('自定义笔法模型已加入名家谱');
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '开始训练 →';
  }
}

function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',')[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

async function restoreJob() {
  let saved = localStorage.getItem('master_job');
  if (!saved) {
    try {
      const latest = await api('/api/latest');
      saved = latest.id;
    } catch (e) {
      return;
    }
  }
  if (!saved) return;
  try {
    const job = await api('/api/job?id=' + encodeURIComponent(saved));
    localStorage.setItem('master_job', saved);
    state.job = saved;
    if (job.status === 'running') {
      pollJob(saved);
      return;
    }
    if (job.status === 'done' && job.previews && job.previews.length) {
      const a = state.meta.artists.find(x => x.id === job.artist);
      if (a) state.artist = a;
      if (job.scene) state.scene = job.scene;
      if (job.mood) state.mood = job.mood;
      if (job.abstraction !== undefined) {
        state.abstraction = job.abstraction;
        $('#abstraction').value = job.abstraction;
        updateAbsLabel(job.abstraction);
      }
      state.previews = job.previews;
      renderArtists(); renderModel(); renderScenes(); renderMoods();
      renderPreviews();
      $('#generate-hint').textContent = '已恢复上次生成的预览，可直接放大选画。';
    }
  } catch (e) {
    localStorage.removeItem('master_job');
  }
}

async function init() {
  $('#today').textContent = new Date().toLocaleDateString('zh-CN', {year:'numeric', month:'2-digit', day:'2-digit'}).replaceAll('/', '.');
  initNav(); initModal(); initControls(); initTrain(); initStrokeDemo(); initCorpus();
  try {
    state.meta = await api('/api/meta');
    state.artist = state.meta.artists[0];
    state.scene = state.artist.default_scene;
    renderArtists(); renderModel(); renderScenes(); renderMoods();
    $('#train-base').innerHTML = state.meta.artists.map(a => `<option value="${a.id}">${a.name} · ${a.style_name}</option>`).join('');
    $('#generate-hint').textContent = `已选 ${state.artist.name}，可生成预览。`;
    await restoreJob();
    await loadCorpus();
    await loadTreeGallery();
  } catch (e) {
    toast('工作台连接失败：' + e.message);
  }
}

init();
async function loadCorpus() {
  try {
    const s = await api('/api/corpus/status?t=' + Date.now());
    const names = {pima:'披麻', zhedai:'折带', fupi:'斧劈', midian:'米点', none:'简笔'};
    $('#corpus-artists').innerHTML = Object.entries(s.artists).map(([id,a]) => {
      const img = (a.image_list && a.image_list[0]) ? a.image_list[0].url : '';
      return `<div class="corpus-card" data-img="${img}" title="${img ? '点击查看馆藏原作' : '尚无图片'}">
        <div class="corpus-thumb">${img ? `<img src="${img}" alt="${a.name}馆藏" loading="lazy">` : '<span>暂无图</span>'}</div>
        <div class="corpus-info"><b>${a.name}</b><span>${a.texture_name} · ${a.works} 幅馆藏</span></div>
        <div class="count">${a.images} 图<br>${a.patches} patch</div>
      </div>`;
    }).join('');
    $$('.corpus-card').forEach(card => card.addEventListener('click', () => {
      if (card.dataset.img) window.open(card.dataset.img, '_blank');
    }));
    if (s.model) {
      let html = `<b>四类皴法模型</b><br>样本 ${s.model.samples} 个 · kNN 留一法 ${Math.round(s.model.loo_accuracy*100)}%<br>${Object.entries(s.model.counts).map(([k,v])=>`${names[k]||k} ${v}`).join(' · ')}`;
      if (s.encoder) html += `<br><br><b>CNN 编码器</b><br>标注 ${s.encoder.samples} patch · 预训练 ${s.encoder.bulk_patches} patch · 验证 ${Math.round(s.encoder.val_accuracy*100)}%`;
      if (s.bulk) html += `<br>公共预训练图库：${s.bulk.images} 幅 / ${s.bulk.patches} patch`;
      if (s.ltfc) html += `<br>中华珍宝馆：${s.ltfc.images} 幅 / ${s.ltfc.patches} patch`;
      if (s.clean_model) html += `<br>人工精选：${s.clean_model.samples} patch · 留一法 ${Math.round(s.clean_model.loo_accuracy*100)}%（已用于渲染）`;
      if (s.total_images) html += `<br><b>合计：${s.total_images} 幅 / ${s.total_patches} patch</b>`;
      if (s.tree) html += `<br><br><b>独立树法模型</b><br>${s.tree.clusters} 类 · ${s.tree.templates} 个骨架模板 · 树 patch 172`;
      if (s.stroke_field) html += `<br><br><b>自研 StrokeField</b><br>${s.stroke_field.patches} patch · ${s.stroke_field.epochs} epochs · 输出墨线/笔向/宽度场<br><a href="/stroke-field-compare.jpg" target="_blank" style="color:var(--seal)">查看原 patch / 墨线场 / 矢量化对比 ↗</a>`;
      $('#corpus-model').innerHTML = html;
    } else {
      $('#corpus-model').innerHTML = `图库：${s.images} 幅，patch：${s.patches} 个。<br>${s.images ? '可以提取 patch 并训练；点击左侧卡片查看馆藏原作。' : '图库为空，先下载公共图库。'}`;
    }
  } catch (e) {
    $('#corpus-model').textContent = '图库状态读取失败：' + e.message;
  }
}

async function corpusAction(kind) {
  const btn = {download:'#corpus-download', extract:'#corpus-extract', train:'#corpus-train'}[kind];
  const el = $(btn);
  const old = el.textContent;
  el.disabled = true;
  el.textContent = kind === 'download' ? '下载中…' : kind === 'extract' ? '提取中…' : '训练中…';
  try {
    $('#corpus-msg').textContent = kind === 'download' ? '正在尝试下载公共图库…' : kind === 'extract' ? '正在提取山石 patch…' : '正在训练四类皴法…';
    const r = await api('/api/corpus/' + kind, {});
    if (kind === 'train') { $('#corpus-msg').textContent = '训练完成：kNN ' + Math.round((r.model.loo_accuracy||0)*100) + '%'; toast('皴法模型训练完成'); }
    else if (kind === 'extract') { $('#corpus-msg').textContent = 'patch 提取完成'; toast('patch 提取完成'); }
    else { $('#corpus-msg').textContent = r.downloaded ? `已下载 ${r.downloaded} 幅` : '下载失败，请用命令行'; toast(r.downloaded ? `已下载 ${r.downloaded} 幅` : '下载失败，请用命令行'); }
    await loadCorpus();
    await loadTreeGallery();
  } catch (e) {
    $('#corpus-msg').textContent = e.message;
    toast(e.message);
  } finally {
    el.disabled = false;
    el.textContent = old;
  }
}

function initCorpus() {
  $('#corpus-download').addEventListener('click', () => corpusAction('download'));
  $('#corpus-extract').addEventListener('click', () => corpusAction('extract'));
  $('#corpus-train').addEventListener('click', () => corpusAction('train'));
}
async function loadTreeGallery() {
  try {
    const g = await api('/api/tree/gallery');
    $('#tree-cards').innerHTML = g.cards.map(c => `
      <article class="tree-card" data-img="${c.files[0] ? c.files[0].url : ''}">
        <h3>${c.name}</h3>
        <div class="tree-thumbs">${c.files.map(f => `<img src="${f.url}" alt="${c.name}" loading="lazy">`).join('')}</div>
        <small>${c.files.length} 个标本 · 点击查看</small>
      </article>`).join('');
    $$('.tree-card').forEach(card => card.addEventListener('click', () => {
      if (card.dataset.img) window.open(card.dataset.img, '_blank');
    }));
    $('#tree-meta').textContent = `独立树法标本 ${g.patches} 个 · 山水预览已移除树木`;
  } catch (e) {
    $('#tree-meta').textContent = '树法模块尚未就绪：' + e.message;
  }
}
