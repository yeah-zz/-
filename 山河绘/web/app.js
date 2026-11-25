const $ = (id) => document.getElementById(id);
const icons = {
  layout: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M9 9h12"/>',
  gallery: '<rect x="3" y="4" width="15" height="15" rx="2"/><path d="M7 22h13a2 2 0 0 0 2-2V8M3 15l4-4 4 4 3-3 4 4"/><circle cx="13" cy="8" r="1"/>',
  screen: '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>',
  'arrow-up-right': '<path d="M6 18 18 6M6 6h12v12"/>',
  'arrow-right': '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 2-2.5 2-2.5 4M12 16h.01"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="m3 16 5-5 5 5 3-3 5 5"/><circle cx="15" cy="8" r="1.5"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  pen: '<path d="m13 3 8 8-9 9-9 1 1-9Z"/><path d="m10 6 8 8M3 21l6-6"/><circle cx="10" cy="14" r="1"/>',
  palette: '<path d="M12 3a9 9 0 0 0 0 18h2c2 0 3-2 1-3-2-2 0-4 2-4h2c4 0 2-11-7-11Z"/><circle cx="7.5" cy="9" r=".8"/><circle cx="12" cy="7" r=".8"/><circle cx="16.5" cy="9" r=".8"/><circle cx="6.5" cy="14" r=".8"/>',
  play: '<path d="m7 4 14 8-14 8Z"/>', pause: '<path d="M8 4v16M16 4v16"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>',
  refresh: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M5 8a7.5 7.5 0 0 1 13-3l2 3M4 16l2 3a7.5 7.5 0 0 0 13-3"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>', upload: '<path d="M12 16V3m-5 5 5-5 5 5M3 15v6h18v-6"/>',
  settings: '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3" fill="var(--paper)"/><circle cx="15" cy="17" r="3" fill="var(--paper)"/>'
};
function icon(name) { return `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.image}</svg>`; }
function hydrateIcons() { document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon)); }
hydrateIcons();
const palette = {title:'#29342d',black: '#29342d', blue: '#44738b', blue2: '#7fa9bb', green: '#537d58', yellow: '#b79a60', red: '#ad6158'};
const screenMode = location.pathname === '/screen';
if (screenMode) document.body.classList.add('screen');
let config, library = [], selected, geometry, state = {status:'idle'}, chosenMode = 'live', showReference = false;
let drawCounts = {}, drawKey = '', displayedStateKey = '', busy = false, polling = false, uploads = 0, toastTimer;
let itemRevision = '', lastLibraryRefresh = 0, renderTicket = 0, pendingOperations = [];
let parameterEditor = null, parametersSaving = false;
const canvas = $('painting'), ctx = canvas.getContext('2d');
$('today').textContent = new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai', month:'2-digit', day:'2-digit'}).format(new Date()).replace('/', '月') + '日';
const escapeHTML = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(message) { $('toast').textContent = message; $('toast').classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('show'), 5200); }
async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json', 'X-Workbench-Token':config.token}, body:JSON.stringify(body)} : {});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '服务请求失败');
  return result;
}
function selectedItem() { return library.find(i => i.id === selected); }
function totalCount(item = selectedItem()) { return item ? Object.values(item.counts).reduce((a,b) => a+b, 0) : 0; }
function taskForSelection() { return state.id === selected; }
function activeTask() { return ['running','starting','paused','waiting','ready'].includes(state.status) && state.id; }
function referenceDisplay(show) {
  showReference = show;
  $('reference').hidden = !show;
  $('reference-toggle').style.background = show ? '#eaf0e8' : '';
  $('reference-toggle').title = show ? '返回实时画布' : '查看参考原图';
  $('brush-marker').style.display = show ? 'none' : $('brush-marker').style.display;
  $('surface-label').textContent = show ? '参考原图 · 点击图片按钮返回画布' : !taskForSelection() || state.status === 'idle' ? '空白画布 · 等待第一笔落下' : state.mode === 'demo' ? '离线预演 · 一笔一画，山河生长' : '与机械臂同步 · 每完成一笔，画布添一笔';
}
function artCard(item) {
  const cover = item.displayImage || item.image;
  return `<div class="artwork-entry"><button class="artwork ${item.id === selected ? 'selected' : ''}" data-artwork="${item.id}" aria-pressed="${item.id === selected}"><div class="artwork-image">${cover ? `<img src="${escapeHTML(cover)}" alt="${escapeHTML(item.title)}" loading="lazy">` : '<div class="artwork-placeholder">山河入画</div>'}<span class="selected-check">✓</span></div><div class="artwork-meta"><strong>${escapeHTML(item.title)}</strong><span>${escapeHTML(item.description || totalCount(item).toLocaleString() + ' 笔 · 已就绪')}</span></div></button><button class="artwork-delete" data-delete-artwork="${escapeHTML(item.id)}">删除画作</button></div>`;
}
function bindArtCards(root) {
  root.querySelectorAll('[data-artwork]').forEach(button => button.onclick = () => selectArtwork(button.dataset.artwork));
  root.querySelectorAll('[data-delete-artwork]').forEach(button => {
    button.disabled = state.parametersLocked || (activeTask() && state.id === button.dataset.deleteArtwork);
    button.onclick = () => deleteArtwork(button.dataset.deleteArtwork);
  });
}
async function deleteArtwork(id) {
  const item = library.find(i => i.id === id);
  if (!item || !confirm(`删除“${item.title}”？\n画作会从网页画库移除，本地 JSON 和原图保留。`)) return;
  try {
    const result = await api('/api/library/delete', {id});
    if (selected === id) {
      selected = null; geometry = null; localStorage.removeItem('shanhehui-selected');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      referenceDisplay(false); $('reference').removeAttribute('src');
      $('current-title').textContent = '请选择画作'; $('vertical-title').textContent = '';
      $('screen-title').textContent = '请选择画作';
    }
    state = await api('/api/state');
    await refreshLibrary(); updateStateUI(); $('modal').close(); toast(result.message);
  } catch (error) { toast(error.message); }
}
function renderLibrary(operations = pendingOperations) {
  pendingOperations = operations;
  $('nav-count').textContent = library.length;
  $('library-count').textContent = library.length;
  const visible = library;
  $('artworks').innerHTML = visible.map(artCard).join('');
  bindArtCards($('artworks'));
  $('operations').innerHTML = operations.filter(i => i.status !== 'ready').map(i => `<div class="operation ${i.status==='error'?'error':''}"><span>${escapeHTML(i.title)}</span><span>${escapeHTML(i.message)}</span></div>`).join('');
}
async function refreshLibrary() {
  const result = await api('/api/library');
  library = result.items;
  const revision = JSON.stringify(result);
  if (revision !== itemRevision) {
    itemRevision = revision; renderLibrary(result.operations);
    for (const item of result.unavailable || []) {
      const notice = document.createElement('div'); notice.className = 'operation error';
      notice.textContent = item.title + '：' + item.message; $('operations').append(notice);
    }
  }
  if (!selected && library.length) await selectArtwork(library[0].id, true);
  return result;
}
async function selectArtwork(id, automatic = false) {
  if (!automatic && activeTask() && id !== state.id) { toast('当前任务尚未结束，请先结束演示，或按笔暂停并结束真实任务。'); return; }
  if (id === selected && geometry) return;
  const ticket = ++renderTicket;
  const data = await api('/api/artwork?id=' + encodeURIComponent(id));
  if (ticket !== renderTicket) return;
  geometry = data; selected = id; localStorage.setItem('shanhehui-selected', id);
  canvas.width = geometry.size[0]; canvas.height = geometry.size[1];
  $('paper').style.aspectRatio = `${geometry.size[0]}/${geometry.size[1]}`;
  const item = selectedItem();
  $('current-title').textContent = item.title;
  $('vertical-title').textContent = item.title.replace(/^\d+_/, '').slice(0, 6);
  $('paper-size').textContent = `${item.mm[0]} × ${item.mm[1]} mm`;
  $('surface-size').textContent = '宣纸画布';
  $('screen-title').textContent = item.title;
  if (item.image) $('reference').src = item.image;
  else $('reference').removeAttribute('src');
  drawKey = ''; drawCounts = {}; displayedStateKey = '';
  ctx.clearRect(0,0,canvas.width,canvas.height);
  referenceDisplay(false);
  renderLibrary(); updateStateUI(); drawCompleted(); fitPaper();
  $('modal').close();
}
function renderStroke(stroke, color, ctx = canvas.getContext('2d')) {
  const points = stroke.points;
  if (!points.length) return;
  const opacity = Array.isArray(stroke.opacity) ? stroke.opacity.reduce((a,b)=>a+b,0)/stroke.opacity.length : stroke.opacity;
  ctx.strokeStyle = palette[color]; ctx.fillStyle = palette[color];
    ctx.globalAlpha = color === 'black' ? .77 : Math.min(.55, Math.max(.01, opacity || .35));
  ctx.lineWidth = Math.max(.65, Math.min(18, stroke.width));
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(points[0][0],points[0][1]);
  points.slice(1).forEach(point => ctx.lineTo(point[0],point[1]));
  if (points.length === 1) { ctx.arc(points[0][0],points[0][1],ctx.lineWidth/2,0,Math.PI*2); ctx.fill(); }
  else ctx.stroke();
  ctx.globalAlpha = 1;
}
function openArtworkPreview() {
  if (!geometry || !selectedItem()) { toast('请先选择画作。'); return; }
  const previewGeometry = geometry, item = selectedItem();
  openModal(item.title + ' · 画稿预览', `<div class="preview-switch mode-switch" role="group" aria-label="预览内容"><button id="preview-full" class="selected">完整画稿</button><button id="preview-black">黑线轮廓</button>${item.image ? '<button id="preview-reference">参考原图</button>' : ''}</div><div class="artwork-preview-paper"><canvas id="artwork-preview"></canvas>${item.image ? `<img id="artwork-preview-reference" src="${escapeHTML(item.image)}" alt="${escapeHTML(item.title)}参考原图" hidden>` : ''}</div><p class="preview-note" id="preview-note"></p>`);
  const preview = $('artwork-preview');
  preview.width = previewGeometry.size[0]; preview.height = previewGeometry.size[1];
  const painter = preview.getContext('2d');
  function show(mode) {
    document.querySelectorAll('.preview-switch button').forEach(button => button.classList.toggle('selected', button.id === 'preview-' + mode));
    preview.hidden = mode === 'reference';
    if ($('artwork-preview-reference')) $('artwork-preview-reference').hidden = mode !== 'reference';
    if (mode !== 'reference') {
      painter.clearRect(0, 0, preview.width, preview.height);
      painter.fillStyle = '#fffefa'; painter.fillRect(0, 0, preview.width, preview.height);
      const order = mode === 'black' ? ['black', 'title'] : ['black', ...Object.keys(previewGeometry.stages).filter(key => !['black', 'title'].includes(key)), 'title'];
      for (const color of order) for (const stroke of previewGeometry.stages[color] || []) renderStroke(stroke, color, painter);
    }
    $('preview-note').textContent = mode === 'reference' ? '参考原图 · 用于对照构图' : mode === 'black' ? '黑线与题字的完整轨迹预览 · 不代表当前绘画进度' : '当前画稿的全部笔触预览 · 纸面墨色与晕染以实物为准';
  }
  $('preview-full').onclick = () => show('full');
  $('preview-black').onclick = () => show('black');
  if ($('preview-reference')) $('preview-reference').onclick = () => show('reference');
  show('full');
}
function drawCompleted() {
  if (!geometry || !taskForSelection()) return;
  const key = state.id + state.mode + state.started;
  const complete = state.completed || {};
  const rebuild = key !== drawKey || Object.keys(drawCounts).some(k => (complete[k] || 0) < drawCounts[k]);
  if (rebuild) { ctx.clearRect(0,0,canvas.width,canvas.height); drawCounts = {}; drawKey = key; }
  let last;
  const order = [...new Set([...(state.stageOrder || []), ...Object.keys(geometry.stages)])];
  for (const color of order) {
    const strokes = geometry.stages[color];
    if (!strokes) continue;
    const count = Math.min(complete[color] || 0, strokes.length);
    for (let index = drawCounts[color] || 0; index < count; index++) { renderStroke(strokes[index], color); last = strokes[index].points.at(-1); }
    drawCounts[color] = count;
  }
  if (last) {
    const marker = $('brush-marker'); marker.style.left = `${last[0]/canvas.width*100}%`; marker.style.top = `${last[1]/canvas.height*100}%`; marker.style.display = showReference || state.status !== 'running' ? 'none' : 'block';
    $('tip-coordinate').textContent = `${last[0].toFixed(0)}, ${last[1].toFixed(0)} px`;
    $('arm-segments').style.transform = state.status === 'running' ? `rotate(${((last[0]/canvas.width-.5)*12).toFixed(1)}deg)` : '';
  } else if (state.status !== 'running') $('brush-marker').style.display = 'none';
}
function updateStateUI() {
  if (parameterEditor && $('parameter-fieldset')) {
    const locked = state.parametersLocked ?? parameterEditor.locked;
    $('parameter-fieldset').disabled = locked || parametersSaving;
    $('save-parameters').disabled = locked || parametersSaving;
    if (locked && !parametersSaving) {
      $('parameter-notice').textContent = '机械臂任务尚未退出，当前只能查看参数。请正常退出任务后再保存。';
      $('parameter-notice').className = 'parameter-notice locked';
    } else if (!parametersSaving && $('parameter-notice').classList.contains('locked')) {
      $('parameter-notice').textContent = '保存后用于下一次网页绘画，原画稿的笔触和下压量保持不变。';
      $('parameter-notice').className = 'parameter-notice';
    }
  }
  document.querySelectorAll('[data-delete-artwork]').forEach(button => {
    button.disabled = state.parametersLocked || (activeTask() && state.id === button.dataset.deleteArtwork);
  });
  if (!selectedItem()) {
    for (const id of ['start','pause','resume','finish']) $(id).disabled = true;
    $('overall-percent').textContent = '0'; $('overall-bar').style.width = '0%';
    $('stroke-count').textContent = '0'; $('stroke-total').textContent = ' / 0 笔';
    $('task-status').textContent = '请选择画作';
    return;
  }
  const relevant = taskForSelection();
  const s = relevant ? state : {status:'idle', completed:{}};
  const counts = relevant ? s.completed || {} : {};
  const taskInProgress = relevant && ['running','starting','paused','waiting','ready'].includes(s.status);
  const total = totalCount();
  const done = Object.values(counts).reduce((a,b)=>a+b,0);
  const percent = total ? Math.floor(done/total*100) : 0;
  const blackTotal = selectedItem().counts.black, blackDone = counts.black || 0;
  const titleTotal=selectedItem().counts.title||0, titleDone=counts.title||0;
  const colorTotal = total - blackTotal-titleTotal, colorDone = done-blackDone-titleDone;
  $('stage-title').hidden=!titleTotal;
  $('stage-color').hidden=!colorTotal;
  $('title-percent').textContent=(titleTotal?Math.floor(titleDone/titleTotal*100):0)+'%';
  $('title-bar').style.width=(titleTotal?titleDone/titleTotal*100:0)+'%';
  $('title-desc').textContent=titleDone?`${titleDone} / ${titleTotal} 笔 · 山河绘`:'可单独题字，准备黑墨后开始';
  $('stage-title').className='stage'+(s.active==='title'?' active':titleTotal&&titleDone===titleTotal?' done':'');
  const statusLabels = {idle:'准备就绪', starting:'正在连接', ready:'等待准备', running:'绘画中', paused:'已暂停', saved:'已保存退出', waiting:s.prompt?.kind === 'pause' ? '已暂停' : s.freeStages ? '等待选择' : '等待洗笔', complete:'绘画完成', stopped:'已结束', error:'执行失败', uncertain:'需现场核查'};
  $('overall-percent').textContent = percent; $('overall-bar').style.width = percent + '%';
  $('stroke-count').textContent = done.toLocaleString(); $('stroke-total').textContent = ' / '+ total.toLocaleString() + ' 笔';
  $('progress-fraction').textContent = done ? `${done.toLocaleString()} / ${total.toLocaleString()} 笔` : '等待开始创作';
  $('task-status').textContent = statusLabels[s.status] || s.status;
  $('canvas-badge').textContent = s.status === 'idle' ? '待绘制' : s.mode === 'demo' ? '演示 · '+ statusLabels[s.status] : '同步 · '+ statusLabels[s.status];
  $('black-percent').textContent = Math.floor(blackDone/blackTotal*100)+'%'; $('black-bar').style.width = blackDone/blackTotal*100+'%';
  $('color-percent').textContent = (colorTotal ? Math.floor(colorDone/colorTotal*100) : 100)+'%'; $('color-bar').style.width = (colorTotal ? colorDone/colorTotal*100 : 100)+'%';
  $('black-desc').textContent = blackDone ? `${blackDone} / ${blackTotal} 笔${blackDone===blackTotal?' · 勾勒完成':''}` : '可单独勾勒山水轮廓';
  $('color-desc').textContent = colorDone ? `${colorDone} / ${colorTotal} 笔 · ${colorDone===colorTotal?'上色完成':config.labels[s.active] || '分色绘制'}` : '可直接选择颜色上色';
  $('stage-black').className = 'stage' + (s.active==='black'?' active':blackDone===blackTotal?' done':'');
  $('stage-color').className = 'stage' + (s.active && !['black','title'].includes(s.active)?' active':colorDone===colorTotal?' done':'');
  $('arm-status').textContent = s.mode==='demo' ? '演示轨迹' : s.mode==='monitor' ? '记录同步' : s.mode==='live' ? statusLabels[s.status] : '待命';
  $('current-stroke').textContent = s.active ? `${config.labels[s.active]} #${counts[s.active] || 0}` : '—';
  $('arm-description').textContent = s.status==='running' ? s.mode==='demo' ? '正在逐笔预演，画布随落笔慢慢呈现。' : s.notice || '机械臂每完成一笔，这里便添上对应的一笔。' : s.error || s.notice || (s.status==='waiting'?'这一阶段已完成，准备好笔与颜料后继续。':'启动任务后，画布与笔触进度同步更新。');
  const elapsed = relevant && s.started && s.status !== 'idle' ? Math.floor(Date.now()/1000-s.started) : 0;
  $('elapsed').textContent = `${String(Math.floor(elapsed/60)).padStart(2,'0')}:${String(elapsed%60).padStart(2,'0')}`;
  $('screen-percent').textContent = percent + '%'; $('screen-bar').style.width = percent+'%';
  $('screen-state').textContent = `${s.mode==='demo'?'演示 · ':''}${statusLabels[s.status]}`;
  $('screen-stage').textContent = s.active ? config.labels[s.active] + ` · ${done} / ${total} 笔` : statusLabels[s.status];
  const normalColors = Object.keys(selectedItem().counts).filter(k => !['black','title'].includes(k) && (counts[k] || 0) < selectedItem().counts[k]);
  const freeStages = !relevant || !!s.freeStages;
  const remainingColors = freeStages ? Object.keys(selectedItem().counts).filter(k => (counts[k] || 0) < selectedItem().counts[k]) : normalColors.length?normalColors:titleDone<titleTotal?['title']:[];
  const waitingTitle=remainingColors.length===1&&remainingColors[0]==='title';
  const waitingForColor = relevant && (s.status === 'waiting' && s.mode === 'demo' || ['color','stage'].includes(s.prompt?.kind));
  const guide = $('preparation-guide');
  guide.className = 'preparation-guide' + (waitingForColor ? ' needs-attention' : s.error || s.status === 'uncertain' ? ' has-error' : '');
  $('guide-title').textContent = waitingForColor ? (waitingTitle?'山水已完成，请洗笔准备最后题字':colorDone ? '本色已画完，请洗笔换色' : '勾勒完成，请洗笔准备上色') : s.error ? '绘画暂未继续' : s.status === 'uncertain' ? '请核查机械臂和纸面' : s.status === 'complete' ? '山河成画，绘制完成' : s.status === 'running' ? (s.active === 'black' ? '正在按部位勾勒山水' : s.active==='title'?'最后题字 · 山河绘':`正在按部位铺染${config.labels[s.active] || ''}`) : s.status === 'paused' || s.prompt?.kind === 'pause' ? '已抬笔停靠，等待继续' : '选一幅山河，准备落笔';
  $('guide-copy').textContent = waitingForColor ? (waitingTitle?'请等墨干，洗净并擦拭毛笔，准备黑墨。确认后最后写山、河、绘。':`请等墨干，洗净并擦拭毛笔，准备${remainingColors.map(k => config.labels[k]).join('或')}颜料。每种颜色均按完整部位依次画完。`) : s.error || (s.status === 'complete' ? '所有笔触已完成。请取下作品并清洗毛笔。' : s.status === 'running' ? s.notice || `已完成 ${done} 笔，画布随绘画进度同步呈现。` : s.status==='paused' || s.prompt?.kind==='pause' ? '点击“继续作画”接着当前任务。结束后不保存续画进度，再次开始将从第一笔画起。' : '放好纸张、毛笔和颜料，点击“开始绘画”。');
  if (freeStages && waitingForColor) {
    $('guide-title').textContent = '本阶段已完成，下一步由你选择';
    $('guide-copy').textContent = '可以继续勾勒、选择颜色上色或题字，也可以结束作画。换墨色前请等墨干、洗笔并准备好对应墨色。';
    if (selectedItem().colorLead && normalColors[0] === 'green' && !(counts.green || 0)) {
      $('guide-copy').textContent = `建议先选绿色，从${selectedItem().colorLead}开始上色，再画其他景物。仍可自由选择阶段；换墨色前请洗笔并准备好对应墨色。`;
    }
  } else if (freeStages && s.status==='running' && s.active==='title') {
    $('guide-title').textContent = '正在题字 · 山河绘';
  } else if (freeStages && ['idle','stopped','complete'].includes(s.status)) {
    $('guide-copy').textContent = s.status==='complete' ? '所有笔触已完成。请取下作品并清洗毛笔。' : selectedItem().colorLead ? `可自由选择勾勒、上色或题字。上色建议先选绿色，从${selectedItem().colorLead}开始。` : '选择本次先画的内容，准备好对应墨色后开始。可以只画一个阶段。';
  }
  // This text is refreshed on every poll, before the controls' cached return.
  if (!taskInProgress && !config.freeStages) {
    $('guide-title').textContent = '自由选阶段已更新，等待后台重启';
    $('guide-copy').textContent = '请先正常结束当前任务，再双击“启动绘画工作台”。运行中的任务仍可暂停、继续和换色。';
  }
  $('color-breakdown').innerHTML = Object.keys(selectedItem().counts).filter(k => !['black','title'].includes(k)).map(k => `<div class="color-progress"><b style="background:${palette[k]}"></b><span>${config.labels[k]}</span><span>${counts[k] || 0} / ${selectedItem().counts[k]} 笔</span><span>${(counts[k] || 0) === selectedItem().counts[k] ? '已完成' : s.active === k ? '绘制中' : '待上色'}</span></div>`).join('');
  const key = JSON.stringify([selected,s.status,s.mode,chosenMode,s.prompt,s.events,s.speed,s.error,s.pauseRequested,s.parametersLocked,state.parametersLocked,busy,remainingColors]);
  if (key === displayedStateKey) return;
  displayedStateKey = key;
  const isLive = relevant && ['live','monitor'].includes(s.mode);
  if (isLive) chosenMode = 'live';
  $('mode-demo').classList.toggle('selected', chosenMode==='demo'); $('mode-live').classList.toggle('selected', chosenMode==='live');
  const firstStage = $('start-stage').value;
  $('start-stage-row').hidden = !!taskInProgress;
  $('start-stage').innerHTML = Object.keys(selectedItem().counts).filter(k => selectedItem().counts[k] > 0).map(k => `<option value="${k}">${k==='black'?'黑线勾勒':k==='title'?'题字 · 山河绘':config.labels[k]+'上色'}</option>`).join('');
  if ([...$('start-stage').options].some(o => o.value === firstStage)) $('start-stage').value = firstStage;
  $('start-stage').disabled = busy || !!activeTask();
  const start = $('start');
  start.hidden = taskInProgress; start.disabled = busy;
  if (!taskInProgress && !config.freeStages) {
    start.disabled = true;
    $('start-stage-row').hidden = true;
  }
  start.querySelector('span').textContent = busy ? '正在准备绘画…' : chosenMode==='demo' ? (s.status==='complete'?'重新预演':'开始离线预演') : s.status==='saved'?'继续绘画':'开始绘画';
  $('mode-demo').disabled = busy || !!activeTask(); $('mode-live').disabled = busy || !!activeTask();
  const controlEnabled = s.mode==='demo' || s.mode==='live';
  $('secondary-controls').hidden = false;
  $('pause').disabled = busy || s.status!=='running' || !controlEnabled || !!s.pauseRequested;
  const resume = s.status==='paused' && s.mode==='demo' || s.prompt?.kind==='pause';
  $('pause').innerHTML = `<i>${icon('pause')}</i>${s.pauseRequested?'等待停靠…':'暂停作画'}`;
  $('resume').disabled = busy || !resume || !controlEnabled;
  $('clear-records').disabled = busy || !!state.parametersLocked;
  $('finish').textContent = '结束作画';
  $('finish').disabled = busy || !relevant || !taskInProgress || s.mode==='live' && !s.prompt;
  $('speed-control').hidden = chosenMode==='live';
  $('monitor').innerHTML = `<i>${icon('refresh')}</i>${s.mode==='monitor'?'退出终端同步':'同步已有终端任务'}`;
  $('task-note').querySelector('span').textContent = s.mode==='monitor' ? '已同步执行记录，请在原终端暂停或换色。' : chosenMode==='live' ? '阶段可自由选择；结束后再开，从所选阶段第一笔开始。暂停可继续本次进度。' : '演示也可自由选择阶段，机械臂不运动。';
  const waitingColor = waitingForColor;
  $('color-choice').hidden = !waitingColor;
  $('color-choice').querySelector('p').textContent=freeStages?'选择下一阶段，或点击“结束作画”。请先准备好对应墨色。':waitingTitle?'请洗净毛笔、准备黑墨，确认后开始最后题字。':'请等墨干、洗笔并准备好颜料，再选色继续。';
  if (waitingColor) {
    $('color-buttons').innerHTML = remainingColors.map(k=>`<button class="color-button" data-color="${k}" ${busy?'disabled':''}><b style="background:${palette[k]}"></b>${k==='black'?'黑墨已准备，开始勾勒':k==='title'?'黑墨已准备，开始题字':'颜料已准备，开始'+config.labels[k]+'上色'}</button>`).join('');
    $('color-buttons').querySelectorAll('[data-color]').forEach(button => button.onclick = ()=>action(freeStages?{command:'stage',stage:button.dataset.color,promptId:state.prompt?.id}:{command:'color',color:button.dataset.color,promptId:state.prompt?.id}));
  }
  document.querySelectorAll('[data-speed]').forEach(el=>el.classList.toggle('selected', +el.dataset.speed===(s.speed||state.speed||4)));
  const events = relevant ? s.events || [] : [];
  $('activity').innerHTML = events.length ? events.slice(-3).reverse().map(event=>`<div class="activity-row"><span class="activity-dot"></span><span title="${escapeHTML(event.message)}">${escapeHTML(event.message)}</span><time>${new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(event.at*1000))}</time></div>`).join('') : '<div class="activity-row"><span class="activity-dot"></span><span>画作已就绪，等待新的创作</span><time>—</time></div>';
}
async function action(body) {
  if (busy) return;
  if (['live','demo'].includes(body.command) && !config.freeStages) {
    toast('请先正常结束当前任务，再重新启动绘画工作台以启用自由选阶段。');
    return;
  }
  busy = true; displayedStateKey = ''; updateStateUI();
  try {
    state = await api('/api/action', {...body, id:body.id || selected});
    if (body.command==='live' || body.command==='monitor' || body.command==='demo') { await refreshLibrary(); geometry=null; await selectArtwork(state.id,true); }
    if (body.command==='demo' || body.command==='live' || body.command==='clear') referenceDisplay(false);
    if (body.command==='clear') { drawKey=''; drawCounts={}; ctx.clearRect(0,0,canvas.width,canvas.height); toast(state.message); }
    if (state.id && state.id!==selected) await selectArtwork(state.id,true);
    drawCompleted();
  } catch(error) { toast(error.message); }
  finally { busy=false; displayedStateKey=''; updateStateUI(); }
}
function openModal(title, html) { $('modal-title').textContent = title; $('modal-body').innerHTML = html; $('modal').showModal(); }

function parameterInput(id, label, value, unit='毫米', attributes='') {
  return `<label class="parameter-field" for="${id}"><span>${label}<small>${unit}</small></span><input id="${id}" type="number" step="${unit==='%'?'1':'any'}" value="${escapeHTML(value)}" required ${attributes}></label>`;
}
async function openParameters() {
  const button = $('parameters'); button.disabled = true;
  try {
    parameterEditor = await api('/api/settings');
    const values = parameterEditor.values;
    const cupNames = {black:'黑墨',blue:'蓝色',green:'绿色'};
    const liquids = Object.keys(cupNames).map(name => parameterInput(`parameter-${name}-surface_z`, cupNames[name]+'液面高度', values.stations[name].surface_z)).join('');
    const details = Object.keys(cupNames).map(name => {
      const row = values.stations[name];
      return `<section class="parameter-cup"><h3><b style="background:${palette[name]}"></b>${cupNames[name]}杯</h3><div class="parameter-grid">${parameterInput(`parameter-${name}-x`, '杯中心 X', row.x)}${parameterInput(`parameter-${name}-y`, '杯中心 Y', row.y)}${parameterInput(`parameter-${name}-rim_z`, '杯沿高度', row.rim_z)}${parameterInput(`parameter-${name}-dip_mm`, '蘸入深度', row.dip_mm, '毫米', 'min="0.001" max="2"')}${parameterInput(`parameter-${name}-dwell_s`, '蘸取停留', row.dwell_s, '秒', 'min="0" max="3"')}${parameterInput(`parameter-${name}-drip_s`, '提笔沥液', row.drip_s, '秒', 'min="0.5" max="5"')}</div></section>`;
    }).join('');
    const compensationFields = Object.keys(cupNames).map(name => {
      const row = values.stations[name];
      return `<section class="parameter-cup"><h3><b style="background:${palette[name]}"></b>${cupNames[name]}液面补偿</h3><label class="ready-check"><input type="checkbox" id="parameter-${name}-level_compensation_enabled" ${row.level_compensation_enabled ? 'checked' : ''}>启用累计行笔长度补偿</label><div class="parameter-grid">${parameterInput(`parameter-${name}-level_every_length_mm`, '每累计画', row.level_every_length_mm, '毫米', 'min="0.001" max="100000"')}${parameterInput(`parameter-${name}-level_step_mm`, '下次蘸取额外下降', row.level_step_mm, '毫米', 'min="0.001" max="0.5"')}${parameterInput(`parameter-${name}-level_max_mm`, '累计下降上限', row.level_max_mm, '毫米', 'min="0.001" max="0.5"')}</div></section>`;
    }).join('');
    openModal('修改现场参数', `<form id="parameter-form"><p class="parameter-intro">填写当前毛笔在示教器上的实测读数，保留数值正负号。保存不会移动机械臂。</p><div id="parameter-notice" class="parameter-notice ${parameterEditor.locked?'locked':''}" role="status">${escapeHTML(parameterEditor.message)}</div><fieldset id="parameter-fieldset" ${parameterEditor.locked?'disabled':''}><p class="parameter-hint">行笔、空移、蘸墨速度及倍率使用原画稿与执行程序的设置；此页面不修改速度或加速度。</p><h3>纸面与液面</h3><p class="parameter-hint">纸面高度：笔尖刚触纸的高度。液面高度：笔尖刚接触对应墨水或颜料的高度。</p><div class="parameter-grid">${parameterInput('parameter-paper_z', '纸面基准高度', values.paper_z)}${liquids}</div><details class="parameter-details"><summary>页面补偿<span>展开后分别调整各颜色</span></summary><p class="parameter-hint">每种颜色独立累计实际完成的纸面行笔长度，不计空中移动。默认每画满500毫米，下次蘸取额外下降0.1毫米，累计最多0.5毫米；不改变纸面作画高度。补偿是用量估算，不是实时液面测量。重新启动执行阶段会重新计数；补液后需停机重测液面。</p>${compensationFields}</details><details class="parameter-details"><summary>杯位与蘸色参数<span>移动杯子或换笔后请核对</span></summary>${details}</details></fieldset><p class="parameter-hint">保存时检查五幅画稿；运行时采用保存后的参数。纸面高度改变后，原纸的续画进度需另外核查。</p><div class="parameter-actions"><button type="button" class="button" id="close-parameters">关闭</button><button type="submit" class="button primary" id="save-parameters" ${parameterEditor.locked?'disabled':''}>保存并检查画稿</button></div></form>`);
    $('close-parameters').onclick = () => { if (!parametersSaving) $('modal').close(); };
    $('parameter-form').addEventListener('invalid', event => {
      const section = event.target.closest('details');
      if (section) section.open = true;
    }, true);
    $('parameter-form').onsubmit = saveParameters;
    updateStateUI();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}
async function saveParameters(event) {
  event.preventDefault();
  if (parametersSaving || !$('parameter-form').reportValidity()) return;
  const values = {paper_z:$('parameter-paper_z').valueAsNumber, stations:{}};
  for (const name of ['black','blue','green']) {
    values.stations[name] = {strokes_per_load:parameterEditor.values.stations[name].strokes_per_load};
    values.stations[name].level_compensation_enabled = $(`parameter-${name}-level_compensation_enabled`).checked;
    for (const key of ['x','y','surface_z','rim_z','dip_mm','dwell_s','drip_s','level_every_length_mm','level_step_mm','level_max_mm']) {
      values.stations[name][key] = $(`parameter-${name}-${key}`).valueAsNumber;
    }
  }
  parametersSaving = true;
  $('parameter-fieldset').disabled = true; $('save-parameters').disabled = true;
  $('close-parameters').disabled = true; $('modal-close').disabled = true;
  $('save-parameters').textContent = '正在检查画稿…';
  $('parameter-notice').textContent = '正在检查纸面、杯位、液面补偿和绘画轨迹，机械臂不会运动。';
  $('parameter-notice').className = 'parameter-notice';
  try {
    const result = await api('/api/settings', {values, revision:parameterEditor.revision});
    parameterEditor = result;
    $('modal').close();
    toast(`参数已保存，${result.checkedArtworks}幅画稿检查通过。下次网页绘画将采用这些参数。`);
  } catch (error) {
    $('parameter-notice').textContent = error.message + ' 原参数保持不变。';
    $('parameter-notice').className = 'parameter-notice error';
  } finally {
    parametersSaving = false;
    $('modal-close').disabled = false;
    if ($('save-parameters')) {
      $('save-parameters').textContent = '保存并检查画稿';
      $('close-parameters').disabled = false;
      updateStateUI();
    }
  }
}
$('parameters').onclick = openParameters;
function gallery() { openModal('我的画作', `<div class="modal-gallery">${library.map(artCard).join('')}</div>`); bindArtCards($('modal-body')); }
function liveConfirmation() {
  const stage = $('start-stage').value;
  const label = stage==='black'?'黑线勾勒':stage==='title'?'题字 · 山河绘':config.labels[stage]+'上色';
  openModal('准备让山河落笔', `<div class="modal-copy"><p>画稿：<strong>${escapeHTML(selectedItem().title)}</strong>。本次先画 <strong>${escapeHTML(label)}</strong>，从该阶段第一笔开始。完成后停靠，再选择下一阶段或结束作画，不会自动补画其他阶段。</p><p>请保持纸张位置与画稿一致，检查毛笔高度与颜料杯位置，准备好${['black','title'].includes(stage)?'黑墨':'对应颜料'}，确认机械臂周围留有活动空间。</p><p>本次不保存退出后的续画进度。暂停会在当前笔完成后抬笔停靠；需要立即停止时，请使用机械臂控制器的急停。</p></div><label class="ready-check"><input type="checkbox" id="confirmation">纸张、毛笔和墨色已准备好，可以开始所选阶段</label><button class="button primary full" id="confirm-live" disabled>开始${escapeHTML(label)}</button>`);
  $('confirmation').onchange = ()=>$('confirm-live').disabled = !$('confirmation').checked;
  $('confirm-live').onclick = ()=>{ $('modal').close(); action({command:'live',confirmation:'RUN',stage}); };
}
$('start').onclick = ()=>chosenMode==='demo'?action({command:'demo',stage:$('start-stage').value}):liveConfirmation();
$('pause').onclick = ()=>action({command:'pause'});
$('resume').onclick = ()=>action({command:'resume',promptId:state.prompt?.id});
$('finish').onclick = ()=>action({command:'finish',promptId:state.prompt?.id});
$('clear-records').onclick = ()=>{
  openModal('清空之前作画记录', '<div class="modal-copy"><p>清空旧任务记录、网页画布和笔触进度。保留五幅画稿与现场参数，旧记录归档备查。</p><p>此操作不会停止或移动机械臂。请先结束绘画程序，确认机械臂已停止、旧指令队列已取消且毛笔已离纸。清空后请换新纸，下次从第一笔开始。</p></div><label class="ready-check"><input type="checkbox" id="clear-confirmation">已确认机械臂停止、旧队列取消、毛笔离纸，准备重新作画</label><button class="button primary full" id="confirm-clear" disabled>清空记录与画布</button>');
  $('clear-confirmation').onchange = ()=>$('confirm-clear').disabled = !$('clear-confirmation').checked;
  $('confirm-clear').onclick = async ()=>{ $('modal').close(); await action({command:'clear',confirmation:'CLEAR',controllerStopped:true}); };
};
function setMode(mode) {
  if (activeTask()) { toast('先结束当前演示，或在真实任务停靠后结束作画，再切换模式。'); return; }
  chosenMode=mode; displayedStateKey=''; updateStateUI();
}
$('mode-demo').onclick = ()=>setMode('demo'); $('mode-live').onclick = ()=>setMode('live');
$('monitor').onclick = ()=>action({command:state.mode==='monitor'?'unmonitor':'monitor'});
document.querySelectorAll('[data-speed]').forEach(button=>button.onclick=()=>action({command:'speed',value:+button.dataset.speed}));
[$('upload-top')].forEach(button=>button.onclick=()=>$('file-input').click());
$('nav-gallery').onclick = gallery; $('all-artworks').onclick = gallery;
$('nav-workspace').onclick = ()=>{ $('modal').close(); scrollTo({top:0,behavior:'smooth'}); };
function openScreen() { window.open('/screen','shanhehui-screen'); }
$('nav-screen').onclick = openScreen; $('open-screen').onclick = openScreen;
$('screen-fullscreen').onclick = ()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen().catch(()=>toast('浏览器未允许全屏，可手动使用 F11。'));
$('artwork-preview-open').onclick = openArtworkPreview;
$('reference-toggle').onclick = ()=>{
  if (!selectedItem()?.image) { toast('这份 JSON 没有附带参考原图，画布仍可逐笔呈现。'); return; }
  referenceDisplay(!showReference);
};
$('download').onclick = ()=>{
  if (!geometry) return;
  const output=document.createElement('canvas'); output.width=canvas.width; output.height=canvas.height;
  const painter=output.getContext('2d'); painter.fillStyle='#fffefa'; painter.fillRect(0,0,output.width,output.height); painter.drawImage(canvas,0,0);
  const link=document.createElement('a'); link.download=selectedItem().title+'_当前画布.png'; link.href=output.toDataURL('image/png'); link.click();
};
$('modal-close').onclick = ()=>{ if (!parametersSaving) $('modal').close(); };
$('modal').onclick = event=>{if(!parametersSaving && event.target===$('modal') && (event.offsetX<0||event.offsetX>$('modal').offsetWidth||event.offsetY<0||event.offsetY>$('modal').offsetHeight)) $('modal').close();};
$('modal').oncancel = event=>{ if (parametersSaving) event.preventDefault(); };
$('help').onclick = ()=>openModal('一幅山河，如何开始', '<div class="modal-copy"><p><strong>1 · 选择作品与阶段</strong><br>选好画稿，在“本次先画”里选择黑线勾勒、某种颜色上色或题字。可只上色、只题字，也可勾勒后直接题字。</p><p><strong>2 · 准备好再开始</strong><br>保持纸张与画稿位置一致，准备所选阶段的墨色，点击开始绘画并确认。每个阶段从第一笔开始。</p><p><strong>3 · 下一步由你选择</strong><br>每阶段完成后停靠，可自由选择任一未完成阶段，或结束作画。换墨色前请等墨干、洗净并擦拭毛笔。</p><p><strong>4 · 暂停或展示</strong><br>暂停会等待当前笔完成后抬笔停靠，可继续当前阶段或结束。结束后不保存续画进度；重新开始会从新选阶段第一笔画起。全画稿进度只统计本次实际完成的笔触。</p><p>离线预演同样支持自由选择阶段。点击大屏模式可展示本次绘画过程。</p></div>');
async function importFiles(files) {
  for (const file of files) {
    if (file.size > 25*1024*1024) { toast(file.name+' 超过 25 MB，已跳过。'); continue; }
    uploads++;
    toast('正在上传 '+file.name+'…');
    try {
      const data = await new Promise((resolve,reject)=>{ const reader=new FileReader(); reader.onload=()=>resolve(reader.result.split(',')[1]); reader.onerror=reject; reader.readAsDataURL(file); });
      const result = await api('/api/upload',{name:file.name,data});
      await refreshLibrary();
      if (result.status==='ready' && !activeTask()) await selectArtwork(result.id);
      toast(result.status==='ready'?'画稿已导入。':'上传成功，后台正在生成绘画任务。');
    } catch(error) { toast(error.message); }
    finally { uploads--; }
  }
  $('file-input').value='';
}
$('file-input').onchange=()=>importFiles([...$('file-input').files]);
document.body.addEventListener('dragover',event=>{event.preventDefault();});
document.body.addEventListener('drop',event=>{event.preventDefault();});
async function poll() {
  if (polling || busy) return;
  polling = true;
  try {
    state = await api('/api/state');
    $('connection-label').textContent = state.mode==='live' ? state.status==='starting' ? '机械臂连接中' : state.status === 'error' || state.status === 'uncertain' ? '请检查机械臂' : '绘画任务已连接' : state.mode==='monitor' ? '绘画进度同步中' : '绘画工作台已就绪';
    document.querySelector('.connection b').style.background='#638f63';
    if (state.id && state.id !== selected && (screenMode || activeTask())) { if (!library.some(i=>i.id===state.id)) await refreshLibrary(); await selectArtwork(state.id,true); referenceDisplay(false); }
    if (taskForSelection()) {
      if (state.status==='running' && showReference && screenMode) referenceDisplay(false);
      drawCompleted();
      if (!showReference) referenceDisplay(false);
    }
    updateStateUI();
    if (Date.now()-lastLibraryRefresh>2500) { lastLibraryRefresh=Date.now(); await refreshLibrary(); }
  } catch(error) {
    $('connection-label').textContent='服务连接中断'; document.querySelector('.connection b').style.background='#b57854';
    if (screenMode) $('screen-state').textContent='连接中断 · 画布保留最后确认状态';
  } finally { polling=false; }
}
async function init() {
  try {
    const saved = localStorage.getItem('shanhehui-selected');
    config = await api('/api/config');
    state = await api('/api/state');
    await refreshLibrary();
    const initial = (screenMode || activeTask() ? library.find(i=>i.id===state.id) : null) || library.find(i=>i.id===saved) || library[0];
    if (initial) await selectArtwork(initial.id,true);
    if (screenMode) referenceDisplay(false);
    await poll(); setInterval(poll,300);
  } catch(error) { toast('工作台初始化失败：'+error.message); }
}
init();

function fitPaper() {
  if (!geometry) return;
  const surface=$('canvas-surface'), style=getComputedStyle(surface);
  const width=surface.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
  const height=surface.clientHeight-parseFloat(style.paddingTop)-parseFloat(style.paddingBottom);
  const fitWidth=Math.min(width,height*geometry.size[0]/geometry.size[1]);
  $('paper').style.width=fitWidth+'px'; $('paper').style.height=fitWidth*geometry.size[1]/geometry.size[0]+'px';
}
new ResizeObserver(fitPaper).observe($('canvas-surface'));

// 山景仅作用于标题装饰，不参与绘画数据与机械臂控制。
(() => {
  const hero = document.querySelector('.ink-hero');
  const toggle = $('ambience-toggle');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let enabled = !reducedMotion.matches;
  try { enabled = localStorage.getItem('shanhehui-ambience') !== 'off' && !reducedMotion.matches; } catch {}
  function updateAmbience() {
    document.body.classList.toggle('ambience-off', !enabled);
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.textContent = enabled ? '山水动效 · 开' : '山水动效 · 关';
    hero.style.removeProperty('--mountain-x');
    hero.style.removeProperty('--mountain-y');
  }
  toggle.addEventListener('click', () => {
    enabled = !enabled;
    try { localStorage.setItem('shanhehui-ambience', enabled ? 'on' : 'off'); } catch {}
    updateAmbience();
  });
  reducedMotion.addEventListener('change', event => { if (event.matches) { enabled = false; updateAmbience(); } });
  hero.addEventListener('pointermove', event => {
    if (!enabled || reducedMotion.matches || event.pointerType === 'touch') return;
    const bounds = hero.getBoundingClientRect();
    hero.style.setProperty('--mountain-x', ((event.clientX - bounds.left) / bounds.width - .5) * 9 + 'px');
    hero.style.setProperty('--mountain-y', ((event.clientY - bounds.top) / bounds.height - .5) * 5 + 'px');
  });
  hero.addEventListener('pointerleave', () => {
    hero.style.setProperty('--mountain-x', '0px');
    hero.style.setProperty('--mountain-y', '0px');
  });
  updateAmbience();
})();
