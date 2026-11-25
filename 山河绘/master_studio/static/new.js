const state = {
  artists: [], status: null, artist: 'huang_gongwang', mode: 'landscape',
  kind: 'pine', mood: 'clear', lineOnly: false, count: 4, previews: [], job: null, busy: false
};

const KINDS = {tree:{pine:'松树',broadleaf:'杂树',bare:'枯枝'},flower:{plum:'梅花',lotus:'荷花',orchid:'兰花',chrysanthemum:'菊花'}};

function $(s){ return document.querySelector(s); }
function esc(s){ return String(s||'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function artist(){ return state.artists.find(a=>a.id===state.artist) || state.artists[0] || {}; }

async function api(url, opts={}){
  const r = await fetch(url, {headers:{'Content-Type':'application/json'}, ...opts});
  const d = await r.json().catch(()=>({error:'返回不是 JSON'}));
  if(!r.ok) throw new Error(d.error || ('HTTP '+r.status));
  return d;
}

async function boot(){
  try{
    const [meta, status] = await Promise.all([api('/api/meta'), api('/api/corpus/status?t='+Date.now())]);
    state.artists = meta.artists || [];
    state.status = status;
    renderArtists();
    renderOptions();
    bind();
    const latest = await api('/api/latest').catch(()=>null);
    if(latest && latest.previews && latest.previews.length){
      state.previews = latest.previews;
      setMain(latest.previews[0].url, (latest.previews[0].label||'最近画稿'));
      renderThumbs();
    } else { $('#canvasEmpty').style.display='grid'; }
    say('ai', '欢迎来到山河绘 · 景区画房。\n\n我是山河，可以带你认识名家，也可以直接把眼前的山水变成一幅画。\n\n你可以说：介绍黄公望 / 画四幅倪瓒的山水 / 我要一棵松树 / 画荷花。');
    chips(['介绍一下黄公望','画四幅倪瓒的山水','我要一棵松树','画一幅荷花']);
  }catch(e){ say('ai','连接不上画房：'+e.message); }
}

function renderArtists(){
  const host = $('#artistList');
  host.innerHTML = state.artists.map(a=>{
    const c = state.status && state.status.artists && state.status.artists[a.id];
    const img = c && c.image_list && c.image_list[0] ? c.image_list[0].url : '';
    return `<button class="artist-card ${a.id===state.artist?'active':''}" data-id="${a.id}">
      ${img?`<img src="${img}" alt="${esc(a.name)}">`:'<div style="width:56px;height:64px;background:#ded3bd"></div>'}
      <span><b>${esc(a.name)}</b><span>${esc(a.dynasty)} · ${esc(a.style_name)}</span><em>${esc((a.tags||[])[0]||'')}</em></span>
    </button>`;
  }).join('');
  host.querySelectorAll('.artist-card').forEach(el=>el.addEventListener('click',()=>{
    state.artist = el.dataset.id; renderArtists(); updateTitle();
    say('ai', `已切换到 ${artist().name}。${artist().intro||''}\n\n可以说“画四幅”，或问我他的皴法。`);
  }));
}

function updateTitle(){
  const a = artist();
  if(state.mode==='tree') return setTitle('树法 v0 · 训练中', `${KINDS.tree[state.kind]} · 树法工坊`);
  if(state.mode==='flower') return setTitle('花鸟 v0 · 训练中', `${KINDS.flower[state.kind]} · 花鸟工坊`);
  if(state.mode==='masters') return setTitle('名家笔意 · 景区导览', '选择一位名家，听山河讲他的故事');
  setTitle(`拟${a.name} · ${a.work||'山水'}`, `${a.school||''} · ${state.previews.length?state.previews.length+' 幅可选':'等待生成'}`);
}
function setTitle(kicker, title){ $('#stageKicker').textContent=kicker; $('#stageTitle').textContent=title; }

function renderOptions(){
  const box = $('#modeOptions');
  if(state.mode==='landscape'){
    const moods = [['clear','清远'],['vast','苍茫'],['heroic','雄浑'],['sparse','萧散']];
    box.innerHTML = moods.map(([id,n])=>`<button class="opt ${state.mood===id?'active':''}" data-mood="${id}">${n}</button>`).join('')
      + `<button class="opt ${state.lineOnly?'active':''}" data-line="1">纯线稿</button>`;
    box.querySelectorAll('[data-mood]').forEach(el=>el.addEventListener('click',()=>{state.mood=el.dataset.mood;renderOptions();}));
    box.querySelector('[data-line]').addEventListener('click',()=>{state.lineOnly=!state.lineOnly;renderOptions();});
  } else if(state.mode==='tree'){
    box.innerHTML = Object.entries(KINDS.tree).map(([id,n])=>`<button class="opt ${state.kind===id?'active':''}" data-kind="${id}">${n}</button>`).join('');
    box.querySelectorAll('.opt').forEach(el=>el.addEventListener('click',()=>{state.kind=el.dataset.kind;renderOptions();updateTitle();}));
  } else if(state.mode==='flower'){
    box.innerHTML = Object.entries(KINDS.flower).map(([id,n])=>`<button class="opt ${state.kind===id?'active':''}" data-kind="${id}">${n}</button>`).join('');
    box.querySelectorAll('.opt').forEach(el=>el.addEventListener('click',()=>{state.kind=el.dataset.kind;renderOptions();updateTitle();}));
  } else {
    box.innerHTML = state.artists.map(a=>`<button class="opt" data-artist="${a.id}">${esc(a.name)}</button>`).join('');
    box.querySelectorAll('.opt').forEach(el=>el.addEventListener('click',()=>{state.artist=el.dataset.artist;renderArtists();updateTitle();say('ai',`${artist().name}：${artist().intro||''}`);}));
  }
  $('#generateBtn').textContent = state.mode==='landscape' ? '生成四幅' : state.mode==='tree' ? '生成树稿' : state.mode==='flower' ? '生成花鸟' : '生成山水';
  updateTitle();
}

function setMode(mode){
  state.mode = mode;
  document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active', b.dataset.mode===mode));
  if(mode==='tree') state.kind='pine';
  if(mode==='flower') state.kind='plum';
  renderOptions();
  if(mode==='landscape') say('ai','山水模式。选一位名家，或直接说“画四幅”。');
  if(mode==='tree') say('ai','树法独立成稿，不再混进山水。选松/杂树/枯枝，先生成一棵。');
  if(mode==='flower') say('ai','花鸟独立成稿。选梅/荷/兰/菊，先生成一幅小品。');
  if(mode==='masters') say('ai','这里是名家导览。点左侧名家，或问我“介绍一下范宽”。');
}

function bind(){
  document.querySelectorAll('.mode').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
  $('#generateBtn').addEventListener('click',()=>generate());
  $('#surpriseBtn').addEventListener('click',()=>{ if(state.mode==='masters') setMode('landscape'); else generate(); });
  $('#chatForm').addEventListener('submit',async e=>{ e.preventDefault(); const v=$('#chatInput').value.trim(); if(!v) return; $('#chatInput').value=''; say('user',v); await ask(v); });
  $('#thumbs').addEventListener('click',e=>{ const t=e.target.closest('.thumb'); if(!t) return; setMain(t.dataset.url,t.dataset.label); document.querySelectorAll('.thumb').forEach(x=>x.classList.toggle('active',x===t)); });
}

async function generate(){
  if(state.busy) return;
  state.busy = true; $('#generateBtn').disabled = true;
  const old = $('#generateBtn').textContent; $('#generateBtn').textContent='生成中…';
  try{
    if(state.mode==='tree'){
      const r = await api('/api/tree/generate',{method:'POST',body:JSON.stringify({kind:state.kind,seed:Math.floor(Math.random()*99999)})});
      state.previews=[{url:r.url,label:r.kind_name}]; setMain(r.url,r.kind_name); renderThumbs(); say('ai',`${r.kind_name}树稿生成好了。树法现在是独立模块，可以单独导出。`);
    } else if(state.mode==='flower'){
      const r = await api('/api/flower/generate',{method:'POST',body:JSON.stringify({kind:state.kind,seed:Math.floor(Math.random()*99999)})});
      state.previews=[{url:r.url,label:r.kind_name}]; setMain(r.url,r.kind_name); renderThumbs(); say('ai',`${r.kind_name}小品生成好了。`);
    } else {
      const job = await api('/api/generate',{method:'POST',body:JSON.stringify({artist:state.artist,scene:artist().default_scene||'fuchun',mood:state.mood,count:4,width:1000,detail:1.0,abstraction:0.0,line_only:state.lineOnly})});
      state.job = job.id;
      for(let i=0;i<90;i++){
        await new Promise(r=>setTimeout(r,1500));
        const j = await api('/api/job?id='+job.id);
        if(j.status==='done'){ state.previews=j.previews||[]; renderThumbs(); if(state.previews.length){setMain(state.previews[0].url,state.previews[0].label);} updateTitle(); return; }
        if(j.status==='error') throw new Error(j.error||'生成失败');
      }
      throw new Error('生成超时');
    }
  }catch(e){ say('ai','生成失败：'+e.message); }
  finally{ state.busy=false; $('#generateBtn').disabled=false; $('#generateBtn').textContent=old; }
}

function setMain(url,label){
  $('#canvasEmpty').style.display='none';
  $('#mainImage').src=url;
  $('#canvasLabel').textContent=label||'';
  updateTitle();
}
function renderThumbs(){
  $('#thumbs').innerHTML = state.previews.map((p,i)=>`<img class="thumb ${i===0?'active':''}" src="${p.url}" data-url="${p.url}" data-label="${esc(p.label||'')}" alt="${esc(p.label||'')}">`).join('');
  updateTitle();
}

function say(who,text){
  const m=document.createElement('div'); m.className='msg '+who;
  m.innerHTML=`<span class="who">${who==='ai'?'山河 · AI':'游客'}</span>${esc(text)}`;
  $('#messages').appendChild(m); $('#messages').scrollTop=$('#messages').scrollHeight;
}
function chips(items){
  $('#chips').innerHTML=items.map(x=>`<button class="chip">${esc(x)}</button>`).join('');
  document.querySelectorAll('.chip').forEach(b=>b.addEventListener('click',async()=>{say('user',b.textContent);await ask(b.textContent);}));
}

async function ask(text){
  try{
    const r = await api('/api/chat',{method:'POST',body:JSON.stringify({message:text,state:{artist:state.artist,mode:state.mode}})});
    say('ai',r.text||'');
    if(r.artist && r.artist!==state.artist && state.artists.some(a=>a.name===r.artist)){
      state.artist = state.artists.find(a=>a.name===r.artist).id; renderArtists();
    }
    const a=r.action;
    if(a && a.type==='landscape'){ setMode('landscape'); state.count=a.count||4; await generate(); }
    if(a && a.type==='tree'){ state.mode='tree'; state.kind=a.kind||'pine'; document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b.dataset.mode==='tree')); renderOptions(); await generate(); }
    if(a && a.type==='flower'){ state.mode='flower'; state.kind=a.kind||'plum'; document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b.dataset.mode==='flower')); renderOptions(); await generate(); }
  }catch(e){ say('ai','助手暂时离线：'+e.message); }
}

window.addEventListener('DOMContentLoaded',boot);
