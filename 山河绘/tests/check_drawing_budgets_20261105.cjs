const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/drawing_budgets_20261105';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const catalog=read('catalog.json'),url=read('server.json').url;
 for(const scene of ['G03_01','G05_03','T02']){
  const ledger=read(scene+'/selection.json'),full=JSON.parse(fs.readFileSync(ledger.complete_plan,'utf8'));
  assert.equal(sha(ledger.complete_plan),ledger.complete_sha256);
  assert.deepEqual(read(scene+'_full/plan.json').curves.map(r=>r.points),full.curves.map(r=>r.points));
  let prior=new Set();
  for(const budget of [300,500,700]){
   const p=read(scene+'_'+budget+'/plan.json'),s=ledger.selections[budget],chosen=new Set(s.selected_ids);
   assert([...prior].every(i=>chosen.has(i)));prior=chosen;
   assert.equal(p.curves.length,Math.min(full.curves.length,Math.max(budget,s.protected_count)));
   assert(s.protected_ids.every(i=>chosen.has(i)));
   p.curves.forEach(r=>assert.deepEqual(r.points,full.curves[r.complete_curve_id].points));
   assert.equal(new Set(p.actions.map(a=>a.curve)).size,p.curves.length);
   assert.equal(new Set(p.actions.map(a=>a.stroke_id)).size,p.curves.length);
   assert.equal(p.ready_for_robot,false);assert.equal(p.complete_extraction_unchanged,true);
   let seen=new Set(),last=null,phase=-1;
   for(const a of p.actions){if(a.module!==last){assert(!seen.has(a.module));seen.add(a.module);last=a.module;phase=-1}assert(a.phase>=phase);phase=a.phase}
  }
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  await p.goto(url);await p.waitForFunction(()=>scene==='G03_01_500'&&sourceImage.complete&&sourceImage.naturalWidth>0);
  assert.equal(await p.locator('#scene option').count(),12);
  assert((await p.locator('#scene option:checked').textContent()).includes('500笔预算'));
  await p.locator('#play').click();await p.waitForFunction(()=>cursor>=2);await p.locator('#play').click();
  await p.locator('#seek').evaluate(e=>{e.value=1000;e.dispatchEvent(new Event('input'))});
  assert.equal(await p.locator('#percent').textContent(),'100%');
  await p.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await p.locator('#live').click();await p.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert.equal(await p.locator('#percent').textContent(),'0%');await p.locator('#offline').click();
  for(const s of catalog.scenes){await p.selectOption('#scene',s.id);await p.waitForFunction(id=>scene===id,s.id);
   assert.equal((await p.locator('#failure').textContent()),'');
   const a=read(s.id+'/plan.json');assert.equal(sha(path.join(root,s.id,'plan.json')),s.plan_sha256);
  }
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,variants:12,
   full_sources_unchanged:true,selected_geometry_unchanged:true,protected_candidates_retained:true,
   nested_tiers:true,phase_order_and_module_lock:true,live_not_bound:true,page_errors:errors,
   new_long_stroke_reconstruction:false,artist_accuracy_verified:false,physical_quality_verified:false},null,2));
  console.log('PASS: 12 variants, nested budgets, protected curves, unchanged full inputs, replay and honest unbound live state');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
