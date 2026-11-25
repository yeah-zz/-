const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/hierarchical_painting_20261105';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const catalog=read('catalog.json'),url=read('server.json').url;
 for(const row of catalog.scenes){
  const p=read(row.id+'/plan.json');assert.equal(hash(path.join(root,row.id,'plan.json')),row.plan_sha256);
  assert.equal(hash(p.source),p.source_sha256);assert.equal(hash(p.parent),p.parent_sha256);
  assert.equal(p.ready_for_robot,false);assert.equal(p.learned_relation_filter_applied,false);
  assert.equal(new Set(p.actions.map(a=>a.curve)).size,p.curves.length);
  const sources=p.curves.flatMap(c=>c.source_curve_ids).sort((a,b)=>a-b);
  assert.deepEqual(sources,Array.from({length:p.stats.before.count},(_,i)=>i));
  assert.equal(p.stats.hierarchy_order.module_reentries,0);
  let seen=new Set(),last=null,lastPhase=-1;
  for(const a of p.actions){if(a.module!==last){assert(!seen.has(a.module));seen.add(a.module);last=a.module;lastPhase=-1}assert(a.phase>=lastPhase);lastPhase=a.phase}
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1540,height:1120},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[],external=[];page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith(url))external.push(r.url())});
  await page.goto(url);await page.waitForFunction(()=>window.document.getElementById('counts').textContent.includes('1080'));
  await page.waitForFunction(()=>sourceImage.complete&&sourceImage.naturalWidth>0);
  assert.equal(await page.locator('canvas').count(),2);
  await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);
  await page.locator('#play').click();const stopped=await page.evaluate(()=>cursor);
  await page.waitForTimeout(200);assert.equal(await page.evaluate(()=>cursor),stopped);
  await page.locator('#next').click();assert(await page.evaluate(()=>cursor)>stopped);
  await page.locator('#seek').evaluate(e=>{e.value=160;e.dispatchEvent(new Event('input'))});
  await page.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await page.locator('#seek').evaluate(e=>{e.value=1000;e.dispatchEvent(new Event('input'))});
  assert.equal(await page.locator('#percent').textContent(),'100%');
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert.equal(await page.locator('#percent').textContent(),'0%');assert(await page.locator('#play').isDisabled());
  await page.screenshot({path:path.join(root,'execution_waiting.png'),fullPage:true});
  await page.locator('#offline').click();
  for(const row of catalog.scenes){await page.selectOption('#scene',row.id);await page.waitForFunction(id=>scene===id,row.id);assert((await page.locator('#failure').textContent())==='')}
  const forbidden=await page.request.post(url+'api/move',{data:{}});assert.equal(forbidden.status(),405);
  const traversal=await page.request.get(url+'source_snapshot/hierarchical_paint_server_20261105.py');assert.equal(traversal.status(),404);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  fs.writeFileSync(path.join(root,'platform_verification.json'),JSON.stringify({passed:true,scene_count:3,
   provenance_partition:true,module_reentries:0,phase_order_verified:true,seek_play_pause_and_module_jump:true,
   offline_not_mixed_with_execution:true,no_robot_commands:true,page_errors:errors,external_requests:external,
   artistic_accuracy_established:false,physical_improvement_verified:false},null,2));
  console.log('PASS: all plans, module locks, phase order, playback, unbound real execution mode, HTTP read-only boundaries');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
