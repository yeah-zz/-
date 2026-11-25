const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/approved_execution_refine_20261107';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const catalog=read('catalog.json'),url=read('server.json').url;
 for(const scene of catalog.scenes){
  const p=read(scene.id+'/plan.json'),old=JSON.parse(fs.readFileSync(p.parent,'utf8')),s=read(scene.id+'/summary.json');
  assert.equal(sha(p.parent),p.parent_sha256);assert.equal(sha(path.join(root,scene.id,'plan.json')),scene.plan_sha256);
  assert.deepEqual(p.curves.flatMap(r=>r.accepted_baseline_ids).sort((a,b)=>a-b),Array.from({length:old.curves.length},(_,i)=>i));
  assert.deepEqual(p.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b),old.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b));
  assert.equal(s.detail_budget_removals,0);assert.equal(p.ready_for_robot,false);assert.equal(p.new_weights_trained,false);
  assert(s.baseline_max_deviation_px<=1.3);assert.equal(p.actions.length,p.curves.length);
  assert.equal(new Set(p.actions.map(a=>a.curve)).size,p.curves.length);
  const seen=new Set();let last=null,phase=-1;
  for(const a of p.actions){if(a.module!==last){assert(!seen.has(a.module));seen.add(a.module);last=a.module;phase=-1}assert(a.phase>=phase);phase=a.phase}
  assert(s.after_risk['1.2'].starts_touching_last12<=s.same_curve_set_control_risk['1.2'].starts_touching_last12);
  for(const w of ['0.6','1.2','2.0']){
   assert(Math.abs(s.after_risk[w].distinct_stroke_overlap_area_px-s.same_curve_set_control_risk[w].distinct_stroke_overlap_area_px)<1e-8);
  }
  const profiles=read(scene.id+'/width_targets.json');assert.equal(profiles.pressure_mm,null);
  assert.equal(profiles.profiles.length,p.curves.length);
  assert(profiles.profiles.every(r=>r.pressure_mm===null&&!r.normal_recentering_applied&&r.observed.length===r.width_px.length));
 }
 assert.equal(read('execution_binding.json').journal,null);
 const context=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>scene==='G03_01_500'&&sourceImage.naturalWidth>0);
  await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);await page.locator('#play').click();
  const stop=await page.evaluate(()=>cursor);await page.waitForTimeout(150);assert.equal(await page.evaluate(()=>cursor),stop);
  await page.locator('#next').click();assert(await page.evaluate(()=>cursor)>stop);
  await page.locator('#seek').evaluate(e=>{e.value=220;e.dispatchEvent(new Event('input'))});
  await page.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert(await page.locator('#play').isDisabled());assert.equal(await page.locator('#percent').textContent(),'0%');
  await page.locator('#offline').click();
  for(const s of catalog.scenes){await page.selectOption('#scene',s.id);await page.waitForFunction(id=>scene===id,s.id)}
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await page.locator('img').count(),3);assert.deepEqual(errors,[]);
  assert.equal((await page.request.post(url+'api/move',{data:{}})).status(),405);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,plans:catalog.scenes.length,
   accepted_input_geometry_and_provenance_preserved:true,no_detail_budget_removal:true,
   source_hashes_unchanged:true,object_phase_order_verified:true,
   contact_comparison_uses_same_curve_set:true,order_does_not_remove_geometric_overlap:true,
   no_pressure_mapping_or_robot_binding:true,page_errors:errors,physical_improvement_verified:false},null,2));
  console.log('PASS: approved geometry, provenance, phase order, same-line-set contact control, width unknowns, playback and offline boundaries');
 }finally{await context.close()}
})().catch(e=>{console.error(e);process.exit(1)});
