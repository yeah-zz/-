const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/landscape_stroke_refinement_20261104';
(async()=>{
 const summary=JSON.parse(fs.readFileSync(path.join(root,'results.json'),'utf8'));
 assert.equal(summary.model_weights_trained,false);assert.equal(summary.robot_connected,false);
 const cases=['G03_01','G05_03','T02'];
 for(const id of cases){
  const d=JSON.parse(fs.readFileSync(path.join(root,id,'offline_candidates.json'),'utf8'));
  const stats=JSON.parse(fs.readFileSync(path.join(root,id,'statistics.json'),'utf8'));
  const ids=d.curves.flatMap(c=>c.source_curve_ids).sort((a,b)=>a-b);
  assert.deepEqual(ids,d.original_curves.map((_,i)=>i));
  assert(d.curves.every(c=>c.points.length>=2&&c.points.flat().every(Number.isFinite)));
  assert(d.curves.every(c=>c.points.slice(1).some((p,k)=>Math.hypot(p[0]-c.points[k][0],p[1]-c.points[k][1])>0)));
  assert.equal(d.ready_for_robot,false);assert.equal(d.pressure_mm,null);assert.equal(d.robot_connected,false);
  assert(d.width_profiles.every(p=>p.pressure_mm===null&&p.normal_recentering_applied===false));
  assert.deepEqual(d.actions.map(a=>a.curve).sort((a,b)=>a-b),d.curves.map((_,i)=>i));
  const seen=new Set();let prior=null;
  for(const a of d.actions){assert.equal(a.module,d.curves[a.curve].module);if(a.module!==prior){assert(!seen.has(a.module));seen.add(a.module);prior=a.module;}}
  assert.equal(stats.input_provenance_partition_verified,true);
  assert(stats.original_sample_distance_to_new_px_quantiles[3]<=1.5);
  assert(stats.candidate.count<=stats.original.count);
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1450,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  await p.goto('file:///'+root+'/index.html');
  await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await p.locator('img').count(),6);
  const links=await p.locator('a').evaluateAll(a=>a.map(x=>x.getAttribute('href')));
  for(const href of links)assert(fs.existsSync(path.join(root,href)));
  await p.screenshot({path:path.join(root,'browser_report.png'),fullPage:true});
  for(const id of cases){
   await p.goto('file:///'+root+'/'+id+'/sequence.html');
   await p.locator('#next').click();assert((await p.locator('#status').innerText()).includes('1 /'));
   await p.locator('#source').check();await p.waitForFunction(()=>bg.complete&&bg.naturalWidth>0);
   await p.locator('#step').evaluate(e=>{e.value=e.max;e.dispatchEvent(new Event('input'))});
   const max=await p.locator('#step').getAttribute('max');assert((await p.locator('#status').innerText()).includes(max+' / '+max));
   await p.locator('#reset').click();await p.locator('#play').click();await p.waitForFunction(()=>+document.getElementById('step').value>=2);await p.locator('#play').click();
   if(id==='G03_01')await p.screenshot({path:path.join(root,'browser_sequence.png'),fullPage:true});
  }
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,geometry_cases:cases,
   source_id_partition:true,nonzero_curves:true,each_candidate_scheduled_once:true,module_reentries:0,
   same_display_width_comparisons:6,interactive_sequences:3,errors,
   stroke_identity_accuracy_verified:false,physical_drawing_improvement_verified:false},null,2));
  console.log('PASS: 3 scene provenance/geometry/schedules, 6 comparisons, 3 interactive sequences.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
