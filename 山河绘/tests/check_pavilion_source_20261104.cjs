const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/pavilion_source_strokes_20261104';
(async()=>{
 const report=JSON.parse(fs.readFileSync(path.join(root,'results.json'),'utf8'));
 assert.equal(report.model_weights_trained,false);assert.equal(report.robot_connected,false);
 for(const id of ['pavilion','pine','rock']){
  const a=JSON.parse(fs.readFileSync(path.join(root,id,'candidate_strokes.json'),'utf8'));
  const s=JSON.parse(fs.readFileSync(path.join(root,id,'statistics.json'),'utf8'));
  const ids=a.new_candidates.flatMap(r=>r.source_curve_ids).sort((x,y)=>x-y);
  assert.deepEqual(ids,a.original_paths.map((_,i)=>i));
  assert.equal(a.ready_for_robot,false);assert.equal(a.physical_pressure_mm,null);
  assert.equal(s.current_brush_calibrated,false);
  assert(a.width_profiles.every(p=>p.pressure_mm===null));
  if(id==='pavilion')assert(a.width_profiles.every(p=>p.normal_recentering_applied===false));
  else assert.equal(a.new_candidates.length,a.original_paths.length);
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1500,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  await p.goto('file:///'+root+'/index.html');
  await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await p.locator('img').count(),3);
  const links=await p.locator('a').evaluateAll(a=>a.map(x=>x.getAttribute('href')));
  for(const href of links)assert(fs.existsSync(path.join(root,href)));
  assert.deepEqual(errors,[]);
  await p.screenshot({path:path.join(root,'browser_preview.png'),fullPage:true});
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,
   local_comparison_images:3,artifact_links:links.length,source_ids_partitioned:true,
   no_pressure_or_robot_commands:true,architecture_not_applied_to_controls:true,
   image_width_accuracy_verified:false,physical_improvement_verified:false,errors},null,2));
  console.log('PASS: report images/links, source ID partition, controls and offline-only metadata.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
