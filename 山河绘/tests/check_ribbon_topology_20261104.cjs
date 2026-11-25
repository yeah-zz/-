const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/ribbon_topology_refinement_20261104';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const summary=read('results.json'),train=read('training_summary.json'),protocol=read('protocol.json');
 assert.equal(train.completed_updates,600);assert(train.gpu.includes('5060'));assert.equal(train.tensor_device,'cuda:0');
 assert.equal(train.annotation_unchanged,true);assert.equal(sha(path.join(root,'model.pt')),train.model_sha256);
 assert.equal(sha(protocol.annotation_file),protocol.annotation_sha256);assert(protocol.excluded_uncertain_or_unreviewed.length>0);
 assert.equal(summary.network_applied_to_outputs,false);assert.equal(summary.network_calibration_passed,false);
 const d=read('G03_01_candidates.json');assert.equal(d.ready_for_robot,false);assert.equal(d.physical_pressure_mm,null);
 assert.equal(sha(d.parent_file),d.parent_sha256);assert.equal(sha(d.source),d.source_sha256);
 const ids=d.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b);
 const parent=JSON.parse(fs.readFileSync(d.parent_file,'utf8'));
 assert.deepEqual(ids,parent.original_curves.map((_,i)=>i));
 assert(d.curves.every(c=>c.points.length>=2&&c.points.flat().every(Number.isFinite)));
 assert(d.curves.every(c=>c.points.slice(1).some((p,k)=>Math.hypot(p[0]-c.points[k][0],p[1]-c.points[k][1])>0)));
 for(const c of d.cycle_diagnostics.small_cycle_candidates.filter(c=>c.eligible)){
  assert(c.hole_brightness_delta<=.025);assert(c.external_branch_vertices>=1);
 }
 const t=read('T02_native_candidates.json');assert.deepEqual(t.original_native_size,[1254,1254]);
 assert.equal(t.ready_for_robot,false);assert.equal(t.physical_pressure_mm,null);assert.equal(t.new_center_network_applied,false);
 for(const row of t.fine_diagnostics.accepted){assert(row.source_bilateral_support>=.8);assert(row.tree_route_fraction>=.85);assert(row.observed_width_fraction>=.85);}
 assert.equal(summary.native_tree_added_candidates,t.new_fine_candidates.length);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1480,height:1100},args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  await p.goto('file:///'+root+'/index.html');
  await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await p.locator('img').count(),3);
  const links=await p.locator('a').evaluateAll(a=>a.map(x=>x.getAttribute('href')));
  for(const href of links)assert(fs.existsSync(path.join(root,href)));
  assert((await p.locator('body').innerText()).includes('未通过校准'));
  await p.screenshot({path:path.join(root,'browser_report.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,
   gpu_training_checkpoint_verified:true,uncertain_labels_excluded:true,source_files_unchanged:true,
   rejected_network_not_applied:true,old_source_id_partition:true,native_image_size_verified:true,
   local_comparisons:3,artifact_links:links.length,errors,
   artist_stroke_accuracy_established:false,physical_ink_improvement_verified:false},null,2));
  console.log('PASS: new GPU checkpoint/provenance, held network gate, source geometry, native fine gates, 3 local reports.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
