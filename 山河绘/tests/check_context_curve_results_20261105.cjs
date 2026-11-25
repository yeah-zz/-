const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/context_curve_selection_20261105';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const summary=read('results.json'),training=read('training_summary.json'),second=read('center_round2/training_summary.json');
 assert.equal(training.completed_updates,1000);assert(training.gpu.includes('5060'));
 assert.equal(sha(path.join(root,'model.pt')),training.model_sha256);
 assert.equal(sha(path.join(root,'center_round2/model.pt')),second.model_sha256);
 assert.equal(summary.center_round2_adopted,false);assert.equal(summary.lower_joint_threshold_adopted,false);
 assert.equal(summary.release_to_robot,false);
 const rel=read('relations/training_summary.json');assert.equal(sha(path.join(root,'relations/model.pt')),rel.model_sha256);
 assert.equal(read('center_round2/adoption_decision.json').adopted,false);
 assert(read('relations/fresh_pipeline_test.json').joint_false_duplicate>0);
 const actual=read('actual_prediction_relation_audit.json');assert(actual.records.every(r=>r.passes_coverage_guard));
 for(const id of ['G03_01','G05_03','T02']){
  const d=read(id+'/candidate.json');assert.equal(d.ready_for_robot,false);assert.equal(d.physical_pressure_mm,null);
  assert.equal(sha(d.source),d.source_sha256);assert.equal(sha(d.parent),d.parent_sha256);
  assert.equal(d.model_sha256,training.model_sha256);assert.notEqual(d.model_sha256,second.model_sha256);
  assert(d.selected_input_ids.every(i=>i>=0&&i<d.input_rows.length));
  assert.equal(new Set(d.selected_input_ids).size,d.selected_input_ids.length);
  assert.deepEqual(d.output_paths.slice(0,d.selected_input_ids.length),d.selected_input_ids.map(i=>d.input_rows[i].points));
  assert(d.relation_selection.events.every(e=>d.relation_selection.selected.includes(e.representative)));
  assert.equal(d.labels_read_at_inference,false);
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  viewport:{width:1450,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  await p.goto('file:///'+root+'/index.html');
  await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await p.locator('img').count(),5);
  const text=await p.locator('body').innerText();assert(text.includes('尚未取得明显去杂突破'));assert(text.includes('误删8条'));
  const links=await p.locator('a').evaluateAll(a=>a.map(x=>x.getAttribute('href')));
  for(const href of links)assert(fs.existsSync(path.join(root,href)));
  await p.screenshot({path:path.join(root,'browser_report.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,
   trained_weights_hashes_verified:3,source_archives_unchanged:true,retained_geometry_unchanged:true,
   failed_round2_not_applied:true,lower_threshold_rejected:true,known_relation_error_disclosed:true,
   actual_prediction_reference_guard_verified:true,report_images:5,errors,
   independent_painting_accuracy_established:false,physical_improvement_verified:false},null,2));
  console.log('PASS: 3 checkpoints, archived sources, rejected variants, actual-output audit and 5 visual panels.');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
