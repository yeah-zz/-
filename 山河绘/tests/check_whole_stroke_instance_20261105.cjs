const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const train='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/whole_stroke_instance_20261105',root=train+'/comparison';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const d=read('decision.json'),catalog=read('catalog.json');assert.equal(d.adopted,false);assert.equal(d.experiment_passed,false);
 assert.equal(d.accepted_whole_replacements,0);assert.equal(d.model_updates,4800);
 for(const suffix of ['', '/ownership_attention']){const p=train+suffix,m=JSON.parse(fs.readFileSync(path.join(p,'training_summary.json'),'utf8'));
  assert.equal(m.updates,2400);assert(m.gpu.includes('5060'));assert.equal(sha(path.join(p,'instance_curve.pt')),m.model_sha256)}
 for(const s of catalog.scenes){const p=read(s.id+'/plan.json'),old=JSON.parse(fs.readFileSync(p.parent,'utf8'));
  assert.equal(sha(p.parent),p.parent_sha256);assert.equal(sha(path.join(root,s.id,'plan.json')),s.plan_sha256);
  assert.deepEqual(p.curves,old.curves);assert.deepEqual(p.actions,old.actions);
  assert.equal(p.new_instance_model_applied,false);assert.equal(p.ready_for_robot,false);
  assert.equal(p.stats.before.count,p.stats.after.count);assert.equal(p.stats.accepted_continuations,0);
 }
 for(const id of ['G03_01','G05_03','T02'])for(const model of ['first','attention']){
  const raw=read(id+'/'+model+'_raw_curves.json');assert.equal(raw.reference_labels_read_at_inference,false);
  assert.equal(raw.postprocessing_stitching_used,false);assert(raw.proposals.every(p=>p.points.length===33));
 }
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[],url=read('server.json').url;page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>scene==='G03_01_500'&&sourceImage.complete&&sourceImage.naturalWidth>0);
  assert.equal(await page.locator('#counts').textContent(),'500 → 500');assert.equal(await page.locator('#joins').textContent(),'0');
  await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);await page.locator('#play').click();
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert.equal(await page.locator('#percent').textContent(),'0%');
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await page.locator('img').count(),4);const text=await page.locator('body').innerText();assert(text.includes('未证明超过'));assert(text.includes('参考短提示与参考框'));
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,trained_updates:4800,
   checkpoints_hash_verified:true,approved_500_700_geometry_and_actions_unchanged:true,raw_33point_outputs_separate:true,
   no_geometric_stitching_claimed_as_learning:true,experiment_failed_and_not_adopted:true,
   reference_conditioned_diagnostic_disclosed:true,live_not_bound:true,page_errors:errors},null,2));
  console.log('PASS: neural checkpoints, direct ownership/33-point outputs, unchanged baselines, explicit failed gate and report');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
