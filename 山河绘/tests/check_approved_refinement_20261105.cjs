const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/approved_baseline_refinement_20261105';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const catalog=read('catalog.json'),adoption=read('neural_adoption.json');assert.equal(adoption.adopted,false);assert.equal(adoption.new_model_updates,4400);
 for(const s of catalog.scenes){
  const p=read(s.id+'/plan.json'),old=JSON.parse(fs.readFileSync(p.parent,'utf8'));
  assert.equal(sha(p.parent),p.parent_sha256);assert.equal(sha(path.join(root,s.id,'plan.json')),s.plan_sha256);
  assert.equal(p.new_neural_model_applied,false);assert.equal(p.ready_for_robot,false);assert.equal(p.no_additional_detail_omission,true);
  assert.deepEqual(p.curves.flatMap(r=>r.complete_input_ids).sort((a,b)=>a-b),Array.from({length:old.curves.length},(_,i)=>i));
  assert.deepEqual(p.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b),old.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b));
  assert(p.stats.baseline_max_sample_distance_px<.8);assert.equal(p.stats.hierarchy_order.module_reentries,0);
  assert.equal(new Set(p.actions.map(a=>a.curve)).size,p.curves.length);
 }
 const neural='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/local_curve_policy_20261105';
 for(const suffix of ['', '/recovery_round2']){
  const folder=neural+suffix,t=JSON.parse(fs.readFileSync(path.join(folder,'training_summary.json'),'utf8'));
  assert.equal(t.updates,2200);assert(t.gpu.includes('5060'));assert.equal(sha(path.join(folder,'policy.pt')),t.model_sha256);
 }
 const url=read('server.json').url,ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{
  headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>scene==='G03_01_500'&&sourceImage.complete&&sourceImage.naturalWidth>0);
  assert((await page.locator('#counts').textContent()).includes('486'));
  await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);await page.locator('#play').click();
  await page.locator('#seek').evaluate(e=>{e.value=1000;e.dispatchEvent(new Event('input'))});
  await page.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert.equal(await page.locator('#percent').textContent(),'0%');
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  const body=await page.locator('body').innerText();assert(body.includes('当前可见改动不来自新模型'));assert(body.includes('未采用'));
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,approved_inputs_unchanged:true,
   no_extra_detail_omission:true,curve_provenance_verified:true,module_reentry:false,
   trained_updates:4400,neural_model_not_adopted:true,physical_quality_verified:false,page_errors:errors},null,2));
  console.log('PASS: 6 conservative variants, immutable approved inputs, 2 archived GPU checkpoints and honest model non-adoption');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
