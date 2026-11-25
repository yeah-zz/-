const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/long_stroke_recovery_20261105';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const length=p=>p.slice(1).reduce((s,q,i)=>s+Math.hypot(q[0]-p[i][0],q[1]-p[i][1]),0);
(async()=>{
 const catalog=read('catalog.json'),url=read('server.json').url;
 for(const item of catalog.scenes){
  const p=read(item.id+'/plan.json'),old=JSON.parse(fs.readFileSync(p.parent,'utf8')),d=read(item.id+'/diagnostics.json');
  assert.equal(sha(p.parent),p.parent_sha256);assert.equal(sha(path.join(root,item.id,'plan.json')),item.plan_sha256);
  const ids=p.curves.flatMap(r=>r.complete_input_ids).sort((a,b)=>a-b);
  assert.deepEqual(ids,Array.from({length:old.curves.length},(_,i)=>i));
  assert.deepEqual(p.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b),old.curves.flatMap(r=>r.source_curve_ids).sort((a,b)=>a-b));
  assert.equal(p.ready_for_robot,false);assert.equal(p.new_weights_trained,false);
  assert(d.accepted.every(a=>old.curves[a.i].module===old.curves[a.j].module));
  const expected=old.curves.reduce((s,r)=>s+length(r.points),0)+d.accepted.reduce((s,a)=>s+length(a.bridge)-a.trim_a-a.trim_b,0);
  const measured=p.curves.reduce((s,r)=>s+length(r.points),0);assert(Math.abs(expected-measured)<1e-5);
  assert.equal(p.actions.length,p.curves.length);assert.equal(new Set(p.actions.map(a=>a.curve)).size,p.curves.length);
  assert.equal(p.stats.hierarchy_order.module_reentries,0);assert(p.stats.old_sample_distance_quantiles_px[3]<.8);
 }
 const synthetic=read('fresh_synthetic_controls.json');assert.equal(synthetic.totals.cases,280);
 assert.equal(synthetic.totals.wrong_merged_groups,0);assert(synthetic.totals.recovered_pairs<synthetic.totals.expected_pairs);
 assert.equal(synthetic.real_stroke_precision,null);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1540,height:1120},
  args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>scene==='G03_01'&&sourceImage.complete&&sourceImage.naturalWidth>0);
  await page.locator('#play').click();await page.waitForFunction(()=>cursor>1);await page.locator('#play').click();
  await page.locator('#seek').evaluate(e=>{e.value=120;e.dispatchEvent(new Event('input'))});
  await page.screenshot({path:path.join(root,'platform_preview.png'),fullPage:true});
  await page.locator('#live').click();await page.waitForFunction(()=>document.getElementById('feed').textContent.includes('尚未绑定'));
  assert.equal(await page.locator('#percent').textContent(),'0%');
  await page.goto(url+'report.html');await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await page.locator('img').count(),3);
  const body=await page.locator('body').innerText();assert(body.includes('漏接仍明显'));assert(body.includes('没有训练新的像素提取模型'));
  const links=await page.locator('a').evaluateAll(a=>a.map(x=>x.href));for(const link of links){const response=await page.request.get(link);assert.equal(response.status(),200)}
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'verification.json'),JSON.stringify({passed:true,full_source_hashes_unchanged:true,
   every_input_curve_accounted_for:true,no_detail_omission:true,arc_length_conservation_verified:true,
   no_cross_module_joins:true,synthetic_controls_only:true,real_stroke_accuracy_verified:false,
   live_not_bound:true,page_errors:errors,physical_improvement_verified:false},null,2));
  console.log('PASS: provenance, arc-length conservation, source coverage, synthetic limits and offline platform/report');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
