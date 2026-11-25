const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/pima_single_stroke_20261109';
const read=n=>JSON.parse(fs.readFileSync(path.join(root,n),'utf8'));
(async()=>{
 const url=read('server.json').url,v=read('verification.json'),r=read('single_curve_training_report.json'),m=read('single_reference_manifest.json');
 assert(v.checkpoint_full_rollout_replayed&&v.approved_plans_unchanged);assert.equal(r.real_training_events,2);assert.equal(r.same_rock_development_events,1);assert.equal(r.independent_test_works,0);
 assert.equal(m.records.filter(x=>x.status==='admitted_candidate').length,3);assert.equal(read('decision.json').adopted,false);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1360,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));await page.goto(url);
  await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  const before=await page.locator('#state').textContent();await page.locator('#play').click();await page.waitForFunction(prev=>document.getElementById('state').textContent!==prev,before);await page.locator('#play').click();
  await page.locator('#case').selectOption('2');assert((await page.locator('#state').textContent()).includes('112'));
  const body=await page.locator('body').textContent();assert(body.includes('并非模型自主检测'));assert(body.includes('完整皴法复刻仍未通过'));
  assert.equal((await page.request.post(url+'api/move',{data:{}})).status(),405);assert.equal((await page.request.get(url+'single_curve_tracer.pt')).status(),404);assert.deepEqual(errors,[]);
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,images_loaded:true,trajectory_playback:true,source_conditioning_disclosed:true,no_robot_control:true,errors},null,2));console.log('PASS: full-curve replay, source/seed limits, excluded labels, read-only report');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
