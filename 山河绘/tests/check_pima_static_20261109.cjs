const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/pima_static_discovery_20261109',read=n=>JSON.parse(fs.readFileSync(path.join(root,n),'utf8'));
(async()=>{
 const v=read('verification.json'),r=read('static_training_report.json'),g=read('guarded_discovery_report.json'),d=read('decision.json'),url=read('server.json').url;
 assert(v.checkpoint_raw_rollout_reproduced&&v.approved_plans_unchanged&&v.blank_source_does_not_create_curve);assert.equal(r.real_training_events,2);assert.equal(r.augmented_training_variants,10);assert.equal(r.external_sources_read_by_training,false);
 assert(g.external_is_now_development_not_blind_test);assert.equal(d.adopted,false);assert.equal(d.actual_new_expert_verified_training_strokes,0);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1360,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));await p.goto(url);await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  const text=await p.locator('body').textContent();assert(text.includes('完整画面提取和皴法复刻仍未通过'));assert(text.includes('不再称独立盲测'));assert(text.includes('不是 10 条真实新笔触'));assert(text.includes('几何墨带预览'));
  assert.equal((await p.request.post(url+'api/move',{data:{}})).status(),405);assert.equal((await p.request.get(url+'static_tracer.pt')).status(),404);assert.deepEqual(errors,[]);
  await p.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,images_loaded:true,real_vs_synthetic_disclosed:true,external_revision_disclosed:true,no_robot_endpoint:true,errors},null,2));console.log('PASS: static prototype report, raw failures, data provenance, no-controller server');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
