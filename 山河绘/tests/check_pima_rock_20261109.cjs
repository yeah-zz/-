const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/pima_one_rock_cycle_20261109';
const read=n=>JSON.parse(fs.readFileSync(path.join(root,n),'utf8'));
(async()=>{
 const manifest=read('reference_manifest.json'),train=read('training_report.json'),decision=read('decision.json'),v=read('verification.json');
 assert.equal(manifest.records.filter(r=>r.status==='reviewed_candidate').length,4);
 assert.equal(manifest.records.filter(r=>r.status==='uncertain_excluded').length,3);
 assert.equal(manifest.complete_rock_reference,false);assert.equal(manifest.unlabelled_pixels_are_negative,false);
 assert.equal(train.real_training_events,4);assert.equal(train.instance_ownership_head_trained,false);assert.equal(train.named_cunfa_classifier_trained,false);
 assert(v.checkpoint_raw_rollout_reproduced&&v.approved_plans_unchanged&&v.autonomous_discovery_not_given_reference_seed);
 assert.equal(v.verified_author_disjoint_test_works,0);assert.equal(decision.adopted,false);
 const url=read('server.json').url,ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1360,height:1000},args:['--disable-background-networking','--no-first-run','--disk-cache-dir='+path.join(root,'qa_cache')]});
 try{
  const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(String(e)));await p.goto(url);const text=await p.locator('body').textContent();
  assert(text.includes('整块山石闭环尚未通过'));assert(text.includes('不是模型自主笔顺'));assert(text.includes('训练输入是54秒时的静态阶段图'));assert(text.includes('不称为独立盲测'));
  for(const image of await p.locator('img').all())await image.scrollIntoViewIfNeeded();
  await p.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await p.locator('img[src="tutorial_excerpt.gif"]').count(),1);assert(text.includes('不是原始帧率'));
  await p.locator('#step').fill('3');await p.locator('#step').dispatchEvent('input');assert((await p.locator('#caption').textContent()).includes('3 / 4'));
  await p.locator('#play').click();await p.waitForFunction(()=>document.querySelector('#caption').textContent.includes('4 / 4'));
  assert.equal((await p.request.post(url+'api/move',{data:{}})).status(),405);assert.equal((await p.request.get(url+'new_work_static_tracer.pt')).status(),404);
  const range=await p.request.get(url+'tutorial.mp4',{headers:{Range:'bytes=0-31'}});assert.equal(range.status(),206);assert.equal((await range.body()).length,32);
  await p.evaluate(()=>window.scrollTo(0,0));await p.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,images_loaded:true,reference_replay_works:true,sampled_tutorial_gif_loaded:true,original_fmp4_browser_demux_failed:true,range_requests_work:true,no_robot_endpoint:true,failures_and_limits_disclosed:true,errors},null,2));
  console.log('PASS: references, source split, raw failure disclosure, replay, ranged video, unchanged approved plans, offline-only report');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
