const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/rock_context_joint_20261109',read=n=>JSON.parse(fs.readFileSync(path.join(root,n),'utf8'));
(async()=>{
 const m=read('data_manifest.json'),e=read('evaluation_report.json'),t=read('training_report.json'),v=read('verification.json'),p=read('protocol_checks.json'),d=read('decision.json');
 assert.equal(m.real_train_works,1);assert.equal(m.structural_visible_segments,4);assert.equal(m.motion_curve_candidates,4);assert.equal(m.expert_verified_strokes,0);assert.equal(m.unlabelled_pixels_are_negative,false);
 assert.equal(t.input,'completed static RGB module only');assert.equal(t.procedural_is_real_cunfa,false);assert.equal(t.named_cunfa_recognition_trained,false);assert.equal(t.postprocess_geometry_stitching,false);
 assert(v.saved_checkpoint_replayed&&v.approved_plans_unchanged&&v.all_joint_heads_trained);assert(p.unknown_anchor_labels_ignored&&p.unknown_instance_pixels_ignored&&p.unknown_curve_points_ignored);
 for(const scene of e.real){assert.equal(scene.manual_stroke_seeds_at_inference,false);assert.equal(scene.manual_module_box_at_input,true);let roles=scene.drawing_order.map(i=>scene.predictions[i].role);assert(roles.every((r,i)=>i===0||r>=roles[i-1]));assert(scene.predictions.every(p=>p.curve.length===24&&p.curve.every(x=>x.every(Number.isFinite))))}
 assert.equal(d.adopted,false);assert.equal(e.verified_author_blind_test_works,0);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1460,height:1050},args:['--disable-background-networking','--no-first-run','--disk-cache-dir='+path.join(root,'qa_cache')]});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',x=>errors.push(String(x)));const url=read('server.json').url;await page.goto(url);const body=await page.locator('body').textContent();
  assert(body.includes('整块山石'));assert(body.includes('不是'+t.procedural_unique_scenes+'幅真实皴法画'));assert(body.includes('本轮由人选定山石模块范围'));assert(body.includes('没有训练整幅画自动分石块'));assert(body.includes('尚未验收'));
  for(const im of await page.locator('img').all())await im.scrollIntoViewIfNeeded();await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  await page.locator('#play').click();const n=e.real[0].predicted_count;
  if(n)await page.waitForFunction(n=>document.querySelector('#status').textContent.includes('已播放 '+n+' / '+n),n,{timeout:25000});
  await page.locator('#scene').selectOption('same_work_right_rock');await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('已播放 0 /'));
  assert.equal((await page.request.post(url+'api/move',{data:{}})).status(),405);assert.equal((await page.request.get(url+'whole_context_step1250.pt')).status(),404);assert.equal((await page.request.get(url+'../data_manifest.json')).status(),200);
  await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'browser_checks.json'),JSON.stringify({passed:true,images_loaded:true,prediction_replay_works:true,predicted_roles_scheduled_contours_first:true,manual_module_selection_disclosed:true,no_oracle_stroke_seed:true,real_partial_and_procedural_limits_disclosed:true,no_robot_endpoint:true,errors},null,2));console.log('PASS: whole-module joint model report, role ordering, prediction replay, sparse labels and offline boundaries');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
