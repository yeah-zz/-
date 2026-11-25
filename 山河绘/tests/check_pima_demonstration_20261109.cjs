const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/pima_demonstration_20261109';
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name),'utf8'));
(async()=>{
 const url=read('server.json').url,report=read('next_group_report.json'),verification=read('verification.json');
 assert(verification.checkpoint_replay);assert.equal(report.independent_test_groups,0);assert.equal(report.adopted,false);
 assert.equal(report.complete_curve_prediction,false);assert.equal(report.phase_input_is_assistant_coarse_hint_not_model_recognition,true);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1360,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  const first=await page.locator('#frame').getAttribute('src');
  await page.locator('#step').fill('4');await page.locator('#step').dispatchEvent('input');
  await page.waitForFunction(()=>document.getElementById('frame').complete&&document.getElementById('frame').naturalWidth>0);
  const fourth=await page.locator('#frame').getAttribute('src');assert.notEqual(first,fourth);
  await page.locator('#play').click();await page.waitForFunction(previous=>document.getElementById('frame').getAttribute('src')!==previous,fourth);
  await page.locator('#play').click();assert((await page.locator('body').textContent()).includes('完整皴法复刻仍未通过'));
  assert.equal((await page.request.post(url+'api/move',{data:{}})).status(),405);
  assert.equal((await page.request.get(url+'next_group_pilot.pt')).status(),404);assert.deepEqual(errors,[]);
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,images_loaded:true,slider_changes_observed_frame:true,playback_changes_frame:true,
   failure_status_visible:true,no_robot_endpoint:true,page_errors:errors},null,2));
  console.log('PASS: observed-process playback, honest failure disclosure, no controller routes, checkpoint replay');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
