const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/cunfa_recognition_reproduction_20261107';
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
(async()=>{
 const url=read('server.json').url,v=read('verification.json'),d=read('decision.json');
 assert(v.passed);assert.equal(v.completed_updates,2200);assert.equal(d.adopted,false);assert.equal(d.real_style_accuracy_established,false);
 const c=read('classification_report.json');assert.equal(c.test.samples,9);assert.equal(c.expert_style_labels,0);
 const ctx=await chromium.launchPersistentContext(path.join(root,'qa_browser'),{headless:true,
  executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',viewport:{width:1440,height:1000},args:['--disable-background-networking','--no-first-run']});
 try{
  const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url);await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  assert.equal(await page.locator('img').count(),3);assert((await page.locator('body').textContent()).includes('未通过真实皴法可靠识别验收'));
  await page.screenshot({path:path.join(root,'report_preview.png'),fullPage:false});
  assert.deepEqual(errors,[]);assert.equal((await page.request.post(url+'api/move',{data:{}})).status(),405);
  assert.equal((await page.request.get(url+'classifier.pt')).status(),404);
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify({passed:true,images_loaded:3,
   failure_status_visible:true,model_files_not_served:true,no_motion_endpoint:true,page_errors:errors},null,2));
  console.log('PASS: report images, disclosed real-image failure, checkpoint verification, isolated read-only service');
 }finally{await ctx.close()}
})().catch(e=>{console.error(e);process.exit(1)});
